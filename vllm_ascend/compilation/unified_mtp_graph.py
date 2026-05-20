# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Unified ACLGraph capture for MTP speculative decoding
#
# This module implements a unified ACLGraph capture that combines:
# - main_model.forward() - main model all layers
# - mtp_model.forward() - MTP decoder layers
# - compute_logits() - MTP logits
# - argmax() - draft token generation
#
# During graph replay, all operations are executed in one launch, eliminating
# the per-step stream sync that the legacy two-graph design needs.
#

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper


@dataclass
class UnifiedGraphOutput:
    hidden_states: torch.Tensor | IntermediateTensors
    logits: torch.Tensor | None = None
    draft_token_ids: torch.Tensor | None = None
    sample_hidden_states: torch.Tensor | None = None
    aux_hidden_states: torch.Tensor | None = None
    # Populated only by the safe runner / model_runner_v1 fallback path,
    # but defined here so both runners and downstream consumers share a
    # single type. ``error`` is the human-readable reason captured from the
    # exception that triggered the fallback.
    error: str | None = None
    fallback_used: bool = False


class UnifiedMTPGraphRunner:
    """Run main model + MTP draft model in a single ACLGraph capture.

    The wrapper produced by :class:`UnifiedMTPACLGraphWrapper` calls
    ``_run_unified_forward`` once per replay, so all operations (main forward,
    draft forward, ``compute_logits`` and ``argmax``) sit on the same NPU
    stream and there is no inter-graph synchronization.

    Note: this is the "fast path" runner without input validation or fallback.
    Production paths use :class:`UnifiedMTPGraphRunnerSafe` which adds
    pre-flight checks and a graceful fallback to the legacy two-graph path.
    """

    def __init__(
        self,
        main_model: nn.Module,
        draft_model: nn.Module,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.main_model = main_model
        self.draft_model = draft_model
        self.vllm_config = vllm_config
        self.device = device

        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
        )
        self.hidden_size = vllm_config.model_config.get_hidden_size()

        self._init_buffers()

    def _init_buffers(self):
        """Allocate persistent buffers shared across captures.

        ``passing_hidden_states`` must accommodate the worst-case combined
        token count for main + draft on a single step, which is
        ``max_num_batched_tokens + num_speculative_tokens * max_num_seqs``.
        Sized this way so a single buffer can be reused across all batch
        sizes in ``cudagraph_capture_sizes``.
        """
        max_num_tokens = (
            self.vllm_config.scheduler_config.max_num_batched_tokens
        )
        max_num_reqs = self.vllm_config.scheduler_config.max_num_seqs

        self.draft_token_ids_buffer = torch.zeros(
            max_num_reqs * self.num_speculative_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        self.passing_hidden_states = torch.zeros(
            max_num_tokens + self.num_speculative_tokens * max_num_reqs,
            self.hidden_size,
            dtype=self.vllm_config.model_config.dtype,
            device=self.device,
        )

    def _run_unified_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
        num_tokens: int,
        num_tokens_padded: int,
        logits_indices: torch.Tensor,
        draft_input_ids: torch.Tensor | None = None,
        draft_positions: torch.Tensor | None = None,
        draft_hidden_states: torch.Tensor | None = None,
        draft_token_indices_to_sample: torch.Tensor | None = None,
        draft_attn_metadata: Any | None = None,
        **model_kwargs,
    ) -> UnifiedGraphOutput:
        # NOTE: ``draft_kv_caches`` was previously accepted and assigned to
        # ``self.draft_model`` modules from inside this function. That is not
        # safe inside an ACLGraph capture (attribute assignment is not
        # replayable) and produced silent KV-cache corruption on replay.
        # KV-cache binding must happen *before* capture, in the model loader.
        hidden_states = self.main_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

        aux_hidden_states = None
        if isinstance(hidden_states, tuple) and len(hidden_states) > 1:
            hidden_states, aux_hidden_states = hidden_states

        if isinstance(hidden_states, IntermediateTensors):
            return UnifiedGraphOutput(
                hidden_states=hidden_states,
                logits=None,
                draft_token_ids=None,
                sample_hidden_states=None,
            )

        sample_hidden_states = hidden_states[logits_indices]

        draft_logits: torch.Tensor | None = None
        draft_token_ids: torch.Tensor | None = None

        if draft_hidden_states is not None and draft_input_ids is not None:
            draft_kwargs = {
                "input_ids": draft_input_ids[
                    : num_tokens + self.num_speculative_tokens
                ],
                "positions": (
                    draft_positions[
                        : num_tokens + self.num_speculative_tokens
                    ]
                    if draft_positions is not None
                    else positions
                ),
                "hidden_states": draft_hidden_states[
                    : num_tokens + self.num_speculative_tokens
                ],
            }

            if inputs_embeds is not None:
                draft_kwargs["inputs_embeds"] = inputs_embeds

            if draft_attn_metadata is not None:
                forward_context = get_forward_context()
                if forward_context is not None and hasattr(
                    forward_context, "attn_metadata"
                ):
                    draft_kwargs["attn_metadata"] = draft_attn_metadata

            draft_hidden_out = self.draft_model(**draft_kwargs)

            if isinstance(draft_hidden_out, tuple):
                draft_last_hidden, _ = draft_hidden_out
            else:
                draft_last_hidden = draft_hidden_out

            if draft_token_indices_to_sample is not None:
                draft_sample_hidden = draft_last_hidden[
                    draft_token_indices_to_sample
                ]
                draft_logits = self.draft_model.compute_logits(
                    draft_sample_hidden
                )
                draft_token_ids = draft_logits.argmax(dim=-1)

        return UnifiedGraphOutput(
            hidden_states=hidden_states,
            logits=draft_logits,
            draft_token_ids=draft_token_ids,
            sample_hidden_states=sample_hidden_states,
            aux_hidden_states=aux_hidden_states,
        )

    def create_runnable(self) -> Callable:
        """Return the callable that ACLGraphWrapper will capture/replay."""
        return self._run_unified_forward


class UnifiedMTPACLGraphWrapper(ACLGraphWrapper):
    """ACLGraph wrapper that captures the unified main + MTP forward."""

    def __init__(
        self,
        unified_runner: "UnifiedMTPGraphRunner | Any",
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode = CUDAGraphMode.FULL,
    ):
        # ``unified_runner`` may be either ``UnifiedMTPGraphRunner`` (fast
        # path) or ``UnifiedMTPGraphRunnerSafe`` (production path with
        # validation + fallback). Both expose ``create_runnable()``.
        if not hasattr(unified_runner, "create_runnable"):
            raise TypeError(
                "unified_runner must implement create_runnable(); "
                f"got {type(unified_runner).__name__}"
            )
        super().__init__(
            runnable=unified_runner.create_runnable(),
            vllm_config=vllm_config,
            runtime_mode=runtime_mode,
        )
        self.unified_runner = unified_runner


def create_unified_mtp_graph_wrapper(
    main_model: nn.Module,
    draft_model: nn.Module,
    vllm_config: VllmConfig,
    device: torch.device,
    safe: bool = True,
    pcp_size: int = 1,
    dcp_size: int = 1,
    use_compress: bool = False,
    is_pp_last_rank: bool = True,
) -> UnifiedMTPACLGraphWrapper | None:
    """Build a unified MTP ACLGraph wrapper if the configuration allows it.

    The eligibility check is delegated to
    :func:`unified_mtp_graph_safe.should_use_unified_graph` so this factory
    cannot disagree with the runtime gate inside ``model_runner_v1``.
    """
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.compilation.unified_mtp_graph_safe import (
        UnifiedMTPGraphRunnerSafe,
        should_use_unified_graph,
    )

    ascend_config = get_ascend_config(vllm_config)
    if not ascend_config.enable_unified_mtp_graph:
        logger.debug("Unified MTP graph disabled by configuration")
        return None

    can_use, reason = should_use_unified_graph(
        vllm_config=vllm_config,
        scheduler_output=None,
        num_tokens=0,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        use_compress=use_compress,
        is_pp_last_rank=is_pp_last_rank,
    )
    if not can_use:
        logger.debug("Unified MTP graph not enabled: %s", reason)
        return None

    logger.info(
        "Creating unified MTP ACLGraph wrapper (safe=%s) for "
        "single-launch execution",
        safe,
    )

    runner_cls = UnifiedMTPGraphRunnerSafe if safe else UnifiedMTPGraphRunner
    unified_runner = runner_cls(
        main_model=main_model,
        draft_model=draft_model,
        vllm_config=vllm_config,
        device=device,
    )

    return UnifiedMTPACLGraphWrapper(
        unified_runner=unified_runner,
        vllm_config=vllm_config,
        runtime_mode=CUDAGraphMode.FULL,
    )
