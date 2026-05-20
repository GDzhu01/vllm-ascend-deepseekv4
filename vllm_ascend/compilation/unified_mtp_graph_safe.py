# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Unified ACLGraph capture for MTP speculative decoding with comprehensive
# error handling. See ``unified_mtp_graph.py`` for the fast-path runner that
# this module wraps with input validation and graceful fallback.
#

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors

# Re-export UnifiedGraphOutput from the fast-path module so both runners
# produce the *same* dataclass type. Importing it twice (once here, once in
# unified_mtp_graph.py) used to break ``isinstance(output, UnifiedGraphOutput)``
# checks across the boundary.
from vllm_ascend.compilation.unified_mtp_graph import UnifiedGraphOutput

__all__ = [
    "UnifiedGraphOutput",
    "UnifiedMTPGraphError",
    "UnifiedMTPGraphRunnerSafe",
    "UnifiedMTPMemoryError",
    "GraphCaptureError",
    "GraphReplayError",
    "InputMismatchError",
    "EdgeCaseHandler",
    "should_use_unified_graph",
    "_is_oom_error",
]


class UnifiedMTPGraphError(Exception):
    """Base exception for unified MTP graph errors."""
    pass


class GraphCaptureError(UnifiedMTPGraphError):
    """Error during graph capture phase."""
    pass


class GraphReplayError(UnifiedMTPGraphError):
    """Error during graph replay phase."""
    pass


class InputMismatchError(UnifiedMTPGraphError):
    """Error when inputs don't match captured graph expectations."""
    pass


class UnifiedMTPMemoryError(UnifiedMTPGraphError):
    """Error due to insufficient memory for graph operations.

    Note: Renamed from ``MemoryError`` to avoid shadowing the built-in
    ``MemoryError`` and to make stack traces clearly attributable to the
    unified MTP graph runner.
    """
    pass


def _is_oom_error(exc: BaseException) -> bool:
    """Best-effort detection of NPU/CUDA out-of-memory errors.

    ``torch.cuda.OutOfMemoryError`` is unavailable on Ascend NPUs, so we fall
    back to a string match on the exception message which is robust across the
    CANN / torch-npu stack.
    """
    cuda_oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "oom" in msg
        or "rtmalloc" in msg
        or "rtmemcpy" in msg
    )


class UnifiedMTPGraphRunnerSafe:
    """
    Unified runner with comprehensive error handling and fallback mechanisms.
    
    Error Handling Strategy:
    1. Pre-flight checks: Validate inputs before graph operations
    2. Try-catch wrappers: Catch and handle runtime errors
    3. Fallback mechanisms: Gracefully degrade to separate execution
    4. Recovery: Clean up state after errors
    
    Edge Cases Covered:
    - Graph capture failures
    - Graph replay timeouts
    - Input shape mismatches
    - Memory allocation failures
    - Missing draft inputs
    - Attention metadata inconsistencies
    - Pipeline parallelism scenarios
    - Multimodal inputs
    - Async scheduling conflicts
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
        
        self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens
        self.hidden_size = vllm_config.model_config.get_hidden_size()
        
        self._init_buffers()
        self._init_error_state()
        
    def _init_buffers(self):
        """Initialize persistent buffers with memory safety checks.

        Buffer sizing rationale (must match
        :class:`UnifiedMTPGraphRunner._init_buffers`):

        ``max_num_batched_tokens + num_speculative_tokens * max_num_seqs``
        — accommodates the worst-case combined token count of main + draft
        on a single step. The runtime forward only ever slices
        ``[: num_tokens + num_speculative_tokens]``, so any oversize is
        wasted memory rather than a correctness issue, but undersizing
        would silently truncate draft inputs at replay time.
        """
        max_num_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.vllm_config.scheduler_config.max_num_seqs
        
        try:
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
        except RuntimeError as e:
            if _is_oom_error(e):
                logger.warning(f"Failed to allocate unified graph buffers: {e}")
                raise UnifiedMTPMemoryError(
                    f"Insufficient memory for unified graph buffers: {e}"
                ) from e
            raise
    
    def _init_error_state(self):
        """Initialize error tracking state."""
        self._capture_failed = False
        self._last_error = None
        self._fallback_mode = False
        self._replay_timeout_count = 0
        self._max_replay_timeout_count = 3
        self._fallback_attempts = 0
        # Stop retrying the unified graph after this many consecutive
        # ``reset_error_state`` cycles to avoid an infinite capture-fail/retry
        # loop on broken environments.
        self._max_fallback_attempts = 3
    
    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int,
        logits_indices: torch.Tensor,
        draft_hidden_states: torch.Tensor | None,
        draft_input_ids: torch.Tensor | None,
    ) -> tuple[bool, str]:
        """
        Pre-flight validation of all inputs before graph operations.
        
        Returns:
            (is_valid, error_message)
        """
        if input_ids is None:
            return False, "input_ids is None"
        
        if positions is None:
            return False, "positions is None"
        
        if num_tokens <= 0:
            return False, f"Invalid num_tokens: {num_tokens}"
        
        if logits_indices is None or len(logits_indices) == 0:
            return False, "logits_indices is empty"
        
        if num_tokens > input_ids.shape[0]:
            return False, f"num_tokens ({num_tokens}) exceeds input_ids shape ({input_ids.shape[0]})"
        
        if draft_hidden_states is not None:
            expected_size = num_tokens + self.num_speculative_tokens
            if draft_hidden_states.shape[0] < expected_size:
                return False, f"draft_hidden_states shape {draft_hidden_states.shape[0]} < expected {expected_size}"
        
        if draft_input_ids is not None:
            if draft_input_ids.shape[0] < num_tokens:
                return False, f"draft_input_ids shape {draft_input_ids.shape[0]} < num_tokens {num_tokens}"
        
        return True, ""
    
    def _check_attention_metadata_consistency(
        self,
        main_attn_metadata: Any,
        draft_attn_metadata: Any | None,
        num_tokens: int,
    ) -> tuple[bool, str]:
        """Check attention metadata consistency between main and draft models."""
        if draft_attn_metadata is None:
            return True, ""
        
        if hasattr(main_attn_metadata, 'num_reqs') and hasattr(draft_attn_metadata, 'num_reqs'):
            if main_attn_metadata.num_reqs != draft_attn_metadata.num_reqs:
                return False, (
                    f"num_reqs mismatch: main={main_attn_metadata.num_reqs}, "
                    f"draft={draft_attn_metadata.num_reqs}"
                )
        
        return True, ""
    
    @contextmanager
    def _error_recovery_context(self):
        """Context manager for error recovery."""
        try:
            yield
        except RuntimeError as e:
            self._last_error = str(e)
            if _is_oom_error(e):
                self._capture_failed = True
                logger.error(f"Unified graph memory error: {e}")
                self._cleanup_after_error()
                raise UnifiedMTPMemoryError(
                    f"Memory allocation failed: {e}"
                ) from e
            if "graph" in str(e).lower() or "capture" in str(e).lower():
                self._capture_failed = True
                logger.error(f"Unified graph runtime error: {e}")
                self._cleanup_after_error()
                raise GraphCaptureError(f"Graph operation failed: {e}") from e
            raise
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Unified graph unexpected error: {e}")
            self._cleanup_after_error()
            raise
    
    def _cleanup_after_error(self):
        """Clean up state after an error."""
        self._fallback_mode = True
        if hasattr(self, '_draft_token_ids_from_unified_graph'):
            self._draft_token_ids_from_unified_graph = None
    
    def _handle_replay_timeout(self) -> bool:
        """
        Handle replay timeout scenarios.
        
        Returns:
            True if should continue with fallback, False if should abort
        """
        self._replay_timeout_count += 1
        
        if self._replay_timeout_count >= self._max_replay_timeout_count:
            logger.error(f"Unified graph replay timeout count exceeded ({self._replay_timeout_count})")
            self._fallback_mode = True
            return False
        
        logger.warning(f"Unified graph replay timeout (count: {self._replay_timeout_count}), retrying...")
        return True
    
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
        """
        Execute unified forward with comprehensive error handling.

        Note: Earlier revisions accepted ``draft_kv_caches`` and assigned them
        onto the draft model from inside this function. That is unsafe inside
        an ACLGraph capture (attribute writes are not replayed) and is now
        forbidden. KV-cache binding must happen *before* graph capture.
        """
        if self._fallback_mode:
            return self._run_fallback_forward(
                input_ids, positions, intermediate_tensors, inputs_embeds,
                num_tokens, logits_indices, **model_kwargs
            )
        
        is_valid, error_msg = self._validate_inputs(
            input_ids, positions, num_tokens, logits_indices,
            draft_hidden_states, draft_input_ids
        )
        if not is_valid:
            logger.warning(f"Unified graph input validation failed: {error_msg}")
            return self._run_fallback_forward(
                input_ids, positions, intermediate_tensors, inputs_embeds,
                num_tokens, logits_indices, error=error_msg, **model_kwargs
            )
        
        attn_valid, attn_error = self._check_attention_metadata_consistency(
            model_kwargs.get('attn_metadata'),
            draft_attn_metadata,
            num_tokens
        )
        if not attn_valid:
            logger.warning(f"Attention metadata inconsistency: {attn_error}")
            draft_attn_metadata = None
        
        with self._error_recovery_context():
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
        
        draft_logits = None
        draft_token_ids = None
        
        if draft_hidden_states is not None and draft_input_ids is not None:
            try:
                draft_token_count = num_tokens + self.num_speculative_tokens
                if draft_positions is not None:
                    draft_pos_slice = draft_positions[:draft_token_count]
                else:
                    draft_pos_slice = positions
                draft_kwargs = {
                    "input_ids": draft_input_ids[:draft_token_count],
                    "positions": draft_pos_slice,
                    "hidden_states": draft_hidden_states[:draft_token_count],
                }
                
                if inputs_embeds is not None:
                    draft_kwargs["inputs_embeds"] = inputs_embeds
                
                if draft_attn_metadata is not None:
                    forward_context = get_forward_context()
                    if forward_context is not None and hasattr(forward_context, 'attn_metadata'):
                        draft_kwargs["attn_metadata"] = draft_attn_metadata
                
                draft_hidden_out = self.draft_model(**draft_kwargs)
                
                if isinstance(draft_hidden_out, tuple):
                    draft_last_hidden, _ = draft_hidden_out
                else:
                    draft_last_hidden = draft_hidden_out
                
                if draft_token_indices_to_sample is not None and len(draft_token_indices_to_sample) > 0:
                    draft_sample_hidden = draft_last_hidden[draft_token_indices_to_sample]
                    draft_logits = self.draft_model.compute_logits(draft_sample_hidden)
                    draft_token_ids = draft_logits.argmax(dim=-1)
                    
            except Exception as e:
                logger.warning(f"Draft model execution failed in unified graph: {e}")
                draft_token_ids = None
        else:
            if draft_hidden_states is None:
                logger.debug("Draft hidden_states not available, skipping draft model")
        
        return UnifiedGraphOutput(
            hidden_states=hidden_states,
            logits=draft_logits,
            draft_token_ids=draft_token_ids,
            sample_hidden_states=sample_hidden_states,
            aux_hidden_states=aux_hidden_states,
        )
    
    def _run_fallback_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
        num_tokens: int,
        logits_indices: torch.Tensor,
        error: str | None = None,
        **model_kwargs,
    ) -> UnifiedGraphOutput:
        """
        Fallback to separate execution when unified graph fails.
        """
        logger.info(f"Using fallback mode for unified graph. Reason: {error or self._last_error or 'unknown'}")
        
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
                fallback_used=True,
                error=error,
            )
        
        sample_hidden_states = hidden_states[logits_indices]
        
        return UnifiedGraphOutput(
            hidden_states=hidden_states,
            logits=None,
            draft_token_ids=None,
            sample_hidden_states=sample_hidden_states,
            aux_hidden_states=aux_hidden_states,
            fallback_used=True,
            error=error,
        )
    
    def reset_error_state(self):
        """Reset transient error state so the next call can retry the graph.

        Note: ``_fallback_mode`` is also cleared here. Without this clear, the
        runner would stay in fallback forever after the first failure even if
        the caller explicitly asks for a retry. ``_fallback_attempts`` keeps a
        counter so we can stop retrying once the failure is persistent.
        """
        self._capture_failed = False
        self._last_error = None
        self._replay_timeout_count = 0
        if self._fallback_mode:
            self._fallback_attempts = getattr(self, "_fallback_attempts", 0) + 1
            if self._fallback_attempts < self._max_fallback_attempts:
                self._fallback_mode = False
            else:
                logger.warning(
                    "Unified MTP graph stays in fallback mode after %d "
                    "consecutive failures; reset_error_state ignored.",
                    self._fallback_attempts,
                )
    
    def is_fallback_mode(self) -> bool:
        """Check if currently in fallback mode."""
        return self._fallback_mode
    
    def get_last_error(self) -> str | None:
        """Get last error message."""
        return self._last_error
    
    def create_runnable(self) -> Callable:
        """Create a callable for ACLGraphWrapper to wrap."""
        return self._run_unified_forward


class EdgeCaseHandler:
    """
    Handler for specific edge cases in unified graph execution.
    """
    
    @staticmethod
    def handle_batch_size_change(
        current_batch_size: int,
        captured_batch_size: int,
    ) -> bool:
        """
        Handle batch size changes between capture and replay.
        
        Returns:
            True if change is acceptable, False if should fallback
        """
        if current_batch_size == captured_batch_size:
            return True
        
        ratio = abs(current_batch_size - captured_batch_size) / captured_batch_size
        if ratio > 0.2:
            logger.warning(f"Batch size change too large: {current_batch_size} vs {captured_batch_size}")
            return False
        
        return True
    
    @staticmethod
    def handle_pipeline_parallel(
        is_pp_rank: bool,
        is_last_rank: bool,
    ) -> tuple[bool, str]:
        """
        Handle pipeline parallelism scenarios.
        
        Returns:
            (can_use_unified, reason)
        """
        if not is_last_rank:
            return False, "Unified graph only supported on last PP rank"
        
        return True, ""
    
    @staticmethod
    def handle_async_scheduling(
        async_scheduling_enabled: bool,
        method: str,
    ) -> tuple[bool, str]:
        """
        Handle async scheduling conflicts.
        
        CRITICAL: Async scheduling is HIGH CONFLICT with unified graph.
        Unified graph requires synchronous execution (single-launch).
        Async scheduling needs CPU-GPU decoupling which breaks graph capture.
        
        See: Issue #5459, Issue #8587 - async + full graph crashes
        
        Returns:
            (can_use_unified, reason)
        """
        if async_scheduling_enabled and method == "mtp":
            return False, "Async scheduling HIGH CONFLICT with unified graph (requires synchronous execution)"
        
        return True, ""
    
    @staticmethod
    def handle_multimodal_inputs(
        has_mm_inputs: bool,
        mm_embeds_available: bool,
    ) -> tuple[bool, str]:
        """
        Handle multimodal input scenarios.
        
        Returns:
            (can_use_unified, reason)
        """
        if has_mm_inputs and not mm_embeds_available:
            return False, "Multimodal inputs require mm_embeds"
        
        return True, ""
    
    @staticmethod
    def handle_pcp_dcp(
        pcp_size: int,
        dcp_size: int,
    ) -> tuple[bool, str]:
        """
        Handle PCP/DCP conflicts.
        
        CRITICAL: PCP/DCP (>1) is HIGH CONFLICT with unified graph.
        PCP distributes tokens across ranks with dynamic slot_mapping updates.
        Unified graph cannot handle dynamic slot_mapping changes during draft steps.
        
        See: eagle_proposer.py:660-721 (slot_mapping += pcp_size each step)
        
        Returns:
            (can_use_unified, reason)
        """
        if pcp_size > 1 or dcp_size > 1:
            return False, (
                f"PCP/DCP HIGH CONFLICT (pcp_size={pcp_size}, "
                f"dcp_size={dcp_size}, dynamic slot_mapping updates "
                "incompatible)"
            )
        
        return True, ""
    
    @staticmethod
    def handle_padded_drafter_batch(
        disable_padded_drafter_batch: bool,
    ) -> tuple[bool, str]:
        """
        Handle disable_padded_drafter_batch conflicts.
        
        CRITICAL: disable_padded_drafter_batch=True is HIGH CONFLICT.
        Draft model uses different batch sizes than main model.
        Unified graph requires consistent shapes across all operations.
        
        See: eagle_proposer.py:135
        
        Returns:
            (can_use_unified, reason)
        """
        if disable_padded_drafter_batch:
            return False, (
                "disable_padded_drafter_batch HIGH CONFLICT (draft uses "
                "unpadded batch, unified graph requires consistent shapes)"
            )
        
        return True, ""
    
    @staticmethod
    def handle_use_compress(
        use_compress: bool,
    ) -> tuple[bool, str]:
        """
        Handle use_compress (DeepSeek) conflicts.
        
        MEDIUM CONFLICT: Compressed hidden states need special handling.
        Need verification of hidden_states passing in unified graph.
        
        Returns:
            (can_use_unified, reason)
        """
        if use_compress:
            return True, "use_compress enabled - requires verification of compressed hidden states passing"
        
        return True, ""
    
    @staticmethod
    def handle_memory_pressure(
        memory_allocated: float,
        memory_reserved: float,
        threshold: float = 0.9,
    ) -> tuple[bool, str]:
        """
        Handle memory pressure scenarios.
        
        Returns:
            (can_continue, reason)
        """
        if memory_allocated / memory_reserved > threshold:
            return False, f"Memory pressure too high: {memory_allocated}/{memory_reserved}"
        
        return True, ""


def should_use_unified_graph(
    vllm_config: VllmConfig,
    scheduler_output: Any,
    num_tokens: int,
    device_memory_info: dict | None = None,
    pcp_size: int = 1,
    dcp_size: int = 1,
    use_compress: bool = False,
    is_pp_last_rank: bool = True,
) -> tuple[bool, str]:
    """
    Comprehensive check for whether unified graph should be used.
    
    Returns:
        (should_use, reason)
    """
    edge_handler = EdgeCaseHandler()
    
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return False, "No speculative config"
    
    if speculative_config.method != "mtp":
        return False, f"Method is {speculative_config.method}, not mtp"
    
    if not vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
        return False, "FULL cudagraph mode not enabled"
    
    if speculative_config.enforce_eager:
        return False, "enforce_eager is True"
    
    pp_can, pp_reason = edge_handler.handle_pipeline_parallel(
        is_pp_rank=True,
        is_last_rank=is_pp_last_rank,
    )
    if not pp_can:
        return False, pp_reason
    
    async_can, async_reason = edge_handler.handle_async_scheduling(
        vllm_config.scheduler_config.async_scheduling,
        speculative_config.method,
    )
    if not async_can:
        return False, async_reason
    
    pcp_can, pcp_reason = edge_handler.handle_pcp_dcp(pcp_size, dcp_size)
    if not pcp_can:
        return False, pcp_reason
    
    padded_can, padded_reason = edge_handler.handle_padded_drafter_batch(
        speculative_config.disable_padded_drafter_batch,
    )
    if not padded_can:
        return False, padded_reason
    
    compress_can, compress_reason = edge_handler.handle_use_compress(use_compress)
    if not compress_can:
        return False, compress_reason
    
    if scheduler_output is not None:
        has_encoder_input = (
            hasattr(scheduler_output, 'scheduled_encoder_inputs')
            and len(scheduler_output.scheduled_encoder_inputs) > 0
        )
        mm_can, mm_reason = edge_handler.handle_multimodal_inputs(
            vllm_config.model_config.is_multimodal_model,
            not has_encoder_input,
        )
        if not mm_can:
            return False, mm_reason
    
    if device_memory_info is not None:
        mem_can, mem_reason = edge_handler.handle_memory_pressure(
            device_memory_info.get('allocated', 0),
            device_memory_info.get('reserved', 1),
        )
        if not mem_can:
            return False, mem_reason
    
    return True, "All conditions met"