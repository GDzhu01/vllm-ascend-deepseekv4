#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

from functools import cached_property

from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.request import Request


def _unwrap_kv_cache_spec(kv_cache_group):
    kv_cache_spec = kv_cache_group.kv_cache_spec
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
    return kv_cache_spec


class CompressorScheduleGroup:
    """Track one homogeneous compressor state group for a scheduler step.

    The model runner uses a different compressed KV-cache write path when a
    decode request reaches a compressed boundary. Mixing boundary and
    non-boundary decode requests in one batch makes the graph/compressor path
    ambiguous. This helper lets the scheduler accept only requests with the same
    per-KV-cache compressed-token delta in the current scheduling step.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        target_decode_key: tuple[int, ...] | None = None,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self._target_decode_key = target_decode_key
        self._decode_key: tuple[int, ...] | None = None

    @cached_property
    def compress_ratios(self) -> tuple[int, ...]:
        ratios: list[int] = []
        for kv_cache_group in self.kv_cache_config.kv_cache_groups:
            kv_cache_spec = _unwrap_kv_cache_spec(kv_cache_group)
            ratio = int(getattr(kv_cache_spec, "compress_ratio", 1))
            if ratio > 1:
                ratios.append(ratio)
        return tuple(ratios)

    @property
    def enabled(self) -> bool:
        return len(self.compress_ratios) > 0

    @property
    def decode_key(self) -> tuple[int, ...] | None:
        return self._decode_key

    def get_key(
        self,
        request: Request,
        num_new_tokens: int,
        num_computed_tokens: int | None = None,
    ) -> tuple[int, ...] | None:
        if not self.enabled:
            return None

        computed_tokens = (
            request.num_computed_tokens
            if num_computed_tokens is None
            else num_computed_tokens
        )
        if computed_tokens < request.num_prompt_tokens:
            return None

        return tuple(
            (computed_tokens + num_new_tokens) // ratio
            - computed_tokens // ratio
            for ratio in self.compress_ratios
        )

    def can_accept(self, key: tuple[int, ...] | None) -> bool:
        if not self.enabled:
            return True

        if key is None:
            # Prefill/chunked-prefill is not a CompressorDecode request.
            return True

        if self._target_decode_key is not None and key != self._target_decode_key:
            return False

        return self._decode_key is None or self._decode_key == key

    def accept(self, key: tuple[int, ...] | None) -> None:
        if not self.enabled:
            return

        if key is not None and self._decode_key is None:
            self._decode_key = key
