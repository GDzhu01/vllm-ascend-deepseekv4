# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
import vllm.v1.kv_cache_interface
from typing_extensions import Self
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


@dataclass(frozen=True)
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """MLAAttentionSpec extended to support DSA models, with optional Sparse C8 support.

    When Sparse C8 is enabled, the KV cache tuple changes from
    (kv_cache[0]: bfloat16, kv_cache[1]: bfloat16, kv_cache[2]: bfloat16)
    to
    (kv_cache[0]: bfloat16, kv_cache[1]: bfloat16, kv_cache[2]: int8, kv_cache[3]: float16).

    The semantic meaning of each KV cache entry is as follows:
    1. kv_cache[0] stores kv_lora.
    2. kv_cache[1] stores k_rope.
    3. kv_cache[2] stores the key tensor from the indexer module.
    4. kv_cache[3] stores the key scale tensor from the indexer module,
       and exists only when Sparse C8 is enabled.

    The main changes are as follows:
    1. The key tensor from the indexer module stored in kv_cache[2] is
       converted from bf16 to int8 to reduce memory usage. It is then
       processed with int8 precision in Lightning_indexer computation
       to improve computational efficiency.
    2. The quantization scale of the key tensor in the indexer module
       must also be stored for the Lightning_indexer_quant operator,
       and is therefore saved in kv_cache[3].
    """

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    sparse_head_dim: tuple[int, ...] | None = None
    cache_sparse_c8: bool = False
    c8_k_cache_dtype: torch.dtype = torch.int8
    c8_k_scale_cache_dtype: torch.dtype = torch.float16

    @property
    def page_size_bytes(self) -> int:
        if self.cache_sparse_c8:
            assert self.sparse_head_dim is not None
            assert len(self.sparse_head_dim) == 3
            num_heads_per_page = self.block_size * self.num_kv_heads
            # kv_cache[0]: bfloat16, kv_cache[1]: bfloat16
            kv_lora_rank, qk_rope_head_dim = self.sparse_head_dim[:2]
            k_pe_nope_bytes = num_heads_per_page * (kv_lora_rank + qk_rope_head_dim) * get_dtype_size(self.dtype)
            # kv_cache[2]: int8
            index_head_dim = self.sparse_head_dim[-1]
            indexer_k_bytes = num_heads_per_page * index_head_dim * get_dtype_size(self.c8_k_cache_dtype)
            # kv_cache[3]: float16
            # since the scale is stored per token, head_dim is set to 1.
            index_scale_head_dim = 1
            indexer_k_scale_bytes = (
                num_heads_per_page * index_scale_head_dim * get_dtype_size(self.c8_k_scale_cache_dtype)
            )
            return k_pe_nope_bytes + indexer_k_bytes + indexer_k_scale_bytes

        return self.block_size * self.num_kv_heads * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))

    @property
    def sparse_kv_cache_ratio(self) -> tuple[float, float, float, float | None]:
        """
        Compute the relative byte share of each KV cache entry.

        Returns:
            A tuple containing the ratios for:
            - kv_cache[0]
            - kv_cache[1]
            - kv_cache[2]
            - kv_cache[3] (None if Sparse C8 is disabled)
        """

        assert self.sparse_head_dim is not None

        def get_sparse_head_dim_virtual() -> tuple[int, int, int, int]:
            assert self.sparse_head_dim is not None
            assert self.cache_sparse_c8 is True

            kv_lora_rank, qk_rope_head_dim, index_k_head_dim = self.sparse_head_dim

            factor = get_dtype_size(self.dtype) // get_dtype_size(self.c8_k_cache_dtype)
            index_k_head_dim_virtual = index_k_head_dim // factor

            assert get_dtype_size(self.dtype) == get_dtype_size(self.c8_k_scale_cache_dtype)
            index_k_scale_head_dim_virtual = 1

            return (
                kv_lora_rank,
                qk_rope_head_dim,
                index_k_head_dim_virtual,
                index_k_scale_head_dim_virtual,
            )

        if self.cache_sparse_c8:
            virtual_dims = get_sparse_head_dim_virtual()
            total_virtual_head_dim = sum(virtual_dims)

            return (
                total_virtual_head_dim / virtual_dims[0],  # kv_cache[0]
                total_virtual_head_dim / virtual_dims[1],  # kv_cache[1]
                total_virtual_head_dim / virtual_dims[2],  # kv_cache[2]
                total_virtual_head_dim / virtual_dims[3],  # kv_cache[3]
            )

        return (
            self.head_size / self.sparse_head_dim[0],  # kv_cache[0]
            self.head_size / self.sparse_head_dim[1],  # kv_cache[1]
            self.head_size / self.sparse_head_dim[2],  # kv_cache[2]
            None,  # kv_cache[3] does not exist
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        scale_dim_set = set(spec.scale_dim for spec in specs)
        scale_dtype_set = set(spec.scale_dtype for spec in specs)
        sparse_head_dim_set = set(spec.sparse_head_dim for spec in specs)
        cache_sparse_c8_set = set(spec.cache_sparse_c8 for spec in specs)
        c8_k_cache_dtype_set = set(spec.c8_k_cache_dtype for spec in specs)
        c8_k_scale_cache_dtype_set = set(
            spec.c8_k_scale_cache_dtype for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(scale_dim_set) == 1
            and len(scale_dtype_set) == 1
            and len(sparse_head_dim_set) == 1
            and len(cache_sparse_c8_set) == 1
            and len(c8_k_cache_dtype_set) == 1
            and len(c8_k_scale_cache_dtype_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version and sparse "
            "cache layout."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            sparse_head_dim=specs[0].sparse_head_dim,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
            scale_dim=scale_dim_set.pop(),
            scale_dtype=scale_dtype_set.pop(),
            cache_sparse_c8=cache_sparse_c8_set.pop(),
            c8_k_cache_dtype=c8_k_cache_dtype_set.pop(),
            c8_k_scale_cache_dtype=c8_k_scale_cache_dtype_set.pop(),
        )


def _init_mla_cache_fields(spec: MLAAttentionSpec | SlidingWindowMLASpec):
    """Shared MLA cache init logic for quantiztion format across different models."""
    FP8_DTYPE = "fp8_ds_mla"
    MODEL_VERSIONS = ["v32", "svf"]
    if spec.cache_dtype_str != FP8_DTYPE:
        return
    assert spec.model_version in MODEL_VERSIONS, "Invalid model version."
    assert (spec.model_version == "v32" and spec.compress_ratio == 1) or (
        spec.model_version == "svf" and spec.compress_ratio in [0, 4, 128]
    ), "Invalid compress ratio."
    if spec.compress_ratio > 1:
        assert spec.block_size % spec.compress_ratio == 0, (
            f"Block size {spec.block_size} must be divisible by compress ratio."
        )

    # See `vllm/v1/attention/backends/mla/flashmla_sparse.py`
    #  for details.
    assert spec.num_kv_heads == 1, "MLAAttentionSpec only supports 1 head."
    # TODO(yifan): move this head size to bytes mapping to a utils file.
    if spec.model_version == "svf":
        if get_ascend_device_type() == AscendDeviceType.A5:
            HEAD_DIM_TO_BLOCK_BYTES: dict[int, int] = {
                128: 128,  # indexer value: 128B fp8, scale handled by scale_dim
                512: 640,  # attention KV: 448B fp8 NoPE + 128B bf16 RoPE + scale/pad
            }
        else:
            HEAD_DIM_TO_BLOCK_BYTES = {
                128: 260,   # SVF: 128*2B NoPE, 4B for fp32 scale = 260B
                512: 1024,  # SVF: 512*2B NoPE + RoPE = 1024B
            }

        if spec.head_size in HEAD_DIM_TO_BLOCK_BYTES:
            actual_head_bytes = HEAD_DIM_TO_BLOCK_BYTES[spec.head_size]
        else:
            actual_head_bytes = spec.head_size
        object.__setattr__(spec, "head_size", actual_head_bytes)
        object.__setattr__(spec, "head_size_v", actual_head_bytes)

        # ====================GPU=======================
        # if spec.alignment is not None:
        #     # Apply 576-byte alignment padding for SVF 512.
        #     # KV cache tensor is allocated with padded page_size,
        #     # but kernels access with shape [num_blocks, real_page_size].
        #     actual_page_size = spec.real_page_size_bytes
        #     padded_page_size = round_up(actual_page_size, spec.alignment)
        #     if padded_page_size != actual_page_size:
        #         object.__setattr__(spec, "page_size_padded", padded_page_size)

        if get_ascend_device_type() == AscendDeviceType.A5:
            # TODO(zyj): FIXME(qcs): this is a bug to just use real_page_size_bytes, 
            # cause the page_size_padded will be overrided by this operation
            actual_page_size = spec.real_page_size_bytes
            padded_page_size = round_up(actual_page_size, 128)
            if padded_page_size != actual_page_size:
                object.__setattr__(spec, "page_size_padded", padded_page_size)
    else:
        raise ValueError(f"Invalid model version: {spec.model_version}")


vllm.v1.kv_cache_interface.MLAAttentionSpec = AscendMLAAttentionSpec
vllm.v1.kv_cache_interface._init_mla_cache_fields = _init_mla_cache_fields


def _iter_kv_cache_specs(spec: KVCacheSpec):
    if isinstance(spec, UniformTypeKVCacheSpecs):
        yield from spec.kv_cache_specs.values()
    else:
        yield spec


def _needs_kv_cache_zeroing(self: KVCacheConfig) -> bool:
    for group in self.kv_cache_groups:
        for spec in _iter_kv_cache_specs(group.kv_cache_spec):
            if spec.__class__.__name__.endswith("MambaSpec"):
                return True
    return False


KVCacheConfig.needs_kv_cache_zeroing = property(_needs_kv_cache_zeroing)


def _patch_grouped_new_block_ids() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    def take_new_block_ids(self: KVCacheManager) -> list[list[int]]:
        ids_by_group = [
            mgr.take_new_block_ids()
            for mgr in self.coordinator.single_type_managers
        ]
        return ids_by_group if any(ids_by_group) else []

    KVCacheManager.take_new_block_ids = take_new_block_ids


_patch_grouped_new_block_ids()


def _patch_attention_new_block_tracking() -> None:
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionSpec,
        SingleTypeKVCacheManager,
        spec_manager_map,
    )
    from vllm_ascend.core.single_type_kv_cache_manager import (
        CompressAttentionManager,
    )

    def should_track_blocks(manager: SingleTypeKVCacheManager) -> bool:
        return isinstance(
            manager.kv_cache_spec,
            (AscendMLAAttentionSpec, SlidingWindowMLASpec, SlidingWindowSpec),
        )

    original_allocate_new_blocks = SingleTypeKVCacheManager.allocate_new_blocks
    original_allocate_new_computed_blocks = (
        SingleTypeKVCacheManager.allocate_new_computed_blocks
    )

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        new_blocks = original_allocate_new_blocks(
            self, request_id, num_tokens, num_tokens_main_model)
        if should_track_blocks(self) and type(self.kv_cache_spec) is not FullAttentionSpec:
            self.new_block_ids.extend(b.block_id for b in new_blocks)
        return new_blocks

    def allocate_new_computed_blocks(
        self,
        request_id,
        new_computed_blocks,
        num_local_computed_tokens,
        num_external_computed_tokens,
    ) -> None:
        num_external_blocks = 0
        if (should_track_blocks(self)
                and type(self.kv_cache_spec) is not FullAttentionSpec
                and request_id not in self.num_cached_block
                and num_external_computed_tokens > 0):
            req_blocks = self.req_to_blocks[request_id]
            num_total_computed_tokens = (
                num_local_computed_tokens + num_external_computed_tokens)
            num_skipped_tokens = self.get_num_skipped_tokens(
                num_total_computed_tokens)
            num_skipped_blocks = num_skipped_tokens // self.block_size
            local_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            num_blocks_before_external = (
                len(req_blocks) + num_skipped_blocks +
                len(local_computed_blocks))
            num_external_blocks = max(
                cdiv(num_total_computed_tokens, self.block_size) -
                num_blocks_before_external,
                0,
            )

        original_allocate_new_computed_blocks(
            self,
            request_id,
            new_computed_blocks,
            num_local_computed_tokens,
            num_external_computed_tokens,
        )

        if num_external_blocks:
            self.new_block_ids.extend(
                b.block_id
                for b in self.req_to_blocks[request_id][-num_external_blocks:])

    SingleTypeKVCacheManager.allocate_new_blocks = allocate_new_blocks
    SingleTypeKVCacheManager.allocate_new_computed_blocks = (
        allocate_new_computed_blocks)
    spec_manager_map[AscendMLAAttentionSpec] = CompressAttentionManager


_patch_attention_new_block_tracking()
