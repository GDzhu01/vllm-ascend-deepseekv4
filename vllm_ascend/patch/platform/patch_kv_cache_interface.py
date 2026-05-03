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
    SlidingWindowMLASpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager

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
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same quantization method."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            sparse_head_dim=specs[0].sparse_head_dim,
            dtype=specs[0].dtype,
            cache_dtype_str=cache_dtype_str_set.pop(),
            cache_sparse_c8=specs[0].cache_sparse_c8,
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
        # we should have a scale dim here.
        # we should have dtype of scale and k of indexer here.
        # NOTE(zyj): patch modifies here
        # TODO(cmq): adapt this for A5
        HEAD_DIM_TO_BLOCK_BYTES: dict[int, int] = {
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


_ORIG_NEEDS_KV_CACHE_ZEROING = KVCacheConfig.needs_kv_cache_zeroing.fget


def _needs_kv_cache_zeroing(self: KVCacheConfig) -> bool:
    assert _ORIG_NEEDS_KV_CACHE_ZEROING is not None
    if _ORIG_NEEDS_KV_CACHE_ZEROING(self):
        return True

    for group in self.kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            specs = group_spec.kv_cache_specs.values()
        else:
            specs = (group_spec,)
        if any(
            isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))
            and getattr(spec, "model_version", None) == "svf"
            and getattr(spec, "compress_ratio", 1) > 1
            for spec in specs
        ):
            return True
    return False


KVCacheConfig.needs_kv_cache_zeroing = property(_needs_kv_cache_zeroing)


_ORIG_BLOCK_POOL_GET_NEW_BLOCKS = BlockPool.get_new_blocks
_ORIG_BLOCK_POOL_FREE_BLOCKS = BlockPool.free_blocks
_ORIG_KV_CACHE_MANAGER_INIT = KVCacheManager.__init__
_ORIG_KV_CACHE_MANAGER_TAKE_NEW_BLOCK_IDS = KVCacheManager.take_new_block_ids


def _get_new_blocks_with_zero_tracking(self: BlockPool, num_blocks: int):
    blocks = _ORIG_BLOCK_POOL_GET_NEW_BLOCKS(self, num_blocks)
    if blocks and getattr(self, "_ascend_track_new_block_ids", False):
        block_ids = getattr(self, "_ascend_new_block_ids_to_zero", None)
        if block_ids is None:
            block_ids = []
            self._ascend_new_block_ids_to_zero = block_ids
        block_ids.extend(block.block_id for block in blocks)
    return blocks


def _prepend_free_blocks(free_block_queue, blocks) -> None:
    if not blocks:
        return

    old_first_block = free_block_queue.fake_free_list_head.next_free_block
    if old_first_block is None:
        raise RuntimeError(
            "next_free_block of fake_free_list_head should always exist")

    last_block = free_block_queue.fake_free_list_head
    for block in blocks:
        block.prev_free_block = last_block
        last_block.next_free_block = block
        last_block = block

    last_block.next_free_block = old_first_block
    old_first_block.prev_free_block = last_block
    free_block_queue.num_free_blocks += len(blocks)


def _free_blocks_with_reuse_first(self: BlockPool, ordered_blocks) -> None:
    if not getattr(self, "_ascend_reuse_freed_blocks_first", False):
        return _ORIG_BLOCK_POOL_FREE_BLOCKS(self, ordered_blocks)

    blocks_list = list(ordered_blocks)
    for block in blocks_list:
        block.ref_cnt -= 1

    free_blocks = [
        block for block in blocks_list
        if block.ref_cnt == 0 and not block.is_null
    ]
    free_blocks.sort(key=lambda block: block.block_id)
    _prepend_free_blocks(self.free_block_queue, free_blocks)


def _take_new_block_ids_from_pool(self: BlockPool) -> list[int]:
    block_ids = getattr(self, "_ascend_new_block_ids_to_zero", None)
    if not block_ids:
        return []
    self._ascend_new_block_ids_to_zero = []
    return block_ids


def _kv_cache_manager_init_with_block_pool_tracking(
    self: KVCacheManager, *args, **kwargs
) -> None:
    _ORIG_KV_CACHE_MANAGER_INIT(self, *args, **kwargs)
    self.block_pool._ascend_track_new_block_ids = (
        self.kv_cache_config.needs_kv_cache_zeroing)


def _take_new_block_ids_with_block_pool(self: KVCacheManager) -> list[int]:
    block_ids = _ORIG_KV_CACHE_MANAGER_TAKE_NEW_BLOCK_IDS(self)
    block_pool = getattr(getattr(self, "coordinator", None), "block_pool", None)
    if block_pool is not None and hasattr(block_pool, "take_new_block_ids"):
        block_ids.extend(block_pool.take_new_block_ids())
    return block_ids


BlockPool.get_new_blocks = _get_new_blocks_with_zero_tracking
BlockPool.free_blocks = _free_blocks_with_reuse_first
BlockPool.take_new_block_ids = _take_new_block_ids_from_pool
KVCacheManager.__init__ = _kv_cache_manager_init_with_block_pool_tracking
KVCacheManager.take_new_block_ids = _take_new_block_ids_with_block_pool
