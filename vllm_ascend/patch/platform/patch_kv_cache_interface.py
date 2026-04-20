# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
import vllm.v1.kv_cache_interface
from typing_extensions import Self
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs, KVCacheSpec
from dataclasses import dataclass
from vllm.config import VllmConfig
from vllm_ascend import envs

from typing_extensions import Self
import torch
import vllm

from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    MambaSpec,
    CrossAttentionSpec
)

USE_MULTI_GROUPS_KV_CACHE = envs.USE_MULTI_GROUPS_KV_CACHE

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

        return self.block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

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

def get_all_kvcache_specs_from_list(
    kv_cache_spec_list: dict[str, list[KVCacheSpec]],
) -> list[KVCacheSpec]:
    all_kv_cache_specs = []
    for layer_name, layer_spec_list in kv_cache_spec_list.items():
        for layer_specs in layer_spec_list:
            all_kv_cache_specs.append(layer_specs)
    return all_kv_cache_specs

@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    """
    A KV cache spec for multiple layers with the same type of attention. Here,
    same types means always need the same number of token slots. For example,
    sliding window attentions with different window sizes are not the same type
    and should not be merged into one UniformTypeKVCacheSpecs.
    """

    kv_cache_specs_list: dict[str, list[KVCacheSpec]]
    kv_cache_specs: dict[str, KVCacheSpec]

    @property
    def page_size_bytes(self) -> int:
        all_specs = get_all_kvcache_specs_from_list(self.kv_cache_specs_list)
        return sum(spec.page_size_bytes for spec in all_specs)

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        all_specs = get_all_kvcache_specs_from_list(self.kv_cache_specs_list)
        max_num_pages = max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in all_specs
        )
        return max_num_pages * self.page_size_bytes

    @classmethod
    def is_uniform_type(cls, kv_cache_specs_list: dict[str, list[KVCacheSpec]]) -> bool:
        """
        Whether all layers have the same type of KV cache spec.
        """
        all_specs = get_all_kvcache_specs_from_list(kv_cache_specs_list)
        block_sizes = set(spec.block_size for spec in all_specs)
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        for _, layer_specs in kv_cache_specs_list.items():
            if len(layer_specs) == 1:
                # Different specs in one layer, not uniform
                return False

        one_spec = next(iter(cls.kv_cache_specs.values()))
        if isinstance(one_spec, FullAttentionSpec):
            return all(
                isinstance(spec, FullAttentionSpec) for spec in cls.kv_cache_specs.values()
            )
        elif isinstance(one_spec, CrossAttentionSpec):
            return all(
                isinstance(spec, CrossAttentionSpec) for spec in cls.kv_cache_specs.values()
            )
        elif isinstance(one_spec, SlidingWindowSpec):
            return all(
                isinstance(spec, SlidingWindowSpec)
                and spec.sliding_window == one_spec.sliding_window
                for spec in cls.kv_cache_specs.values()
            )
        elif isinstance(one_spec, ChunkedLocalAttentionSpec):
            return all(
                isinstance(spec, ChunkedLocalAttentionSpec)
                and spec.attention_chunk_size == one_spec.attention_chunk_size
                for spec in cls.kv_cache_specs.values()
            )
        elif isinstance(one_spec, MambaSpec):
            return all(
                isinstance(spec, MambaSpec)
                and spec.num_speculative_blocks == one_spec.num_speculative_blocks
                for spec in cls.kv_cache_specs.values()
            )
        else:
            # NOTE(Chen): Please add new branches for new KV cache spec types.
            raise NotImplementedError(
                f"Unsupported KV cache spec type: {type(one_spec)}"
            )

    @classmethod
    def from_specs(cls, kv_cache_specs_list: dict[str, list[KVCacheSpec]]) -> Self | None:
        """
        Return a SameTypeKVCacheSpecs object if all layers have the same type
        of KV cache spec. Return None if not.
        """
        if cls.is_uniform_type(kv_cache_specs_list):

            kv_cache_specs: dict[str, KVCacheSpec] = {}
            for layer_name, layer_specs in kv_cache_specs_list.items():
                kv_cache_specs[layer_name] = layer_specs[0]

            block_size = next(iter(kv_cache_specs.values())).block_size
            return cls(block_size=block_size, kv_cache_specs=kv_cache_specs, kv_cache_specs_list=kv_cache_specs_list)
        else:
            return None


if USE_MULTI_GROUPS_KV_CACHE:
    # vllm.v1.kv_cache_interface.AttentionSpec = AttentionSpec
    # logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>> patched KV Cache Spec")
    # vllm.v1.kv_cache_interface.KVCacheSpec = PatchedKVCacheSpec
    vllm.v1.kv_cache_interface.get_all_kvcache_specs_from_list = get_all_kvcache_specs_from_list
    vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs

vllm.v1.kv_cache_interface.MLAAttentionSpec = AscendMLAAttentionSpec
