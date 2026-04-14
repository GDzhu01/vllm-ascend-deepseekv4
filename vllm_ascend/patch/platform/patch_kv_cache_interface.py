from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs, KVCacheSpec
from dataclasses import dataclass
from vllm.config import VllmConfig
from vllm_ascend import envs

from typing_extensions import Self
import torch
import vllm

from vllm_ascend.worker.v2.model_runner import logger
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
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>> patched KV Cache Spec")
    # vllm.v1.kv_cache_interface.KVCacheSpec = PatchedKVCacheSpec
    vllm.v1.kv_cache_interface.get_all_kvcache_specs_from_list = get_all_kvcache_specs_from_list
    vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs

