# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager, SlidingWindowManager, FullAttentionManager, spec_manager_map)
from vllm.v1.kv_cache_interface import KVCacheSpec, AttentionSpec
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_spec import (CompressAttentionSpec, # --
                                            Compress4AttentionSpec,
                                            Compress128AttentionSpec,
                                            SWAAttentionSpec,  # --
                                            C4AttnKVStateSpec,
                                            C4AttnScoreStateSpec,
                                            C128AttnKVStateSpec,
                                            C128AttnScoreStateSpec,
                                            C4IndexerKVStateSpec,
                                            C4IndexerScoreStateSpec,
                                            C4IndexerSpec)


class CompressAttentionManager(FullAttentionManager):

    def __init__(self, kv_cache_spec: CompressAttentionSpec,
                 block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.compress_ratio = kv_cache_spec.compress_ratio
        self._null_block = block_pool.null_block

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        assert isinstance(self.kv_cache_spec, (CompressAttentionSpec, C4IndexerSpec))

        num_tokens //= self.compress_ratio
        num_tokens_main_model //= self.compress_ratio

        return super().get_num_blocks_to_allocate(request_id, num_tokens,
                                                  new_computed_blocks, total_computed_tokens, num_tokens_main_model)

    def allocate_new_blocks(self, request_id: str,
                            num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        num_tokens //= self.compress_ratio
        ## TODO: check spec decode
        num_tokens_main_model //= self.compress_ratio

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(
                num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
        """
        num_tokens //= self.compress_ratio

        return super().cache_blocks(request, num_tokens)

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(
            kv_cache_spec, Compress4AttentionSpec | Compress128AttentionSpec | C4IndexerSpec
        ), (
            "CompressAttentionManager can only be used for compressor attention groups"
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            # Need to drop the last matched block if eagle is enabled.
            for computed in computed_blocks:
                computed.pop()

        # NOTE: Div the compress ratio when finding the longest cache hit token length.
        alignment_tokens = cdiv(alignment_tokens, kv_cache_spec.compress_ratio)
        while (
            block_size != alignment_tokens  # Faster for common case.
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        print(f"{kv_cache_spec=}")
        print(f"{kv_cache_group_ids=}")
        print(f"{computed_blocks=}")
        return computed_blocks


def get_manager_for_kv_cache_spec(kv_cache_spec: KVCacheSpec,
                                  **kwargs) -> SingleTypeKVCacheManager:
    spec_manager_map.update({
        Compress4AttentionSpec: CompressAttentionManager,
        Compress128AttentionSpec: CompressAttentionManager,
        SWAAttentionSpec: SlidingWindowManager,
        C4AttnKVStateSpec: SlidingWindowManager,
        C4AttnScoreStateSpec: SlidingWindowManager,
        C128AttnKVStateSpec: SlidingWindowManager,
        C128AttnScoreStateSpec: SlidingWindowManager,
        C4IndexerKVStateSpec: SlidingWindowManager,
        C4IndexerScoreStateSpec: SlidingWindowManager,
        C4IndexerSpec: CompressAttentionManager
    })
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager