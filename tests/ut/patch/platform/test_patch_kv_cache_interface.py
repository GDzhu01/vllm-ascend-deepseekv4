# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.block_pool import BlockPool

import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa: F401


def test_block_pool_reuses_freed_blocks_first_when_enabled():
    block_pool = BlockPool(num_gpu_blocks=10,
                           enable_caching=True,
                           hash_block_size=128)
    block_pool._ascend_reuse_freed_blocks_first = True

    first_blocks = block_pool.get_new_blocks(3)
    second_blocks = block_pool.get_new_blocks(2)
    assert [block.block_id for block in first_blocks] == [1, 2, 3]
    assert [block.block_id for block in second_blocks] == [4, 5]

    block_pool.free_blocks(reversed(first_blocks))

    reused_blocks = block_pool.get_new_blocks(3)
    assert [block.block_id for block in reused_blocks] == [1, 2, 3]
