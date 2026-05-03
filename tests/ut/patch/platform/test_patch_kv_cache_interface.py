# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.block_pool import BlockPool

import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa: F401


def test_block_pool_tracks_new_block_ids_when_enabled():
    block_pool = BlockPool(num_gpu_blocks=10,
                           enable_caching=True,
                           hash_block_size=128)
    block_pool._ascend_track_new_block_ids = True

    blocks = block_pool.get_new_blocks(3)
    assert [block.block_id for block in blocks] == [1, 2, 3]
    assert block_pool.take_new_block_ids() == [1, 2, 3]
    assert block_pool.take_new_block_ids() == []

    more_blocks = block_pool.get_new_blocks(2)
    assert [block.block_id for block in more_blocks] == [4, 5]
    assert block_pool.take_new_block_ids() == [4, 5]
