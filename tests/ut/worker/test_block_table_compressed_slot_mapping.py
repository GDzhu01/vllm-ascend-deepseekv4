#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase


class TestCompressedBlockTableSlotMapping(TestBase):
    def test_svf_compressed_slot_mapping_uses_storage_block_size(self):
        with patch("vllm_ascend.worker.block_table.get_dcp_group") as mock_get_dcp_group, \
             patch("vllm_ascend.worker.block_table.get_pcp_group") as mock_get_pcp_group:
            mock_dcp_group = MagicMock(spec=GroupCoordinator)
            mock_dcp_group.world_size = 1
            mock_dcp_group.rank_in_group = 0
            mock_get_dcp_group.return_value = mock_dcp_group

            mock_pcp_group = MagicMock(spec=GroupCoordinator)
            mock_pcp_group.world_size = 1
            mock_pcp_group.rank_in_group = 0
            mock_get_pcp_group.return_value = mock_pcp_group

            from vllm_ascend.worker.block_table import BlockTable

            block_table = BlockTable(
                block_size=256,
                max_num_reqs=1,
                max_num_blocks_per_req=8,
                max_num_batched_tokens=64,
                pin_memory=False,
                device=torch.device("cpu"),
                kernel_sizes=[256],
                cp_kv_cache_interleave_size=1,
                num_speculative_tokens=0,
                kv_cache_group=SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(compress_ratio=4)
                ),
            )

            block_table.add_row([5, 7], 0)

            req_indices = np.zeros(10, dtype=np.int32)
            positions = np.arange(60, 70, dtype=np.int32)
            block_table.compute_slot_mapping(req_indices, positions)

            expected = np.array(
                [
                    5 * 64 + 60,
                    5 * 64 + 61,
                    5 * 64 + 62,
                    5 * 64 + 63,
                    7 * 64 + 0,
                    7 * 64 + 1,
                    7 * 64 + 2,
                    7 * 64 + 3,
                    7 * 64 + 4,
                    7 * 64 + 5,
                ],
                dtype=np.int32,
            )
            np.testing.assert_array_equal(block_table.slot_mapping.np[:10], expected)
