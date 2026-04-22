import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.core.kv_cache_spec import C4IndexerSpec
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_compress = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.vllm_config.speculative_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        runner.model_config.get_vocab_size.return_value = 32000
        runner.block_size = 128
        runner.cache_config = SimpleNamespace(block_size=128)
        runner.offload_config = SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
        )
        runner.max_num_reqs = 8
        runner.max_model_len = 128
        runner.max_encoder_len = 0
        runner.max_num_tokens = 128
        runner.pin_memory = False
        runner.is_pooling_model = False
        runner.input_batch = SimpleNamespace(logitsprocs=[])
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))

    def test_reshape_kv_cache_uses_selected_kernel_block_size_for_hybrid_group(self):
        runner = self._build_runner()
        runner.use_hybrid_blocks = True
        runner.kernel_block_sizes = [[32]]
        runner.attn_backend.get_supported_kernel_block_sizes.return_value = [128, 32]

        kv_cache_spec = FullAttentionSpec(
            block_size=32,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["draft_attn"],
                    kv_cache_spec=kv_cache_spec,
                )
            ],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_group_id=0,
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(
            kv_cache_config,
            kv_cache_raw_tensors,
        )
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 32, 8, 64))
        self.assertEqual(v_cache.shape, (2, 32, 8, 64))
        runner.attn_backend.get_kv_cache_shape.assert_any_call(2, 32, 8, 64)

    def test_reshape_kv_cache_keeps_physical_block_size_for_compressed_c4_indexer_group(self):
        runner = self._build_runner()
        runner.use_compress = True
        runner.use_hybrid_blocks = True
        runner.kernel_block_sizes = [[128]]
        runner.attn_backend = MagicMock()
        runner.attn_backend.get_supported_kernel_block_sizes.return_value = [128]
        runner.attn_backend.get_kv_cache_shape.side_effect = (
            lambda num_blocks, block_size, num_kv_heads, head_size: (
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
            )
        )

        kv_cache_spec = C4IndexerSpec(
            block_size=1024,
            num_kv_heads=1,
            head_size=128,
            indexer_scale_dim=1,
            dtype=torch.int8,
            page_size_padded=0,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=kv_cache_spec.page_size_bytes * 2,
                    shared_by=["draft_attn"],
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["draft_attn"],
                    kv_cache_spec=kv_cache_spec,
                )
            ],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_group_id=0,
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(
            kv_cache_config,
            kv_cache_raw_tensors,
        )
        indexer_k_cache, indexer_scale_cache = kv_caches["draft_attn"]

        self.assertEqual(indexer_k_cache.shape, (2, 1024, 1, 128))
        self.assertEqual(indexer_scale_cache.shape, (2, 1024, 1, 1))
        runner.attn_backend.get_kv_cache_shape.assert_any_call(2, 1024, 1, 128)
        runner.attn_backend.get_kv_cache_shape.assert_any_call(2, 1024, 1, 1)

    def test_may_reinitialize_input_batch_uses_backend_supported_sizes_for_normal_model(self):
        runner = self._build_runner()
        runner.use_hybrid_blocks = True
        backend = MagicMock()
        backend.get_supported_kernel_block_sizes.return_value = [128]
        runner.attn_groups = [[SimpleNamespace(backend=backend)]]
        kv_cache_spec = FullAttentionSpec(
            block_size=32,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["draft_attn"],
                    kv_cache_spec=kv_cache_spec,
                )
            ],
        )

        with patch(
            "vllm_ascend.worker.model_runner_v1.select_common_block_size",
        ) as mock_select_common_block_size:
            with patch("vllm_ascend.worker.model_runner_v1.NPUInputBatch") as mock_input_batch:
                runner.may_reinitialize_input_batch(kv_cache_config)

        mock_select_common_block_size.assert_not_called()
        self.assertEqual(runner.kernel_block_sizes, [[128]])
        self.assertEqual(
            mock_input_batch.call_args.kwargs["kernel_block_sizes"],
            [[128]],
        )

    def test_initialize_kv_cache_passes_first_kernel_block_size_group_to_drafter(self):
        runner = self._build_runner()
        runner.attn_groups = [[SimpleNamespace(kv_cache_spec=object())]]
        runner.kernel_block_sizes = [[128], [32], [0]]
        runner.speculative_config = MagicMock()
        runner.speculative_config.use_eagle.return_value = True
        runner.speculative_config.uses_draft_model.return_value = False
        runner.drafter = AscendEagleProposer.__new__(AscendEagleProposer)
        runner.drafter.initialize_attn_backend = MagicMock()
        runner.initialize_attn_backend = MagicMock()
        runner.may_add_encoder_only_layers_to_kv_cache_config = MagicMock()
        runner.maybe_add_kv_sharing_layers_to_kv_cache_groups = MagicMock()
        runner.may_reinitialize_input_batch = MagicMock()
        runner.initialize_kv_cache_tensors = MagicMock(return_value={})
        runner.model_config.enable_return_routed_experts = False
        kv_cache_config = SimpleNamespace(kv_cache_groups=[])

        with patch("vllm_ascend.worker.model_runner_v1.has_kv_transfer_group", return_value=False):
            runner.initialize_kv_cache(kv_cache_config)

        self.assertEqual(
            runner.drafter.initialize_attn_backend.call_args.args[1],
            [128],
        )


class TestNPUModelRunnerEPLB(unittest.TestCase):

    def test_sync_parallel_eplb_config_copies_redundant_expert_count(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.eplb_enable = True
        runner.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                enable_eplb=False,
                eplb_config=SimpleNamespace(num_redundant_experts=0),
            )
        )
        runner.ascend_config = SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=2),
        )

        runner._sync_parallel_eplb_config()

        self.assertTrue(runner.vllm_config.parallel_config.enable_eplb)
        self.assertEqual(
            runner.vllm_config.parallel_config.eplb_config.num_redundant_experts,
            2,
        )

    def test_sync_parallel_eplb_config_noops_when_eplb_disabled(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.eplb_enable = False
        runner.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                enable_eplb=False,
                eplb_config=SimpleNamespace(num_redundant_experts=0),
            )
        )
        runner.ascend_config = SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=2),
        )

        runner._sync_parallel_eplb_config()

        self.assertFalse(runner.vllm_config.parallel_config.enable_eplb)
        self.assertEqual(
            runner.vllm_config.parallel_config.eplb_config.num_redundant_experts,
            0,
        )

if __name__ == "__main__":
    unittest.main()
