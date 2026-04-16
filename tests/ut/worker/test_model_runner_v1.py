from collections import defaultdict
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

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

    def test_store_layer_attn_metadata_keeps_single_object_for_normal_model(self):
        runner = self._build_runner()
        attn_metadata_dict = {}
        metadata_a = SimpleNamespace(name="a")

        runner._store_layer_attn_metadata(attn_metadata_dict, "layer", metadata_a)
        self.assertIs(attn_metadata_dict["layer"], metadata_a)

    def test_store_layer_attn_metadata_appends_for_compress_model(self):
        runner = self._build_runner()
        runner.use_compress = True
        attn_metadata_dict = defaultdict(list)
        metadata_a = SimpleNamespace(name="a")
        metadata_b = SimpleNamespace(name="b")

        runner._store_layer_attn_metadata(attn_metadata_dict, "layer", metadata_a)
        runner._store_layer_attn_metadata(attn_metadata_dict, "layer", metadata_b)

        self.assertEqual(attn_metadata_dict["layer"], [metadata_a, metadata_b])

    def test_may_reinitialize_input_batch_keeps_group_block_size_when_common_selector_has_no_match(self):
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
            side_effect=ValueError("No common block size for 32."),
        ):
            with patch("vllm_ascend.worker.model_runner_v1.NPUInputBatch") as mock_input_batch:
                runner.may_reinitialize_input_batch(kv_cache_config)

        self.assertEqual(runner.kernel_block_sizes, [[32]])
        self.assertEqual(
            mock_input_batch.call_args.kwargs["kernel_block_sizes"],
            [[32]],
        )

if __name__ == "__main__":
    unittest.main()
