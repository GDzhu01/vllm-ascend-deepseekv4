from types import SimpleNamespace
from unittest.mock import patch

import torch

import vllm_ascend.patch.worker.patch_deepseek_compressor as patch_deepseek_compressor
from vllm_ascend.utils import AscendDeviceType


def _fake_swa_init(self, head_dim, window_size, dtype, prefix, cache_config):
    self.kv_cache = torch.tensor([])
    self.head_dim = head_dim
    self.window_size = window_size
    self.prefix = prefix
    self.cache_config = cache_config
    self.dtype = dtype
    self.block_size = 64


def test_ascend_svf_swa_cache_uses_64_token_blocks():
    cache_config = SimpleNamespace(cache_dtype="auto")
    vllm_config = SimpleNamespace(cache_config=cache_config)

    with patch.object(
        patch_deepseek_compressor.SVFSWACache,
        "__init__",
        _fake_swa_init,
    ), patch.object(
        patch_deepseek_compressor,
        "get_ascend_device_type",
        return_value=AscendDeviceType.A3,
    ):
        cache = patch_deepseek_compressor.AscendSVFSWACache(
            head_dim=512,
            window_size=256,
            dtype=torch.float16,
            prefix="layer0.swa_cache",
            cache_config=cache_config,
        )
        spec = cache.get_kv_cache_spec(vllm_config)

    assert cache.block_size == 64
    assert spec.block_size == 64
