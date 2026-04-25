import torch
import sys

import vllm
from vllm.config import get_current_vllm_config
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.config.cache import CacheConfig
from vllm_ascend.models.deepseek_v4_kv_cache_utils import (
    get_deepseek_svf_block_size,
)
from vllm_ascend.models.layer.deepseek_compressor import (
    AscendDeepseekV32IndexerCache,
    CompressorStateCache,
    SVFSWACache,
)
from vllm_ascend.patch.platform.patch_kv_cache_interface import (
    SlidingWindowMLASpec,
)

from vllm.config import VllmConfig
from vllm_ascend.attention.dsa_v1 import AscendDSABackend


class AscendCompressorStateCache(CompressorStateCache):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        block_size: int,
        prefix: str,
        alignment: int = 0,
        page_size_padded: int | None = None,
    ):
        torch.nn.Module.__init__(self)
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        self.compress_ratio = compress_ratio
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        self.block_size = block_size
        self.alignment = alignment
        self.page_size_padded = page_size_padded


    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=self.alignment,
            page_size_padded=self.page_size_padded
        )

    def forward(self): ...

    def get_attn_backend(self):
        return AscendDSABackend


class AscendSVFSWACache(SVFSWACache):
    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        alignment: int = 0,
    ):
        torch.nn.Module.__init__(self)
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.window_size = window_size
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.block_size = get_deepseek_svf_block_size(
            cache_config.block_size if cache_config is not None else None
        )
        self.alignment = alignment

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            model_version="svf",
            alignment=self.alignment,
        )

    def forward(self): ...

    def get_attn_backend(self):
        return AscendDSABackend

vllm.model_executor.models.deepseek_v2.DeepseekV32IndexerCache = AscendDeepseekV32IndexerCache

_deepseek_compressor_mod = sys.modules.get(
    "vllm.model_executor.layers.deepseek_compressor"
)
if _deepseek_compressor_mod is not None:
    _deepseek_compressor_mod.CompressorStateCache = AscendCompressorStateCache

try:
    import vllm.v1.attention.backends.mla.sparse_swa as _sparse_swa
except ModuleNotFoundError:
    _sparse_swa = None

if _sparse_swa is not None:
    _sparse_swa.SVFSWACache = AscendSVFSWACache

