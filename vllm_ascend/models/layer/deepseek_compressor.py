import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.models.deepseek_v4_kv_cache_utils import (
    get_deepseek_svf_block_size,
)


class CompressorStateCache(nn.Module, AttentionLayerBase):
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
        super().__init__()
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

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=self.alignment,
            page_size_padded=self.page_size_padded,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendDSABackend


class SVFSWACache(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        alignment: int = 0,
    ):
        super().__init__()
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

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendDSABackend
