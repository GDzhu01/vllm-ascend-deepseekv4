import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional, Union
from vllm.platforms import current_platform
from vllm.config import VllmConfig

import torch_npu


# =========================================================================
# 1. 全局状态容器 (Global State)
# =========================================================================

class RopeGlobalState:
    def __init__(self):
        # 静态全量表: {config_key: (cos_full, sin_full)} [MaxSeq, 1, 1, Dim]
        self.static_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # 运行时固定Buffer (CUDA Graph专用): {config_key: (cos_buf, sin_buf)} [MaxBatch, 1, 1, Dim]
        self.runtime_buffer: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # 层级映射: {layername: config_key}
        self.layer_map: Dict[str, str] = {}

# 实例化全局单例
_ROPE_STATE = RopeGlobalState()

# =========================================================================
# 2. 智能代理类 (Proxy Class)
# =========================================================================

class RopeDataProxy:
    """
    通用代理类。
    它可以存储 {key: Tensor} 或 {key: (Tensor, Tensor)}，
    并对其中的 value 进行透传切片。
    """
    def __init__(self, data_map):
        self._data = data_map

    def __getitem__(self, index):
        # 场景 A: 按层取值 (Forward 阶段) -> 传入 layername (str)
        if isinstance(index, str):
            layername = index
            key = _ROPE_STATE.layer_map.get(layername)
            if key is None:
                raise KeyError(f"Layer {layername} uses RoPE but was not registered.")
            
            # 直接返回对应的数据 (可能是 Tensor，也可能是 Tuple，取决于初始化时存的啥)
            return self._data[key]
        
        # 场景 B: 切片操作 (Metadata Build 阶段) -> 传入 slice/tuple
        else:
            new_map = {}
            # 修正点：不要强制解包 (c, s)，而是直接作为 item 处理
            for key, item in self._data.items():
                # 如果 item 是 Tensor，直接切片
                if isinstance(item, torch.Tensor):
                    new_map[key] = item[index]
                # 如果 item 是 Tuple/List (防御性编程，万一以后你想存 tuple)，则分别切片
                elif isinstance(item, (tuple, list)):
                    new_map[key] = type(item)(x[index] for x in item)
                else:
                    raise TypeError(f"Unsupported type in RopeDataProxy: {type(item)}")
            
            return RopeDataProxy(new_map)

# =========================================================================
# 3. 核心功能函数
# =========================================================================

def get_cos_and_sin_dsa(positions: torch.Tensor, use_cache: bool = False):
    batch_map = {} # 这里暂存 {key: (cos, sin)}
    num_tokens = positions.size(0)

    for key, (static_cos, static_sin) in _ROPE_STATE.static_cache.items():
        if static_cos.device != positions.device:
            static_cos = static_cos.to(positions.device)
            static_sin = static_sin.to(positions.device)

        current_cos = static_cos[positions]
        current_sin = static_sin[positions]

        if use_cache:
            if key not in _ROPE_STATE.runtime_buffer:
                raise RuntimeError(f"RoPE buffer for key {key} not initialized.")
            
            buf_cos, buf_sin = _ROPE_STATE.runtime_buffer[key]
            
            # In-place copy
            buf_cos[:num_tokens].copy_(current_cos)
            buf_sin[:num_tokens].copy_(current_sin)
            
            # 存入 View
            batch_map[key] = (buf_cos[:num_tokens], buf_sin[:num_tokens])
        else:
            batch_map[key] = (current_cos, current_sin)

    # 拆分成两个独立的 map，这样 cos_proxy 内部存的就是 {key: cos_tensor}
    cos_map = {k: v[0] for k, v in batch_map.items()}
    sin_map = {k: v[1] for k, v in batch_map.items()}

    # 返回两个代理对象，分别管理 cos 和 sin
    return RopeDataProxy(cos_map), RopeDataProxy(sin_map)


# =========================================================================
# 4. 修改后的 Embedding 类
# =========================================================================

class ComplexExpRotaryEmbedding(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layername: str,               # 必填：用于区分层
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int, # 用于计算静态数学表
        base: int,
        scaling_factor: float,
        **extra_kwargs,
    ) -> None:
        super().__init__()
        self.layername = layername
        self.rotary_dim = rotary_dim
        # dtype = torch.get_default_dtype()
        dtype = torch.float32
        # 1. 生成 Config Key
        beta_fast = extra_kwargs.get("beta_fast", 32)
        beta_slow = extra_kwargs.get("beta_slow", 1)
        # Key 包含了所有影响数值计算的参数
        config_key = (f"rotary_dim{rotary_dim}_max_position_embeddings{max_position_embeddings}_"
                      f"base{base}_scaling_factor{scaling_factor}_beta_fast{beta_fast}_beta_slow{beta_slow}")
        # print(f'config_key: {config_key}')
        # 2. 注册 Layer -> Key
        _ROPE_STATE.layer_map[layername] = config_key

        # 3. 初始化静态数学表 (如果该 Key 尚未初始化)
        # 这里实现了“多层复用”：只有第一个遇到该配置的层会执行计算
        if config_key not in _ROPE_STATE.static_cache:
            # print(f"[RoPE] Initializing Static Cache for key: {config_key}")
            complex_cis = self.precompute_freqs_cis(
                rotary_dim, max_position_embeddings, max_position_embeddings,
                base, scaling_factor, beta_fast, beta_slow
            )
            # 转换为实部/虚部并 reshape 为 [MaxSeq, 1, 1, Dim]
            cos = complex_cis.real.repeat_interleave(2, dim=-1).to(dtype)
            sin = complex_cis.imag.repeat_interleave(2, dim=-1).to(dtype)
            
            cos = cos.to(current_platform.device_type)
            sin = sin.to(current_platform.device_type)
            
            # 预先 unsqueeze，方便后续 gather 和 broadcast
            # [Seq, Dim] -> [Seq, 1, 1, Dim]
            _ROPE_STATE.static_cache[config_key] = (
                cos.unsqueeze(1).unsqueeze(1), 
                sin.unsqueeze(1).unsqueeze(1)
            )

        # 4. 初始化 Runtime Buffer (如果该 Key 尚未初始化)
        # 这就是你要的 "init_rope_buffers" 逻辑，现在移到了这里
        if config_key not in _ROPE_STATE.runtime_buffer:
            # print(f"[RoPE] Allocating CUDA Buffer for key: {config_key}, size: {max_batch_size}")
            # 确保 buffer 在正确的 device 上
            target_device = current_platform.device_type
            max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

            buffer_cos = torch.ones(
                max_num_batched_tokens, 1, 1, rotary_dim,
                dtype=dtype, device=target_device
            )
            buffer_sin = torch.zeros(
                max_num_batched_tokens, 1, 1, rotary_dim,
                dtype=dtype, device=target_device
            )
            
            _ROPE_STATE.runtime_buffer[config_key] = (buffer_cos, buffer_sin)

    @staticmethod
    def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
        """保持原有的 DeepSeek/MLA 计算逻辑"""
        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return (dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base)))

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min, max, dim):
            if min == max: max += 0.001
            linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
            return torch.clamp(linear_func, 0, 1)

        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if original_seq_len > 0:
            low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        t = torch.arange(seqlen)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor, # 必须传入
        sin: torch.Tensor, # 必须传入
    ) -> torch.Tensor:
        
        # 此时传入的 cos/sin 已经是通过 layername 获取到的具体 tensor
        
        ori_shape = x.shape
        y = x # In-place or copy based on need, assuming x can be modified or y is new tensor
        
        # 维度对齐
        if x.dim() == 2: x = x.unsqueeze(-2)
        if x.dim() == 3: x = x.unsqueeze(1)
        
        x = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")

        y.copy_(x.view(ori_shape))
        return y

    def extra_repr(self) -> str:
        return f"layername={self.layername}, rotary_dim={self.rotary_dim}"