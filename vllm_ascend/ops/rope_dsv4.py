import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Union, List, Optional
from vllm.platforms import current_platform
from vllm.config import VllmConfig

import torch_npu


# =========================================================================
# 1. 全局状态容器 (Global State)
# =========================================================================

class RopeGlobalState:
    def __init__(self):
        # 1. 静态计算表 (只跟数学参数有关): {config_key: (cos_full, sin_full)}
        self.static_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # 2. 运行时固定 Buffer (跟 Group 有关): 
        # {config_key: {group_name: (cos_buf, sin_buf)}}
        self.runtime_buffer: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
        
        # 3. 层级注册信息: {layername: (config_key, [required_groups])}
        self.layer_info: Dict[str, Tuple[str, List[str]]] = {}
        
        # 4. 辅助统计: 记录某个 Config 下注册了哪些 Group，防止重复分配
        # {config_key: Set(group_names)}
        self.registry_summary: Dict[str, set] = {}

_ROPE_STATE = RopeGlobalState()

class RopeDataProxy:
    def __init__(self, data_map, is_cos=True):
        # data_map 结构: {config_key: {group_name: (cos, sin)}}
        self._data = data_map
        self.idx = 0 if is_cos else 1 

    def __getitem__(self, index):
        # === 场景 A: Metadata Build 阶段 (切片) ===
        # 输入: slice 或 tuple ([:num_tokens, ...])
        if not isinstance(index, str):
            new_map = {}
            for config_k, groups_map in self._data.items():
                new_map[config_k] = {}
                for group_name, item in groups_map.items():
                    # item 是 (cos_tensor, sin_tensor) 或 单个 tensor (如果是多次切片后)
                    # 为了稳健，我们总是假设 data_map 存的是原始的 tuple 结构
                    c_val = item[0][index]
                    s_val = item[1][index]
                    new_map[config_k][group_name] = (c_val, s_val)
            
            # 返回一个新的 Proxy，保持内部结构不变
            return RopeDataProxy(new_map, is_cos=(self.idx == 0))

        # === 场景 B: Forward 阶段 (按层取值) ===
        # 输入: layername (str)
        else:
            layername = index
            info = _ROPE_STATE.layer_info.get(layername)
            if info is None:
                raise KeyError(f"Layer {layername} not registered.")
            
            config_key, required_groups = info
            
            # 获取该配置下的所有 group 数据
            config_data = self._data.get(config_key, {})
            
            # 收集该层需要的数据
            layer_result = {}
            for grp in required_groups:
                if grp in config_data:
                    # config_data[grp] 是 (cos, sin)，根据 self.idx 取一个
                    layer_result[grp] = config_data[grp][self.idx]
                else:
                    # 如果这层要 "special_group" 但输入没给，可能需要处理异常或留空
                    pass
            
            # === 关键体验优化 ===
            # 如果该层只注册了 1 个 Group (绝大多数情况)，直接返回 Tensor
            # 这样你原本的代码 cos = metadata.cos[layername] 依然跑得通
            if len(layer_result) == 1:
                return list(layer_result.values())[0]
            
            # 如果注册了多个 Group，返回字典 {'default': t1, 'special': t2}
            return layer_result

# =========================================================================
# get_cos_and_sin_dsa for get all sin and cos
# =========================================================================

def get_cos_and_sin_dsa(
    positions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
    use_cache: bool = False
):
    """
    Args:
        positions: 
            - 如果是单个 Tensor，默认视为 {"default": tensor}
            - 如果是 Dict，则格式为 {"group_name": tensor}
    """
    # 1. 规范化输入
    if isinstance(positions, torch.Tensor):
        pos_map = {"default": positions}
    else:
        pos_map = positions

    # 结果容器: {config_key: {group_name: (cos, sin)}}
    batch_result = {}

    # 遍历所有存在的 Config Key
    for config_key, registered_groups in _ROPE_STATE.registry_summary.items():
        
        # 获取该 Config 的全量静态数学表
        if config_key not in _ROPE_STATE.static_cache:
            continue
        static_cos, static_sin = _ROPE_STATE.static_cache[config_key]
        
        batch_result[config_key] = {}

        # 遍历当前 batch 提供的所有 Group 数据
        for group_name, pos_tensor in pos_map.items():
            
            # 优化：如果这个 Config 根本没注册过这个 Group，直接跳过
            if group_name not in registered_groups:
                continue
            
            curr_cos = static_cos[pos_tensor]
            curr_sin = static_sin[pos_tensor]
            
            # --- 2. ACLGraph Buffer 处理 ---
            if use_cache:
                # 找到那个固定的坑位
                group_buffers = _ROPE_STATE.runtime_buffer.get(config_key, {}).get(group_name)
                
                if group_buffers is None:
                    # 这种情况通常是初始化漏了，或者是用了未注册的 group name
                    # TODO: 加warning或者删掉？
                    continue
                
                buf_cos, buf_sin = group_buffers
                num_tokens = pos_tensor.size(0)
                
                # In-place Copy (固定地址写入)
                buf_cos[:num_tokens].copy_(curr_cos)
                buf_sin[:num_tokens].copy_(curr_sin)
                
                # 保存 View
                batch_result[config_key][group_name] = (buf_cos[:num_tokens], buf_sin[:num_tokens])
            else:
                batch_result[config_key][group_name] = (curr_cos, curr_sin)

    # 返回 Proxy，分开 Cos 和 Sin
    return RopeDataProxy(batch_result, is_cos=True), RopeDataProxy(batch_result, is_cos=False)


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
        rope_groups: List[str] = ("default",),
        **extra_kwargs,
    ) -> None:
        super().__init__()
        self.layername = layername
        self.rotary_dim = rotary_dim
        dtype = torch.get_default_dtype()
        # dtype = torch.float32
        # 1. 生成 Config Key
        beta_fast = extra_kwargs.get("beta_fast", 32)
        beta_slow = extra_kwargs.get("beta_slow", 1)
        # Key 包含了所有影响数值计算的参数，不能包含layername和group
        config_key = (f"rotary_dim{rotary_dim}_max_position_embeddings{max_position_embeddings}_"
                      f"base{base}_scaling_factor{scaling_factor}_beta_fast{beta_fast}_beta_slow{beta_slow}")

        # 2. 注册 Layer 信息 (记录这层需要哪些 Group)
        _ROPE_STATE.layer_info[layername] = (config_key, rope_groups)

        # 3. 更新全局 Group 注册表
        if config_key not in _ROPE_STATE.registry_summary:
            _ROPE_STATE.registry_summary[config_key] = set()
        for grp in rope_groups:
            _ROPE_STATE.registry_summary[config_key].add(grp)

        # 4. 初始化静态表
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
            
        # 5. 初始化 Runtime Buffer (按 Group 分配)
        if config_key not in _ROPE_STATE.runtime_buffer:
            _ROPE_STATE.runtime_buffer[config_key] = {}
        
        target_device = current_platform.device_type
        max_batch_size = vllm_config.scheduler_config.max_num_batched_tokens
        # 遍历这层需要的 Group，如果没有 Buffer 就分配
        for grp in rope_groups:
            if grp not in _ROPE_STATE.runtime_buffer[config_key]:
                # print(f"Allocating Buffer for Key={config_key}, Group={grp}")
                buf_cos = torch.ones(max_batch_size, 1, 1, rotary_dim, dtype=dtype, device=target_device)
                buf_sin = torch.zeros(max_batch_size, 1, 1, rotary_dim, dtype=dtype, device=target_device)
                _ROPE_STATE.runtime_buffer[config_key][grp] = (buf_cos, buf_sin)

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