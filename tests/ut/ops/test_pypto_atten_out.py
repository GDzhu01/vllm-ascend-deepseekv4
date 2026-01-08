import os
import sys
import unittest
from typing import ClassVar

import torch
import torch_npu
import pypto
from vllm_ascend.ops.pypto.attention_post_impl import npu_attention_post_v4

hc, d, sinkhorn_iters, norm_eps, hc_eps = 4, 4096, 20, 1e-6, 1e-6
mix_hc = (2 + hc) * hc


def compare(t: torch.Tensor, t_ref: torch.Tensor, name, atol, rtol, max_error_ratio=0.005, max_error_count=10):
    """
    比较两个张量的差异，超过阈值时打印错误点并抛出断言错误
    Args:
        t: 待比较张量
        t_ref: 参考张量
        name: 张量名称（用于日志）
        atol: 绝对容差
        rtol: 相对容差
        max_error_ratio: 误差点占总元素数的最大比例
        max_error_count: 显示的最大误差点数量（同时也是误差点阈值的上限）
    """
    def check_is_nan_inf():
        # ========== 核心新增：检测t中的NaN和Inf并直接报错 ==========
        # 1. 检测NaN
        nan_mask = torch.isnan(t)
        nan_count = nan_mask.sum().item()

        # 2. 检测Inf（包含+Inf和-Inf）
        inf_mask = torch.isinf(t)
        inf_count = inf_mask.sum().item()
 
        # 若存在NaN或Inf，拼接错误信息并报错
        if nan_count > 0 or inf_count > 0:
            error_msg = f"\n========== 张量 {name} 检测到非法值（禁止存在NaN/Inf）=========="

            # 打印NaN信息
            if nan_count > 0:
                nan_positions = torch.nonzero(nan_mask, as_tuple=False)
                show_nan_count = min(nan_count, max_error_count)
                error_msg += f"\n- NaN数量：{nan_count}，前 {show_nan_count} 个位置："
                for i in range(show_nan_count):
                    pos_tuple = tuple(p.item() for p in nan_positions[i])
                    error_msg += f"\n  位置 {pos_tuple}"

            # 打印Inf信息（区分+Inf/-Inf）
            if inf_count > 0:
                inf_positions = torch.nonzero(inf_mask, as_tuple=False)
                show_inf_count = min(inf_count, max_error_count)
                error_msg += f"\n- Inf数量：{inf_count}，前 {show_inf_count} 个位置（值类型）："
                for i in range(show_inf_count):
                    pos = inf_positions[i]
                    pos_tuple = tuple(p.item() for p in pos)
                    inf_val = t[pos_tuple].item()
                    inf_type = "+Inf" if inf_val == float('inf') else "-Inf"
                    error_msg += f"\n  位置 {pos_tuple}：{inf_type}"
            error_msg += "\n" + "="*80 + "\n"

            # 抛出断言错误，终止函数执行
            assert False, error_msg
 
    # check 是否是nan 或 inf
    check_is_nan_inf()
 
    # 先验证张量的基本属性一致
    assert t.shape == t_ref.shape, f"张量形状不一致：t.shape={t.shape}, t_ref.shape={t_ref.shape}"
    assert t.dtype == t_ref.dtype, f"张量数据类型不一致：t.dtype={t.dtype}, t_ref.dtype={t_ref.dtype}"
    assert t.device == t_ref.device, f"张量设备不一致：t.device={t.device}, t_ref.device={t_ref.device}"

    # 计算误差点数量的阈值（取比例计算值和最大数量的较小值）
    error_count_threshold = round(max_error_ratio * t_ref.numel())

    # 计算误差掩码（超过阈值的位置为True）
    diff_abs = (t - t_ref).abs()
    tolerance = atol + rtol * t_ref.abs()
    diff_mask = diff_abs > tolerance
    error_count = diff_mask.sum().item()

    # 计算最大误差和其位置
    max_diff, flat_max_pos = torch.max(diff_abs.flatten(), dim=0)
    max_pos = torch.unravel_index(flat_max_pos, t.shape)
    max_pos = tuple(idx.item() for idx in max_pos)

    # 打印错误点的逻辑（如果有误差点）
    if error_count > 0:
        print(f"\n========== 张量 {name} 存在 {error_count} 个误差点（阈值：{error_count_threshold}）==========")
 
        # 获取所有误差点的位置
        error_positions = torch.nonzero(diff_mask, as_tuple=False)  # shape: [error_count, dims]

        # 限制显示的误差点数量（避免数据量过大）
        show_count = min(error_count, max_error_count)
        print(f"显示前 {show_count} 个误差点（位置 | 待比较值 | 参考值 | 绝对误差 | 允许阈值）：")

        # 遍历前N个误差点打印详细信息
        for i in range(show_count):
            pos = error_positions[i]

            # 转换为元组格式的位置（如 (0, 2, 3)）
            pos_tuple = tuple(p.item() for p in pos)

            # 获取对应位置的数值
            t_val = t[pos_tuple].item()
            t_ref_val = t_ref[pos_tuple].item()
            diff_val = diff_abs[pos_tuple].item()
            tol_val = tolerance[pos_tuple].item()

            # 格式化输出，保留足够小数位
            print(f"  位置 {pos_tuple}: {t_val:.8f} vs {t_ref_val:.8f} | 误差={diff_val:.8f} | 阈值={tol_val:.8f}")

        # 打印最大误差点
        print(f"\n最大误差点：位置 {max_pos} | 误差={max_diff.item():.8f} | 阈值={tolerance[max_pos].item():.8f}")
        print("=" * 80 + "\n")

    # 断言误差点数量不超过阈值
    assert error_count <= error_count_threshold, \
        (f"compare fail: {name}, max diff: {max_diff.item():.8f} at {max_pos}, "
         f"error_count: {error_count}, error_count_threshold: {error_count_threshold}")
    
    print("compare success !!!!")


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    """
    PyTorch版本的均匀分布数据生成, 与NumPy版本行为完全一致
    严格保持 [min_value, max_value) 左闭右开区间特性
    """
    # 特殊情况：全零张量
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    # 布尔类型处理：等概率生成True/False
    if dtype == torch.bool:
        # 生成[0,2)的整数，转换为bool即等概率True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)
    # 浮点类型：[min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand生成[0,1)，缩放后得到[min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    # 整数类型：[min_value, max_value)
    else:
        # torch.randint的high参数为开区间，直接对应[min_value, max_value)
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin):
    """
    q: (t, n_q, rope_dim), bf16
    cos: (t, rope_dim), bf16
    sin: (t, rope_dim), bf16
    """
    input_dtype = q.dtype
    q = q.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    cos = torch.unsqueeze(cos, dim=1)  # [t, 1, rope_dim]
    sin = torch.unsqueeze(sin, dim=1)  # [t, 1, rope_dim]

    t, n, d = q.shape
    q = q.reshape(t, n, d // 2, 2).permute(0, 1, 3, 2).reshape(t, n, d)

    # (t, n_q, rope_dim), (t, 1, rope_dim) = (t, n_q, rope_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)

    if input_dtype != torch.float32:
        q_embed = q_embed.to(input_dtype)
    return q_embed


def compute_attention_post(inputs, params):
    atten_res = inputs[0]
    cos = inputs[1]
    sin = inputs[2]
    wo_a = inputs[3]
    wo_b = inputs[4]

    t = params.get("t")
    n_q = params.get("n_q")
    d = params.get("d")
    n_g = params.get("n_g")
    o_lora_rank = params.get("o_lora_rank")

    rope_in = atten_res[:, :, (d - cos.shape[-1]): ]  # (t, n_q, rope_dim), bf16
    nope_res = atten_res[:, :, 0: (d - cos.shape[-1])]  # (t, n_q, d - rope_dim), bf16
    rope_res = apply_rotary_pos_emb(rope_in, cos, sin)  # (t, n_q, rope_dim), bf16
    atten_res_new = torch.cat((nope_res, rope_res), dim=-1)

    # batch_matmul
    mm1_left_trans = atten_res_new.reshape(
        t, n_g, n_q * d // n_g).transpose(1, 0)  # (n_g, t, n_q * d // n_g)
    # (n_g, t, n_q * d // n_g) @ (n_g, n_q * d // n_g, o_lora_rank) = (n_g, t, o_lora_rank)
    bmm_1_res = torch.bmm(mm1_left_trans.to(torch.float32),
                        wo_a.to(torch.float32)).to(torch.bfloat16)
    bmm_res = bmm_1_res.transpose(1, 0)  # (t, n_g, o_lora_rank)

    # matmul
    bmm_reshpe = bmm_res.reshape(t, n_g * o_lora_rank)
    # (t, n_g * o_lora_rank) @ (n_g * o_lora_rank, h) = (t, n_q, h)
    mm_res = torch.mm(bmm_reshpe.to(torch.float32),
                      wo_b.to(torch.float32)).to(torch.bfloat16)

    return rope_res, bmm_res, mm_res, nope_res


def gen_attention_post_v4_golden(dtype, params):
    torch.manual_seed(42)
    t = params.get("t")
    n_q = params.get("n_q")
    d = params.get("d")
    rope_dim = params.get("rope_dim")
    n_g = params.get("n_g")
    o_lora_rank = params.get("o_lora_rank")
    h = params.get("h")
    attn_res = gen_uniform_data([t, n_q, d], -1, 1, dtype)
    cos = gen_uniform_data([t, rope_dim], -1, 1, dtype)
    sin = gen_uniform_data([t, rope_dim], -1, 1, dtype)
    wo_a = gen_uniform_data([n_g, n_q * d // n_g, o_lora_rank], -1, 1, dtype)
    wo_b = gen_uniform_data([n_g * o_lora_rank, h], -1, 1, dtype)
    hidden_states = torch.zeros([t, h]).to(dtype)
    inputs = [attn_res, cos, sin, wo_a, wo_b, hidden_states]
    rope_res, bmm_res, mm_res, nope_res = compute_attention_post(inputs, params)
    return inputs, rope_res, bmm_res, mm_res, nope_res


def do_attention_post_func(inputs, params, golden_list):
    """
    atten_res: (t, n_q, d), bf16
    cos: (t, rope_dim), bf16
    sin: (t, rope_dim), bf16
    wo_a: (n_g, n_q * d // n_g, o_lora_rank), bf16
    wo_b: (n_g * o_lora_rank, h)
    """
    torch_npu.npu.config.allow_internal_format = True
    # rope + batch_matmul + matmul
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    atten_res = inputs[0].npu()
    cos = inputs[1].npu()
    sin = inputs[2].npu()
    wo_a = inputs[3].npu()
    wo_b = inputs[4].npu()
    wo_b_nz = torch_npu.npu_format_cast(wo_b, torch_npu.Format.FRACTAL_NZ)

    t = params.get("t")
    rope_dim = params.get("rope_dim")
    n_q = params.get("n_q")
    d = params.get("d")
    n_g = params.get("n_g")
    o_lora_rank = params.get("o_lora_rank")
    h = params.get("h")

    hidden_states = npu_attention_post_v4(atten_res, cos, sin, wo_a, wo_b)

    torch_npu.npu.synchronize()
    compare(hidden_states.cpu(),
            golden_list[2], "hidden_states", atol=0.0001, rtol=0.005)


def do_attention_post_func_graph(inputs, params, golden_list):
    """
    atten_res: (t, n_q, d), bf16
    cos: (t, rope_dim), bf16
    sin: (t, rope_dim), bf16
    wo_a: (n_g, n_q * d // n_g, o_lora_rank), bf16
    wo_b: (n_g * o_lora_rank, h)
    """
    torch_npu.npu.config.allow_internal_format = True
    # rope + batch_matmul + matmul
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    atten_res = inputs[0].npu()
    cos = inputs[1].npu()
    sin = inputs[2].npu()
    wo_a = inputs[3].npu()
    wo_b = inputs[4].npu()
    wo_b_nz = torch_npu.npu_format_cast(wo_b, torch_npu.Format.FRACTAL_NZ)

    t = params.get("t")
    rope_dim = params.get("rope_dim")
    n_q = params.get("n_q")
    d = params.get("d")
    n_g = params.get("n_g")
    o_lora_rank = params.get("o_lora_rank")
    h = params.get("h")

    # capture model
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        hidden_states = npu_attention_post_v4(atten_res, cos, sin, wo_a, wo_b)
    g.replay()
    

    torch_npu.npu.synchronize()
    compare(hidden_states.cpu(),
            golden_list[2], "hidden_states", atol=0.0001, rtol=0.005)


class TestPyPtoAttnOut(unittest.TestCase):
    def test_pypto_attn_out(self):
        dtype = torch.bfloat16
        params = {"t": 249, "n_q": 64, "d": 512, "rope_dim": 64, "n_g": 8, "o_lora_rank": 1024, "h": 4096}
        inputs, rope_golden, bmm_golden, mm_golden, nope_res = gen_attention_post_v4_golden(
        dtype, params)
        do_attention_post_func(
            inputs, params, [rope_golden, bmm_golden, mm_golden, nope_res])
        return True

    def test_pypto_attn_out_in_graph(self):
        dtype = torch.bfloat16
        params = {"t": 249, "n_q": 64, "d": 512, "rope_dim": 64, "n_g": 8, "o_lora_rank": 1024, "h": 4096}
        inputs, rope_golden, bmm_golden, mm_golden, nope_res = gen_attention_post_v4_golden(
        dtype, params)
        do_attention_post_func_graph(
            inputs, params, [rope_golden, bmm_golden, mm_golden, nope_res])
        return True
    
    
if __name__ == '__main__':
    unittest.main(verbosity=2)
