import os
import sys
import unittest
from typing import ClassVar

import torch
import torch_npu
import pypto
from vllm_ascend.ops.pypto.hc_pre_impl import npu_hc_pre

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


def gen_rms_norm_denom(x):
    _, d = x.shape
    # print("rms norm x.shape", x.shape)
    x = x.square()
    x = x.sum(-1, True) / d
    x = x + norm_eps
    x = x.sqrt()
    return x


def gen_sigmoid(x):
    x = -x
    x = x.exp()
    x = 1 / (1 + x)
    return x


def gen_hc_split_sinkhorn(x, hc_scale, hc_base):
    t, _ = x.shape # (t, 24)

    pre = x[:, :hc] * hc_scale[0] + hc_base[:, :hc] # (t, 4)
    pre = gen_sigmoid(pre) + hc_eps # (t, 4)

    post = x[:, hc: 2*hc] * hc_scale[1] + hc_base[:, hc: 2*hc] # (t, 4)
    post = 2.0 * gen_sigmoid(post) # (t, 4)

    comb_flag = (x[:, 2*hc: ] * hc_scale[2] + hc_base[:, 2*hc: ]).reshape(t, hc, hc) # (t, 4, 4)
    row_max = comb_flag.amax(-1, keepdim=True) # (t, 4, 1)
    comb_flag = (comb_flag - row_max).exp() # (t, 4, 4)

    row_sum = comb_flag.sum(-1, keepdim=True) # (t, 4, 1)
    comb_flag = comb_flag / row_sum + hc_eps # (t, 4, 4)
    col_sum = comb_flag.sum(-2, keepdim=True) # (t, 1, 4)
    comb_flag = comb_flag / (col_sum + hc_eps) # (t, 4, 4)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb_flag.sum(-1, keepdim=True) # (t, 4, 4)
        comb_flag = comb_flag / (row_sum + hc_eps) # (t, 4, 4)
        col_sum = comb_flag.sum(-2, keepdim=True) # (t, 4, 4)
        comb_flag = comb_flag / (col_sum + hc_eps) # (t, 4, 4)
    return pre, post, comb_flag


def gen_hc_pre(x, hc_fn, hc_scale, hc_base):
    t = x.shape[0]
    x_16 = x.reshape((t, hc * d))
    hc_base = hc_base.reshape(1, mix_hc)
    x = x_16.to(torch.float32)

    hc_fn = hc_fn.to(torch.float32)

    res = torch.matmul(x, hc_fn.transpose(0, 1)) # (t, hc*d) @ (mix_hc, hc*d)^t = (t, mix_hc)
    # mm_res = res

    res = res.to(torch.bfloat16).to(torch.float32)

    res = res / gen_rms_norm_denom(x) # (t, mix_hc) / (t, 1) = (t, mix_hc)
    mm_res = res

    pre, post, comb = gen_hc_split_sinkhorn(res, hc_scale, hc_base) # (t, hc), (t, hc), (t, hc, hc)
    mul_res = pre.reshape(t, hc, 1) * x.reshape(t, hc, d)
    res = mul_res.sum(-2) # (t,mul_res d)
    res = res.to(torch.bfloat16)
    return res, post, comb, mm_res


def gen_hc_pre_data(t = 16):
    # print("t is ", t)
    x = torch.empty((t, hc, d), dtype=torch.bfloat16).uniform_(-1, 1)
    hc_fn = torch.empty((mix_hc, hc*d), dtype=torch.bfloat16).uniform_(-1, 1)
    hc_scale = torch.empty((3,), dtype=torch.float32).uniform_(-1, 1)
    hc_base = torch.empty((mix_hc, ), dtype=torch.float32).uniform_(-1, 1)
    res, post, comb, mm_res = gen_hc_pre(x, hc_fn, hc_scale, hc_base)
    # print("res", res.shape, res)
    # print("post", post.shape, post)
    # print("comb", comb.shape, comb)
    return x, hc_fn, hc_scale, hc_base, res, post, comb, mm_res


class TestPyPtoHcPre(unittest.TestCase):
    def test_pypto_hc_pre(self):
        device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
        torch.npu.set_device(int(device_id))
        torch.manual_seed(42)

        x, hc_fn, hc_scale, hc_base, y_gd, post_gd, comb_gd, mm_res_gd = gen_hc_pre_data(16)
        print("gen golden success !!!")
        x = x.to(device=f'npu:{device_id}')
        hc_fn = hc_fn.to(device=f'npu:{device_id}')
        hc_scale = hc_scale.to(device=f'npu:{device_id}')
        hc_base = hc_base.to(device=f'npu:{device_id}')

        y = torch.zeros_like(y_gd).to(device=f'npu:{device_id}')
        post = torch.zeros_like(post_gd).to(device=f'npu:{device_id}')
        comb = torch.zeros_like(comb_gd).to(device=f'npu:{device_id}')
        # mm_res = torch.zeros_like(mm_res_gd).to(device=f'npu:{device_id}')

        y, post, comb = npu_hc_pre(x, hc_fn, hc_scale, hc_base)
        torch_npu.npu.synchronize()

        # mm_res = mm_res.cpu()
        y = y.cpu()
        post = post.cpu()
        comb = comb.cpu()

        compare(y, y_gd, "y", atol=0.0001, rtol=0.0078125)
        compare(post, post_gd, "post", atol=0.000025, rtol=0.005)
        compare(comb, comb_gd, "comb", atol=0.000025, rtol=0.005)
    
    def test_pypto_hc_pre_graph(self):
        device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
        torch.npu.set_device(int(device_id))
        torch.manual_seed(42)

        x, hc_fn, hc_scale, hc_base, y_gd, post_gd, comb_gd, mm_res_gd = gen_hc_pre_data(16)
        print("gen golden success !!!")

        y = torch.zeros_like(y_gd).to(device=f'npu:{device_id}')
        post = torch.zeros_like(post_gd).to(device=f'npu:{device_id}')
        comb = torch.zeros_like(comb_gd).to(device=f'npu:{device_id}')
        # mm_res = torch.zeros_like(mm_res_gd).to(device=f'npu:{device_id}')
        
        ### to device
        x = x.to(device=f'npu:{device_id}')
        hc_fn = hc_fn.to(device=f'npu:{device_id}')
        hc_scale = hc_scale.to(device=f'npu:{device_id}')
        hc_base = hc_base.to(device=f'npu:{device_id}')
        
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            y, post, comb = npu_hc_pre(x, hc_fn, hc_scale, hc_base)
        g.replay()
        pypto.runtime._device_synchronize()#内部接口，不推荐使用
        torch_npu.npu.synchronize()
        # mm_res = mm_res.cpu()
        y = y.cpu()
        post = post.cpu()
        comb = comb.cpu()

        compare(y, y_gd, "y", atol=0.0001, rtol=0.0078125)
        compare(post, post_gd, "post", atol=0.000025, rtol=0.005)
        compare(comb, comb_gd, "comb", atol=0.000025, rtol=0.005)
        

if __name__ == '__main__':
    unittest.main(verbosity=2)
