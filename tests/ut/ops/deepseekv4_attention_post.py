#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import math
import os
import torch
import torch_npu
import pypto
import logging
import pytest
import numpy as np
from attention_post_impl import npu_attention_post_v4, attention_post_decode, AttnPostConfig, Rope3dTileConfig
from utils.compare import compare


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define("attn_post(Tensor atten_res, Tensor cos, Tensor sin, Tensor wo_a, Tensor wo_b) -> (Tensor)")

@torch.library.impl(pyptolib, "attn_post", "Meta")
def attn_post(atten_res, cos, sin, wo_a, wo_b):
    y = torch.empty([atten_res.size(0), wo_b.size(1)], dtype=atten_res.dtype, device=atten_res.device)
    return y

@torch.library.impl(pyptolib, "attn_post", "NPU")
def attn_post(atten_res, cos, sin, wo_a, wo_b):
    return npu_attention_post_v4(atten_res, cos, sin, wo_a, wo_b)

class AttentionPostV4(torch.nn.Module):
    def forward(self, attn_res, cos, sin, wo_a, wo_b):
        for i in range(20):
            torch.add(attn_res, 0)
        return torch.ops.pypto.attn_post(attn_res, cos, sin, wo_a, wo_b)

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

    # define npu outputs
    hidden_states = torch.zeros([t, h]).to(torch.bfloat16).npu()

    atten_res_pto = pypto.from_torch(
        atten_res, dynamic_axis=[0], name="atten_res")
    cos_pto = pypto.from_torch(cos, dynamic_axis=[0], name="cos")
    sin_pto = pypto.from_torch(sin, dynamic_axis=[0], name="sin")
    wo_a_pto = pypto.from_torch(wo_a, name="wo_a")
    wo_b_pto = pypto.from_torch(wo_b_nz, name="wo_b")

    hidden_states_pto = pypto.from_torch(
        hidden_states, dynamic_axis=[0], name="hidden_states")

    pto_inputs = [atten_res_pto, cos_pto, sin_pto, wo_a_pto, wo_b_pto]
    pto_outputs = [hidden_states_pto]

    tile_config = AttnPostConfig(
        unroll_list=[128, 64, 32, 16, 8, 1],
        rope3d_tile_config = Rope3dTileConfig(
            [1, 64],
            [1, 64, 64],
            [1, 64, 128, 128]
        ),
        c1_tile = [[64, 64], [64, 64], [512, 512]],
        c2_tile = [[128, 128], [128, 128], [256, 256]]
    )

    # call main function
    attention_post_decode(*pto_inputs, *pto_outputs, tile_config)

    torch_npu.npu.synchronize()
    compare(hidden_states.cpu(),
            golden_list[2], "hidden_states", atol=0.0001, rtol=0.005)


def do_attention_post_func_torch_graph(inputs, params, golden_list):
    """
    atten_res: (t, n_q, d), bf16
    cos: (t, rope_dim), bf16
    sin: (t, rope_dim), bf16
    wo_a: (n_g, n_q * d // n_g, o_lora_rank), bf16
    wo_b: (n_g * o_lora_rank, h)
    """
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    t = params.get("t")
    h = params.get("h")

    # define npu inputs
    atten_res_npu = inputs[0].npu()
    cos_npu = inputs[1].npu()
    sin_npu = inputs[2].npu()
    wo_a_npu = inputs[3].npu()
    wo_b_npu = inputs[4].npu()
    wo_b_nz = torch_npu.npu_format_cast(wo_b_npu, torch_npu.Format.FRACTAL_NZ)

    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    compiler_config = CompilerConfig()
    compiler_config.mode = "reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    model = torch.compile(AttentionPostV4(), dynamic=False, fullgraph=True, backend=npu_backend)
    
    hidden_states = model(atten_res_npu, cos_npu, sin_npu, wo_a_npu, wo_b_nz)
    pypto.runtime._device_synchronize()

    compare(hidden_states.cpu(), golden_list[2], "hidden_states", atol=0.0001, rtol=0.005)


def get_case_config(case_name: str):
    test_case_config = {
        "test_attention_post_v4_impl_perf":
            {"t": 16, "n_q": 64, "d": 512, "rope_dim": 64, "n_g": 8, "o_lora_rank": 1024, "h": 4096},
        "test_attention_post_v4_impl_prec":
            {"t": 249, "n_q": 64, "d": 512, "rope_dim": 64, "n_g": 8, "o_lora_rank": 1024, "h": 4096},
    }
    case_config = test_case_config.get(case_name)
    return case_config


def do_attention_post_entry(case_name: str, is_torch_graph: bool = False):
    dtype = torch.bfloat16
    params = get_case_config(case_name)
    if not params:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

    inputs, rope_golden, bmm_golden, mm_golden, nope_res = gen_attention_post_v4_golden(
        dtype, params)

    if is_torch_graph:
        print("\n =============== torch graph ====================")
        do_attention_post_func_torch_graph(
            inputs, params, [rope_golden, bmm_golden, mm_golden, nope_res])
    else:
        print("\n =============== st ====================")
        do_attention_post_func(
            inputs, params, [rope_golden, bmm_golden, mm_golden, nope_res])

    return True


def test_attention_post_v4_impl_prec():
    '''
    attention post v4 泛化精度用例
    '''
    do_attention_post_entry("test_attention_post_v4_impl_prec", is_torch_graph=False)


def test_attention_post_v4_impl():
    '''
    attention post v4 testcase
    '''
    do_attention_post_entry("test_attention_post_v4_impl_perf", is_torch_graph=False)


def test_attention_post_v4_impl_torch_graph():
    '''
    attention post v4 torch graph
    '''
    do_attention_post_entry("test_attention_post_v4_impl_perf", is_torch_graph=True)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_attention_post_v4_impl()
