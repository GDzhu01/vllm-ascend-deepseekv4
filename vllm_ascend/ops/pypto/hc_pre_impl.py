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
Hello World Example for PyPTO

This example demonstrates the simplest tensor addition.
"""

import pypto
import torch
from torch._dynamo import allow_in_graph

def rms_norm_denom(x: pypto.Tensor) -> pypto.Tensor:
    norm_eps = 1e-6
    # print("rms_norm_denom input shape ", x.shape)
    # Compute RMS: sqrt(mean(x^2) + eps)
    squared = x * x
    mean_sq = pypto.sum(squared, dim=-1, keepdim=True)
    mean_sq = mean_sq / x.shape[-1]
    rms = pypto.sqrt((mean_sq + norm_eps))
    return rms


def sigmoid(x: pypto.Tensor) -> pypto.Tensor:
    # sigmoid(x) = 1 / (1 + exp(-x))
    x_neg = pypto.mul(x, -1.0)
    exp_neg = pypto.exp(x_neg)
    ones = pypto.full(exp_neg.shape, 1.0, exp_neg.dtype, valid_shape=exp_neg.shape)
    sigmoid = pypto.div(ones, exp_neg + 1.0)
    return sigmoid


def hc_split_sinkhorn(x: pypto.Tensor, hc_scale: pypto.Tensor, hc_base: pypto.Tensor, hc, hc_eps) \
    -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    sinkhorn_iters = 20
    tile_t, _ = x.shape # (tile_t, 24)
    # print("x.shape ", x.shape)

    comb_flag = (x[:, 2*hc: ] * (hc_scale[2:3].reshape([1, 1]).expand_clone([tile_t, 1])) + hc_base[:, 2*hc: ])
    comb_flag = comb_flag.reshape([tile_t, hc, hc]) # (tile_t, 4, 4)

    if tile_t <= 20:
        pypto.set_vec_tile_shapes(1, 16, 16)
    elif tile_t <= 64:
        pypto.set_vec_tile_shapes(4, 16, 16)
    else:
        pypto.set_vec_tile_shapes(128, 16, 16)

    row_max = pypto.amax(comb_flag, -1, True)   # (tile_t, 4, 1)
    comb_flag = pypto.exp(comb_flag - row_max)    # (tile_t, 4, 4)

    row_sum = pypto.sum(comb_flag, -1, True)    # (tile_t, 4, 1)
    comb_flag = comb_flag / row_sum + hc_eps # (tile_t, 4, 4)
    col_sum = pypto.sum(comb_flag, -2, True) # (tile_t, 1, 4)
    comb_flag = comb_flag / (col_sum + hc_eps) # (tile_t, 4, 4)

    for _ in range(sinkhorn_iters - 1):
        row_sum = comb_flag.sum(-1, keepdim=True) # (tile_t, 4, 4)
        comb_flag = comb_flag / (row_sum + hc_eps) # (tile_t, 4, 4)
        col_sum = comb_flag.sum(-2, keepdim=True) # (tile_t, 4, 4)
        comb_flag = comb_flag / (col_sum + hc_eps) # (tile_t, 4, 4)
    return comb_flag


@pypto.jit(
    host_options={"only_codegen": True}
    # for acl graph
    # runtime_options={
    #    "stitch_cfgcache_size": 2500000                 
    # }
)
def hc_pre_kernel(x: pypto.Tensor, hc_fn: pypto.Tensor, hc_scale: pypto.Tensor, hc_base_: pypto.Tensor,
                y: pypto.Tensor, post: pypto.Tensor, comb: pypto.Tensor,
):
    # pypto.set_debug_options(runtime_debug_mode=1)
    # pypto.set_debug_options(runtime_debug_mode=2)   ## for acl graph

    t = x.shape[0]
    hc = x.shape[1]
    d = x.shape[2]
    mix_hc = (2 + hc) * hc
    hc_eps = 1e-6

    ### check shape
    assert hc == 4, f"hc is {hc}, expected 4"
    assert d == 4096, f"d is {d}, expected 4096"
    assert mix_hc == hc_fn.shape[0], f"mix_hc is {hc_fn.shape[0]}, expected 24"
    assert hc_scale.shape[0] == 3, f"hc_scale.shape[0] is {hc_scale.shape[0]}, expected 3"

    # unroll_list = [16, 1]
    unroll_list=[1024, 256, 64, 16, 4, 1]

    for _ in pypto.loop(1):
        x_2d = pypto.reshape(x, [t, hc*d], inplace=True)
        hc_base= pypto.reshape(hc_base_, [1, mix_hc], inplace=True)
    # print("t in kernel is ", t)
    for t_idx, unrollLength in pypto.loop_unroll(0, t, 1, name="t_loop", idx_name="t_idx", unroll_list=unroll_list):
        tile_t = unrollLength
        # print("========================= tile_t: ", tile_t)
        # print("========================= t_idx: ", t_idx)

        pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128])
        tile_shapes_1 = [16, 512]
        tile_shape_2 = 64
        if tile_t <= 16:
            tile_shapes_1 = [2, 1024]
            tile_shape_2 = 128
            pypto.set_cube_tile_shapes([8, 8], [1024, 1024], [128, 128])
        elif tile_t <= 64:
            tile_shapes_1 = [8, 1024]
            tile_shape_2 = 32
        else:
            tile_shapes_1 = [16, 512]
            tile_shape_2 = 128

        pypto.set_vec_tile_shapes(tile_shapes_1[0], tile_shapes_1[1])

        x_view = pypto.view(x_2d, [tile_t, hc*d], [t_idx, 0])
        x_fp32 = pypto.cast(x_view, pypto.DT_FP32)
        rms_res = rms_norm_denom(x_fp32)    ## (t, hc*d) -> (t, 1)

        pypto.set_vec_tile_shapes(tile_shape_2, 16)
        mm_res = pypto.matmul(x_view, hc_fn, pypto.DT_BF16, b_trans=True)   # (t, hc*d) @ (mix_hc, hc*d)^t = (t, mix_hc)
        mm_res = pypto.cast(mm_res, pypto.DT_FP32)

        rms_res = mm_res / rms_res  ## t, mix_hc

        pre = rms_res[:, :hc] * (hc_scale[0:1].reshape([1, 1]).expand_clone([tile_t, 1])) + hc_base[:, :hc] # (tile_t, 4)
        pre = sigmoid(pre) + hc_eps # (tile_t, 4)

        pre_3d = pre.reshape([tile_t, hc, 1])
        x_fp32_3d = x_fp32.reshape([tile_t, hc, d])
        pypto.set_vec_tile_shapes(tile_shape_2, 16, 16)

        mul_res = pre_3d * x_fp32_3d
        res_fp32 = pypto.sum(mul_res, dim=-2)
        res_bf16 = pypto.cast(res_fp32, pypto.DT_BF16)
        pypto.assemble(res_bf16, [t_idx, 0], y)

        post_ = rms_res[:, hc: 2*hc] * (hc_scale[1:2].reshape([1, 1]).expand_clone([tile_t, 1])) + hc_base[:, hc: 2*hc] # (tile_t, 4)
        post_ = sigmoid(post_) * 2.0 # (tile_t, 4)
        pypto.assemble(post_, [t_idx, 0], post)

        comb_ = hc_split_sinkhorn(rms_res, hc_scale, hc_base, hc, hc_eps)   # (tile_t, hc), (tile_t, hc), (tile_t, hc, hc)
        pypto.assemble(comb_, [t_idx, 0, 0], comb)

def check_input_output_shape_dtype(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
    assert x.dim() == 3 and x.size(1) == 4 and x.size(2) == 4096,\
        f"expected x dim num {x.dim()}, x axis1 {x.size(1)}, x axis2 {x.size(2)}"
    assert hc_fn.dim() == 2 and hc_fn.size(0) == 24 and hc_fn.size(1) == 4 * 4096,\
        f"expected hc_fn dim num 2, hc_fn axis0 24, hc_fn axis1 12384"
    assert hc_scale.dim() == 1 and hc_scale.size(0) == 3, f"expected hc_scale dim num 1, hc_scale axis0 3"
    assert hc_base.dim() == 1 and hc_base.size(0) == 24, f"expected hc_scale dim num 1, hc_scale axis0 24"

    assert x.dtype == torch.bfloat16, f"x.dtype is {x.dtype}, expected torch.bfloat16"
    assert hc_fn.dtype == torch.bfloat16, f"hc_fn.dtype is {hc_fn.dtype}, expected torch.bfloat16"
    assert hc_scale.dtype == torch.float32, f"hc_scale.dtype is {hc_scale.dtype}, expected torch.float32"
    assert hc_base.dtype == torch.float32, f"hc_base.dtype is {hc_base.dtype}, expected torch.float32"


@allow_in_graph
def npu_hc_pre(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor)\
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # print("x.shape in npu_hc_pre", x.shape)
    ### check dtype
    check_input_output_shape_dtype(x, hc_fn, hc_scale, hc_base)

    y = torch.zeros([x.size(0), x.size(2)], dtype=x.dtype, device=f'{x.device}')
    post = torch.zeros([x.size(0), x.size(1)], dtype=hc_scale.dtype, device=f'{x.device}')
    comb = torch.zeros([x.size(0), x.size(1), x.size(1)], dtype=hc_scale.dtype, device=f'{x.device}')

    in_outs = {
        x: [0],
        hc_fn: None,
        hc_scale: None,
        hc_base: None,
        y:[0],
        post:[0],
        comb:[0],
    }

    pto_in_outs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in in_outs.items()]
    hc_pre_kernel(*pto_in_outs)

    return y, post, comb


