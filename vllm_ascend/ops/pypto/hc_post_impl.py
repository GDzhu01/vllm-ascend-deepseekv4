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
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch_npu
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto


@dataclass
class HcPostTileConfig: 
    def __init__(self): 
        self.tile_b = 8
        self.unroll_list = [32, 16, 8, 4, 2, 1]


def hc_post_compute(
    x: pypto.tensor,
    residual: pypto.tensor, 
    post: pypto.tensor, 
    comb: pypto.tensor,
    y: pypto.tensor,
    tile_config: HcPostTileConfig):
    assert len(x.shape) == 2 and len(residual.shape) == 3 and len(post.shape) == 2 and len(comb.shape) == 3

    dtype = x.dtype
    t = x.shape[0]
    hc = residual.shape[1]
    d = residual.shape[2]

    post_reshape = pypto.reshape(post, [t, hc, 1], inplace=True)
    x_reshape = pypto.reshape(x, [t, 1, d], inplace=True)
    comb_reshape = pypto.reshape(comb, [t, hc, hc, 1], inplace=True)
    residual_reshape = pypto.reshape(residual, [t, hc, 1, d], inplace=True)

    assert hc == 4 and d == 4096
    for t_idx, unrollLength in pypto.loop_unroll(0, t, 1, name="LI_LOOP_BATCH", idx_name="t_idx",
                                                unroll_list=tile_config.unroll_list, ):
        t_tile = unrollLength
        pypto.set_vec_tile_shapes(1, 4, 1)
        post_slice = pypto.view(post_reshape, [t_tile, hc, 1], [t_idx, 0, 0])
        pypto.set_vec_tile_shapes(1, 4, 2048)
        x_slice = pypto.view(x_reshape, [t_tile, 1, d], [t_idx, 0, 0])
        x_slice_fp32 = pypto.cast(x_slice, pypto.DT_FP32)
        x_slice_expand = pypto.expand_clone(x_slice_fp32, [t_tile, 4, d])
        post_res = post_slice * x_slice_expand

        pypto.set_vec_tile_shapes(1, 4, 1, 2048)
        residual_slice = pypto.view(residual_reshape, [t_tile, hc, 1, d], [t_idx, 0, 0, 0])
        pypto.set_vec_tile_shapes(1, 4, 4, 2048)
        comb_slice = pypto.view(comb_reshape, [t_tile, hc, hc, 1], [t_idx, 0, 0, 0])

        residual_slice_expand = pypto.expand_clone(residual_slice, [t_tile, hc, 4, d])
        residual_res = residual_slice_expand * comb_slice

        residual_reduce = pypto.sum(residual_res, 1)
        pypto.set_vec_tile_shapes(1, 4, 2048)
        y_tmp = pypto.add(post_res, residual_reduce)
        y_dtype = pypto.cast(y_tmp, pypto.DT_BF16)
        pypto.assemble(y_dtype, [t_idx, 0, 0], y)


def check_input_output_shape_dtype(x: torch.tensor, residual: torch.tensor, post: torch.tensor, comb: torch.tensor, y: torch.tensor):
    assert x.size(1) == 4096 and x.dim() == 2, f"expected x dim num 2, x axis1 4096"
    assert residual.dim() == 3 and residual.size(1) == 4 and residual.size(2) == 4096, f"expected residual dim num 3, residual axis1 4, residual axis1 4096"
    assert post.dim() == 2 and post.size(1) == 4, f"expected post dim num 2, post axis1 4"
    assert comb.dim() == 3 and comb.size(1) == 4 and comb.size(2) == 4, f"expected comb dim num 3, comb axis1 4, comb axis2 4"
    assert y.dim() == 3 and y.size(1) == 4 and y.size(2) == 4096, f"expected y dim num 3, y axis1 4, y axis2 4096"

    assert x.dtype == torch.bfloat16, f"x.dtype is {x.dtype}, expected torch.bfloat16"
    assert residual.dtype == torch.float32, f"residual.dtype is {residual.dtype}, expected torch.float32"
    assert post.dtype == torch.float32,  f"post.dtype is {post.dtype}, expected torch.float32"
    assert comb.dtype == torch.float32, f"comb.dtype is {comb.dtype}, expected torch.float32"
    assert y.dtype == torch.bfloat16, f"y.dtype is {y.dtype}, expected torch.bfloat16"


@pypto.jit(
    host_options={"only_codegen": True},
    # runtime_options={
    #     "stitch_cfgcache_size": 2500000
    # }
)
def hc_post_kernel(
    x: pypto.tensor,
    residual: pypto.tensor,
    post: pypto.tensor,
    comb: pypto.tensor,
    y: pypto.tensor,
    tile_config: HcPostTileConfig):
    pypto.experimental.set_operation_config(combine_axis=True)
    hc_post_compute(x, residual, post, comb, y, tile_config)


@allow_in_graph
def npu_hc_post(
    x: torch.tensor,
    residual: torch.tensor,
    post: torch.tensor,
    comb: torch.tensor):

    tile_config = HcPostTileConfig()
    y = torch.zeros([x.size(0), residual.size(1), residual.size(2)], dtype=x.dtype, device=f'{x.device}')
    inputs = {
        x: [0],
        residual: [0],
        post: [0],
        comb: [0],
    }
    outputs = {
        y: [0]
    }

    check_input_output_shape_dtype(x, residual, post, comb, y)
    if not isinstance(x, FakeTensor):
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        hc_post_kernel(*pto_inputs, *pto_outputs, tile_config)

    return y