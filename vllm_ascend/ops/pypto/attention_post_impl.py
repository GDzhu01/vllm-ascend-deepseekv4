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
Attention Post Module

This module implements Attention Post with RoPE for DeepSeek V4.

Main Functions:
    - attention_post_compute: Attention post with RoPE computation
    - attention_post_decode: JIT-compiled decode version

Example:
    See testdV4_attention_post.py for usage examples.
"""
from dataclasses import dataclass
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
import torch

SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3


@dataclass
class Rope3dTileConfig:
    two_dim_tile: list
    three_dim_tile: list
    four_dim_tile: list


@dataclass
class AttnPostConfig:
    unroll_list: list
    rope3d_tile_config: Rope3dTileConfig
    c1_tile: list
    c2_tile: list


def rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    """Rotate half of the tensor dimensions for RoPE computation.

    Splits the last dimension in half and applies rotation transformation:
    [-x2, x1] where x1 is the first half and x2 is the second half.
    This is a key component of RoPE (Rotary Position Embedding).

    Args:
        input_tensor: Input tensor with last dimension divisible by 2

    Returns:
        Rotated tensor with same shape as input

    Raises:
        AssertionError: If input dimension is less than 1 or last dimension
                       is not divisible by 2
    """
    chunk_size = 2
    shape = input_tensor.shape
    shape_size = len(shape)
    assert shape_size >= 1, "rope rotate_half input dim less than 1"
    assert shape[shape_size - 1] % chunk_size == 0, "rope rotate_half last dim shape is even"

    new_shape = list(shape)
    new_shape[shape_size - 1] //= chunk_size

    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = new_shape[shape_size - 1]

    x1 = pypto.view(input_tensor, new_shape, offset1)
    x2 = pypto.view(input_tensor, new_shape, offset2)

    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def interleaved_rope_3d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor, rope_3d_config: Rope3dTileConfig) -> pypto.Tensor:
    """Apply 3D Rotary Position Embedding (RoPE).

    Implements RoPE transformation for 3D tensors with shape (batch, heads, dim).
    The RoPE is applied independently to each head using broadcasted cos/sin values.

    Args:
        x: Input tensor of shape (batch, heads, rope_dim)
        cos: Cosine values for RoPE, shape (batch, rope_dim)
        sin: Sine values for RoPE, shape (batch, rope_dim)

    Returns:
        Tensor with RoPE applied, same shape as input x

    Note:
        The function broadcasts cos and sin to match the head dimension,
        then applies rotation: x_rotated = x * cos + rotate_half(x) * sin
    """
    assert (len(x.shape) == SHAPE_DIM_3 and len(cos.shape) == SHAPE_DIM_2 and len(sin.shape) == SHAPE_DIM_2)

    pypto.set_vec_tile_shapes(*rope_3d_config.two_dim_tile) # (1, 64)
    cast_cos = pypto.cast(cos, pypto.DataType.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DataType.DT_FP32)

    pypto.set_vec_tile_shapes(*rope_3d_config.three_dim_tile) # (1, 64, 64)
    cast_x = pypto.cast(x, pypto.DataType.DT_FP32)
    cast_cos = pypto.reshape(cast_cos, [x.shape[0], 1, x.shape[2]])
    cast_sin = pypto.reshape(cast_sin, [x.shape[0], 1, x.shape[2]])

    pypto.set_vec_tile_shapes(*rope_3d_config.four_dim_tile)  # (1, 64, 128, 128)
    x_view = pypto.reshape(cast_x, [x.shape[0], x.shape[1], x.shape[2] // 2, 2])
    x_trans = pypto.transpose(x_view, 2, 3)
    x_re_second = pypto.reshape(x_trans, x.shape)
    x_embed = x_re_second * cast_cos + rotate_half(x_re_second) * cast_sin

    return pypto.cast(x_embed, x.dtype)


def attention_post_compute(attn_res: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor,
                           wo_a: pypto.Tensor, wo_b: pypto.Tensor, hidden_states: pypto.Tensor,
                           tile_config: AttnPostConfig):
    """Attention Post compute.

    Args:
        group       name           dtype     shape                              format
        INPUT 0	    attn_res	   DT_BF16	 (t, n_q, dim)	                    ND	 
        INPUT 1	    cos	           DT_BF16	 (t, rope_dim)	                    ND	 
        INPUT 2	    sin	           DT_BF16	 (t, rope_dim)	                    ND	 
        INPUT 3	    wo_a	       DT_BF16	 (ng, n_q*dim // ng, o_lora_rank)	ND	 
        INPUT 4	    wo_b	       DT_BF16	 (ng*o_lora_rank, h)	            ND	 
        OUTPUT 1	hidden_states  DT_BF16	 (t, h)	                            ND
    Note:

    """
    assert len(attn_res.shape) == 3 and len(cos.shape) == 2 and len(sin.shape) == 2
    assert len(wo_a.shape) == 3 and len(wo_b.shape) == 2 and len(hidden_states.shape) == 2

    dtype = attn_res.dtype
    t = attn_res.shape[0]
    n_q = attn_res.shape[1]
    d = attn_res.shape[2]
    
    n_g = wo_a.shape[0]
    o_lora_rank = wo_a.shape[2]

    rope_dim = cos.shape[1]
    nope_dim = d - rope_dim

    h = hidden_states.shape[1]

    unroll_list = tile_config.unroll_list
    rope3d_tile_config = tile_config.rope3d_tile_config
    c1_tile = tile_config.c1_tile
    c2_tile = tile_config.c2_tile

    for t_idx, unrollLength in pypto.loop_unroll(0, t, 1, name="ATTN_POST_T_LOOP", idx_name="t_idx",
                                                 unroll_list=unroll_list):
        tile_t = unrollLength

        # for nope+rope
        tmp_tensor = pypto.tensor([tile_t, n_q, d], dtype, "tmp_tensor")

        # copy nope to tmp_tensor
        pypto.set_semantic_label("Rope_nope")
        pypto.set_vec_tile_shapes(1, 32, 512)
        atten_res_nope = pypto.view(attn_res, [tile_t, n_q, nope_dim], [t_idx, 0, 0]) # nope: (tile_t, n_q, 448)
        pypto.assemble(pypto.clone(atten_res_nope), [0, 0, 0], tmp_tensor)

        # apply rope to tmp_tensor
        pypto.set_semantic_label("Rope_rope")
        atten_res_rope = pypto.view(attn_res, [tile_t, n_q, rope_dim], [t_idx, 0, nope_dim]) # rope: (tile_t, n_q, 64)
        cos_in = pypto.view(cos, [tile_t, rope_dim], [t_idx, 0])
        sin_in = pypto.view(sin, [tile_t, rope_dim], [t_idx, 0])
        rope_result = interleaved_rope_3d(atten_res_rope, cos_in, sin_in, rope3d_tile_config)
        pypto.assemble(rope_result, [0, 0, nope_dim], tmp_tensor) # (tile_t, n_q, d)

        # bmm1 left transpose: (tile_t, n_q, d) -> (n_g, tile_t, n_q * d / n_g)
        pypto.set_semantic_label("bmm1_transpose")
        pypto.set_vec_tile_shapes(1, 32, 512)
        atten_reshape = pypto.reshape(tmp_tensor, [tile_t, n_g, n_q * d // n_g])
        pypto.set_vec_tile_shapes(1, 4, 4096)
        atten_trs = pypto.transpose(atten_reshape, 0, 1)

        # batch_matmul: (n_g, tile_t, n_q * d / n_g) @ (n_g, n_q * d / n_g, o_lora_rank) --> (n_g, tile_t, o_lora_rank)
        pypto.set_semantic_label("bmm1_cube")
        pypto.set_cube_tile_shapes(*c1_tile)
        attn_res_l = pypto.view(atten_trs, [n_g, tile_t, n_q * d // n_g], [0, 0, 0])
        bmm1_res = pypto.matmul(attn_res_l, wo_a, dtype, a_trans=False, b_trans=False)

        # bmm1 output transpose: (n_g, tile_t, o_lora_rank) -> (tile_t, n_g * o_lora_rank)
        pypto.set_semantic_label("mm2_transpose")
        pypto.set_vec_tile_shapes(1, 16, 1024)
        bmm1_res_tmp = pypto.view(bmm1_res, [n_g, tile_t, o_lora_rank], [0, 0, 0])
        bmm1_res_trs = pypto.transpose(bmm1_res_tmp, 0, 1)
        bmm1_res_reshape = pypto.reshape(bmm1_res_trs, [tile_t, n_g * o_lora_rank])

        # matmul: (t, n_g * o_lora_rank) @ (n_g * o_lora_rank, h) --> (tile_t, h)
        pypto.set_semantic_label("mm2_cube")
        pypto.set_cube_tile_shapes(*c2_tile)
        mm2_l = pypto.view(bmm1_res_reshape, [tile_t, n_g * o_lora_rank], [0, 0])
        mm2_res = pypto.matmul(mm2_l, wo_b, dtype, a_trans=False, b_trans=False)

        pypto.set_vec_tile_shapes(16, 1024)
        pypto.assemble(mm2_res, [t_idx, 0], hidden_states)


@pypto.jit(
    pass_options={
        "mg_copyin_upper_bound": 16 * 1024 * 1024,
        "pg_upper_bound": 80000,
        "pg_lower_bound": 512,
        "pg_parallel_lower_bound": 40,
        # "vec_nbuffer_mode": 2,
        # "vec_nbuffer_setting": {-1: 2},
    },
    runtime_options={
        "stitch_function_inner_memory": 128,
        "stitch_function_outcast_memory": 128,
        "stitch_cfgcache_size": 2500000
    }
)
def attention_post_decode(attn_res: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor,
                          wo_a: pypto.Tensor, wo_b: pypto.Tensor, hidden_states: pypto.Tensor,
                          tile_config: AttnPostConfig):
    """JIT-compiled attention post for decode phase.

    Args:

    Note:

    """
    attention_post_compute(attn_res, cos, sin, wo_a, wo_b, hidden_states, tile_config)

def check_input_output_shape_dtype(attn_res: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                            wo_a: torch.Tensor, wo_b: torch.Tensor, hidden_states: torch.Tensor):
    assert attn_res.size(1) == 64 and attn_res.size(2) == 512 and attn_res.dim() == 3, f"expected attn_res dim num 3, attn_res axis1 64, attn_res axis2 512"
    assert cos.size(1) == 64 and sin.size(1) == 64 and cos.dim() == 2 and sin.dim() == 2,\
        f"expected cos dim num 2, sin dim num 2, cos axis1 64, sin axis1 64"
    assert wo_a.size(0) == 8 and wo_a.size(1) == 4096 and wo_a.size(2) == 1024 and wo_a.dim() == 3,\
        f"expected wo_a dim num 3, wo_a axis0 8, wo_a axis1 4096, wo_a axis2 1024"
    assert wo_b.size(0) == 8 * 1024 and wo_b.size(1) == 4096 and wo_b.dim() == 2,\
        f"expected wo_b dim num 2, wo_b axis0 8192, wo_b axis1 4096"
    assert hidden_states.size(1) == 4096,\
        f"expected hidden_states dim num 2, hidden_states axis1 4096"

    assert attn_res.dtype == torch.bfloat16, f"attn_res.dtype is {attn_res.dtype}, expected torch.bfloat16"
    assert cos.dtype == torch.bfloat16, f"cos.dtype is {cos.dtype}, expected torch.bfloat16"
    assert sin.dtype == torch.bfloat16, f"sin.dtype is {sin.dtype}, expected torch.bfloat16"
    assert wo_a.dtype == torch.bfloat16, f"wo_a.dtype  is {wo_a.dtype}, expected torch.bfloat16"
    assert wo_b.dtype == torch.bfloat16, f"wo_b.dtype  is {wo_b.dtype}, expected torch.bfloat16"
    assert hidden_states.dtype == torch.bfloat16, f"hidden_states.dtype is {hidden_states.dtype}, expected torch.bfloat16"
    

@allow_in_graph
def npu_attention_post_v4(attn_res: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                          wo_a: torch.Tensor, wo_b: torch.Tensor):
    """
    torch npu graph interface

    """
    # mark dynamic_axis
    # define npu outputs
    hidden_states = torch.zeros([attn_res.size(0), wo_b.size(1)], dtype=attn_res.dtype, device=f'{attn_res.device}')

    check_input_output_shape_dtype(attn_res, cos, sin, wo_a, wo_b, hidden_states)
    atten_res_pto = pypto.from_torch(attn_res, dynamic_axis=[0], name="attn_res")
    cos_pto = pypto.from_torch(cos, dynamic_axis=[0], name="cos")
    sin_pto = pypto.from_torch(sin, dynamic_axis=[0], name="sin")
    wo_a_pto = pypto.from_torch(wo_a, name="wo_a")
    wo_b_pto = pypto.from_torch(wo_b, name="wo_b")
    hidden_states_pto = pypto.from_torch(hidden_states, dynamic_axis=[0], name="hidden_states")

    # tiling
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

    # kernel
    if not isinstance(attn_res, FakeTensor):
        pto_inputs = [atten_res_pto, cos_pto, sin_pto, wo_a_pto, wo_b_pto]
        pto_outputs = [hidden_states_pto]
        attention_post_decode(*pto_inputs, *pto_outputs, tile_config)
        
    return hidden_states
