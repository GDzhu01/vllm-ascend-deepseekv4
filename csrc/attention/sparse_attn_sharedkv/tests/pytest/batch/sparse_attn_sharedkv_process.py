#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
import numpy as np
import os
import pytest
import random
import torch
import torch_npu
import custom_ops as ops

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

def call_npu(input_data):
    params = input_data['params']
    metadata_input = input_data['metadata_input']
    tensor_input = input_data['input']
    print("用例参数: ", params)

    # metadata解析
    K = metadata_input['K']
    cmp_ratio = metadata_input['cmp_ratio']
    N1 = metadata_input['N1']
    N2 = metadata_input['N2']
    D = metadata_input['D']
    B = metadata_input['B']

    # tensor解析
    q = tensor_input['q'].npu()
    ori_kv = tensor_input['ori_kv'].npu()
    ori_block_table = tensor_input['ori_block_table'].npu()
    if 'cu_seqlens_q' in tensor_input and tensor_input['cu_seqlens_q'] is not None:
        cu_seqlens_q = tensor_input['cu_seqlens_q']
    else:
        cu_seqlens_q = torch.tensor([])
    cu_seqlens_q = cu_seqlens_q.npu()
    used_seqused_q_flag = False
    if 'seqused_q' in tensor_input and tensor_input['seqused_q'] is not None:
        seqused_q = tensor_input['seqused_q']
        used_seqused_q_flag = True
    else:
        seqused_q = torch.tensor([])
    seqused_q = seqused_q.npu()
    seqused_kv = tensor_input['seqused_kv'].npu()
    sinks = tensor_input['sinks'].npu()
    softmax_scale = tensor_input['softmax_scale']
    ori_mask_mode = tensor_input['ori_mask_mode']
    cmp_mask_mode = tensor_input['cmp_mask_mode']
    ori_win_left = tensor_input['ori_win_left']
    ori_win_right = tensor_input['ori_win_right']
    layout_q = tensor_input['layout_q'] if type(tensor_input['layout_q']) == type('TND') else tensor_input['layout_q'][0]
    layout_kv = tensor_input['layout_kv']
    ori_k_in_pa_shape = tensor_input['ori_kv'].npu()
    cmp_k_in_pa_shape = tensor_input['cmp_kv'].npu() if tensor_input['cmp_kv'] is not None else None
    max_seqlen_q = metadata_input['max_seqlen_q']
    ori_max_s2 = metadata_input['max_seqlen_kv']
    cmp_sparse_indices = tensor_input['cmp_sparse_indices']
    cmp_block_table = tensor_input['cmp_block_table']

    # 路由到三个算子的逻辑：
    template_idx = 0
    if K is None :
        if cmp_ratio is None:
            template_idx = 0  # SWA
        else:
            template_idx = 1  # CFA
    else:
        template_idx = 2  # SCFA

    if template_idx == 1 or template_idx == 2:
        cmp_k_in_pa_shape = cmp_k_in_pa_shape.npu()
        if cmp_block_table is not None:
            cmp_block_table = cmp_block_table.npu()
    if template_idx == 2:
        cmp_sparse_indices = cmp_sparse_indices.npu()

    if template_idx == 1 or template_idx == 2:
        cmp_k_in_pa_shape = cmp_k_in_pa_shape.npu()
        if cmp_block_table is not None:
            cmp_block_table = cmp_block_table.npu()
    if template_idx == 2:
        cmp_sparse_indices = cmp_sparse_indices.npu()

    q = q.npu()
    ori_k_in_pa_shape = ori_k_in_pa_shape.npu()
    ori_block_table = ori_block_table.npu()
    cu_seqlens_q = cu_seqlens_q.npu()
    seqused_kv = seqused_kv.npu()
    sinks = sinks.npu()

    if template_idx == 0:
        metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=N1,
            num_heads_kv=N2,
            head_dim=D,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=torch.tensor([]).npu(),
            cu_seqlens_cmp_kv=torch.tensor([]).npu(),
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            batch_size=B,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=ori_max_s2,
            ori_mask_mode=ori_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            has_ori_kv=ori_k_in_pa_shape != None,
            has_cmp_kv=cmp_k_in_pa_shape != None,
            device = "npu:0")
        npu_result, softmax_lse = torch.ops.custom.npu_sparse_attn_sharedkv(q,
                                                            ori_kv=ori_k_in_pa_shape,
                                                            ori_block_table=ori_block_table,
                                                            cu_seqlens_q=cu_seqlens_q if layout_q == 'TND' else None,
                                                            seqused_q=seqused_q if used_seqused_q_flag else None,
                                                            seqused_kv=seqused_kv,
                                                            sinks=sinks,
                                                            metadata=metadata,
                                                            softmax_scale=softmax_scale,
                                                            ori_mask_mode=ori_mask_mode,
                                                            ori_win_left=ori_win_left,
                                                            ori_win_right=ori_win_right,
                                                            layout_q=layout_q,
                                                            layout_kv=layout_kv)
    elif template_idx == 1:
        metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=N1,
            num_heads_kv=N2,
            head_dim=D,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=torch.tensor([]).npu(),
            cu_seqlens_cmp_kv=torch.tensor([]).npu(),
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            batch_size=B,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=ori_max_s2,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            has_ori_kv=ori_k_in_pa_shape != None,
            has_cmp_kv=cmp_k_in_pa_shape != None,
            device = "npu:0")
        npu_result, softmax_lse = torch.ops.custom.npu_sparse_attn_sharedkv(q,
                                                            ori_kv=ori_k_in_pa_shape,
                                                            cmp_kv=cmp_k_in_pa_shape,
                                                            ori_block_table=ori_block_table,
                                                            cmp_block_table=cmp_block_table,
                                                            cu_seqlens_q=cu_seqlens_q if layout_q == 'TND' else None,
                                                            seqused_q=seqused_q if used_seqused_q_flag else None,
                                                            seqused_kv=seqused_kv,
                                                            sinks=sinks,
                                                            metadata=metadata,
                                                            softmax_scale=softmax_scale,
                                                            cmp_ratio=cmp_ratio,
                                                            ori_mask_mode=ori_mask_mode,
                                                            cmp_mask_mode=cmp_mask_mode,
                                                            ori_win_left=ori_win_left,
                                                            ori_win_right=ori_win_right,
                                                            layout_q=layout_q,
                                                            layout_kv=layout_kv)
    else:
        metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=N1,
            num_heads_kv=N2,
            head_dim=D,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=torch.tensor([]).npu(),
            cu_seqlens_cmp_kv=torch.tensor([]).npu(),
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            batch_size=B,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=ori_max_s2,
            cmp_topk=K,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            has_ori_kv=ori_k_in_pa_shape != None,
            has_cmp_kv=cmp_k_in_pa_shape != None,
            device = "npu:0")
        npu_result, softmax_lse = torch.ops.custom.npu_sparse_attn_sharedkv(q,
                                                                ori_kv=ori_k_in_pa_shape,
                                                                cmp_kv=cmp_k_in_pa_shape,
                                                                cmp_sparse_indices=cmp_sparse_indices,
                                                                ori_block_table=ori_block_table,
                                                                cmp_block_table=cmp_block_table,
                                                                cu_seqlens_q=cu_seqlens_q if layout_q == 'TND' else None,
                                                                seqused_q=seqused_q if used_seqused_q_flag else None,
                                                                seqused_kv=seqused_kv,
                                                                sinks=sinks,
                                                                metadata=metadata,
                                                                softmax_scale=softmax_scale,
                                                                cmp_ratio=cmp_ratio,
                                                                ori_mask_mode=ori_mask_mode,
                                                                cmp_mask_mode=cmp_mask_mode,
                                                                ori_win_left=ori_win_left,
                                                                ori_win_right=ori_win_right,
                                                                layout_q=layout_q,
                                                                layout_kv=layout_kv)

    torch.npu.synchronize()
    return npu_result, softmax_lse