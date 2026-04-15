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

def create_tensor_with_stride_padding(src_tensor, pad_len):
    device = src_tensor.device
    dtype = src_tensor.dtype
    shape = list(src_tensor.shape) # [65, 128, 1, 512]
    
    # 1. 计算 Stride
    row_logical_size = shape[1] * shape[2] * shape[3]
    stride_0 = row_logical_size + pad_len
    new_strides = [stride_0, shape[2] * shape[3], shape[3], 1]
    
    # 2. 直接在 NPU 上分配物理存储并填充 NaN
    # 注意：使用 torch.full 确保底层分配后立即初始化
    physical_numel = stride_0 * shape[0]
    storage_tensor = torch.full((physical_numel,), float('nan'), 
                                dtype=dtype, device=device)
    
    # 获取原始存储引用
    raw_storage = storage_tensor.untyped_storage()

    # 3. 构造视图并绑定 Storage
    target_tensor = torch.empty((0,), dtype=dtype, device=device)
    target_tensor.set_(raw_storage, 0, shape, new_strides)

    # 4. 核心修正：逐行拷贝 + 强制同步
    # 在 NPU 上，连续的 copy_ 可能会被合并。通过循环并确保每一行是独立 view 拷贝。
    for i in range(shape[0]):
        # .narrow(0, i, 1) 或者直接 target_tensor[i]
        # 确保 src[i] 到 target_tensor[i] 的拷贝严格遵守 stride
        target_tensor[i].copy_(src_tensor[i])

    # 5. 关键点：NPU 是异步的，打印前必须同步，否则可能看到的是内存旧值
    torch_npu.npu.synchronize()

    # --- 验证打印 ---
    # 同样在打印前，我们需要确保拿到的 storage_tensor 反映了最新状态
    # 逻辑结束索引 (第一行) 65536
    gap_start = row_logical_size
    gap_end = row_logical_size + pad_len
    
    print(f"--- NPU Padding Check (Pad Len: {pad_len}) ---")
    # 打印那段物理间隙
    print(f"NPU Physical Gap ({gap_start-2}:{gap_end+2}):\n", storage_tensor[gap_start-2:gap_end+2])
    
    # 如果还是没有 nan，尝试下面的“强制间隙置 nan”法
    return target_tensor

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
    if 'cu_seqlens_kv' in tensor_input and tensor_input['cu_seqlens_kv'] is not None:
        cu_seqlens_kv = tensor_input['cu_seqlens_kv']
    else:
        cu_seqlens_kv = torch.tensor([])
    if 'cu_seqlens_cmp_kv' in tensor_input and tensor_input['cu_seqlens_cmp_kv'] is not None:
        cu_seqlens_cmp_kv = tensor_input['cu_seqlens_cmp_kv']
    else:
        cu_seqlens_cmp_kv = torch.tensor([])
    seqused_q = seqused_q.npu()
    cu_seqlens_kv = cu_seqlens_kv.npu()
    cu_seqlens_cmp_kv = cu_seqlens_cmp_kv.npu()
    seqused_kv = tensor_input['seqused_kv'].npu()
    sinks = tensor_input['sinks'].npu()
    softmax_scale = tensor_input['softmax_scale']
    ori_mask_mode = tensor_input['ori_mask_mode']
    cmp_mask_mode = tensor_input['cmp_mask_mode']
    ori_win_left = tensor_input['ori_win_left']
    ori_win_right = tensor_input['ori_win_right']
    layout_q = tensor_input['layout_q'] if type(tensor_input['layout_q']) == type('TND') else tensor_input['layout_q'][0]
    layout_kv = tensor_input['layout_kv']
    if layout_kv == "PA_ND":
        ori_k_in_pa_shape = create_tensor_with_stride_padding(tensor_input['ori_kv'].npu(), 64)
        cmp_k_in_pa_shape = create_tensor_with_stride_padding(tensor_input['cmp_kv'].npu(), 64) if tensor_input['cmp_kv'] is not None else None
    else:
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
            cu_seqlens_kv=torch.tensor([]).npu() if layout_kv != "TND" else cu_seqlens_kv,
            cu_seqlens_cmp_kv=torch.tensor([]).npu() if layout_kv != "TND" else cu_seqlens_cmp_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv if layout_kv != "TND" else torch.tensor([]).npu(),
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
                                                            cu_seqlens_kv=None if tensor_input['cu_seqlens_kv'] is None else cu_seqlens_kv,
                                                            cu_seqlens_cmp_kv=None if tensor_input['cu_seqlens_cmp_kv'] is None else cu_seqlens_cmp_kv,
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
            cu_seqlens_kv=torch.tensor([]).npu() if layout_kv != "TND" else cu_seqlens_kv,
            cu_seqlens_cmp_kv=torch.tensor([]).npu() if layout_kv != "TND" else cu_seqlens_cmp_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv if layout_kv != "TND" else torch.tensor([]).npu(),
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
                                                            cu_seqlens_kv=None if tensor_input['cu_seqlens_kv'] is None else cu_seqlens_kv,
                                                            cu_seqlens_cmp_kv=None if tensor_input['cu_seqlens_cmp_kv'] is None else cu_seqlens_cmp_kv,
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
            cu_seqlens_kv=torch.tensor([]).npu() if layout_kv != "TND" else cu_seqlens_kv,
            cu_seqlens_cmp_kv=torch.tensor([]).npu() if layout_kv != "TND" else cu_seqlens_cmp_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv if layout_kv != "TND" else torch.tensor([]).npu(),
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
                                                                cu_seqlens_kv=None if tensor_input['cu_seqlens_kv'] is None else cu_seqlens_kv,
                                                                cu_seqlens_cmp_kv=None if tensor_input['cu_seqlens_cmp_kv'] is None else cu_seqlens_cmp_kv,
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