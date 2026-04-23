
#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import pandas as pd 
import numpy as np
import torch
import torch_npu
import pytest
import random
import math
import ast
import custom_ops as ops


def test_compressor_process(filepath, device_id=0):
    # 加载测试数据
    # test_data = torch.load(filepath, map_location="cpu")
    test_data = torch.load(filepath, map_location="cpu",weights_only=False)

    params = test_data['params']
    cpu_result = test_data['cpu_result']
    kv_mask_result = test_data['kv_mask_result']
    update_kv = test_data['update_kv']
    update_score = test_data['update_score']
    cpu_kv_state = test_data['cpu_kv_state']
    cpu_score_state = test_data['cpu_score_state']

    print("执行用例：", filepath)
    torch_npu.npu.set_device(device_id)

    x =test_data['x']
    wkv = test_data['wkv']
    wgate = test_data['wgate']
    kv_state = test_data['kv_state']
    score_state = test_data['score_state']
    ape = test_data['ape']
    norm_weight = test_data['norm_weight']
    rope_sin = test_data['rope_sin']
    rope_cos = test_data['rope_cos']
    block_table = test_data['block_table']
    cu_seqlens = test_data['cu_seqlens']
    seqused = test_data['seqused']
    start_pos = test_data['start_pos']
    rope_head_dim = test_data['rope_head_dim']
    cmp_ratio = test_data['cmp_ratio']
    coff = test_data['coff']
    norm_eps = test_data['norm_eps']
    rotary_mode = test_data['rotary_mode']
    cache_mode = test_data['cache_mode']

    if cache_mode == 1:  # 连续buffer
        state_cache = torch.zeros((kv_state.shape[0], kv_state.shape[1], 2*kv_state.shape[2]))
        # state_cache = state_cache.to("npu:%s" % DEVICE_ID)
        state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()
    else:
        layer_num = random.randint(1, 50)
        layer_start_idx = random.randint(0, layer_num-1)
        print(f"layer_num: {layer_num}")
        print(f"layer_start_idx: {layer_start_idx}")
        state_cache_pad = torch.zeros((kv_state.shape[0], layer_num*kv_state.shape[1]*kv_state.shape[2]*2))
        print(f"state_cache_pad: shape {state_cache_pad.shape}")
        # state_cache_pad = state_cache_pad.to("npu:%s" % DEVICE_ID)
        state_cache = state_cache_pad[:, layer_start_idx*kv_state.shape[1]*kv_state.shape[2]*2: (layer_start_idx+1)*kv_state.shape[1]*kv_state.shape[2]*2].view(-1, kv_state.shape[1], kv_state.shape[2]*2)
        # state_cache = state_cache.to("npu:%s" % DEVICE_ID)
        state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()
        print(f"state_cache: shape {state_cache.shape}, dtype: {state_cache.dtype}, is_contiguous: {state_cache.is_contiguous()}, stride0: {state_cache.stride(0)}")


    state_cache = state_cache.npu()
    if cu_seqlens is not None: 
        cu_seqlens = cu_seqlens.npu()
    if seqused is not None: 
        seqused = seqused.npu()
    if start_pos is not None: 
        start_pos = torch.tensor(start_pos).to(torch.int32).npu()
    if block_table is not None: 
        kv_block_table = block_table.npu()
        score_block_table = block_table.npu()

    #调用compressor算子
    npu_result = torch.ops.custom.compressor(
                x.npu(),
                wkv.npu(),
                wgate.npu(),
                state_cache,
                ape.npu(),
                norm_weight.npu(), 
                rope_sin.npu(),
                rope_cos.npu(),
                rope_head_dim = rope_head_dim,
                cmp_ratio = cmp_ratio,
                state_block_table = kv_block_table,
                cu_seqlens = cu_seqlens,
                seqused = seqused,
                start_pos = start_pos,
                coff = coff,
                norm_eps = norm_eps,
                rotary_mode = rotary_mode,
                cache_mode = cache_mode
            )
    return cpu_result, kv_mask_result, npu_result ,cpu_kv_state, update_kv, cpu_score_state, update_score, state_cache, params
