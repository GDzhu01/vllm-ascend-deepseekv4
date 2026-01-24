#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import itertools
import torch
import torch_npu
from testcases_sas import ENABLED_PARAMS
import check_result
import sparse_attn_sharedkv_process
import pytest

# 处理所有参数组合
param_combinations = []

for params in ENABLED_PARAMS:
    # 确保所有参数都存在，缺失的用默认值填充
    param_values = {
        "layout_q": params.get("layout_q"),
        "layout_kv": params.get("layout_kv"),
        "q_type": params.get("q_type"),
        "ori_kv_type": params.get("ori_kv_type"),
        "cmp_kv_type": params.get("cmp_kv_type", [None]),
        "B": params.get("B"),
        "S1": params.get("S1"),
        "T1": params.get("T1"),
        "N1": params.get("N1"),
        "N2": params.get("N2"),
        "D": params.get("D"),
        "K": params.get("K", [None]),
        "block_num1": params.get("block_num1"),
        "block_num2": params.get("block_num2", [None]),
        "block_size1": params.get("block_size1"),
        "block_size2": params.get("block_size2", [None]),
        "cu_seqlens_q": params.get("cu_seqlens_q"),
        "seqused_kv": params.get("seqused_kv"),
        "softmax_scale": params.get("softmax_scale"),
        "cmp_ratio": params.get("cmp_ratio", [None]),
        "ori_mask_mode": params.get("ori_mask_mode"),
        "cmp_mask_mode": params.get("cmp_mask_mode", [None]),
        "ori_win_left": params.get("ori_win_left"),
        "ori_win_right": params.get("ori_win_right"),
    }

    # 生成参数名和值列表
    param_names = list(param_values.keys())
    values_lists = [param_values[name] for name in param_names]

    # 生成所有组合
    for combo in itertools.product(*values_lists):
        combination = dict(zip(param_names, combo))
        param_combinations.append(combination)

@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_example(param_combinations):
    layout_q = param_combinations['layout_q']
    layout_kv = param_combinations['layout_kv']
    q_type = param_combinations['q_type']
    ori_kv_type = param_combinations['ori_kv_type']
    cmp_kv_type = param_combinations['cmp_kv_type']
    B = param_combinations['B']
    S1 = param_combinations['S1']
    T1 = param_combinations['T1']
    N1 = param_combinations['N1']
    N2 = param_combinations['N2']
    D = param_combinations['D']
    K = param_combinations['K']
    block_num1 = param_combinations['block_num1']
    block_num2 = param_combinations['block_num2']
    block_size1 = param_combinations['block_size1']
    block_size2 = param_combinations['block_size2']
    cu_seqlens_q = param_combinations['cu_seqlens_q']
    seqused_kv = param_combinations['seqused_kv']
    softmax_scale = param_combinations['softmax_scale']
    cmp_ratio = param_combinations['cmp_ratio']
    ori_mask_mode = param_combinations['ori_mask_mode']
    cmp_mask_mode = param_combinations['cmp_mask_mode']
    ori_win_left = param_combinations['ori_win_left']
    ori_win_right = param_combinations['ori_win_right']

    torch_npu.npu.set_device(0)

    test_data = layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K, block_num1, \
                block_num2, block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, \
                ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right
    print("test_data:", test_data)

    # # 输入参数的合法性校验
    # try:
    #     check_valid_param.check_valid_param(test_data)
    # except ValueError as e:
    #     pytest.skip(f"输入参数校验失败:{e}")

    # 获得cpu结果(真值)和算子结果（测试值）
    npu_result, cpu_result = sparse_attn_sharedkv_process.test_sas_process(test_data)
    print("npu_result.size():", npu_result.size())

    # 结果精度对比
    check_result.check_result(cpu_result, npu_result)
