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

import ast
import utils
import itertools
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pytest
import random
import sparse_attn_sharedkv_process
import sparse_attn_sharedkv_golden
import torch

excel_path = os.getenv("SAS_EXCEL_PATH", "./excel/example.xlsx")
excel_sheet = os.getenv("SAS_EXCEL_SHEET", "Sheet1")
ENABLED_PARAMS_FROM_FILE = utils.load_excel_test_cases(excel_path, excel_sheet)
save_path = os.getenv("SAS_PT_SAVE_PATH", "./data")

param_combinations = []
for params in ENABLED_PARAMS_FROM_FILE:
    # 确保所有参数都存在，缺失的用默认值填充
    param_values = {
        "layout_q": params.get("layout_q"),
        "layout_kv": params.get("layout_kv"),
        "q_type": params.get("q_type"),
        "ori_kv_type": params.get("ori_kv_type"),
        "cmp_kv_type": params.get("cmp_kv_type", [None]),
        "B": params.get("B"),
        "S1": params.get("S1"),
        "S2": params.get("S2"),
        "T1": params.get("T1"),
        "N1": params.get("N1"),
        "N2": params.get("N2"),
        "D": params.get("D"),
        "K": params.get("K", [None]),
        "block_num1": params.get("block_num1"),
        "block_num2": params.get("block_num2", [None]),
        "block_size1": params.get("block_size1"),
        "block_size2": params.get("block_size2", [None]),
        "cu_seqlens_q": params.get("cu_seqlens_q", [None]),
        "seqused_q": params.get("seqused_q", [None]),
        "seqused_kv": params.get("seqused_kv", [None]),
        "softmax_scale": params.get("softmax_scale"),
        "cmp_ratio": params.get("cmp_ratio", [None]),
        "ori_mask_mode": params.get("ori_mask_mode"),
        "cmp_mask_mode": params.get("cmp_mask_mode", [None]),
        "ori_win_left": params.get("ori_win_left"),
        "ori_win_right": params.get("ori_win_right"),
        "q_datarange": params.get("q_datarange", [None]),
        "ori_kv_datarange": params.get("ori_kv_datarange", [None]),
        "cmp_kv_datarange": params.get("cmp_kv_datarange", [None]),
        "testcase_name": params.get("testcase_name"),
    }

    # 生成参数名和值列表
    param_combinations.append(param_values)


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_sparse_attn_sharedkv(param_combinations):   # 初始化参数和tensor
    layout_q = param_combinations['layout_q']
    layout_kv = param_combinations['layout_kv']
    q_type = torch.bfloat16 if param_combinations['q_type'] == "torch.bfloat16" else torch.float16
    ori_kv_type = torch.bfloat16 if param_combinations['ori_kv_type'] == "torch.bfloat16" else torch.float16
    cmp_kv_type = torch.bfloat16 if param_combinations['cmp_kv_type'] == "torch.bfloat16" else torch.float16
    B = int(param_combinations['B'])
    S1 = int(param_combinations['S1'])
    S2 = None if param_combinations['S2'] is None else int(param_combinations['S2'])
    T1 = None if param_combinations['T1'] is None else int(param_combinations['T1'])
    N1 = int(param_combinations['N1'])
    N2 = int(param_combinations['N2'])
    D = int(param_combinations['D'])
    K = None if param_combinations['K'] is None else int(param_combinations['K'])
    block_num1 = param_combinations['block_num1']
    block_num2 = None if param_combinations['block_num2'] is None else int(param_combinations['block_num2'])
    block_size1 = param_combinations['block_size1']
    block_size2 = None if param_combinations['block_size2'] is None else int(param_combinations['block_size2'])
    cu_seqlens_q = None if param_combinations['cu_seqlens_q'] is None else ast.literal_eval(param_combinations['cu_seqlens_q'])
    seqused_q = None if param_combinations['seqused_q'] is None else ast.literal_eval(param_combinations['seqused_q'])
    seqused_kv = ast.literal_eval(param_combinations['seqused_kv'])
    softmax_scale = param_combinations['softmax_scale']
    cmp_ratio = None if param_combinations['cmp_ratio'] is None else int(param_combinations['cmp_ratio'])
    ori_mask_mode = int(param_combinations['ori_mask_mode'])
    cmp_mask_mode = None if param_combinations['cmp_mask_mode'] is None else int(param_combinations['cmp_mask_mode'])
    ori_win_left = param_combinations['ori_win_left']
    ori_win_right = param_combinations['ori_win_right']
    testcase_name = param_combinations['testcase_name']
    q_datarange = ast.literal_eval(param_combinations['q_datarange']) if param_combinations['q_datarange'] is not None else [-10,10]
    ori_kv_datarange = ast.literal_eval(param_combinations['ori_kv_datarange']) if param_combinations['ori_kv_datarange'] is not None else [-10,10]
    cmp_kv_datarange = ast.literal_eval(param_combinations['cmp_kv_datarange']) if param_combinations['cmp_kv_datarange'] is not None else [-10,10]

    # maxSeqLen / block_size向上取整
    ori_block_num_per_batch = []
    ori_block_num_sum = 0
    cmp_block_num_per_batch = []
    cmp_block_num_sum = 0
    
    test_data = layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K, block_num1, \
                block_num2, block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, \
                ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, testcase_name, S2, q_datarange, \
                ori_kv_datarange, cmp_kv_datarange, seqused_q

    print("data parsed.", test_data)
    print("strat to generate data")
    # 生成测试数据
    input_data = sparse_attn_sharedkv_golden.gen_data(test_data)    
    print("strat to save data")
    sparse_attn_sharedkv_golden.save_test_case(input_data, save_path)
