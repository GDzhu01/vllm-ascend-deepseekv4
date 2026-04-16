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

import itertools
import torch
import check_valid_param
import kv_quant_sparse_attn_sharedkv_golden
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import math
import os
import utils
import argparse
import concurrent.futures

# 读取表格
save_path = "qsas_testcase"
ENABLED_PARAMS = utils.load_excel_test_cases("./excel/example.xlsx", "decode")

locals()["param_combinations"] = []
for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        if value == 'torch.bfloat16':
            value = torch.bfloat16
        elif value == 'torch.float8_e4m3fn':
            value = torch.float8_e4m3fn
        elif value == "FALSE":
            value = False
        elif value == "TRUE":
            value = True
        locals()[f"param_{key}"] = [value]
    locals()[f"param_template_run_mode"] = locals()["param_template_run_mode"][0].split(',')

    param_names = [
    "Testcase_Name", "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type", "B", "S1", "S2", "N1", "N2", "D", "K",
    "block_size1", "block_size2", "softmax_scale", "cmp_ratio",
    "ori_mask_mode", "cmp_mask_mode", "ori_win_left", "ori_win_right", "kv_quant_mode", "tile_size", "rope_head_dim", "template_run_mode", "actlen_mode","S1EQS2"
    ]

    param_values = [
        locals()["param_Testcase_Name"],
        locals()["param_layout_q"],
        locals()["param_layout_kv"],
        locals()["param_q_type"],
        locals()["param_ori_kv_type"],
        locals()["param_cmp_kv_type"],
        locals()["param_B"],
        locals()["param_S1"],
        locals()["param_S2"],
        locals()["param_N1"],
        locals()["param_N2"],
        locals()["param_D"],
        locals()["param_K"],
        locals()["param_block_size1"],
        locals()["param_block_size2"],
        locals()["param_softmax_scale"],
        locals()["param_cmp_ratio"],
        locals()["param_ori_mask_mode"],
        locals()["param_cmp_mask_mode"],
        locals()["param_ori_win_left"],
        locals()["param_ori_win_right"],
        locals()["param_kv_quant_mode"],
        locals()["param_tile_size"],
        locals()["param_rope_head_dim"],
        locals()["param_template_run_mode"],
        locals()["param_actlen_mode"],
        locals()["param_S1EQS2"],
    ]

    # 生成所有的组合，并转换为字典列表
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)
    print(locals()["param_combinations"])

case_id = 0
def sas(param_combinations):   # 初始化参数和tensor
    global case_id
    Testcase_Name = param_combinations['Testcase_Name']
    layout_q = param_combinations['layout_q']
    layout_kv = param_combinations['layout_kv']
    q_type = param_combinations['q_type']
    ori_kv_type = param_combinations['ori_kv_type']
    cmp_kv_type = param_combinations['cmp_kv_type']
    B = param_combinations['B']
    S1 = param_combinations['S1']
    S2 = param_combinations['S2']
    N1 = param_combinations['N1']
    N2 = param_combinations['N2']
    D = param_combinations['D']
    K = param_combinations['K']
    block_size1 = param_combinations['block_size1']
    block_size2 = param_combinations['block_size2']
    softmax_scale = param_combinations['softmax_scale']
    cmp_ratio = param_combinations['cmp_ratio']
    ori_mask_mode = param_combinations['ori_mask_mode']
    cmp_mask_mode = param_combinations['cmp_mask_mode']
    ori_win_left = param_combinations['ori_win_left']
    ori_win_right = param_combinations['ori_win_right']
    kv_quant_mode = param_combinations['kv_quant_mode']
    tile_size = param_combinations['tile_size']
    rope_head_dim = param_combinations['rope_head_dim']
    template_run_mode = param_combinations['template_run_mode']
    actlen_mode = param_combinations['actlen_mode']
    S1EQS2 = param_combinations['S1EQS2']

    ops_mode = 'prefill' if S1 > 4 else "decode"
    q_type_str = "BF16"
    if q_type == torch.float16:
        q_type_str = "FP16"
    if Testcase_Name ==None :
        Testcase_Name = f"kvquantSparseAttenShardkv_{template_run_mode}_{ops_mode}_{layout_q}_{q_type_str}_{B}_{N1}_{N2}_{S1}_{S2}_{D}_{K}_{rope_head_dim}_{case_id:06d}"

    # 生成actLen 
    QS = [0]
    KVS = []
    if layout_q == "TND":
        for i in range(B):
            if actlen_mode == "random":
                tmpS1 = S1 - random.randint(0, S1 - 1)
                QS.append(tmpS1 + QS[-1])
                if S1EQS2: #  qs=kvs
                    tmpS2 = tmpS1
                else: # qs != kvs
                    tmpS2 = S2 - random.randint(0, math.ceil(S2/2) - 1)
                KVS.append(tmpS2)
            elif actlen_mode == "full":  ## 走该分支
                tmpS1 = S1
                QS.append(tmpS1 + QS[-1])
                KVS.append(S2)
    elif layout_q == "BSND":
        for i in range(B):
            QS.append(S1 + QS[-1])
            if actlen_mode == "random":
                tmpS2 = S2 - random.randint(0, math.ceil(S2/2) - 1)
            else:
                tmpS2 = S2
            KVS.append(tmpS2)

    cu_seqlens_q = QS
    seqused_kv = KVS
    T1 = QS[-1] if layout_q == "TND" else None

    # 计算blockNum
    ori_block_num_per_batch = []
    ori_block_num_sum = 0
    cmp_block_num_per_batch = []
    cmp_block_num_sum = 0
    for cur_ori_act_kv in seqused_kv:
        cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
        ori_block_num_per_batch.append(cur_ori_kv_block_num)
        ori_block_num_sum += cur_ori_kv_block_num

        cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
        cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
        cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
        cmp_block_num_sum += cur_cmp_kv_block_num
    block_num1 = ori_block_num_sum
    block_num2 = cmp_block_num_sum
    
    params = Testcase_Name, layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K, block_num1, \
                block_num2, block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, ori_mask_mode, \
                cmp_mask_mode, ori_win_left, ori_win_right, kv_quant_mode, tile_size, rope_head_dim, template_run_mode
    print("input_params:", params)
    
    # 输入参数的合法性校验
    try:
        check_valid_param.check_valid_param(params)
    except ValueError as e:
        pytest.skip(f"输入参数校验失败:{e}")

    # 生成测试数据
    input_data = kv_quant_sparse_attn_sharedkv_golden.generate_and_save_testdata(params, save_pt=True, save_path=save_path)
    case_id += 1

@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
def test_sparse_attn_sharedkv(param_combinations):   # 初始化参数和tensor
    # 线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = executor.submit(sas, param_combinations)
        # 等待并获取结果
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"当前用例线程执行失败")
