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
from testcases_qli import ENABLED_PARAMS
import check_result
import qli_single
import pytest



for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value

    # 生成所有参数组合
    param_names = [
        "batch_size", "q_seq", "k_seq", "q_t_size", "k_t_size", "q_head_num", "k_head_num","head_dim", 
        "block_size", "block_num", "qk_dtype", "dequant_dtype", "actual_seq_dtype", "act_seq_q","act_seq_k",
        "query_quant_mode", "key_quant_mode", "layout_query","layout_key", "sparse_count", "sparse_mode", "cmp_ratio"
    ]

    param_values = [
        locals()["param_batch_size"],
        locals()["param_q_seq"],
        locals()["param_k_seq"],
        locals()["param_q_t_size"],
        locals()["param_k_t_size"],
        locals()["param_q_head_num"], 
        locals()["param_k_head_num"],
        locals()["param_head_dim"],
        locals()["param_block_size"],
        locals()["param_block_num"],
        locals()["param_qk_dtype"],
        locals()["param_dequant_dtype"],
        locals()["param_actual_seq_dtype"],
        locals()["param_act_seq_q"],
        locals()["param_act_seq_k"],
        locals()["param_query_quant_mode"],
        locals()["param_key_quant_mode"],
        locals()["param_layout_query"],
        locals()["param_layout_key"],
        locals()["param_sparse_count"],
        locals()["param_sparse_mode"],
        locals()["param_cmp_ratio"]
    ]

    # 生成所有的组合，并转换为字典列表
    locals()["param_combinations"] = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)


    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_sparse_flash_attention(param_combinations):   # 初始化参数和tensor
        batch_size = param_combinations['batch_size']
        q_seq = param_combinations['q_seq']
        k_seq = param_combinations['k_seq']
        q_t_size = param_combinations['q_t_size']
        k_t_size = param_combinations['k_t_size']
        q_head_num = param_combinations['q_head_num']
        k_head_num = param_combinations['k_head_num']
        head_dim = param_combinations['head_dim']
        block_size = param_combinations['block_size']
        block_num = param_combinations['block_num']
        qk_dtype= param_combinations['qk_dtype']
        dequant_dtype = param_combinations['dequant_dtype']
        actual_seq_dtype = param_combinations['actual_seq_dtype']
        act_seq_q = param_combinations['act_seq_q']
        act_seq_k = param_combinations['act_seq_k']
        query_quant_mode = param_combinations['query_quant_mode']
        key_quant_mode = param_combinations['key_quant_mode']
        layout_query = param_combinations['layout_query']
        layout_key = param_combinations['layout_key']
        sparse_count = param_combinations['sparse_count']
        sparse_mode = param_combinations['sparse_mode']
        cmp_ratio = param_combinations['cmp_ratio']


        torch_npu.npu.set_device(0)


        test_data = batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num,\
                    qk_dtype, dequant_dtype, actual_seq_dtype, act_seq_q, act_seq_k, query_quant_mode,\
                    key_quant_mode, layout_query, layout_key, sparse_count, sparse_mode, cmp_ratio
        

        print("test_data:", test_data)

        # 输入参数的合法性校验
        # try:
        #     check_valid_param.check_valid_param(test_data)
        # except ValueError as e:
        #     pytest.skip(f"输入参数校验失败:{e}")

        # 获得cpu结果(真值)和算子结果（测试值）
        cpu_result, npu_result = qli_single.qli_output_single(test_data)

        
        print("npu_result", npu_result)
        print("cpu_result:", cpu_result)

        # 结果精度对比
        check_result.check_result(cpu_result, npu_result)