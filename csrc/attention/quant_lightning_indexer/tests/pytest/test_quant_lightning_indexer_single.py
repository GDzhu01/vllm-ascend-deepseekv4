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
import torch_npu
from test_quant_lightning_indexer_paramset import ENABLED_PARAMS
import result_compare_method
import quant_lightning_indexer_golden
import pytest

for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value
    # 生成所有参数组合
    param_names = [
        "batch_size", "q_seq", "k_seq", "q_t_size", "k_t_size", "q_head_num", "k_head_num","head_dim", 
        "block_size", "block_num", "qk_dtype", "dequant_dtype", "actual_seq_dtype", "act_seq_q","act_seq_k",
        "query_quant_mode", "key_quant_mode", "layout_query","layout_key", "sparse_count", "sparse_mode", 
        "query_datarange","key_datarange","weights_datarange","q_scale_datarange","k_scale_datarange","cmp_ratio"
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
        locals()["param_query_datarange"],
        locals()["param_key_datarange"],
        locals()["param_weights_datarange"],
        locals()["param_q_scale_datarange"],
        locals()["param_k_scale_datarange"],
        locals()["param_cmp_ratio"]
    ]

    # 生成所有的组合，并转换为字典列表
    locals()["param_combinations"] = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)

    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_qli(param_combinations):   # 初始化参数和tensor
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
        query_datarange = param_combinations['query_datarange']
        key_datarange = param_combinations['key_datarange']
        weights_datarange = param_combinations['weights_datarange']
        q_scale_datarange = param_combinations['q_scale_datarange']
        k_scale_datarange = param_combinations['k_scale_datarange']
        cmp_ratio = param_combinations['cmp_ratio']
        torch_npu.npu.set_device(0)
        test_data = batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num,\
                    qk_dtype, dequant_dtype, actual_seq_dtype, act_seq_q, act_seq_k, query_quant_mode,key_quant_mode, layout_query,\
                    layout_key, sparse_count, sparse_mode, query_datarange, key_datarange, weights_datarange, q_scale_datarange,\
                    k_scale_datarange, cmp_ratio

        # 获得cpu结果(真值)和算子结果（测试值）
        cpu_result, npu_result, topk_value = quant_lightning_indexer_golden.qli_output_single(test_data)
        print("npu_result", npu_result)
        print("cpu_result:", cpu_result)
        # 结果精度对比
        result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result, topk_value, test_data)
        print("result", result)
        print("result", fulfill_percent)