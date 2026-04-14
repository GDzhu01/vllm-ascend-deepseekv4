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


def test_qli_process(filepath, device_id=0):
    # 加载测试数据
    test_data = torch.load(filepath, map_location="cpu")

    params = test_data['params']
    cpu_result = test_data['cpu_result']
    topk_value = test_data['topk_value']
    print("执行用例：", filepath)
    torch_npu.npu.set_device(device_id)

    if params[10] == 'FLOAT8_E4M3FN':
        query = test_data['query'].to(dtype=torch.float8_e4m3fn).npu()
        key = test_data['key'].to(dtype=torch.float8_e4m3fn).npu()
    else:
        query = test_data['query'].npu()
        key = test_data['key'].npu()

    
    weights =test_data['weights'].npu()
    query_dequant_scale = test_data['query_dequant_scale'].npu()
    key_dequant_scale = test_data['key_dequant_scale'].npu()
    actual_seq_lengths_query = test_data['actual_seq_lengths_query'].npu()
    actual_seq_lengths_key = test_data['actual_seq_lengths_key'].npu()
    block_table = test_data['block_table'].npu()
    metadata = test_data['metadata'].npu()
    query_quant_mode = test_data['query_quant_mode']
    key_quant_mode = test_data['key_quant_mode']
    layout_query = test_data['layout_query']
    layout_key = test_data['layout_key']
    sparse_count = test_data['sparse_count']
    sparse_mode = test_data['sparse_mode']
    cmp_ratio = test_data['cmp_ratio']

    #调用SFA算子
    npu_result,_ = torch.ops.custom.npu_quant_lightning_indexer(query, key, weights, 
                                                    query_dequant_scale,
                                                    key_dequant_scale,
                                                    actual_seq_lengths_query=actual_seq_lengths_query,
                                                    actual_seq_lengths_key=actual_seq_lengths_key,
                                                    block_table=block_table,
                                                    metadata = metadata,
                                                    query_quant_mode=query_quant_mode,
                                                    key_quant_mode=key_quant_mode,
                                                    layout_query=layout_query,
                                                    layout_key=layout_key, 
                                                    sparse_count=sparse_count,
                                                    sparse_mode=sparse_mode,
                                                    pre_tokens = (1<<63)-1,
                                                    next_tokens = (1<<63)-1,
                                                    cmp_ratio = cmp_ratio,
                                                    return_value = False)
    
    torch.npu.synchronize()

    return cpu_result, npu_result, topk_value, params

