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

import torch

# 定义测试参数组合
TEST_PARAMS = {
    # 基础场景
    "quant_li_default":{
        "batch_size": [4],
        "q_seq": [4],
        "k_seq": [8],
        "q_t_size":[64],
        "k_t_size":[256],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[16],
        "qk_dtype": [torch.int8],
        "dequant_dtype": [torch.float16],
        "actual_seq_dtype": [torch.int32],
        # "act_seq_q":[None], 用于BSND场景
        # "act_seq_k":[None], 用于BSND场景
        "act_seq_q": [[16, 32, 48, 64]], # TND场景下为前缀和
        "act_seq_k": [[256, 512, 512, 1024]], #TND场景下为前缀和；PA场景非前缀和表示每个batch_size的实际token数
        "query_quant_mode": [0],
        "key_quant_mode": [0],
        "layout_query": ["TND"],
        "layout_key":["PA_BSND"],
        "sparse_count": [2048],
        "sparse_mode": [3],
        "cmp_ratio":[2] #1/2/4/8/16/32/64/128
    }

}

# 按需选择要启用的测试参数（例如默认启用所有）
ENABLED_PARAMS = [TEST_PARAMS["quant_li_default"]] 