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

import torch

# 定义测试参数组合
TEST_PARAMS = {
    "decode_first":{
        "Testcase_Name": [None],
        "layout_q": ["TND"],
        "layout_kv": ["PA_ND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.float8_e4m3fn],
        "cmp_kv_type": [torch.float8_e4m3fn],
        "B": [1],
        "S1": [1],
        "S2": [8193],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K": [512],
        "block_num1": [None],
        "block_num2": [None],
        "block_size1": [128],
        "block_size2": [128],
        "cu_seqlens_q": [None],
        "seqused_kv": [None],
        "softmax_scale": [0.04419417],
        "cmp_ratio": [1],
        "ori_mask_mode": [4],
        "cmp_mask_mode": [3],
        "ori_win_left": [127],
        "ori_win_right": [0],
        "kv_quant_mode": [1],
        "tile_size": [64],
        "rope_head_dim": [64],
        "template_run_mode": ["SWA"], # SWA SCFA CFA
        "actlen_mode":["full"],       # random full
        "S1EQS2":[False]
    },

    "prefill_first":{
        "Testcase_Name": [None],
        "layout_q": ["TND"],
        "layout_kv": ["PA_ND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.float8_e4m3fn],
        "cmp_kv_type": [torch.float8_e4m3fn],
        "B": [1],
        "S1": [8192],
        "S2": [8192], 
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K": [512],
        "block_num1": [None],
        "block_num2": [None],
        "block_size1": [128],
        "block_size2": [128],
        "cu_seqlens_q": [None], 
        "seqused_kv": [None], 
        "softmax_scale": [0.04419417],
        "cmp_ratio": [1],
        "ori_mask_mode": [4],
        "cmp_mask_mode": [3],
        "ori_win_left": [127],
        "ori_win_right": [0],
        "kv_quant_mode": [1],
        "tile_size": [64],
        "rope_head_dim": [64],
        "template_run_mode": ["SWA"], # SWA SCFA CFA
        "actlen_mode":["full"],       # random full
        "S1EQS2":[False]
    },
}

# 填入启用的测试参数
ENABLED_PARAMS = [TEST_PARAMS["decode_first"]]