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
    "scfa_decode":{
        "layout_q": ["TND"],
        "layout_kv": ["PA_ND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.bfloat16],
        "cmp_kv_type": [torch.bfloat16],
        "B": [1],
        "S1": [1],
        "T1": [1],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K": [512],
        "block_num1": [65],
        "block_num2": [17],
        "block_size1": [128],
        "block_size2": [128],
        "cu_seqlens_q": [[0, 1]],
        "seqused_kv": [[8193]],
        "softmax_scale": [0.04419417],
        "cmp_ratio": [4],
        "ori_mask_mode": [4],
        "cmp_mask_mode": [3],
        "ori_win_left": [127],
        "ori_win_right": [0]
    },
    "scfa_prefill":{
        "layout_q": ["TND"],
        "layout_kv": ["PA_ND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.bfloat16],
        "cmp_kv_type": [torch.bfloat16],
        "B": [1],
        "S1": [8192],
        "T1": [8192],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K": [512],
        "block_num1": [65],
        "block_num2": [17],
        "block_size1": [128],
        "block_size2": [128],
        "cu_seqlens_q": [[0, 8192]],
        "seqused_kv": [[8192]],
        "softmax_scale": [0.04419417],
        "cmp_ratio": [4],
        "ori_mask_mode": [4],
        "cmp_mask_mode": [3],
        "ori_win_left": [127],
        "ori_win_right": [0]
    },
    "swa_decode":{
        "layout_q": ["TND"],
        "layout_kv": ["PA_ND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.bfloat16],
        "B": [1],
        "S1": [1],
        "T1": [1],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "block_num1": [65],
        "block_size1": [128],
        "cu_seqlens_q": [[0, 1]],
        "seqused_kv": [[8193]],
        "softmax_scale": [0.04419417],
        "ori_mask_mode": [4],
        "ori_win_left": [127],
        "ori_win_right": [0]
    },
    "swa_prefill":{
        "layout_q": ["TND"],
        "layout_kv": ["PA_ND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.bfloat16],
        "B": [1],
        "S1": [8192],
        "T1": [8192],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "block_num1": [65],
        "block_size1": [128],
        "cu_seqlens_q": [[0, 8192]],
        "seqused_kv": [[8192]],
        "softmax_scale": [0.04419417],
        "ori_mask_mode": [4],
        "ori_win_left": [127],
        "ori_win_right": [0]
    },
    "cfa_decode":{
        "layout_q": ["TND"],
        "layout_kv": ["PA_ND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.bfloat16],
        "cmp_kv_type": [torch.bfloat16],
        "B": [1],
        "S1": [1],
        "T1": [1],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "block_num1": [65],
        "block_num2": [17],
        "block_size1": [128],
        "block_size2": [128],
        "cu_seqlens_q": [[0, 1]],
        "seqused_kv": [[8193]],
        "softmax_scale": [0.04419417],
        "cmp_ratio": [128],
        "ori_mask_mode": [4],
        "cmp_mask_mode": [3],
        "ori_win_left": [127],
        "ori_win_right": [0]
    },
    "cfa_prefill":{
        "layout_q": ["TND"],
        "layout_kv": ["PA_ND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.bfloat16],
        "cmp_kv_type": [torch.bfloat16],
        "B": [1],
        "S1": [8192],
        "T1": [8192],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "block_num1": [65],
        "block_num2": [17],
        "block_size1": [128],
        "block_size2": [128],
        "cu_seqlens_q": [[0, 8192]],
        "seqused_kv": [[8192]],
        "softmax_scale": [0.04419417],
        "cmp_ratio": [128],
        "ori_mask_mode": [4],
        "cmp_mask_mode": [3],
        "ori_win_left": [127],
        "ori_win_right": [0]
    }
}

# 按需选择要启用的测试参数（例如默认启用所有）
ENABLED_PARAMS = [TEST_PARAMS[key] for key in TEST_PARAMS.keys()]