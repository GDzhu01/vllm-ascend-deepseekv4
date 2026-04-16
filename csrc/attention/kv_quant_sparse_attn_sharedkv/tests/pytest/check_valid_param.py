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

import math
import random
import logging 
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)


def check_valid_param(params):
    Testcase_Name, layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K, block_num1, \
    block_num2, block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, ori_mask_mode, \
    cmp_mask_mode, ori_win_left, ori_win_right, kv_quant_mode, tile_size, rope_head_dim, template_run_mode = params
    
    # 依次校验参数合法性
    
    if q_type not in [torch.bfloat16]:
        raise ValueError("q_type should be: bfloat16")

    if layout_q not in ["BSND", "TND"]:
        raise ValueError(f"不支持的Q shape: {layout_q}")
    
    if layout_kv not in ["PA_ND"]:
        raise ValueError(f"不支持的KV shape: {layout_kv}")

    if template_run_mode not in ["SCFA","CFA","SWA"]:
        raise ValueError(f"不支持的template_run_mode: {template_run_mode}")
    


