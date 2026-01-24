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

import math
import random
import logging 
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)


def check_valid_param(params):
    # batch_size, q_t_size, k_t_size, block_size, q_seq, kv_seq, q_head_num, kv_head_num, head_dim, rope_dim, \
    # q_dtype, idx_dtype, sparse_block_size, sparse_block_count, kv_seq_act, layout_kv, layout_query = params
    
    # # 依次校验参数合法性
    
    # if q_dtype not in [torch.bfloat16, torch.float16]:
    #     raise ValueError("q_dtype should be: float16/bfloat16")

    # if idx_dtype not in [torch.int32]:
    #     raise ValueError("sparse idxtype should be: int32")

    # if layout_query not in ["BSND", "TND"]:
    #     raise ValueError(f"不支持的Q shape: {layout_query}")
    
    # if layout_kv not in ["BSND","TND","PA_BSND"]:
    #     raise ValueError(f"不支持的KV shape: {layout_kv}")
    


