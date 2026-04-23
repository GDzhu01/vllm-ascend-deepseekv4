# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import test
import torch
import torch_npu
import pytest
import random
import numpy as np
import math
import math, copy
import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn

from compressor_golden import run_compressor_eager, cpu_compressor

class Generalized_operator():
    def forward(self,
                x,
                wkv,
                wgate,
                kv_state,
                score_state,
                ape,
                norm_weight, 
                rope_sin,
                rope_cos,
                block_table,
                cu_seqlens,
                seqused,
                start_pos,
                rope_head_dim,
                cmp_ratio,
                coff,
                norm_eps,
                rotary_mode,
                cache_mode):

        return cpu_compressor(
            x, wkv, wgate, kv_state, score_state, ape, norm_weight, rope_sin, rope_cos,
            block_table=block_table, cu_seqlens=cu_seqlens, seqused=seqused, start_pos=start_pos,
            rope_head_dim=rope_head_dim, cmp_ratio=cmp_ratio, coff=coff, norm_eps=norm_eps, rotary_mode=rotary_mode, cache_mode=cache_mode)

def output_operator(params):
    #构造输入
    batch_size, hidden_size, Seq_len, head_dim, block_size, rope_head_dim, cmp_ratio, coff, norm_eps, \
    start_p, rotary_mode, cache_mode, layout_x, data_type, cu_seqlens, seqused, start_pos, \
    x_datarange, wkv_datarange,  wgate_datarange, ape_datarange, norm_weight_datarange, kv_state_datarange, score_state_datarange = params
    S_max = 0
    save_state_seqlens = None
    bs_combine_flag = False
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32)
    if start_pos is not None:
        start_pos = torch.tensor(start_pos).to(torch.int32)
    elif start_p is not None:
        start_pos = torch.full((batch_size,), start_p, dtype=torch.int32)

    if layout_x == "TH":
        bs_combine_flag = True
        if cu_seqlens is None:
            T = batch_size * Seq_len
            if T !=0:
                cu_seqlens = torch.arange(0, T + 1, Seq_len, dtype=torch.int32)
            else:
                cu_seqlens = torch.zeros((batch_size+1), dtype=torch.int32)
        else:
            cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
        for i in range(batch_size):
            if start_pos is not None:
                if start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i] > S_max:
                    S_max = start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i] 
            else:
                if cu_seqlens[i + 1] - cu_seqlens[i] > S_max:
                    S_max = cu_seqlens[i + 1] - cu_seqlens[i] 

    else:
        cu_seqlens = None
        if start_pos is not None:
            S_max = max(start_pos) + Seq_len
        else:
            S_max = Seq_len
    print(f"params = {params}")

    run_compressor_eager(batch_size, S_max, head_dim, coff, cmp_ratio, bs_combine_flag, S = Seq_len, start_pos = start_pos, \
    seqused = seqused, cu_seqlens = cu_seqlens, block_size = block_size, rotary_mode = rotary_mode, cache_mode = cache_mode, data_type = data_type, \
    hidden_size = hidden_size, rope_head_dim = rope_head_dim, norm_eps = norm_eps, save_state_seqlens = save_state_seqlens, \
    x_datarange = x_datarange, wkv_datarange = wkv_datarange, wgate_datarange = wgate_datarange, ape_datarange = ape_datarange, \
    norm_weight_datarange = norm_weight_datarange, kv_state_datarange = kv_state_datarange, score_state_datarange = score_state_datarange)