#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import argparse
import torch
import numpy as np


def gen_data_fp8(row, col):
    data = torch.randn((row, col), dtype=torch.float32)
    data_e4m3 = data.to(torch.float8_e4m3fn)
    return data_e4m3


def gen_data(m, n, k, trans_a, trans_b):
    os.makedirs('./input', exist_ok=True)
    os.makedirs('./output', exist_ok=True)

    a_fp8 = gen_data_fp8(m, k)
    b_fp8 = gen_data_fp8(k, n)

    if(trans_a == 1):
        a_fp8 = a_fp8.t()
    if(trans_b == 1):
        b_fp8 = b_fp8.t()

    a_np = torch.tensor(a_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_np = torch.tensor(b_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    a_np.tofile('./input/a_8.bin')
    b_np.tofile('./input/b_8.bin')

    a_fp16 = a_fp8.to(torch.float16)
    b_fp16 = b_fp8.to(torch.float16)
    if(trans_a == 1):
        a_fp16 = a_fp8.t().to(torch.float16)
    if(trans_b == 1):
        b_fp16 = b_fp16.t().to(torch.float16)

    c_fp32 = a_fp16.to(torch.float32) @ b_fp16.to(torch.float32)
    c_np = c_fp32.numpy()
    c_np.tofile('./output/expected_data.bin')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)    
    parser.add_argument('k', type=int)
    parser.add_argument('trans_a', type=int)
    parser.add_argument('trans_b', type=int)
    args = parser.parse_args()
    gen_data(args.m, args.n, args.k, args.trans_a, args.trans_b)
