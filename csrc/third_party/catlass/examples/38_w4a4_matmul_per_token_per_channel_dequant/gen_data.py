#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys
import numpy as np

WORKSPACE = os.getcwd()

os.environ["WORKSPACE"] = WORKSPACE
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"

class OpParam:
    def __init__(self) -> None:
        self.m = 0
        self.k = 0
        self.n = 0
        self.transA = False
        self.transB = False


def pack_4bit_to_bytes(data):
    data = data.astype(np.int8)
    data[data < 0] += 16
    packed = (data[..., 1::2] << 4) | (data[..., ::2] & 0x0F)
    return packed

def cpu_golden(x: np.ndarray, weight: np.ndarray, scale: np.ndarray, perTokenScale: np.ndarray):
    atomic = np.float16
    mm = np.matmul(x.astype(np.float32), weight.astype(np.float32)).astype(atomic)
    mm = mm.astype(np.float32) * scale
    mm = mm * perTokenScale

    return mm.astype(np.float32)

def gen_testcase(path: str, param: OpParam) -> None:
    m, k, n = param.m, param.k, param.n
    transA, transB = param.transA, param.transB

    assert not transA, "Transposition is not supported for matrices A"

    # 原始模型输入
    x = np.random.randint(-8, 8, [m, k]).astype(np.int8)
    weight = np.random.randint(-8, 8, [k, n]).astype(np.int8)
    scale = np.random.normal(0, 0.01, [1, n]).astype(np.float32)
    perTokenScale = np.random.normal(0, 0.01, [m, 1]).astype(np.float32)

    # cpu计算golden
    scaleFloat32 = scale.reshape(1, n)
    golden = cpu_golden(x, weight, scale, perTokenScale).reshape(-1, n) 

    # 随路转换需要uint64类型的融合scale
    scaleFloat32.dtype = np.uint32
    scaleUint64 = np.zeros((1, 2 * n), dtype=np.uint32)
    scaleUint64[..., ::2] = scaleFloat32
    scaleUint64.dtype = np.uint64 # Actually uint64

    # 进行Pack压缩
    xInt4 = pack_4bit_to_bytes(x)
    xInt4.dtype = np.int8

    if transB:       # trans: layoutB: nZ
        assert k % 64 == 0, "k must be divisible by 64"
        assert n % 16 == 0, "n must be divisible by 16"
        weightInt4 = pack_4bit_to_bytes(weight.transpose(1, 0))
        weightInt4.dtype = np.int8
        weightInt4 = weightInt4.reshape([n, -1])
        weightInt4 = weightInt4.reshape(n//16, 16, k//64, 32).transpose(2, 0, 1, 3)
    else:                  # no-trans: layoutB: zN
        assert k % 16 == 0, "k must be divisible by 16"
        assert n % 64 == 0, "n must be divisible by 64"
        weightInt4 = pack_4bit_to_bytes(weight)
        weightInt4.dtype = np.int8
        weightInt4 = weightInt4.reshape([k, -1])
        weightInt4 = weightInt4.reshape(k//16, 16, n//64, 32).transpose(2, 0, 1, 3)
    
    weightInt4 = weightInt4.reshape(k, -1)
    scaleUint64 = scaleUint64.squeeze()
    perTokenScale = perTokenScale.squeeze()

    # 存储输入输出和golden
    xInt4.tofile(os.path.join(path, "inputA.dat"))
    weightInt4.tofile(os.path.join(path, "inputB.dat"))
    scaleUint64.tofile(os.path.join(path, "inputScale.dat"))
    perTokenScale.tofile(os.path.join(path, "inputPerTokenScale.dat"))
    golden.tofile(os.path.join(path, "expected.dat"))

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    param = OpParam()
    param.m = int(sys.argv[1])
    param.n = int(sys.argv[2])
    param.k = int(sys.argv[3])
    param.transA = 0
    param.transB = 0 if len(sys.argv) < 5 else int(sys.argv[4])
    # no-trans: zN; trans: nZ

    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    gen_testcase(data_dir, param)
