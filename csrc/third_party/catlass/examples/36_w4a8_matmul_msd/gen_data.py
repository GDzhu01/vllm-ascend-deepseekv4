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
import sys
import time
import torch
import random
import numpy as np

WORKSPACE = os.getcwd()

os.environ["WORKSPACE"] = WORKSPACE
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"

class OpParam:
    def __init__(self) -> None:
        self.kGroupSize = 0 
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

def gen_testcase(path: str, param: OpParam) -> None:

    # golden函数
    def msd_cpu(x, weight, bias, scale, perTokenScale):
        k, n = weight.shape
        quantGroupNum = scale.shape[0]
        kGroupSize = k // quantGroupNum
        weightGroup = weight.reshape(quantGroupNum, kGroupSize, n).astype(np.int32)
        xGroup = x.reshape(-1, quantGroupNum, kGroupSize).astype(np.int32)
        weightGroup = weight.reshape(quantGroupNum, kGroupSize, n).astype(np.int32)

        mmOut = []
        atomic = np.float16
        mmi = np.zeros([xGroup.shape[0], n], dtype=atomic)
        for i in range(quantGroupNum):
            # numpy int32矩阵乘法非常慢，此处用float32的矩阵乘法代替
            mm = np.matmul(xGroup[:, i, :].astype(np.float32), weightGroup[i, ...].astype(np.float32))
            mm = mm.astype(np.float32) * scale[i, :]
            mmi = (mmi.astype(atomic) + mm.astype(atomic)).astype(atomic)
        mmi = mmi.reshape(-1, 2, n).astype(np.float32)
        mmi = mmi[:, 0, :] * 16 + mmi[:, 1, :] + bias.reshape(1, n)
        mmi = mmi * perTokenScale
        mmOut.append(mmi)

        golden = np.concatenate(mmOut, axis=0)
        return golden.astype(np.float32)

    kGroupSize, m, k, n = param.kGroupSize, param.m, param.k, param.n
    transA, transB = param.transA, param.transB
    quantGroupNum = k // kGroupSize

    assert not transA and not transB, "Transposition is not supported for matrices A and B"
    assert k % 2 == 0 and n % 2 == 0, "k and n should be even"
    assert k % kGroupSize == 0, "kGroupSize should be divisible by k"

    # 原始模型输入
    x = np.random.randint(-100, 100, [m, k]).astype(np.int8)
    weight = np.random.randint(-8, 8, [k, n]).astype(np.int8)
    scale = np.random.normal(0, 0.01, [1, n]).astype(np.float32)
    perGroupScale = np.random.normal(0, 12, [quantGroupNum, n]).astype(np.float32)
    perTokenScale = np.random.normal(0, 0.01, [m, 1]).astype(np.float32)

    # 计算精度补齐bias
    weightHigh = weight.astype(np.float32).reshape([quantGroupNum, -1, n]) *\
        perGroupScale.reshape([quantGroupNum, 1, n])
    weightHigh = weightHigh.reshape([k, n])
    bias = 8 * (weightHigh * scale).sum(axis=0)

    # 计算per group和per channel的融合量化系数
    scaleFloat32 = scale * perGroupScale
    # 输入x高低位拼接
    xC12 = np.concatenate([x.reshape(m, 1, k) // 16, (x.reshape(m, 1, k) & 0x0F) - 8], axis=1).reshape(2 * m, k)

    # cpu计算golden
    golden = msd_cpu(xC12, weight, bias, scaleFloat32, perTokenScale).reshape(-1, n)
    
    # 随路转换需要uint64类型的融合scale
    scaleFloat32.dtype = np.uint32
    scaleUint64 = np.zeros((quantGroupNum, 2 * n), dtype=np.uint32)
    scaleUint64[..., ::2] = scaleFloat32
    scaleUint64.dtype = np.uint64

    # npu int4量化
    xInt4 = pack_4bit_to_bytes(xC12)
    xInt4 = xInt4.reshape(2 * m, -1)
    weightInt4 = pack_4bit_to_bytes(weight)
    weightInt4.dtype = np.int8
    weightInt4 = weightInt4.reshape([k, -1])

    # 存储输入输出和golden
    xInt4.tofile(os.path.join(path, "inputA.dat"))
    weightInt4.tofile(os.path.join(path, "inputB.dat"))
    bias.tofile(os.path.join(path, "inputBias.dat"))
    scaleUint64.tofile(os.path.join(path, "inputScale.dat"))
    perTokenScale.tofile(os.path.join(path, "inputPerTokenScale.dat"))
    golden.tofile(os.path.join(path, "expected.dat"))

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    param = OpParam()
    param.kGroupSize = int(sys.argv[1])
    param.m = int(sys.argv[2])
    param.n = int(sys.argv[3])
    param.k = int(sys.argv[4])
    param.transA = 0
    param.transB = 0

    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    gen_testcase(data_dir, param)