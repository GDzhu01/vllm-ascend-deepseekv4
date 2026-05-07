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
import torch
import numpy as np
import pandas as pd
from enum import Enum

CONST_16 = 16
RECORD_COUNT = 10
DATA_RANGE = (-1.0, 1.0)
WORKSPACE = os.getcwd()

os.environ["WORKSPACE"] = WORKSPACE
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"


class CubeFormat(Enum):
    ND = 0
    NZ = 1
    ZN = 2
    ZZ = 3
    NN = 4
    VECTOR = 5

    def __repr__(self) -> str:
        return self.__name__

class OpParam:
    def __init__(self) -> None:
        self.b = 0
        self.m = 0
        self.k = 0
        self.n = 0
        self.transA = False
        self.transB = False
        self.enBias = False
        self.enScale = False
        self.enResidual = False
        self.layoutA = CubeFormat.ND
        self.layoutB = CubeFormat.ND
        self.layoutC = CubeFormat.ND

    def __str__(self) -> str:
        return f"Shape: ({self.b}, {self.m}, {self.k}, {self.n}) \n" + \
               f"Transpose: A {self.transA}, B {self.transB} \n" + \
               f"(De)Quant: Bias {self.enBias}, Scale {self.enScale}, Residual {self.enResidual} \n" + \
               f"Layout: layoutA {self.layoutA}, layoutB {self.layoutB}, layoutC {self.layoutC}"
# 生成一个mXn的矩阵，其中的值介于low和high之间
def gen_rand(msize, nsize, low, high):
    return low + (high - low) * torch.rand((msize, nsize),dtype=torch.float32)

# 生成一个row x col的二维数组，数值介于-8到7之间
def gen_data_int8(row, col):
    data = np.random.randint(-8, 8, size=(row, col), dtype=np.int8)
    return data

# 生成数据并做数据压缩
def gen_data_int4(row, col, trans):
    # 生成原始数据，范围严格在 [-8, 7]
    data_int8_origin = np.random.randint(-8, 8, size=(row, col), dtype=np.int8)
    if trans:
        data_int8_origin = data_int8_origin.T
        # 如果列数为奇数，则补零列
        data_int8 = data_int8_origin
        if row % 2 != 0:
            zero_row = np.zeros((col, 1), dtype=np.int8)
            data_int8 = np.hstack((data_int8_origin, zero_row))

        quantized = data_int8.reshape(-1, 2)
        high_quantized = (quantized[:, 0] & 0x0F)
        low_quantized = (quantized[:, 1] & 0x0F) << 4
        data_int4 = low_quantized | high_quantized

        data_int4_array = np.array(data_int4, dtype=np.int8)
        return data_int8_origin.T, data_int4_array

    else:
        data_int8 = data_int8_origin
        if col % 2 != 0:
            zero_column = np.zeros((row, 1), dtype=np.int8)
            data_int8 = np.hstack((data_int8_origin, zero_column))

        quantized = data_int8.reshape(-1, 2)
        high_quantized = (quantized[:, 0] & 0x0F)
        low_quantized = (quantized[:, 1] & 0x0F) << 4
        data_int4 = low_quantized | high_quantized

        data_int4_array = np.array(data_int4, dtype=np.int8)
        return data_int8_origin, data_int4_array

def gen_testcase(path: str, param: OpParam) -> None:
    bsize, msize, ksize, nsize = param.b, param.m, param.k, param.n
    transA, transB = param.transA, param.transB

    a_int8 = gen_data_int8(msize, ksize)

    b_int8, b_int4 = gen_data_int4(ksize, nsize, transB)
    b_int4.tofile(os.path.join(path, "inputB.dat"))

    # numpy int32矩阵乘法非常慢，此处用float32的矩阵乘法代替
    c_int32 = np.dot(a_int8.astype(np.float32), b_int8.astype(np.float32))
    c_int32 = np.float32(1.5) * c_int32

    if transA:
        a_int8 = a_int8.T
    a_int8.tofile(os.path.join(path, "inputA.dat"))

    c_float = c_int32.astype(np.float32)
    c_half = c_int32.astype(np.float16)
    c_half.tofile(os.path.join(path, "inputC.dat"))
    c_float.tofile(os.path.join(path, "expected.dat"))

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    param = OpParam()
    param.b = 1
    param.m = int(sys.argv[1])
    param.n = int(sys.argv[2])
    param.k = int(sys.argv[3])
    param.transA = 0
    param.transB = 0

    data_dir = os.path.join(current_dir, "data")

    os.makedirs(data_dir, exist_ok=True)
    gen_testcase(data_dir, param)