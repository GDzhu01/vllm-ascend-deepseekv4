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
import logging

import torch
import numpy as np

IS_TRANS_A = False
IS_TRANS_B = True
IS_BIAS = False
# support "int8_int32" for sparse matmul
DATA_TYPE_STR = "int8_int32"
IS_OUTPUT_TXT = False


class SparseMatmulGenData():
    def __init__(self, m, n, k, b):
        self.m = m
        self.n = n
        self.k = k
        self.b = b
        self.is_trans_a = IS_TRANS_A
        self.is_trans_b = IS_TRANS_B
        self.is_bias = IS_BIAS
        self.data_type_str = DATA_TYPE_STR


    @staticmethod
    def generate_sparse_matrix_b(shape, dtype=np.int8):
        """生成一个指定形状的稀疏矩阵B，每行的每4个元素块至少包含2个零"""
        n, k = shape
        b = np.zeros((n, k), dtype=dtype)  # 初始化矩阵B为全零

        for row in range(n):
            for i in range(0, k, 4):
                block = np.zeros(4, dtype=dtype)

                # 随机选择2个位置放置非零元素
                non_zero_positions = np.random.choice(4, 2, replace=False)
                block[non_zero_positions[0]] = np.random.randint(1, 10, dtype=dtype)
                block[non_zero_positions[1]] = np.random.randint(1, 10, dtype=dtype)

                # 放置到矩阵B的当前行
                b[row, i:i + 4] = block
        return b

    @staticmethod
    def densify_and_generate_index(b):
        """稠密化稀疏矩阵B，并生成索引矩阵"""
        n, k = b.shape
        dense_b = np.zeros((n, k // 2), dtype=b.dtype)  # 稠密化后的矩阵
        index_matrix = np.zeros((n, k // 8), dtype=np.uint8)  # 索引矩阵
        index_mask_matrix = np.zeros((n, k // 2), dtype=np.uint32)  # index mask矩阵

        for row in range(n):
            dense_row = []
            index_row = []
            index_mask_row = []

            for i in range(0, k, 4):
                block = b[row, i:i + 4]
                nonzero_positions = [j for j in range(4) if block[j] != 0]
                # 记录第1和第2个非零元素的索引
                if len(nonzero_positions) == 0:
                    index_1 = 0
                    index_2 = 0
                    index_mask_row.extend([i, i])
                elif len(nonzero_positions) == 1:
                    index_1 = nonzero_positions[0] if nonzero_positions[0] < 3 else 0
                    index_2 = 0 if nonzero_positions[0] < 3 else 2
                    index_mask_row.extend([nonzero_positions[0] + i, i])
                else:
                    index_1 = nonzero_positions[0]
                    index_2 = nonzero_positions[1] - 1
                    index_mask_row.extend([nonzero_positions[0] + i, nonzero_positions[1] + i])

                # 记录稠密化后的块
                dense_block = [block[pos] for pos in nonzero_positions[:2]]
                if len(dense_block) < 2:
                    dense_block += [0] * (2 - len(dense_block))
                dense_row.extend(dense_block)

                # 记录索引
                index_row.extend([index_1, index_2])

            # 将索引逆序排列并打包为 int8
            index_bytes = []
            for j in range(0, len(index_row), 4):
                indices = index_row[j:j + 4]
                int8_value = sum((index << (2 * bit_pos)) for bit_pos, index in enumerate(indices))
                index_bytes.append(int8_value)

            dense_b[row, :] = dense_row
            index_matrix[row, :] = index_bytes
            index_mask_matrix[row, :] = index_mask_row

        return dense_b, index_matrix, index_mask_matrix

    @staticmethod
    def gen_sparse_golden(a, dense_b, index_mask_matrix):
        result_type = np.int32
        m = a.shape[0]
        n = dense_b.shape[0]
        c = np.zeros((m, n), dtype=result_type)
        # 遍历 b 和 index 的每一行
        for r in range(n):
            # 从 a 中根据 index 的第 r 行提取数据
            selected_columns = index_mask_matrix[r]  # 第 r 行的索引
            a_selected = a[:, selected_columns]  # 提取对应列

            # 当前 b 第 r 行与提取后的 a_selected 计算矩阵乘法
            c[:, r] = np.dot(a_selected.astype(result_type), dense_b[r].astype(result_type)).astype(result_type)
        return c

    @staticmethod
    def index_nd_to_nz(index_matrix):
        # 将nd格式index矩阵转换为nz格式，分型size为（16， 8）      uint8 = 4*index     16 * 4* 8
        n = index_matrix.shape[0]
        k = index_matrix.shape[1]

        ceil_n = int(np.ceil(n / 16) * 16)
        ceil_k = int(np.ceil(k / 8) * 8)

        index_matrix_nz = np.zeros((ceil_n, ceil_k), dtype=np.uint8)
        index_matrix_nz[:n, :k] = index_matrix

        new_shape = (ceil_n // 16, 16, ceil_k // 8, 8)
        index_matrix_nz = index_matrix_nz.reshape(new_shape)
        index_matrix_nz = index_matrix_nz.transpose(2, 0, 1, 3)
        return index_matrix_nz

    def check_params(self):
        if self.data_type_str != "int8_int32":
            logging.info("[ERROR] can't support data type %s" % (self.data_type_str))
            return -1
        if self.k % 8 != 0:
            logging.info("[ERROR] sparse k %d must be multiple of 8" % (self.k))
            return -1
        if self.b != 1:
            logging.info("[ERROR] sparse batch %d must be 1" % (self.b))
            return -1
        return 0

    def gen_golden_data(self, work_dir):
        if self.check_params() != 0:
            return -1

        # A
        a_gm = np.random.randint(-10, 10, [self.m, self.k], dtype=np.int8)

        # B
        b_sparse = self.generate_sparse_matrix_b((self.n, self.k)).astype(np.int8)
        b_gm, index_matrix, index_mask_matrix = self.densify_and_generate_index(b_sparse)

        # index
        index_gm = self.index_nd_to_nz(index_matrix)

        # bias
        if self.is_bias:
            bias_gm = np.random.randint(-10, 10, (self.b, 1, self.n), dtype=np.int32)

        # C
        c_gm = self.gen_sparse_golden(a_gm, b_gm, index_mask_matrix)
        if self.is_bias:
            c_gm = c_gm + bias_gm

        if self.is_trans_a:
            a_gm = a_gm.T

        # save to file
        a_gm.tofile(work_dir + "/input/x1_gm.bin")
        b_gm.tofile(work_dir + "/input/x2_gm.bin")
        index_gm.tofile(work_dir + "/input/index_gm.bin")
        c_gm.tofile(work_dir + "/output/golden.bin")
        if self.is_bias:
            bias_gm.tofile(work_dir + "/input/bias_gm.bin")

        # save to txt
        if IS_OUTPUT_TXT:
            np.savetxt(work_dir + "/input/x1_gm.txt", a_gm, fmt='%d', newline='\n')
            np.savetxt(work_dir + "/input/x2_gm.txt", b_gm, fmt='%d', newline='\n')
            np.savetxt(work_dir + "/input/index_gm.txt", index_gm.flatten(), fmt='%d', newline='\n')
            np.savetxt(work_dir + "/output/golden.txt", c_gm, fmt='%d', newline='\n')
            if self.is_bias:
                np.savetxt(work_dir + "/input/bias_gm.txt", bias_gm.flatten(), fmt='%d', newline='\n')
        return 0

    def gen_fake_golden_data(self, work_dir):
        if self.check_params() != 0:
            return -1

        data_type_bytes_ab = 1 # int8
        data_type_bytes_c = 4  # int32

        file_byte = self.m * self.k * data_type_bytes_ab
        with open(work_dir + "/input/x1_gm.bin", 'wb') as file:
            file.truncate(file_byte)

        file_byte = self.k * self.n * data_type_bytes_ab // 2
        with open(work_dir + "/input/x2_gm.bin", 'wb') as file:
            file.truncate(file_byte)

        file_byte = self.k * self.n * data_type_bytes_ab // 8
        with open(work_dir + "/input/index_gm.bin", 'wb') as file:
            file.truncate(file_byte)

        if self.is_bias:
            file_byte = 1 * self.n * data_type_bytes_c
            with open(work_dir + "/input/bias_gm.bin", 'wb') as file:
                file.truncate(file_byte)
        return 0

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    shape_m = int(sys.argv[1])
    shape_n = int(sys.argv[2])
    shape_k = int(sys.argv[3])
    shape_b = 1
    matmul_gen_data = SparseMatmulGenData(shape_m, shape_n, shape_k, shape_b)

    data_dir = os.path.join(current_dir, "input")
    out_dir = os.path.join(current_dir, "output")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    matmul_gen_data.gen_golden_data(current_dir)