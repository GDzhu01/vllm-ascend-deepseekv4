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

import random
from typing import Union
import unittest

import torch
import torch_catlass
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


def generate_sequence_split(n, sum_num):
    split_points = sorted([random.randint(0, sum_num) for _ in range(n - 1)])
    split_points = [0] + split_points + [sum_num]
    sequence = [split_points[i + 1] - split_points[i] for i in range(n)]
    sequence[-1] = sum_num - sum(sequence[:-1])
    return sequence


def calculate_prefix_sum(sequence):
    prefix_sum = []
    current_sum = 0
    for num in sequence:
        current_sum += num
        prefix_sum.append(current_sum)
    return prefix_sum


class CatlassTest(TestCase):

    def _run_case_basic(
        self,
        m: int,
        n: int,
        k: int,
        trans_a: bool = False,
        trans_b: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        shape1 = (m, k) if not trans_a else (k, m)
        shape2 = (k, n) if not trans_b else (n, k)

        a = torch.rand(shape1, device="npu").to(dtype)
        b = torch.rand(shape2, device="npu").to(dtype)

        a = a if not trans_a else a.T
        b = b if not trans_b else b.T

        torch.npu.synchronize()
        result = torch_catlass.basic_matmul(a, b, str(dtype).split(".")[-1])
        golden = torch.mm(a, b)
        torch.npu.synchronize()
        if dtype == torch.bfloat16:
            result = result.to(torch.float32)
            golden = golden.to(torch.float32)
        self.assertRtolEqual(result, golden)

    def test_basic_matmul_pybind(self):
        self._run_case_basic(2, 3, 4)

    @unittest.skip("Not ready")
    def test_basic_matmul_pybind_cr(self):
        self._run_case_basic(2, 3, 4, trans_a=True)

    @unittest.skip("Not ready")
    def test_basic_matmul_pybind_rc(self):
        self._run_case_basic(2, 3, 4, trans_b=True)

    @unittest.skip("Not ready")
    def test_basic_matmul_pybind_cc(self):
        self._run_case_basic(2, 3, 4, trans_a=True, trans_b=True)

    @unittest.skip("Not ready")
    def test_basic_matmul_pybind_bf16(self):
        self._run_case_basic(2, 3, 4, trans_a=True, trans_b=True)

    def __grouped_matmul_golden_split_m(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        group_list: Union[torch.Tensor, list[int]],
        dtype: str,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ):
        self.assertEqual(transpose_a, False)
        if isinstance(group_list, list):
            a_list = torch.split(a, group_list)
            # print(a_list)
            result_list = []
            for i in range(len(group_list)):
                a_g = a_list[i] if not transpose_a else a_list[i].T
                b_g = b[i] if not transpose_b else b[i].T
                result_list.append(a_g @ b_g)
            return torch.cat(result_list).to(eval(f"torch.{dtype}"))

    def __grouped_matmul_golden_split_k(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        group_list: Union[torch.Tensor, list[int]],
        dtype: str,
        transpose_a: bool = True,
        transpose_b: bool = False,
    ):
        self.assertEqual(transpose_a and not transpose_b, True)
        assert transpose_a and not transpose_b
        if isinstance(group_list, list):
            a_list = torch.split(a, group_list)
            b_list = torch.split(b, group_list)
            # print(a_list)
            result_list = []
            for i in range(len(group_list)):
                a_g = a_list[i].T
                b_g = b_list[i]
                result_list.append(a_g @ b_g)
            return torch.stack(result_list).to(eval(f"torch.{dtype}"))

    def __grouped_matmul_golden(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        group_list: Union[torch.Tensor, list[int]],
        dtype: str,
        transpose_a: bool,
        transpose_b: bool,
        split_k: bool,
    ):
        if not split_k:
            return self.__grouped_matmul_golden_split_m(
                a, b, group_list, dtype, transpose_a, transpose_b
            )
        else:
            return self.__grouped_matmul_golden_split_k(
                a, b, group_list, dtype, transpose_a, transpose_b
            )

    def test_grouped_matmul_list_m(self):
        g = 128
        group_list = generate_sequence_split(g, random.randint(256, 4096))
        group_list_prefix_sum = calculate_prefix_sum(group_list)
        m_sum, k, n = group_list_prefix_sum[-1], 256, 256
        a = torch.randn((m_sum, k), device="npu").to(torch.float16)
        b = torch.randn((g, k, n), device="npu").to(torch.float16)
        group_list_prefix_sum_tensor = torch.tensor(
            group_list_prefix_sum, device="npu"
        ).to(torch.int64)
        # input, weight, group_list, dtype, transpose_a, transpose_b, 是否为切K
        result = torch_catlass.grouped_matmul(
            a, b, group_list_prefix_sum_tensor, "float16", False, True, False
        )
        golden = self.__grouped_matmul_golden(
            a, b, group_list, "float16", False, True, False
        )
        self.assertRtolEqual(result, golden)

    def test_grouped_matmul_list_k(self):
        g = 128
        group_list = generate_sequence_split(g, random.randint(256, 4096))
        group_list_prefix_sum = calculate_prefix_sum(group_list)
        m, k_sum, n = 256, group_list_prefix_sum[-1], 256
        a = torch.randn((k_sum, m), device="npu").to(torch.float16)
        b = torch.randn((k_sum, n), device="npu").to(torch.float16)
        group_list_prefix_sum_tensor = torch.tensor(
            group_list_prefix_sum, device="npu"
        ).to(torch.int64)
        # input, weight, group_list, dtype, transpose_a, transpose_b, 是否为切K
        result = torch_catlass.grouped_matmul(
            a, b, group_list_prefix_sum_tensor, "float16", True, False, True
        )
        golden = self.__grouped_matmul_golden(
            a, b, group_list, "float16", True, False, True
        )
        self.assertRtolEqual(result, golden)

    def test_optimized_matmul_pybind(self):
        a = torch.rand((2, 3), device="npu").to(torch.float16)
        b = torch.rand((3, 4), device="npu").to(torch.float16)
        torch.npu.synchronize()
        result = torch_catlass.optimized_matmul(a, b, "float16")
        golden = torch.mm(a, b)
        torch.npu.synchronize()
        self.assertRtolEqual(result, golden)


if __name__ == "__main__":
    run_tests()
