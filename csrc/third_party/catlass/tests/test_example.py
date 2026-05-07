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
import re
import subprocess
import unittest
from typing import List, Type

CMAKE_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "output", "bin")
CMAKE_EXAMPLES_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "examples")
os.environ['LD_LIBRARY_PATH'] = os.path.join(CMAKE_BINARY_PATH,
    "..", "shared_lib:") + os.environ.get('LD_LIBRARY_PATH', '')

class CatlassExampleTest(unittest.TestCase):
    def run_case(self, executable_name: str, args: List):
        args = [str(arg) for arg in args]

        ret = subprocess.run(
            [os.path.join(CMAKE_BINARY_PATH, executable_name)] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        for error_log_line in ret.stderr.decode().splitlines():
            acl_match = re.match(
                r"^.*aclError:.*([1-9][0-9]{5})", error_log_line)
            rt_match = re.match(
                r"^.*rtError:.*([1-9][0-9]{5})", error_log_line)
            compare_match = re.match(
                r"^.*Compare failed. Error count: :.*([1-9][0-9]{5})", error_log_line)
            acl_code = 0 if acl_match is None else int(acl_match.group(1))
            rt_code = 0 if rt_match is None else int(rt_match.group(1))
            compare_code = 0 if compare_match is None else int(
                compare_match.group(1))
            self.assertEqual(
                acl_code, 0, f"There is an ACL error: {acl_code}")
            self.assertEqual(rt_code, 0, f"There is an RT error: {rt_code}")
            self.assertEqual(compare_code, 0,
                             f"There is a compare error: {compare_code}")
        self.assertEqual(
            ret.returncode, 0, f"Return code is not zero: {ret.returncode}")

    def test_19_mla(self):
        case_base = [str(i) for i in [1, 1, 128, 16, 16, 128]]
        case_py = case_base + ["half"]
        ret = subprocess.run(["python", os.path.join(
            CMAKE_EXAMPLES_PATH, "19_mla", "gen_data.py")]+case_py)
        case_cpp = case_base + ["--dtype", "half", "--datapath",
                                os.path.join(CMAKE_EXAMPLES_PATH, "19_mla", "data")]
        self.run_case("19_mla", case_cpp)

    def test_24_conv_bias(self):
        case_base = [str(i) for i in [32, 64, 1, 32, 48, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
        case_py = case_base + ["float16"]
        ret = subprocess.run(["python", os.path.join(
            CMAKE_EXAMPLES_PATH, "24_conv_bias", "gen_data.py")]+case_py)        
        case_cpp = [str(i) for i in [32, 1, 4, 32, 48, 16, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
        self.run_case("24_conv_bias", case_cpp)

    def test_29_a2_fp8_e4m3_matmul(self):
        case_py = [str(i) for i in [256, 512, 1024, 0, 0]]
        ret = subprocess.run(["python", os.path.join(
            CMAKE_EXAMPLES_PATH, "29_a2_fp8_e4m3_matmul", "gen_data.py")] + case_py)
        case_cpp = [str(i) for i in [256, 512, 1024, 0]]
        self.run_case("29_a2_fp8_e4m3_matmul", case_cpp)

    def test_32_w4a8_matmul(self):
        case_py = [str(i) for i in [860, 5712, 4535, 0]]
        ret = subprocess.run(["python", os.path.join(
            CMAKE_EXAMPLES_PATH, "32_w4a8_matmul", "gen_data.py")] + case_py)
        case_cpp = [str(i) for i in [860, 5712, 4535, 0]]
        self.run_case("32_w4a8_matmul", case_cpp)
    
    def test_38_w4a4_matmul(self):
        case_py = [str(i) for i in [96, 4096, 1280]] # (M, N, K)
        ret = subprocess.run(["python", os.path.join(
            CMAKE_EXAMPLES_PATH, "38_w4a4_matmul_per_token_per_channel_dequant", "gen_data.py")] + case_py)
        case_cpp = case_py
        self.run_case("38_w4a4_matmul_per_token_per_channel_dequant", case_cpp)
    
    def test_41_sparse_matmul_tla(self):
        case_py = [str(i) for i in [160, 320, 64]] # (M, N, K)
        ret = subprocess.run(["python", os.path.join(
            CMAKE_EXAMPLES_PATH, "41_sparse_matmul_tla", "sparse_gen_data.py")] + case_py)
        case_cpp = case_py
        self.run_case("41_sparse_matmul_tla", case_cpp)

normal_cases = ["00_basic_matmul 256 512 1024 0",
                "01_batched_matmul 5 256 512 1024 0",
                "02_grouped_matmul_slice_m 128 512 1024 2048 0",
                "03_matmul_add 256 512 1024 0",
                "04_padding_matmul 256 512 1024 0",
                "05_grouped_matmul_slice_k 128 512 1024 32 0",
                "06_optimized_matmul 256 512 1024 0",
                "07_grouped_matmul_slice_m_per_token_dequant_moe 128 512 1024 2048 0",
                "08_grouped_matmul 128 512 1024 2048 0",
                "09_splitk_matmul 256 512 1024 0",
                "10_grouped_matmul_slice_m_per_token_dequant 128 512 1024 2048 0",
                "11_grouped_matmul_slice_k_per_token_dequant 128 512 1024 2048 0",
                "12_quant_matmul 256 512 1024 0",
                "13_basic_matmul_tla 256 512 1024 0",
                "14_optimized_matmul_tla 256 512 1024 0",
                "15_gemm 256 512 1024 0",
                "16_group_gemm 3 '128,256,512' '256,512,128' '512,256,128' 0",
                "17_gemv_aiv 256 512 0",
                "18_gemv_aic 256 512 0",
                "20_matmul_bias 256 512 1024 0",
                "21_basic_matmul_preload_zN 256 512 1024 0",
                "22_padding_splitk_matmul 256 512 1024 0",
                "25_matmul_full_loadA 256 512 1024 0",
                "26_matmul_relu 256 512 1024 0",
                "27_matmul_gelu 256 512 1024 0",
                "28_matmul_silu 256 512 1024 0",
                "30_w8a16_matmul 256 512 1024 0",
                "31_small_matmul 256 1024 256 0",
                "33_basic_conv2d 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0",
                "37_streamk_matmul 256 512 1024 0",
                "34_single_core_splitk_matmul 256 512 1024 0",
                "42_quant_optimized_matmul_tla 256 512 1024 0",
                "44_quant_matmul_full_loadA_tla 256 512 1024 0",
                "102_dynamic_optimized_matmul 256 512 1024 0 0 0"
                "103_dynamic_optimized_quant_matmul_per_token_basic 256 512 1024 0 0 0"
                ]


def set_case(case: str):
    case_splited = case.split()
    case_executable_name = case_splited[0]
    case_args = case_splited[1:]

    def __(self: Type[CatlassExampleTest]):
        self.run_case(case_executable_name, case_args)

    setattr(CatlassExampleTest, f"test_{case_executable_name}", __)


for normal_case in normal_cases:
    set_case(normal_case)
if __name__ == '__main__':
    unittest.main()
