
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
import subprocess
import shutil
import unittest


class MsTunerCatlassTest(unittest.TestCase):

    CATLASS_OUTPUT_BINARY_PATH = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "output", "bin"
    )
    CATLASS_OUTPUT_LIB_PATH = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "output", "lib64"
    )
    MSTUNER_TEST_TEMP_PATH = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "mstuner_test_temp"
    )

    @classmethod
    def setUpClass(cls):
        # create temp dir for mstuner_test
        if not os.path.exists(MsTunerCatlassTest.MSTUNER_TEST_TEMP_PATH):
            os.mkdir(MsTunerCatlassTest.MSTUNER_TEST_TEMP_PATH)

        # add ./output/lib64 to LD_LIBRARY_PATH
        if 'LD_LIBRARY_PATH' in os.environ:
            os.environ['LD_LIBRARY_PATH'] = MsTunerCatlassTest.CATLASS_OUTPUT_LIB_PATH + \
                ':' + os.environ['LD_LIBRARY_PATH']
        else:
            os.environ['LD_LIBRARY_PATH'] = MsTunerCatlassTest.CATLASS_OUTPUT_LIB_PATH

    @classmethod
    def tearDownClass(cls):
        # remove temp dir for mstuner_test
        if os.path.exists(MsTunerCatlassTest.MSTUNER_TEST_TEMP_PATH):
            shutil.rmtree(MsTunerCatlassTest.MSTUNER_TEST_TEMP_PATH, ignore_errors=True)


    def compile_lib_catlass_kernels(self, kernel_name: str):
        macro_str = '-DCATLASS_LIBRARY_KERNELS=' + kernel_name
        compile_cmd = [
            'bash', 'scripts/build.sh', '--clean', macro_str, 'mstuner_catlass'
        ]
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)
        return result


    def run_one_case(self, case):
        case_name = case[0]
        case_args = case[1:]
        csv_file_name = case_name + '.csv'
        csv_file_path = os.path.join(MsTunerCatlassTest.MSTUNER_TEST_TEMP_PATH, csv_file_name)

        result = self.compile_lib_catlass_kernels(case_name)
        self.assertEqual(
            result.returncode, 0,
            f'build libcatlass_kernels.so for {case_name} failed: {result.stderr}'
        )

        mstuner_path = os.path.join(MsTunerCatlassTest.CATLASS_OUTPUT_BINARY_PATH, 'mstuner_catlass')
        mstuner_cmd = [
            mstuner_path,
            f'--output={csv_file_path}'
        ]
        mstuner_cmd.extend(case_args)
        result = subprocess.run(mstuner_cmd, capture_output=True, text=True, timeout=600)
        self.assertEqual(
            result.returncode, 0,
            f'running mstuner_catlass for {case_name} with {case_args} failed: {result.stderr}'
        )

        self.assertEqual(
            os.path.exists(csv_file_path), True,
            f'csv file not generated for {case_name} with {case_args} failed: {result.stdout}'
        )


    # add custom test cases below
    mstuner_cases = [
        ['00_basic_matmul', '--m=256', '--n=512', '--k=1024'],
        ['02_grouped_matmul_slice_m', '--m=512', '--n=1024', '--k=2048', '--group_count=128'],
        ['06_optimized_matmul_padding_ab', '--m=555', '--n=322', '--k=1111'],
        ['06_optimized_matmul_padding_a_only', '--m=655', '--n=256', '--k=1111'],
        ['06_optimized_matmul_padding_b_only', '--m=555', '--n=322', '--k=1024'],
        ['06_optimized_matmul_without_padding', '--m=512', '--n=256', '--k=1024'],
        ['08_grouped_matmul', '--m=512', '--n=1024', '--k=2048', '--group_count=128'],
        ['12_grouped_matmul', '--m=256', '--n=512', '--k=1024'],
        ['27_matmul_gelu', '--m=256', '--n=512', '--k=1024'], # Add matmul gelu mstuner (m, n, k)
    ]


    def test_all_cases(self):
        for case in self.mstuner_cases:
            self.run_one_case(case)


if __name__ == '__main__':
    unittest.main()