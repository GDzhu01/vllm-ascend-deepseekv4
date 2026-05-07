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

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

script_path = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(
    os.path.join(script_path, "../output/python_extension/libcatlass_torch.so"))  # 手动指定so路径


class CatlassTest(TestCase):

    def test_basic_matmul_torch_lib(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        result = torch.ops.CatlassTorch.basic_matmul(a, b, "float16")
        torch.npu.synchronize()
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)


if __name__ == "__main__":
    run_tests()
