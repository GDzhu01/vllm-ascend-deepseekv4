#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import result_compare_method
import utils
from batch import sparse_attn_sharedkv_process
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

pt_dir = os.getenv("SAS_PT_LOAD_PATH", "./data")
result_path = Path(os.getenv("SAS_RESULT_SAVE_PATH", './result/sas_result.xlsx'))

# 生成所有的组合，并转换为字典列表
locals()["testcase_files"] = []
if os.path.isdir(pt_dir):
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
    if not pt_files:
        print(f"错误: 目录中没有找到.pt文件: {pt_dir}")
    else:
        print(f"找到 {len(pt_files)} 个测试用例文件")
        for pt_file in pt_files:  
            filepath = os.path.join(pt_dir, pt_file)
            locals()["testcase_files"].append(filepath)
else:
    print(f"错误: 输出目录不存在: {pt_dir}")

print("files:", locals()["testcase_files"])

def call_sas_npu(testcase_files):   # 初始化参数和tensor
    print("执行文件: ", testcase_files)
    torch_npu.npu.set_device(0)
    test_data = torch.load(testcase_files, map_location="cpu")
    npu_result = None
    try:
        npu_result, softmax_lse = sparse_attn_sharedkv_process.call_npu(test_data)
    except Exception as e:
        utils.save_result('Exception', 0, test_data['params'], result_path)
        raise e
    if npu_result != None:
        result, fulfill_percent = result_compare_method.check_result(test_data['cpu_output'], npu_result)
    else:
        result = "Failed"
        fulfill_percent = 0
    utils.save_result(result, fulfill_percent, test_data['params'], result_path)

@pytest.mark.ci
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"])
def test_sparse_attn_sharedkv(testcase_files):   # 初始化参数和tensor
    with ProcessPoolExecutor(max_workers=1) as executor:
        # 创建当前用例子进程
        future1 = executor.submit(call_sas_npu, testcase_files)
        # 检查退出码
        for future in as_completed([future1]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"❌ 当前用例子进程执行失败：{e}")