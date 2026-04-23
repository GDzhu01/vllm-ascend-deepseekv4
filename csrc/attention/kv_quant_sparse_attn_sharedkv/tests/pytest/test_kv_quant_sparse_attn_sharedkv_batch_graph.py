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

import itertools
import torch
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import math
import os
import multiprocessing as mp
import concurrent.futures
import result_compare_method
import check_valid_param
from batch import kv_quant_sparse_attn_sharedkv_process
import utils

testcase_path = "qsas_testcase"
result_path = Path('result.xlsx')  # 或使用传入的result_path
device_id=0

locals()["testcase_files"] = []
if os.path.isdir(testcase_path):
    pt_files = [f for f in os.listdir(testcase_path) if f.endswith('.pt')]
    if not pt_files:
        print(f"错误: 目录中没有找到.pt文件: {testcase_path}")
    else:
        print(f"找到 {len(pt_files)} 个测试用例文件")
        for pt_file in pt_files:  
            filepath = os.path.join(testcase_path, pt_file)
            locals()["testcase_files"].append(filepath)
else:
    print(f"错误: 输出目录不存在: {testcase_path}")

def sas_aclgraph(testcase_files):
    test_data = torch.load(testcase_files, map_location="cpu")
    try:
        npu_result, cpu_quant_result = kv_quant_sparse_attn_sharedkv_process.test_sas_quant_process_graph(test_data, device_id=device_id)
        result, fulfill_percent = result_compare_method.check_result(cpu_quant_result, npu_result)
    except Exception as e:
        print("NPU ERROR：", e)
        result = "NPU ERROR"
        fulfill_percent = 0
    
    utils.save_result(test_data['params'], result, fulfill_percent, result_path)
    
    if(result == "NPU ERROR"):
        pytest.fail(f"用例执行失败:{test_data['Testcase_Name']}")


@pytest.mark.graph
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"])
def test_sparse_attn_sharedkv(testcase_files):
    # 线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(sas_aclgraph, testcase_files)
        # 等待并获取结果
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"当前用例线程执行失败")

