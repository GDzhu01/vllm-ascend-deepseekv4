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
import result_compare_method
from batch import quant_lightning_indexer_pt_loadprocess
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import math
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

TEST_INPUT_PATH = "__PATH__"
pt_dir = TEST_INPUT_PATH
result_path = Path('result.xlsx')  # 或使用传入的result_path

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

def qli(testcase_files):   # 初始化参数和tensor
    cpu_result, npu_result, topk_value, params = quant_lightning_indexer_pt_loadprocess.test_qli_process(testcase_files, device_id=0)
    if npu_result != None:
        result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result, topk_value, params)
    else:
        result = "Failed"
        fulfill_percent = 0
    
    row_data = {
        "Testcase_Name": Path(testcase_files).stem,
        "batch_size": params[0],
        "q_seq": params[1],
        "k_seq": params[2],
        "q_t_size": params[3],
        "k_t_size": params[4],
        "q_head_num": params[5],
        "k_head_num": params[6],
        "head_dim": params[7],
        "block_size": params[8],
        "block_num": params[9],
        "qk_dtype": params[10],
        "dequant_dtype": params[11],
        "actual_seq_dtype": params[12],
        "act_seq_q": params[13],
        "act_seq_k": params[14],
        "query_quant_mode": params[15],
        "key_quant_mode": params[16],
        "layout_query": params[17],
        "layout_key": params[18],
        "sparse_count":params[19],
        "sparse_mode":params[20],
        "query_datarange":params[21],
        "key_datarange":params[22],
        "weights_datarange":params[23],
        "q_scale_datarange":params[24],
        "k_scale_datarange":params[25],
        "cmp_ratio":params[26],
        "result":result,
        "fulfill_percent":fulfill_percent
    }

    # 检查文件是否存在
    if result_path.exists():
        # 读取现有数据
        df = pd.read_excel(result_path)
        
        # 检查列名是否一致
        if set(df.columns) != set(row_data.keys()):
            print("警告：变量名与Excel列名不匹配！")
            print(f"Excel列名: {list(df.columns)}")
            print(f"变量名: {list(row_data.keys())}")
            print("请检查变量名或Excel文件")
            return False
        
        # 追加新行
        new_df = pd.DataFrame([row_data])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        # 文件不存在，创建新的DataFrame
        df = pd.DataFrame([row_data])
    
    # 保存到Excel
    df.to_excel(result_path, index=False)

@pytest.mark.ci
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"])
def test_qli(testcase_files):   # 初始化参数和tensor
    with ProcessPoolExecutor(max_workers=1) as executor:
        # 创建当前用例子进程
        future1 = executor.submit(qli, testcase_files)
        # 检查退出码
        for future in as_completed([future1]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"❌ 当前用例子进程执行失败：{e}")

