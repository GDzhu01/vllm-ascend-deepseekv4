# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools
import torch
from batch import compressor_pt_loadprocess
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import math
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from compressor_golden import check_result

TEST_INPUT_PATH = "./pt_path"
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

def compressor(testcase_files):   # 初始化参数和tensor
    cpu_result, kv_mask_result, npu_result ,cpu_kv_state, mask_cpu_kv_state, cpu_score_state, mask_cpu_score_state, state_cache, params = compressor_pt_loadprocess.test_compressor_process(testcase_files, device_id=0)
    if npu_result != None:
        cpu_kv_state_update = cpu_kv_state[mask_cpu_kv_state]
        cpu_kv_state_origin =  cpu_kv_state[~mask_cpu_kv_state]
        cpu_score_state_update = cpu_score_state[mask_cpu_score_state]
        cpu_score_state_origin = cpu_score_state[~mask_cpu_score_state]
        data_type = str(npu_result.dtype)
        print("--------------------------------------------------------------check result-------------------------------------------------------------")
        result_percent, result= check_result(cpu_result[kv_mask_result].to(torch.float32), npu_result.cpu()[kv_mask_result].to(torch.float32), data_type)
        print("--------------------------------------------------------------check kv state update-------------------------------------------------------------")
        kv_state_result_percent, kv_state_result= check_result(cpu_kv_state_update.to(torch.float32), state_cache.cpu()[:, :, :state_cache.shape[2]//2][mask_cpu_kv_state].to(torch.float32), data_type)
        print("--------------------------------------------------------------check score state update-------------------------------------------------------------")
        score_state_result_percent, score_state_result = check_result(cpu_score_state_update.to(torch.float32), state_cache.cpu()[:, :, state_cache.shape[2]//2:][mask_cpu_score_state].to(torch.float32), data_type)
        print("--------------------------------------------------------------check kv state origin-------------------------------------------------------------")
        kv_state_origin_result_percent, kv_state_origin_result = check_result(cpu_kv_state_origin.to(torch.float32), state_cache.cpu()[:, :, :state_cache.shape[2]//2][~mask_cpu_kv_state].to(torch.float32), data_type, 0.0)
        print("--------------------------------------------------------------check score state origin-------------------------------------------------------------")
        score_state_origin_result_percent, score_state_origin_result = check_result(cpu_score_state_origin.to(torch.float32), state_cache.cpu()[:, :, state_cache.shape[2]//2:][~mask_cpu_score_state].to(torch.float32), data_type, 0.0)
    else:
        result = "Failed"
        result_percent = 0
        kv_state_result = "Failed"
        kv_state_result_percent = 0
        score_state_result = "Failed"
        score_state_result_percent = 0
        kv_state_origin_result = "Failed"
        kv_state_origin_result_percent = 0
        score_state_origin_result = "Failed"
        score_state_origin_result_percent = 0

    
    row_data = {
        "Testcase_Name": Path(testcase_files).stem,
        "batch_size": params[0],
        "hidden_size": params[1],
        "Seq_len": params[2],
        "head_dim": params[3],
        "block_size": params[4],
        "rope_head_dim": params[5],
        "cmp_ratio": params[6],
        "coff": params[7],
        "norm_eps": params[8],
        "start_p": params[9],
        "rotary_mode": params[10],
        "cache_mode": params[11],
        "layout_x": params[12],
        "data_type": params[13],
        "cu_seqlens": params[14],
        "seqused": params[15],
        "start_pos": params[16],
        "x_datarange": params[17],
        "wkv_datarange": params[18],
        "wgate_datarange": params[19],
        "ape_datarange": params[20],
        "norm_weight_datarange": params[21],
        "kv_state_datarange": params[22],
        "score_state_datarange": params[23],
        "result":result,
        "result_percent":result_percent,
        "kv_state_update":kv_state_result,
        "kv_state_update_percent":kv_state_result_percent,
        "score_state_update":score_state_result,
        "score_state_update_percent":score_state_result_percent,
        "kv_state_origin":kv_state_origin_result,
        "kv_state_origin_percent":kv_state_origin_result_percent,
        "score_state_origin":score_state_origin_result,
        "score_state_origin_percent":score_state_origin_result_percent
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
def test_compressor(testcase_files):   # 初始化参数和tensor
    with ProcessPoolExecutor(max_workers=1) as executor:
        # 创建当前用例子进程
        future1 = executor.submit(compressor, testcase_files)
        # 检查退出码
        for future in as_completed([future1]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"❌ 当前用例子进程执行失败：{e}")
