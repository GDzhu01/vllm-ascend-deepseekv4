#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import numpy as np
import os
import pandas as pd
from pathlib import Path
import pytest

def load_excel_test_cases(excel_file_path: str, sheetname: str):

    """
    从 Excel 文件加载测试用例。

    参数:
        excel_file_path (str): Excel 文件的路径。
        sheetname (str, optional): 工作表名称。若未提供，则默认 'Sheet1'。

    返回:
        list[tuple]: 测试用例元组列表，每个元组包含 20+ 个字段。
                       若失败或跳过，则返回空列表。
    """
    # 优先使用传入的 sheetname，否则尝试从环境变量获取
    if sheetname is None:
        sheetname = 'Sheet1'

    # 检查文件是否存在
    if not os.path.exists(excel_file_path):
        pytest.skip(f"Excel file not found: {excel_file_path}", allow_module_level=True)

    try:
        # 读取 Excel 文件的指定 sheet
        df = pd.read_excel(excel_file_path, sheet_name=sheetname)
        df = df.replace({np.nan: None, pd.NA: None})

        # 定义必需的列名
        required_columns = [
            "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type", "B", "S1", "S2", "N1", "N2", "D", "K",
            "block_size1", "block_size2", "softmax_scale", "cmp_ratio",
            "ori_mask_mode", "cmp_mask_mode", "ori_win_left", "ori_win_right", "testcase_name"
        ]
        # 检查是否缺少必要列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            pytest.skip(f"Missing required columns in Excel: {missing_cols}", allow_module_level=True)

        # 构建测试用例列表
        test_cases = []
        for _, row in df.iterrows():
            test_cases.append(row.to_dict())

        return test_cases

    except Exception as e:
        pytest.skip(f"Failed to read Excel file: {e}", allow_module_level=True)
        return None

def save_result(result, fulfill_percent, params, result_path='./result/sas_result.xlsx'):
    result_path = Path(result_path)
    row_data = {
        "testcase_name": params[24] ,
        "layout_q": params[0],
        "layout_kv": params[1],
        "q_type": params[2],
        "ori_kv_type": params[3],
        "cmp_kv_type": params[4],
        "B": params[5],
        "S1": params[6],
        "S2": None if len(params) == 25 else params[25],
        "T1": params[7],
        "N1": params[8],
        "N2": params[9],
        "D": params[10],
        "K": params[11],
        "block_num1": params[12],
        "block_num2": params[13],
        "block_size1": params[14],
        "block_size2": params[15],
        "cu_seqlens_q": params[16],
        "seqused_q": None if len(params) < 30 else params[29],
        "seqused_kv": params[17],
        "softmax_scale": params[18],
        "cmp_ratio": params[19],
        "ori_mask_mode": params[20],
        "cmp_mask_mode": params[21],
        "ori_win_left": params[22],
        "ori_win_right": params[23],
        "q_datarange": None if len(params) == 25 else params[26],
        "ori_kv_datarange": None if len(params) == 25 else params[27],
        "cmp_kv_datarange": None if len(params) == 25 else params[28],
        "result": result,
        "fulfill_percent": fulfill_percent,
    }
    # 检查文件是否存在
    result_path.parent.mkdir(parents=True, exist_ok=True)
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
