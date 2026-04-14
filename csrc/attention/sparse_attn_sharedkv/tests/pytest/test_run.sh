#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 显示帮助信息
show_help() {
    cat << EOF
使用方法: $0 {single|save|load} [参数]

脚本选项:
  single         执行单跑功能
    -R           结果保存路径
    示例1: bash $0 single
    示例2: bash $0 single -R "./result/sas_result.xlsx"

  save           执行将excel文件中用例批量转化为PT文件的功能
    -E           excel表地址
    -S           sheet名
    -P           pt文件保存地址
    示例1: bash $0 save
    示例2: bash $0 save -E "./excel/example.xlsx" -S "Sheet1" -P "./data"

  load           执行批量执行PT形式保存的用例的功能
    -P           pt文件读取地址
    -R           结果保存路径
    示例1: bash $0 load
    示例2: bash $0 load -P "./data" -R "./result/sas_result.xlsx"

通用选项:
  -h, --help   显示此帮助信息
EOF
}

# 检查参数数量
if [ $# -lt 1 ]; then
    echo "错误: 需要至少一个参数"
    show_help
    exit 1
fi

# 获取脚本类型
SCRIPT_TYPE=$1
shift  # 移除第一个参数，剩下的都是选项

# 处理 help 请求
if [[ "$SCRIPT_TYPE" == "-h" ]] || [[ "$SCRIPT_TYPE" == "--help" ]]; then
    show_help
    exit 0
fi

# 运行 单跑 脚本的函数
run_script_single() {    
    echo "准备运行 单跑 脚本..."
    
    # 解析 test_sparse_attn_sharedkv_single.py 的参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -R)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -R 参数需要值"
                    exit 1
                fi
                R_VALUE="$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "test_sparse_attn_sharedkv_single.py 脚本支持的参数: -R"
                exit 1
                ;;
        esac
    done
    
    # 打印参数信息
    echo "==============================="
    echo "参数配置:"
    
    if [ -n "$R_VALUE" ]; then
        echo "  结果存储至文件 $R_VALUE"
        export RESULT_SAVE_PATH="$R_VALUE"
    else
        echo "  结果存储至文件 ./result/sas_result.xlsx"
    fi
    
    echo "==============================="
    
    # 运行 Python 脚本
    if [ "$VERBOSE" = true ]; then
        echo "正在运行 test_sparse_attn_sharedkv_single.py..."
    fi

    # 检查脚本是否存在
    if [ ! -f "test_sparse_attn_sharedkv_single.py" ]; then
        echo "错误: 找不到 test_sparse_attn_sharedkv_single.py 脚本"
        exit 1
    fi
    # 运行脚本，传递环境变量
    python3 -m pytest -rA -s test_sparse_attn_sharedkv_single.py  -v -m ci
}

# 运行 批跑生成pt文件 脚本的函数
run_script_save() {
    echo "准备运行 sparse_attn_sharedkv_pt_save.py 脚本..."
    
    # 解析 sparse_attn_sharedkv_pt_save.py 的参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -E)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -E 参数需要值"
                    exit 1
                fi
                E_VALUE="$2"
                shift 2
                ;;
            -S)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -S 参数需要值"
                    exit 1
                fi
                S_VALUE="$2"
                shift 2
                ;;
            -P)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -P 参数需要值"
                    exit 1
                fi
                P_VALUE="$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "sparse_attn_sharedkv_pt_save.py 脚本支持的参数: -E, -S, -P"
                exit 1
                ;;
        esac
    done
    
    # 打印参数信息
    echo "==============================="
    echo "脚本: sparse_attn_sharedkv_pt_save.py"
    echo "参数配置:"
    
    if [ -n "$E_VALUE" ]; then
        echo "  输入excel文件路径 $E_VALUE"
        export SAS_EXCEL_PATH="$E_VALUE"
    else
        echo "  默认输入excel文件路径 excel/example.xlsx"
    fi
    
    if [ -n "$S_VALUE" ]; then
        echo "  使用sheet名 $S_VALUE"
        export SAS_EXCEL_SHEET="$S_VALUE"
    else
        echo "  默认使用 Sheet1"
    fi
    
    if [ -n "$P_VALUE" ]; then
        echo "  PT文件保存地址 $P_VALUE"
        export SAS_PT_SAVE_PATH="$P_VALUE"
    else
        echo "  默认PT文件保存地址 ./data"
    fi
    
    echo "==============================="
    # 检查脚本是否存在
    if [ ! -f "batch/sparse_attn_sharedkv_pt_save.py" ]; then
        echo "错误: 找不到 sparse_attn_sharedkv_pt_save.py 脚本"
        exit 1
    fi
    
    # 运行脚本，传递环境变量
    python3 -m pytest -rA -s batch/sparse_attn_sharedkv_pt_save.py  -v -m ci
}

# 运行 批跑执行pt文件 脚本的函数
run_script_load() {
    echo "准备运行 test_sas_load.py 脚本..."
    
    # 解析 test_sparse_attn_sharedkv_batch.py 的参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -P)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -P 参数需要值"
                    exit 1
                fi
                P_VALUE="$2"
                shift 2
                ;;
            -R)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -R 参数需要值"
                    exit 1
                fi
                R_VALUE="$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "test_sparse_attn_sharedkv_batch.py 脚本支持的参数: -P, -R"
                exit 1
                ;;
        esac
    done
    
    # 打印参数信息
    echo "==============================="
    echo "脚本: test_sparse_attn_sharedkv_batch.py"
    echo "参数配置:"
    
    if [ -n "$P_VALUE" ]; then
        echo "  PT文件读取地址 $P_VALUE"
        export SAS_PT_LOAD_PATH="$P_VALUE"
    else
        echo "  默认PT文件读取地址 ./data"
    fi

    if [ -n "$R_VALUE" ]; then
        echo "  结果存储至文件 $R_VALUE"
        export SAS_RESULT_SAVE_PATH="$R_VALUE"
    else
        echo "  结果存储至文件 ./result/sas_result.xlsx"
    fi
    
    echo "==============================="
    
    # 检查脚本是否存在
    if [ ! -f "test_sparse_attn_sharedkv_batch.py" ]; then
        echo "错误: 找不到 test_sparse_attn_sharedkv_batch.py 脚本"
        exit 1
    fi
    
    # 运行脚本，传递环境变量
    python3 -m pytest -rA -s test_sparse_attn_sharedkv_batch.py  -v -m ci
}

# 根据脚本类型调用相应的函数
case "$SCRIPT_TYPE" in
    single)
        run_script_single "$@"
        ;;
    save)
        run_script_save "$@"
        ;;
    load)
        run_script_load "$@"
        ;;
    *)
        echo "错误: 未知的脚本类型 '$SCRIPT_TYPE'"
        echo "可用类型: single, save, load"
        show_help
        exit 1
        ;;
esac

echo "==============================="
echo "所有任务完成"