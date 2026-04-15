#!/bin/bash
# ====================== 配置区======================
# 需要读取的用例excel表格路径，如下：
PATH1="./excel/*"
# 用例pt的文件存放路径，如下：
PATH2="./pt_path"

# 脚本路径
Compressor_PT_SAVE_SCRIPT="./batch/compressor_pt_save.py"
TEST_Compressor_BATCH_SCRIPT="test_compressor_batch.py"
REPLACE_PATH_SCRIPT="./batch/replace_path.py"
TEST_Compressor_SINGLE_SCRIPT="test_compressor_single.py"

# ====================== 执行区======================

# 单用例算子调测
run_single() {
    echo "===== 执行单用例算子调测 ====="
    python3 -m pytest -rA -s $TEST_Compressor_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 用例批量生成调试
run_batch() {
    echo "===== 执行用例批量生成测试 ====="

    echo -e "\n===== 第一步：执行compressor_pt_save.py ====="
    python3 $Compressor_PT_SAVE_SCRIPT $PATH1 $PATH2
    if [ $? -ne 0 ]; then
        echo "compressor_pt_save.py 执行失败，退出"
        exit 1
    fi

    echo -e "\n===== 第二步：替换test_compressor_batch.py中的路径 ====="
    python3 $REPLACE_PATH_SCRIPT $TEST_Compressor_BATCH_SCRIPT $PATH2
    if [ $? -ne 0 ]; then
        echo "替换路径失败，退出"
        exit 1
    fi

    echo -e "\n===== 第三步：执行pytest命令 ====="
    python3 -m pytest -rA -s $TEST_Compressor_BATCH_SCRIPT -v -m ci
    if [ $? -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    cp test_compressor_batch.py.bak test_compressor_batch.py

    echo -e "\n=====执行完成！====="
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single    执行单算子用例调测"
    echo "  batch     执行用例批量生成调试"
    echo "  help      显示本帮助信息"
    echo "示例："
    echo "  $0 single  # 执行single模式"
    echo "  $0 batch   # 执行batch模式"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -ne 1 ]; then
    echo "错误：必须传入且仅传入一个参数（single/batch/help）"
    show_help
    exit 1
fi

# 根据参数执行对应函数
case "$1" in
    single)
        run_single
        ;;
    batch)
        run_batch
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/batch/help"
        show_help
        exit 1
        ;;
esac

exit 0