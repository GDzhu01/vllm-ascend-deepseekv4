#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -o errexit
set -o nounset
set -o pipefail

NC="\033[0m"
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"

ERROR="${RED}[ERROR]"
INFO="${GREEN}[INFO]"
WARN="${YELLOW}[WARN]"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CMAKE_SOURCE_DIR=$(realpath "$SCRIPT_DIR/..")
BUILD_DIR="$CMAKE_SOURCE_DIR/build"
OUTPUT_DIR="$CMAKE_SOURCE_DIR/output"
EXAMPLES_DIR="$CMAKE_SOURCE_DIR/examples"

TARGET=""
CMAKE_BUILD_TYPE="Release"
declare -a CMAKE_OPTIONS=()
CLEAN=false
POST_BUILD_INFO=""

echo -e "  ____    _  _____ _        _    ____ ____  "
echo -e " / ___|  / \|_   _| |      / \  / ___/ ___| "
echo -e "| |     / _ \ | | | |     / _ \ \___ \___ \ "
echo -e "| |___ / ___ \| | | |___ / ___ \ ___) |__) |"
echo -e " \____/_/   \_\_| |_____/_/   \_\____/____/ "

function show_help() {
    echo -e "${GREEN}Usage:${NC} $0 [options] <target>"
    echo -e "\n${BLUE}Options:${NC}"
    echo "  --clean         Clean build directories"
    echo "  --debug         Build in debug mode"
    echo "  --msdebug       Enable msdebug support"
    echo "  --simulator     Compile example in simulator mode"
    echo "  --enable_profiling Enable profiling"
    echo "  --enable_print  Enable built-in compiler print feature"
    echo "  --enable_ascendc_dump   Enable AscendC dump API"
    echo "  --tests         Enable building targets in tests"
    echo "  -DCATLASS_ARCH  NPU arch. Supports 2201(AtlasA2/A3)/3510(Ascend950PR/DT)."
    echo "  -D<option>      Additional CMake options"
    echo -e "\n${BLUE}Targets:${NC}"
    echo "  catlass_examples  Build Catlass examples"
    echo "  python_extension  Build Python extension"
    echo "  torch_library     Build Torch library"
    echo "  mstuner_catlass   Build msTuner for CATLASS. Use it with -DCATLASS_LIBRARY_KERNELS=<kernel_name>"
    echo "  <other>           Other specific targets, e.g. 00_basic_matmul"
    echo -e "\n{BLUE}Test targets:${NC}"
    echo "  test_self_contained_includes  Test for self contained includes"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

if [[ ! -v ASCEND_HOME_PATH ]]; then
    echo -e "${ERROR}ASCEND_HOME_PATH environment variable is not set!${NC}"
    echo -e "${ERROR}Please set ASCEND_HOME_PATH before running this script.${NC}"
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            ;;
        --debug)
            CMAKE_OPTIONS+=("-DCMAKE_BUILD_TYPE=Debug")
            echo -e "${WARN}Debug mode enabled"
            ;;
        --msdebug)
            CMAKE_OPTIONS+=("-DASCEND_ENABLE_MSDEBUG=True")
            ;;
        --simulator)
            CMAKE_OPTIONS+=("-DENABLE_SIMULATOR=True")
            ;;
        --tests)
            CMAKE_OPTIONS+=("-DBUILD_TESTS=True")
            ;;
        --enable_profiling)
            CMAKE_OPTIONS+=("-DASCEND_ENABLE_MSPROF=True")
            ;;
        --enable_ascendc_dump)
            CMAKE_OPTIONS+=("-DENABLE_ASCENDC_DUMP=True")
            ;;
        --enable_print)
            CMAKE_OPTIONS+=("-DENABLE_PRINT=True")
            ;;
        --enable_mssanitizer)
            CMAKE_OPTIONS+=("-DENABLE_MSSANITIZER=True")
            ;;
        -D*)
            CMAKE_OPTIONS+=("$1")
            ;;
        *)
            if [[ -z "$TARGET" ]]; then
                TARGET="$1"
            else
                echo -e "${ERROR}Multiple targets specified${NC}" >&2
                show_help
                exit 1
            fi
            ;;
    esac
    shift
done

if [[ "$CLEAN" == true ]]; then
    echo -e "${INFO}Cleaning build directories...\c"
    rm -rf "$BUILD_DIR" "$OUTPUT_DIR"
    if [[ -d "$EXAMPLES_DIR/python_extension/build" ]]; then
        rm -rf "$EXAMPLES_DIR/python_extension/build"
    fi
    echo -e "Complete!${NC}"
fi

if [[ -z "$TARGET" ]]; then
    echo -e "${ERROR}No target specified${NC}" >&2
    show_help
    exit 1
fi

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"

function build_python_extension() {
    echo -e "${INFO}Building Python extension...${NC}"
    export CATLASS_CMAKE_OPTIONS="${CMAKE_OPTIONS[@]}"
    cd "$CMAKE_SOURCE_DIR/examples/python_extension" || exit 1
    rm -rf build dist ./*.egg-info
    python3 setup.py bdist_wheel --dist-dir "$OUTPUT_DIR/python_extension"
    unset CATLASS_CMAKE_OPTIONS
    echo -e "${INFO}Python extension built successfully${NC}"
}

function build_torch_library() {
    echo -e "${INFO}Building Torch library...${NC}"
    cmake -B build \
        -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR/python_extension" \
        -DCATLASS_INCLUDE_DIR="$CMAKE_SOURCE_DIR/include" \
        -DPython3_EXECUTABLE="$(which python3)" \
        -DBUILD_TORCH_LIB=True \
        "${CMAKE_OPTIONS[@]}"
    cmake --build build --target catlass_torch -j
    cmake --install build --component catlass_torch
    echo -e "${INFO}Torch library built successfully${NC}"
}

function build_mstuner_catlass() {
    echo -e "${INFO}Building mstuner_catlass...${NC}"
    cmake -S "$CMAKE_SOURCE_DIR" -B "$BUILD_DIR" "${CMAKE_OPTIONS[@]}"
    cmake --build build --target mstuner_catlass -j
    cmake --install build --component catlass_kernels
    cmake --install build --component mstuner_catlass
    echo -e "${INFO}mstuner_catlass built successfully${NC}"
}

# 执行构建
case "$TARGET" in
    python_extension)
        build_python_extension
        ;;
    torch_library)
        build_torch_library
        ;;
    mstuner_catlass)
        build_mstuner_catlass
        ;;
    *)
        echo -e "${INFO}Building target: $TARGET...${NC}"
        if [[ -d ${BUILD_DIR} ]]; then
            if [[ "$TARGET" == "102_dynamic_optimized_matmul" ]] || [[ "$TARGET" == "catlass_examples" ]]; then
                cmake -S "$CMAKE_SOURCE_DIR" -B "$BUILD_DIR" \
                    -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
                    -DCOMPILE_DYNAMIC_OPTIMIZED_MATMUL=ON \
                    "${CMAKE_OPTIONS[@]}"
            elif [[ "$TARGET" == "103_dynamic_optimized_quant_matmul_per_token_basic" ]] || [[ "$TARGET" == "catlass_examples" ]]; then
                cmake -S "$CMAKE_SOURCE_DIR" -B "$BUILD_DIR" \
                    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
                    -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
                    -DCOMPILE_DYNAMIC_QBMM_OPTIMIZED_MATMUL=ON \
                    "${CMAKE_OPTIONS[@]}"
            else
                cmake -S "$CMAKE_SOURCE_DIR" -B "$BUILD_DIR" \
                    -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
                    "${CMAKE_OPTIONS[@]}"
            fi
        fi
        cmake --build "$BUILD_DIR" --target "$TARGET" -j
        cmake --install "$BUILD_DIR" --component "$TARGET"
        echo -e "${INFO}Target '$TARGET' built successfully${NC}"
        ;;
esac

echo -e "$POST_BUILD_INFO"
