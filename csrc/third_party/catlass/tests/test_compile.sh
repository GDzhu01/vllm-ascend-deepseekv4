#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

SCRIPT_PATH=$(dirname "$(realpath "$0")")
BUILD_SCRIPT_PATH=$(realpath "$SCRIPT_PATH"/../scripts/build.sh)

# self contained include
bash "$BUILD_SCRIPT_PATH" --clean --tests test_self_contained_includes || exit 1

# msSanitizer
bash "$BUILD_SCRIPT_PATH" --clean --enable_mssanitizer catlass_examples || exit 1

# ascendc_dump
bash "$BUILD_SCRIPT_PATH" --clean --enable_ascendc_dump catlass_examples || exit 1

# msopgen package build
bash "$BUILD_SCRIPT_PATH" --clean basic_matmul_aclnn || exit 1

# example test. Do not replace or the test will fail.
bash "$BUILD_SCRIPT_PATH" --clean catlass_examples || exit 1