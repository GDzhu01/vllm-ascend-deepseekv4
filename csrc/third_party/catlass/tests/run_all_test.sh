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

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_PATH=$(dirname "$(realpath "$0")")
BUILD_SCRIPT_PATH=$(realpath "$SCRIPT_PATH"/../scripts/build.sh)

bash "$SCRIPT_PATH/test_compile.sh"

# example test
python3 "$SCRIPT_PATH/test_example.py"

# python extension
bash "$BUILD_SCRIPT_PATH" --clean python_extension || exit 1
WHEEL_DIR="$SCRIPT_PATH/../output/python_extension/"
WHEEL_FILE=$(find "$WHEEL_DIR" -type f -name "torch_catlass-*.whl" 2>/dev/null | head -n 1)
if [ -z "$WHEEL_FILE" ]; then
    echo "Error: No .whl file found in $WHEEL_DIR"
    exit 1
fi
pip install "$WHEEL_FILE"
python3 "$SCRIPT_PATH/test_python_extension.py"
pip uninstall torch_catlass -y

# torch lib
bash "$BUILD_SCRIPT_PATH" --clean torch_library || exit 1
python3 "$SCRIPT_PATH/test_torch_lib.py"

# mstuner_catlass
python3 "$SCRIPT_PATH/test_mstuner.py"