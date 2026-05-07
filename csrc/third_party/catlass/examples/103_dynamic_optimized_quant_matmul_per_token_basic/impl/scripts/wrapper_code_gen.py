#!/usr/bin/env python3
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

import os
from utils.config import Config
from templates.per_token_matmul_template import PerTokenMatmulTemplate
from templates.launch_map_template import LaunchMapTemplate

if __name__ == "__main__":

    kernel_info = {}

    os.makedirs(Config.WRAPPER_CODE_PATH, exist_ok=True)
    
    PerTokenMatmulTemplate.gen_code("int8_t", kernel_info)
    
    LaunchMapTemplate.gen_code(kernel_info)