#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
from utils.config import Config

from templates.common_matmul_template import CommonMatmulTemplate
from templates.aiv_matmul_template import AivMatmulTemplate
from templates.small_matmul_template import SmallMatmulTemplate
from templates.padding_common_matmul_template import PaddingCommonMatmulTemplate
from templates.padding_multi_core_splitk_matmul_template import PaddingMultiCoreSplitkMatmulTemplate
from templates.padding_streamk_matmul_template import PaddingStreamkMatmulTemplate
from templates.single_core_splitk_for_small_k_matmul_template import SingleCoreSplitkForSmallKMatmulTemplate
from templates.padding_single_core_splitk_for_small_k_matmul_template import PaddingSingleCoreSplitkForSmallKMatmulTemplate
from templates.padding_single_core_splitk_k_loop_outer_matmul_template import PaddingSingleCoreSplitkKLoopOuterMatmulTemplate
from templates.padding_single_core_splitk_k_loop_middle_matmul_template import PaddingSingleCoreSplitkKLoopMiddleMatmulTemplate
from templates.local_padding_c_padding_common_matmul_template import LocalPaddingCPaddingCommonMatmulTemplate
from templates.launch_map_template import LaunchMapTemplate

if __name__ == "__main__":

    kernel_info = {}

    os.makedirs(Config.WRAPPER_CODE_PATH, exist_ok=True)
    CommonMatmulTemplate.gen_code("half", kernel_info)
    SmallMatmulTemplate.gen_code("half", kernel_info)
    PaddingCommonMatmulTemplate.gen_code("half", kernel_info)
    PaddingMultiCoreSplitkMatmulTemplate.gen_code("half", kernel_info)
    PaddingStreamkMatmulTemplate.gen_code("half", kernel_info)
    LocalPaddingCPaddingCommonMatmulTemplate.gen_code("half", kernel_info)
    SingleCoreSplitkForSmallKMatmulTemplate.gen_code("half", kernel_info)
    PaddingSingleCoreSplitkForSmallKMatmulTemplate.gen_code("half", kernel_info)
    PaddingSingleCoreSplitkKLoopOuterMatmulTemplate.gen_code("half", kernel_info)
    PaddingSingleCoreSplitkKLoopMiddleMatmulTemplate.gen_code("half", kernel_info)
    AivMatmulTemplate.gen_code("half", kernel_info)
    LaunchMapTemplate.gen_code(kernel_info)