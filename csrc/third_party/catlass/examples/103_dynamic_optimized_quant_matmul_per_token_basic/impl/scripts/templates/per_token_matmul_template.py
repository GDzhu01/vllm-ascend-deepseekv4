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
import itertools

from utils.config import Config

class PerTokenMatmulTemplate:

    TEMPLATE = """
#include "kernel/per_token_matmul_kernel.h"
void {launch_kernel_func_name}(aclrtStream& stream, uint64_t fftsAddr,
    uint8_t* dA, uint8_t* dB, uint8_t* dC, uint8_t* dW, uint8_t* dScale, uint8_t* dPerTokenScale, uint8_t* dTilingParams, TilingParams& tilingParams)
{{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementA = {element_a};
    using ElementB = {element_b};
    using ElementC = {element_c};
    using LayoutA = {layout_a};
    using LayoutB = {layout_b};
    using LayoutC = {layout_c};
    LaunchPerTokenMatmulKernel<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
        stream, fftsAddr, dA, dB, dC, dW, dScale, dPerTokenScale, dTilingParams, tilingParams);
}}

size_t {get_workspace_func_name}(TilingParams& tilingParams)
{{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementA = {element_a};
    using ElementB = {element_b};
    using ElementC = {element_c};
    using LayoutA = {layout_a};
    using LayoutB = {layout_b};
    using LayoutC = {layout_c};
    return PerTokenMatmulKernelGetWorkspaceSize<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(tilingParams);
}}
"""
    KERNEL_NAME = "PerTokenMatmulKernel"

    @staticmethod
    def gen_code(dtype, kernel_info):

        kernel_serial = Config.KERNEL_SERIAL_MAP[PerTokenMatmulTemplate.KERNEL_NAME]

        combinations = list(
            itertools.product(
                Config.LAYOUT_TAG_SET, Config.LAYOUT_TAG_SET
            )
        )
        for l_tag_a, l_tag_b in combinations:
            # kernel_fun_name can be PerTokenMatmulHalfLayout00
            kernel_func_name = (
                PerTokenMatmulTemplate.KERNEL_NAME
                + dtype.capitalize()
                + "Layout"
                + str(l_tag_a)
                + str(l_tag_b)
            )
            # store tilingKey and kernel name
            kernel_info[
                Config.get_tiling_key(kernel_serial, dtype, l_tag_a, l_tag_b, 0)
            ] = kernel_func_name
            # launch_kernel_fun_name can be LaunchPerTokenMatmulHalfLayout00
            launch_kernel_func_name = "Launch" + kernel_func_name
            # get_workspace_fun_name can be PerTokenMatmulHalfLayout00GetWorkspaceSize
            get_workspace_func_name = kernel_func_name + "GetWorkspaceSize"
            # file name can be per_token_matmul_kernel_half_layout_00.cpp
            file_name = Config.camel_to_snake(kernel_func_name) + ".cpp"

            element_a = "int8_t"
            element_b = "int8_t"
            element_c = "half"
            layout_a = Config.LAYOUT_TAG_MAP[l_tag_a]
            layout_b = Config.LAYOUT_TAG_MAP[l_tag_b]
            layout_c = "Catlass::layout::RowMajor"

            content = PerTokenMatmulTemplate.TEMPLATE.format(
                launch_kernel_func_name=launch_kernel_func_name,
                get_workspace_func_name=get_workspace_func_name,
                element_a=element_a,
                element_b=element_b,
                element_c=element_c,
                layout_a=layout_a,
                layout_b=layout_b,
                layout_c=layout_c
            )

            with open(os.path.join(Config.WRAPPER_CODE_PATH, file_name), "w") as f:
                f.write(content)
