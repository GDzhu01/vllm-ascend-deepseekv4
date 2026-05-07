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

import os
import itertools

from utils.config import Config

class AivMatmulTemplate:

    TEMPLATE = """
#include "kernel/aiv_matmul_kernel.h"
void {launch_kernel_func_name}(aclrtStream& stream, uint64_t fftsAddr,
    uint8_t* dA, uint8_t* dB, uint8_t* dC, uint8_t* dW, uint8_t* dTilingParams, TilingParams& tilingParams)
{{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementA = {element_a};
    using ElementB = {element_b};
    using ElementC = {element_c};
    using LayoutA = {layout_a};
    using LayoutB = {layout_b};
    using LayoutC = {layout_c};
    constexpr DispatchPolicyTag dispatchPolicyTag = {dispatch_policy_tag};
    LaunchAivMatmulKernel<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, dispatchPolicyTag>(
        stream, fftsAddr, dA, dB, dC, dTilingParams, tilingParams);
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
    constexpr DispatchPolicyTag dispatchPolicyTag = {dispatch_policy_tag};
    return AivMatmulKernelGetWorkspaceSize<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, dispatchPolicyTag>(tilingParams);
}}
"""

    KERNEL_NAME = "AivMatmulKernel"

    DISPATCH_POLICY_TAG_MAP = {
        0: "DispatchPolicyTag::DEFAULT",
        1: "DispatchPolicyTag::MATMUL_AIV_SIMPLE",
        2: "DispatchPolicyTag::MATMUL_AIV_TRANS"
    }

    @staticmethod
    def gen_code(dtype, kernel_info):
        kernel_serial = Config.KERNEL_SERIAL_MAP[AivMatmulTemplate.KERNEL_NAME]
        DISPATCH_POLICY_SET = [0, 1, 2]
        for d_tag in DISPATCH_POLICY_SET:
            kernel_func_name = (
                AivMatmulTemplate.KERNEL_NAME
                + dtype.capitalize()
                + "Policy"
                + str(d_tag)
            )
            kernel_info[
                Config.get_tiling_key(kernel_serial, dtype, d_tag, 0, 0, 0, 0, 0)
            ] = kernel_func_name
            launch_kernel_func_name = "Launch" + kernel_func_name
            get_workspace_func_name = kernel_func_name + "GetWorkspaceSize"
            file_name = Config.camel_to_snake(kernel_func_name) + ".cpp"

            element_a = dtype
            element_b = dtype
            element_c = dtype
            layout_a = "Catlass::layout::VectorLayout"
            layout_b = "Catlass::layout::VectorLayout"
            layout_c = "Catlass::layout::RowMajor"
            dispatch_policy_tag = AivMatmulTemplate.DISPATCH_POLICY_TAG_MAP[d_tag]

            content = AivMatmulTemplate.TEMPLATE.format(
                launch_kernel_func_name=launch_kernel_func_name,
                get_workspace_func_name=get_workspace_func_name,
                element_a=element_a,
                element_b=element_b,
                element_c=element_c,
                layout_a=layout_a,
                layout_b=layout_b,
                layout_c=layout_c,
                dispatch_policy_tag=dispatch_policy_tag
            )

            fname = os.path.join(Config.WRAPPER_CODE_PATH, file_name)
            try: os.remove(fname)
            except FileNotFoundError: pass

            fd = os.open(fname, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o550) # r-xr-x---
            with os.fdopen(fd, "w") as f:
                f.write(content)
