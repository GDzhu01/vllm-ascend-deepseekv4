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

import library
import search_space
from manifest import OperationRegistry


"""
If more customization of search space configuration is required, you
need unregister this configuration by COMMENTING this line below:

@OperationRegistry.register_high_priority('00_basic_matmul')

More customizable configurations are provided in search_space.py
in form of functions that are marked by @OperationRegistry.register
e.g. register_gemm_00_basic_matmul_operation
Pruning strategies and parameter combination strategies are customizable.
"""


@OperationRegistry.register_high_priority('00_basic_matmul')
def register(manifest):
    config = search_space.SearchSpaceConfiguration(
        kernel_type='00_basic_matmul',

        data_type_a=library.DataType.fp16,
        data_type_b=library.DataType.fp16,
        data_type_c=library.DataType.fp16,

        layout_a=library.LayoutType.RowMajor,
        layout_b=library.LayoutType.RowMajor,
        layout_c=library.LayoutType.RowMajor,

        l1_tile_m_range=(32, 128),  # min and max of a range are set here
        l1_tile_n_range=(128, 256),
        l1_tile_k_range=(128, 256),

        block_swizzle='Gemm::Block::GemmIdentityBlockSwizzle<3, 0>',
    )

    search_space.register_custom_kernel(config, manifest)
