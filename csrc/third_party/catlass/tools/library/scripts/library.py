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

from enum import Enum, auto


class ArchTag(Enum):
    A2 = auto(),

    def to_code(self):
        code_map = {
            ArchTag.A2: 'Arch::AtlasA2'
        }
        if self in code_map.keys():
            return code_map[self]
        else:
            return 'unknown_type'


class DataType(Enum):
    uint8 = auto(),
    int8 = auto(),
    int32 = auto(),
    fp16 = auto(),
    bf16 = auto(),
    fp32 = auto(),
    invalid = auto(),

    def get_name(self):
        return self.name

    def to_code(self):
        code_map = {
            DataType.uint8: 'uint8_t',
            DataType.int8: 'int8_t',
            DataType.int32: 'int32_t',
            DataType.fp16: 'half',
            DataType.bf16: 'bfloat16_t',
            DataType.fp32: 'float32_t',
        }
        if self in code_map.keys():
            return code_map[self]
        else:
            return 'unknown_type'


class LayoutType(Enum):
    # matrix layout
    ColumnMajor = auto(),
    RowMajor = auto(),
    nZ = auto(),
    zN = auto(),
    zZ = auto(),
    nN = auto(),
    PaddingRowMajor = auto(),
    PaddingColumnMajor = auto(),
    # vector
    VectorLayout = auto(),

    invalid = auto(),

    def get_name(self):
        return self.name

    def to_code(self):
        code_map = {
            LayoutType.ColumnMajor: 'layout::ColumnMajor',
            LayoutType.RowMajor: 'layout::RowMajor',
            LayoutType.nZ: 'layout::nZ',
            LayoutType.zN: 'layout::zN',
            LayoutType.zZ: 'layout::zZ',
            LayoutType.nN: 'layout::nN',
            LayoutType.PaddingRowMajor: 'layout::PaddingRowMajor',
            LayoutType.PaddingColumnMajor: 'layout::PaddingColumnMajor',
            LayoutType.VectorLayout: 'layout::VectorLayout',
        }
        if self in code_map.keys():
            return code_map[self]
        else:
            return 'unknown_layout'


class OperationType(Enum):
    Gemm = auto(),


class TileDescription:

    def __init__(self, l1_tile_shape: list, l0_tile_shape: list):
        self.l1_tile_shape = l1_tile_shape
        self.l0_tile_shape = l0_tile_shape

    def get_name(self):
        if len(self.l1_tile_shape) != 3 or len(self.l0_tile_shape) != 3:
            raise Exception('Invalid tile shape')
        name = 'x'.join(str(val) for val in self.l1_tile_shape) + '_' + \
               'x'.join(str(val) for val in self.l0_tile_shape)
        return name


class GemmTypeDescription:
    def __init__(
        self,
        element_type: DataType = DataType.invalid,
        layout: LayoutType = LayoutType.invalid
    ):
        self.element_type = element_type
        self.layout = layout
        self.position = 'GM'

    def to_code(self):
        if self.element_type is DataType.invalid or self.layout is LayoutType.invalid:
            return 'void'
        else:
            return 'Gemm::GemmType<{}, {}>'.format(self.element_type.to_code(), self.layout.to_code())
