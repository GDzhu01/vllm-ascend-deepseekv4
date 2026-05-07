/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_LIBRARY_LIBRARY_UTILS_H
#define CATLASS_LIBRARY_LIBRARY_UTILS_H

#include "catlass/library/operation.h"

namespace Catlass {
namespace Library {


template <typename T> struct DataTypeMap {};

template <> struct DataTypeMap<uint8_t> {
    static DataType const typeId = DataType::U8;
};

template <> struct DataTypeMap<int8_t> {
    static DataType const typeId = DataType::Int8;
};

template <> struct DataTypeMap<int32_t> {
    static DataType const typeId = DataType::Int32;
};

template <> struct DataTypeMap<float16_t> {
    static DataType const typeId = DataType::Fp16;
};

template <> struct DataTypeMap<bfloat16_t> {
    static DataType const typeId = DataType::Bf16;
};

template <> struct DataTypeMap<float32_t> {
    static DataType const typeId = DataType::Fp32;
};

template <typename T> struct LayoutMap {};

template <> struct LayoutMap<Catlass::layout::RowMajor> {
    static LayoutType const typeId = LayoutType::RowMajor;
};

template <> struct LayoutMap<Catlass::layout::ColumnMajor> {
    static LayoutType const typeId = LayoutType::ColumnMajor;
};

template <> struct LayoutMap<Catlass::layout::nZ> {
    static LayoutType const typeId = LayoutType::nZ;
};

template <> struct LayoutMap<Catlass::layout::zN> {
    static LayoutType const typeId = LayoutType::zN;
};

template <> struct LayoutMap<Catlass::layout::zZ> {
    static LayoutType const typeId = LayoutType::zZ;
};

template <> struct LayoutMap<Catlass::layout::PaddingRowMajor> {
    static LayoutType const typeId = LayoutType::PaddingRowMajor;
};

template <> struct LayoutMap<Catlass::layout::PaddingColumnMajor> {
    static LayoutType const typeId = LayoutType::PaddingColumnMajor;
};

template <> struct LayoutMap<Catlass::layout::nN> {
    static LayoutType const typeId = LayoutType::nN;
};

template <> struct LayoutMap<Catlass::layout::VectorLayout> {
    static LayoutType const typeId = LayoutType::VectorLayout;
};

template <typename Element, typename Layout>
TensorDescription MakeTensorDescription()
{
    TensorDescription desc;
    desc.element = DataTypeMap<Element>::typeId;
    desc.layout = LayoutMap<Layout>::typeId;
    return desc;
}
}
}

#endif // CATLASS_LIBRARY_LIBRARY_UTILS_H
