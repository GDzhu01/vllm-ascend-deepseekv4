/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "library_helper.h"

#include <opdev/bfloat16.h>

namespace Catlass {

namespace {

using namespace Library;

template<typename T>
void ConstructLayoutT(DataType dataType, uint32_t a, uint32_t b, uint8_t *data)
{
    switch (dataType) {
        case DataType::U8:
            *reinterpret_cast<T*>(data) = T::template MakeLayout<uint8_t>(a, b);
            break;
        case DataType::Int8:
            *reinterpret_cast<T*>(data) = T::template MakeLayout<int8_t>(a, b);
            break;
        case DataType::Int32:
            *reinterpret_cast<T*>(data) = T::template MakeLayout<int32_t>(a, b);
            break;
        case DataType::Fp16:
            *reinterpret_cast<T*>(data) = T::template MakeLayout<op::fp16_t>(a, b);
            break;
        case DataType::Bf16:
            *reinterpret_cast<T*>(data) = T::template MakeLayout<op::bfloat16>(a, b);
            break;
        case DataType::Fp32:
            *reinterpret_cast<T*>(data) = T::template MakeLayout<float>(a, b);
            break;
        default:
            break;
    }
}
} // namespace

size_t LibraryHelper::GetDataTypeSize(LibraryHelper::DataType dataType)
{
    constexpr size_t BYTE = 1;
    constexpr size_t WORD = 2;
    constexpr size_t DWORD = 4;
    switch (dataType) {
        case DataType::U8:
        case DataType::Int8:
            return BYTE;
        case DataType::Fp16:
        case DataType::Bf16:
            return WORD;
        case DataType::Int32:
        case DataType::Fp32:
            return DWORD;
        default:
            return 0;
    }
}

size_t LibraryHelper::GetLayoutSize(LibraryHelper::LayoutType layoutType)
{
    switch (layoutType) {
        case LayoutType::RowMajor:
            return sizeof(layout::RowMajor);
        case LayoutType::ColumnMajor:
            return sizeof(layout::ColumnMajor);
        case LayoutType::nZ:
            return sizeof(layout::nZ);
        case LayoutType::zN:
            return sizeof(layout::zN);
        case LayoutType::zZ:
            return sizeof(layout::zZ);
        case LayoutType::PaddingRowMajor:
            return sizeof(layout::PaddingRowMajor);
        case LayoutType::PaddingColumnMajor:
            return sizeof(layout::PaddingColumnMajor);
        case LayoutType::nN:
            return sizeof(layout::nN);
        default:
            return 0;
    }
}

std::string_view LibraryHelper::GetDataTypeStr(LibraryHelper::DataType dataType)
{
    switch (dataType) {
        case DataType::U8:
            return "u8";
        case DataType::Int8:
            return "int8";
        case DataType::Int32:
            return "int32";
        case DataType::Fp16:
            return "fp16";
        case DataType::Bf16:
            return "bf16";
        case DataType::Fp32:
            return "fp32";
        default:
            return "";
    }
}

std::string_view LibraryHelper::GetLayoutStr(LibraryHelper::LayoutType layoutType)
{
    switch (layoutType) {
        case LayoutType::RowMajor:
            return "row";
        case LayoutType::ColumnMajor:
            return "column";
        case LayoutType::nZ:
            return "nZ";
        case LayoutType::zN:
            return "zN";
        case LayoutType::zZ:
            return "zZ";
        case LayoutType::PaddingRowMajor:
            return "padding_row_major";
        case LayoutType::PaddingColumnMajor:
            return "padding_column_major";
        case LayoutType::nN:
            return "nN";
        default:
            return "";
    }
}

LibraryHelper::DataType LibraryHelper::GetDataTypeEnum(std::string_view str)
{
    static std::unordered_map<std::string_view, DataType> STR_TO_DTYPE = {
        {"u8", DataType::U8},
        {"int8", DataType::Int8},
        {"int32", DataType::Int32},
        {"fp16", DataType::Fp16},
        {"bf16", DataType::Bf16},
        {"fp32", DataType::Fp32},
    };
    auto it = STR_TO_DTYPE.find(str);
    if (it == STR_TO_DTYPE.end()) {
        return DataType::Invalid;
    }
    return it->second;
}

LibraryHelper::LayoutType LibraryHelper::GetLayoutEnum(std::string_view str)
{
    static std::unordered_map<std::string_view, LayoutType> STR_TO_LAYOUT = {
        {"row", LayoutType::RowMajor},
        {"column", LayoutType::ColumnMajor},
        {"nZ", LayoutType::nZ},
        {"zN", LayoutType::zN},
        {"zZ", LayoutType::zZ},
        {"padding_row_major", LayoutType::PaddingRowMajor},
        {"padding_column_major", LayoutType::PaddingColumnMajor},
        {"nN", LayoutType::nN},
    };
    auto it = STR_TO_LAYOUT.find(str);
    if (it == STR_TO_LAYOUT.end()) {
        return LayoutType::Invalid;
    }
    return it->second;
}

void LibraryHelper::ConstructLayout(LayoutType layoutType, DataType dataType, uint32_t a, uint32_t b, uint8_t *data)
{
    switch (layoutType) {
        case LayoutType::RowMajor:
            *reinterpret_cast<layout::RowMajor*>(data) = layout::RowMajor(a, b);
            break;
        case LayoutType::ColumnMajor:
            *reinterpret_cast<layout::ColumnMajor*>(data) = layout::ColumnMajor(a, b);
            break;
        case LayoutType::nZ:
            ConstructLayoutT<layout::nZ>(dataType, a, b, data);
            break;
        case LayoutType::zN:
            ConstructLayoutT<layout::zN>(dataType, a, b, data);
            break;
        case LayoutType::zZ:
            ConstructLayoutT<layout::zZ>(dataType, a, b, data);
            break;
        case LayoutType::PaddingRowMajor:
            *reinterpret_cast<layout::PaddingRowMajor*>(data) = layout::PaddingRowMajor(a, b);
            break;
        case LayoutType::PaddingColumnMajor:
            *reinterpret_cast<layout::PaddingColumnMajor*>(data) = layout::PaddingColumnMajor(a, b);
            break;
        case LayoutType::nN:
            ConstructLayoutT<layout::nN>(dataType, a, b, data);
            break;
        default:
            break;
    }
}

} // namespace Catlass
