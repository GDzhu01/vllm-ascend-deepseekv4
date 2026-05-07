/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_LIBRARY_HELPER_H
#define CATLASS_TUNER_LIBRARY_HELPER_H

#include <unordered_map>
#include "catlass/library/operation.h"
#include "catlass/layout/matrix.hpp"
#include "catlass/layout/vector.hpp"

namespace Catlass {

class LibraryHelper {
public:
    using DataType = Library::DataType;
    using LayoutType = Library::LayoutType;

    static size_t GetDataTypeSize(DataType dataType);
    static size_t GetLayoutSize(LayoutType layoutType);
    static std::string_view GetDataTypeStr(DataType dataType);
    static std::string_view GetLayoutStr(LayoutType layoutType);
    static DataType GetDataTypeEnum(std::string_view str);
    static LayoutType GetLayoutEnum(std::string_view str);
    static void ConstructLayout(LayoutType layoutType, DataType dataType, uint32_t a, uint32_t b, uint8_t *data);
};

}
#endif // CATLASS_TUNER_LIBRARY_HELPER_H
