/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EXAMPLES_ADVANCED_ACLNN_TILING_H
#define CATLASS_EXAMPLES_ADVANCED_ACLNN_TILING_H
#include <register/tilingdata_base.h>

namespace optiling {
BEGIN_TILING_DATA_DEF(CatlassBasicMatmulTilingData)
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, k);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CatlassBasicMatmul, CatlassBasicMatmulTilingData)
} // namespace optiling
#endif