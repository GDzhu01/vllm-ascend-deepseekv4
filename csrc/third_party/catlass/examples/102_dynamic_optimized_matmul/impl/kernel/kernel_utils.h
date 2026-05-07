/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_UTILS_H
#define KERNEL_UTILS_H

#include "acl/acl.h"
#include "catlass/catlass.hpp"

CATLASS_DEVICE
void ReadTilingParams(uint8_t* dst, __gm__ uint8_t * src, uint32_t bytesNum)
{
    // 64-bit combined read
    uint32_t offset = 0;
    uint32_t b64s = bytesNum / 8;
    for (uint32_t i = 0; i < b64s; ++i) {
        *(uint64_t *)(dst + offset + i * 8) = *(reinterpret_cast<__gm__ uint64_t *>(src + offset + i * 8));
    }
    offset += b64s * 8;
    uint32_t b32s = (bytesNum - b64s * 8) / 4;
    // 32-bit combined read
    for (uint32_t i = 0; i < b32s; ++i) {
        *(uint32_t *)(dst + offset + i * 4) = *(reinterpret_cast<__gm__ uint32_t *>(src + offset + i * 4));
    }
    offset += b32s * 4;
    uint32_t b16s = (bytesNum - b64s * 8 - b32s * 4) / 2;
    // 16-bit combined read
    for (uint32_t i = 0; i < b16s; ++i) {
        *(uint16_t *)(dst + offset + i * 2) = *(reinterpret_cast<__gm__ uint16_t *>(src + offset + i * 2));
    }
    offset += b16s * 2;
    uint32_t b8s = (bytesNum - b64s * 8 - b32s * 4 - b16s * 2);
    for (uint32_t i = 0; i < b8s; ++i) {
        *(uint8_t *)(dst + offset + i) = *(reinterpret_cast<__gm__ uint8_t *>(src + offset + i));
    }
}

# endif // KERNEL_UTILS_H