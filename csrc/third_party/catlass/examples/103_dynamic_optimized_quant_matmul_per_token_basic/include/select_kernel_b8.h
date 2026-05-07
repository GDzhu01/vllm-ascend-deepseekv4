/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_KERNEL_B8_H
#define SELECT_KERNEL_B8_H

#include <limits>
#include "platform_info.h"

bool PerTokenMatmulB8Handler(TilingParams &params, PlatformInfo& platformInfo)
{
    uint8_t kernelSerial = 11;
    uint32_t taskBlocks = CeilDiv(params.m, params.m1) * CeilDiv(params.n, params.n1);
    params.blockDim = taskBlocks > platformInfo.coreNum ? platformInfo.coreNum : taskBlocks;
    
    // dtype int8_t -> 2
    params.tilingKey.SetTilingKey(kernelSerial, 
        params.layoutTagA, params.layoutTagB, 0, 2); 
    return true;
}

void SetSwizzleParams(TilingParams &tilingParams)
{
    if (tilingParams.m > tilingParams.n) {
        tilingParams.swizzleOffset = 3;
        tilingParams.swizzleDirection = 0;
    } else {
        tilingParams.swizzleOffset = 3;
        tilingParams.swizzleDirection = 1;
    }
}

void SetL0Tile(TilingParams &tilingParams, PlatformInfo &platformInfo)
{
    uint32_t bytePerC0 = 32;
    uint32_t c0NumPerFractal = 16;
    uint32_t elePerC0 = bytePerC0;
    uint32_t m0 = tilingParams.m1, n0 = tilingParams.n1, k0 = 0;
    uint32_t kTileMaxA = platformInfo.l0ASize / 2 / m0 / elePerC0
                        * elePerC0;
    uint32_t kTileMaxB = platformInfo.l0BSize / 2 / n0 / elePerC0
                        * elePerC0;
    k0 = kTileMaxA > kTileMaxB ? kTileMaxB : kTileMaxA;

    k0 = k0 / c0NumPerFractal * c0NumPerFractal;
    tilingParams.m0 = m0;
    tilingParams.n0 = n0;
    tilingParams.k0 = k0;
}

void SelectKernelB8(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    // Temporarily store the original layoutTagA and layoutTagB
    uint8_t layoutTagATmp = tilingParams.layoutTagA;
    uint8_t layoutTagBTmp = tilingParams.layoutTagB;
    // When m=1 or n=1, the row-major and column-major matrix layouts are identical, the matrix can be stored
    // in either format. In such cases, the layout with higher memory transfer bandwidth should be selected.
    if (static_cast<LayoutTag>(tilingParams.layoutTagA) == LayoutTag::TagColumnMajor &&
        tilingParams.m == 1 && tilingParams.strideA == 1) {
        tilingParams.layoutTagA = static_cast<uint8_t>(LayoutTag::TagRowMajor);
    }
    if (static_cast<LayoutTag>(tilingParams.layoutTagB) == LayoutTag::TagRowMajor &&
        tilingParams.n == 1 && tilingParams.strideB == 1) {
        tilingParams.layoutTagB = static_cast<uint8_t>(LayoutTag::TagColumnMajor);
    }

    using HandlerPtr = bool (*)(TilingParams& tilingParams, PlatformInfo& platformInfo);
    HandlerPtr handlers[] = {
        PerTokenMatmulB8Handler
    };

    for (auto handler : handlers) {
        if (handler(tilingParams, platformInfo)) {
            break;
        }
    }

    // Restore to the original layout
    tilingParams.layoutTagA = layoutTagATmp;
    tilingParams.layoutTagB = layoutTagBTmp;

    SetL0Tile(tilingParams, platformInfo);
    SetSwizzleParams(tilingParams);
}

#endif  // SELECT_KERNEL_B8_H
