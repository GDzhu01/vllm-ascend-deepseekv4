/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_DYNAMIC_OPTIMIZED_MATMUL_H
#define CATLASS_DYNAMIC_OPTIMIZED_MATMUL_H

#include <iostream>
#include <iomanip>

#include "do_tiling_b16.h"
#include "select_kernel_b16.h"
#include "launch_map.h"

template <class DType>
void DoTiling(TilingParams &tilingParams, PlatformInfo &platformInfo)
{
    uint32_t layoutTagA = tilingParams.layoutTagA;
    uint32_t layoutTagB = tilingParams.layoutTagB;

    DoTilingB16[layoutTagA][layoutTagB](tilingParams, platformInfo);
}

template <class DType>
void SelectKernel(TilingParams &tilingParams, PlatformInfo &platformInfo)
{
    SelectKernelB16(tilingParams, platformInfo);
}

template <class DType>
void DoTilingAndSelectKernel(TilingParams &tilingParams, PlatformInfo &platformInfo)
{
    DoTiling<DType>(tilingParams, platformInfo);
    SelectKernel<DType>(tilingParams, platformInfo);
}

size_t DynamicOptimizedMatmulGetWorkspace(TilingParams &tilingParams)
{
    return getWorkspaceFuncMap[tilingParams.tilingKey.value](tilingParams);
}

void ExecuteDynamicOptimizedMatmul(aclrtStream &stream, uint64_t fftsAddr, uint8_t *dA, uint8_t *dB, uint8_t *dC,
    uint8_t *dW, uint8_t *dTilingParams, TilingParams &tilingParams)
{

    launchKernelFuncMap[tilingParams.tilingKey.value](stream, fftsAddr, dA, dB, dC, dW, dTilingParams, tilingParams);
}

template <class DType>
void PrintTilingParams(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t bytePerC0 = 32;
    uint32_t c0NumPerFractal = 16;
    uint32_t elePerC0 = bytePerC0 / sizeof(DType);
    uint32_t m0 = tilingParams.m1, n0 = tilingParams.n1, k0 = 0;
    if (m0 && n0) {
        uint32_t kTileMaxA = platformInfo.l0ASize / 2 / sizeof(DType) / m0 / elePerC0 * elePerC0;
        uint32_t kTileMaxB = platformInfo.l0BSize / 2 / sizeof(DType) / n0 / elePerC0 * elePerC0;
        k0 = kTileMaxA > kTileMaxB ? kTileMaxB : kTileMaxA;
        if constexpr (std::is_same_v<DType, float>) {
            k0 = k0 / c0NumPerFractal * c0NumPerFractal;
        }
    }
    // AivMatmul
    if (tilingParams.k == 1) {
        m0 = n0 = k0 = 0;
    }
    std::cout << std::dec << "┌─────────────────────────────────────────────┐\n"
              << "│            Tiling Parameters                │\n"
              << "├───────────────────┬─────────────────────────┤\n"
              << "│ m:           " << std::setw(30) << tilingParams.m << " │\n"
              << "│ n:           " << std::setw(30) << tilingParams.n << " │\n"
              << "│ k:           " << std::setw(30) << tilingParams.k << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ layoutTagA:  " << std::setw(30) << static_cast<uint32_t>(tilingParams.layoutTagA) << " │\n"
              << "│ layoutTagB:  " << std::setw(30) << static_cast<uint32_t>(tilingParams.layoutTagB) << " │\n"
              << "│ layoutTagC:  " << std::setw(30) << static_cast<uint32_t>(tilingParams.layoutTagC) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ mTileInL1:   " << std::setw(30) << static_cast<uint32_t>(tilingParams.m1) << " │\n"
              << "│ nTileInL1:   " << std::setw(30) << static_cast<uint32_t>(tilingParams.n1) << " │\n"
              << "│ kTileInL1:   " << std::setw(30) << static_cast<uint32_t>(tilingParams.k1) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ mTileInL0:   " << std::setw(30) << static_cast<uint32_t>(m0) << " │\n"
              << "│ nTileInL0:   " << std::setw(30) << static_cast<uint32_t>(n0) << " │\n"
              << "│ kTileInL0:   " << std::setw(30) << static_cast<uint32_t>(k0) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ paddingTagA: " << std::setw(30) << static_cast<uint32_t>(tilingParams.paddingTagA) << " │\n"
              << "│ paddingTagB: " << std::setw(30) << static_cast<uint32_t>(tilingParams.paddingTagB) << " │\n"
              << "│ paddingTagC: " << std::setw(30) << static_cast<uint32_t>(tilingParams.paddingTagC) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ swizzleOfs:  " << std::setw(30) << static_cast<uint32_t>(tilingParams.swizzleOffset) << " │\n"
              << "│ swizzleDir:  " << std::setw(30) << static_cast<uint32_t>(tilingParams.swizzleDirection) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ blockDim:    " << std::setw(30) << static_cast<uint32_t>(tilingParams.blockDim) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ TilingKey:   " << std::hex << std::setw(30) << tilingParams.tilingKey.value << " │\n"
              << "└───────────────────┴─────────────────────────┘" << std::endl;
    std::cout << "Kernel Func Name : " << funcNameMap[tilingParams.tilingKey.value] << std::endl;
}

#endif  // CATLASS_DYNAMIC_OPTIMIZED_MATMUL_H