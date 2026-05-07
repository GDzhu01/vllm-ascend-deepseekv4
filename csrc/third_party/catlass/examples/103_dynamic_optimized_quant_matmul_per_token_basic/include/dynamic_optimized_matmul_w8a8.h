/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_DYNAMIC_OPTIMIZED_MATMUL_W8A8_H
#define CATLASS_DYNAMIC_OPTIMIZED_MATMUL_W8A8_H

#include <iostream>
#include <iomanip>

#include "do_tiling_b8.h"
#include "select_kernel_b8.h"
#include "launch_map.h"

template <class DType>
void DoW8A8Tiling(TilingParams &tilingParams, PlatformInfo &platformInfo)
{
    uint32_t layoutTagA = tilingParams.layoutTagA;
    uint32_t layoutTagB = tilingParams.layoutTagB;

    DoTilingB8[layoutTagA][layoutTagB](tilingParams, platformInfo);
}

template <class DType>
void SelectW8A8Kernel(TilingParams &tilingParams, PlatformInfo &platformInfo)
{
    SelectKernelB8(tilingParams, platformInfo);
}

template <class DType>
void DoTilingAndSelectKernel(TilingParams &tilingParams, PlatformInfo &platformInfo)
{
    DoW8A8Tiling<DType>(tilingParams, platformInfo);
    SelectW8A8Kernel<DType>(tilingParams, platformInfo);
}

size_t DynamicOptimizedMatmulGetWorkspace(TilingParams &tilingParams)
{
    return getWorkspaceFuncMap[tilingParams.tilingKey.value](tilingParams);
}

void ExecuteDynamicOptimizedMatmul(aclrtStream &stream, uint64_t fftsAddr, uint8_t *dA, uint8_t *dB, uint8_t *dC,
    uint8_t *dW, uint8_t *dScale, uint8_t *dPerTokenScale, uint8_t *dTilingParams, TilingParams &tilingParams)
{

    launchKernelFuncMap[tilingParams.tilingKey.value](stream, fftsAddr, dA, dB, dC, dW, dScale, dPerTokenScale, dTilingParams, tilingParams);
}

template <class DType>
void PrintTilingParams(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
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
              << "│ mTileInL0:   " << std::setw(30) << static_cast<uint32_t>(tilingParams.m0) << " │\n"
              << "│ nTileInL0:   " << std::setw(30) << static_cast<uint32_t>(tilingParams.n0) << " │\n"
              << "│ kTileInL0:   " << std::setw(30) << static_cast<uint32_t>(tilingParams.k0) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ swizzleOfs:  " << std::setw(30) << static_cast<uint32_t>(tilingParams.swizzleOffset) << " │\n"
              << "│ swizzleDir:  " << std::setw(30) << static_cast<uint32_t>(tilingParams.swizzleDirection) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ blockDim:    " << std::setw(30) << static_cast<uint32_t>(tilingParams.blockDim) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ TilingKey:   " << std::hex << std::setw(30) << tilingParams.tilingKey.value << " │\n"
              << "└───────────────────┴─────────────────────────┘" << std::endl;
    std::cout << std::dec << "Kernel Func Name : " << funcNameMap[tilingParams.tilingKey.value] << std::endl;
}

#endif  // CATLASS_DYNAMIC_OPTIMIZED_MATMUL_W8A8_H
