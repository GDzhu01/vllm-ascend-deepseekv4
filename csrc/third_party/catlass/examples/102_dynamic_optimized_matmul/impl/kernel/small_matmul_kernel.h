/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SMALL_MATMUL_KERNEL_H
#define SMALL_MATMUL_KERNEL_H

#include "kernel_utils.h"
#include "tiling_params.h"
#include "acl/acl.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/dynamic_small_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"

template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void>
struct TileCopyDynamicOptimized : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    using CopyGmToL1A = typename Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, AType>;
    using CopyGmToL1B = typename Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, BType>;
};

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_GLOBAL __attribute__((aic)) void SmallMatmulKernel(__gm__ uint8_t *__restrict__ gmA,
    __gm__ uint8_t *__restrict__ gmB, __gm__ uint8_t *__restrict__ gmC, __gm__ uint8_t *__restrict__ tilingData)
{
    Catlass::Arch::Resource<ArchTag> resource;

    /*
    * Load tiling parameters from global memory (tilingData) to local array tilingParams
    * 
    * tilingData memory layout corresponds to tilingParams as follows:
    * --------------------------------------------------------------------------------
    * | Offset | Size | Variable         | Type      | Description                   |
    * |--------|------|------------------|-----------|-------------------------------|
    * | 0-7    | 8    | strideA          | uint64_t  | matrix A stride               |
    * | 8-15   | 8    | strideB          | uint64_t  | matrix B stride               |
    * | 16-23  | 8    | strideC          | uint64_t  | matrix C stride               |
    * | 24-27  | 4    | m                | uint32_t  | matrix M dimension            |
    * | 28-31  | 4    | n                | uint32_t  | matrix N dimension            |
    * | 32-35  | 4    | k                | uint32_t  | matrix K dimension            |
    * | 36-37  | 2    | m1               | uint16_t  | l1 mTile(16-bit to save space)|
    * | 38-39  | 2    | n1               | uint16_t  | l1 nTile(16-bit to save space)|
    * | 40-41  | 2    | k1               | uint16_t  | l1 kTile(16-bit to save space)|
    * | 42-47  | 4    | (reserved)       | -         | unused                        |
    * --------------------------------------------------------------------------------
    */

    // This kernel only needs to read TILING_PARAMS_BYTES bytes of data.
    constexpr uint32_t TILING_PARAMS_BYTES = 42;
    uint8_t tilingParams[TILING_PARAMS_BYTES];
    ReadTilingParams(tilingParams, tilingData, TILING_PARAMS_BYTES);

    // The byte size of the TilingParams structure may exceed TILING_PARAMS_BYTES. 
    // Please avoid using pointers to access data beyond TILING_PARAMS_BYTES !!!
    TilingParams* tiling = (TilingParams*)(tilingParams);

    int64_t strideA = static_cast<int64_t>(tiling->strideA);
    int64_t strideB = static_cast<int64_t>(tiling->strideB);
    int64_t strideC = static_cast<int64_t>(tiling->strideC);
    uint32_t m = tiling->m;
    uint32_t n = tiling->n;
    uint32_t k = tiling->k;

    uint32_t m1 = static_cast<uint32_t>(tiling->m1);
    uint32_t n1 = static_cast<uint32_t>(tiling->n1);
    uint32_t k1 = static_cast<uint32_t>(tiling->k1);

    Catlass::GemmCoord problemShape(m, n, k);
    Catlass::GemmCoord l1TileShape(m1, n1, k1);
    LayoutA layoutA{m, k, strideA};
    LayoutB layoutB{k, n, strideB};
    LayoutC layoutC{m, n, strideC};

    constexpr bool enableUnitFlag = false;
    static constexpr uint32_t stages = 1;
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2DynamicSmall<stages, enableUnitFlag>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using TileCopy = TileCopyDynamicOptimized<ArchTag, AType, BType, CType>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, void, void, AType, BType, CType, void, TileCopy>;
    using BlockEpilogue = void;
    using BlockScheduler = void;
    // kernel level
    using MatmulKernel = Catlass::Gemm::Kernel::DynamicSmallMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
    typename MatmulKernel::Params params{problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};
    // call a kernel
    MatmulKernel matmul;
    matmul(params, resource);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
void LaunchSmallMatmulKernel(aclrtStream &stream, uint64_t fftsAddr, uint8_t *dA, uint8_t *dB, uint8_t *dC,
    uint8_t *dTilingParams, TilingParams &tilingParams)
{
    SmallMatmulKernel<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>
        <<<tilingParams.blockDim, nullptr, stream>>>(dA, dB, dC, dTilingParams);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
size_t SmallMatmulKernelGetWorkspaceSize(TilingParams &tilingParams)
{
    return 0;
}

#endif  // SMALL_MATMUL_KERNEL_H