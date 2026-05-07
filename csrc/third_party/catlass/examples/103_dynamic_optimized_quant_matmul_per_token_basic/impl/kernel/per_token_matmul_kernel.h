/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PER_TOKEN_MATMUL_KERNEL_H
#define PER_TOKEN_MATMUL_KERNEL_H

#include "kernel_utils.h"
#include "tiling_params.h"
#include "acl/acl.h"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/dynamic_w8a8_matmul_per_token_per_channel_dequant.hpp"
#include "catlass/gemm/gemm_type.hpp"

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
[[bisheng::core_ratio(1, 2)]] CATLASS_GLOBAL void PerTokenMatmulKernel(uint64_t fftsAddr, __gm__ uint8_t *__restrict__ gmA,
    __gm__ uint8_t *__restrict__ gmB, __gm__ uint8_t *__restrict__ gmC, __gm__ uint8_t *__restrict__ gmWQuant, __gm__ uint8_t *__restrict__ gmScale, __gm__ uint8_t *__restrict__ gmPerTokenScale, __gm__ uint8_t *__restrict__ tilingData)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
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
    * | 42-42  | 1    | swizzleOffset    | uint8_t   | swizzle offset                |
    * | 43-43  | 1    | swizzleDirection | uint8_t   | swizzle direction             |
    * | 44-45  | 2    | m0               | uint16_t  | l0 mTile(16-bit to save space)|
    * | 46-47  | 2    | n0               | uint16_t  | l0 nTile(16-bit to save space)|
    * | 48-49  | 2    | k0               | uint16_t  | l0 kTile(16-bit to save space)|
    * | 50-55  | 6    | (reserved)       | -         | unused                        |
    * --------------------------------------------------------------------------------
    */

    // This kernel only needs to read TILING_PARAMS_BYTES bytes of data.
    constexpr uint32_t TILING_PARAMS_BYTES = 56;
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

    uint32_t swizzleOffset = static_cast<uint32_t>(tiling->swizzleOffset);
    uint32_t swizzleDirection = static_cast<uint32_t>(tiling->swizzleDirection);

    uint32_t m0 = static_cast<uint32_t>(tiling->m0);
    uint32_t n0 = static_cast<uint32_t>(tiling->n0);
    uint32_t k0 = static_cast<uint32_t>(tiling->k0);

    Catlass::GemmCoord problemShape(m, n, k);
    Catlass::GemmCoord l1TileShape(m1, n1, k1);
    Catlass::GemmCoord l0TileShape(m0, n0, k0);

    LayoutA layoutA{m, k, strideA};
    LayoutB layoutB{k, n, strideB};
    LayoutC layoutC{m, n, strideC};

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using MatMulType = Catlass::Gemm::GemmType<int32_t, Catlass::layout::RowMajor>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using LayoutScale = Catlass::layout::VectorLayout;
    using LayoutPerTokenScale = Catlass::layout::VectorLayout;

    // --------- Epilogue
    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Catlass::Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleType = Catlass::Gemm::GemmType<float, LayoutScale>;
    using PerTokenScaleType = Catlass::Gemm::GemmType<float, LayoutPerTokenScale>;
    using RowBroadcastMulType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using BroadcastOneBlkType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using EpilogueTileShape = Catlass::MatrixShape<32, 256>;
    using TileRowBroadcastMul = Catlass::Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Catlass::Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Catlass::Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, MatMulType, ScaleType, PerTokenScaleType, CType>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, MatMulType, ScaleType, PerTokenScaleType, CType, TileRowBroadcastMul, TileBroadcastOneBlk,
        TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;
    
    // --------- Epilogue
    
    // --------- BlockMmad
    using BlockScheduler = typename Catlass::Gemm::Block::DynamicGemmIdentityBlockSwizzle;
    
    constexpr uint32_t workspaceStages = 2;
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2DynamicPreloadAsyncWithCallback<
        preloadStages, l1Stages, l0AStages, l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;
    
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, void, void, AType, BType, MatMulType>;
    // --------- BlockMmad

    // --------- Kernel
    using ElementScale = float;
    using ElementPerTokenScale = float;

    LayoutScale layoutScale{n};
    LayoutPerTokenScale layoutPerTokenScale{m};
    
    using MatmulKernel = Catlass::Gemm::Kernel::DynamicW8A8MatmulPerTokenPerChannelDequant<
        BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages, ElementScale, LayoutScale>;
    typename MatmulKernel::Params params{
        problemShape, 
        l1TileShape,
        l0TileShape,
        gmA, layoutA, 
        gmB, layoutB,  
        gmScale, layoutScale, 
        gmPerTokenScale, layoutPerTokenScale, 
        gmC, layoutC, 
        gmWQuant};
    // call a kernel
    MatmulKernel matmul;
    // --------- Kernel
    matmul(params, resource);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
void LaunchPerTokenMatmulKernel(aclrtStream &stream, uint64_t fftsAddr, uint8_t *dA, uint8_t *dB, uint8_t *dC,
    uint8_t *dW, uint8_t *dScale, uint8_t *dPerTokenScale, uint8_t *dTilingParams, TilingParams &tilingParams)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = static_cast<uint32_t>(tilingParams.m1);
    uint32_t n1 = static_cast<uint32_t>(tilingParams.n1);
    uint32_t k1 = static_cast<uint32_t>(tilingParams.k1);
    uint32_t m0 = static_cast<uint32_t>(tilingParams.m0);
    uint32_t n0 = static_cast<uint32_t>(tilingParams.n0);
    uint32_t k0 = static_cast<uint32_t>(tilingParams.k0);
    uint8_t *dwQuant = nullptr;
    
    size_t sizeWQuant = 0;

    constexpr uint32_t workspaceStages = 2;
    constexpr uint32_t CORENUM = 24;

    dwQuant = dW;
    // LiTileShape::M * LiTileShape::N * coreNum * workspaceStages * sizeof(int32_t)
    sizeWQuant = m1 * n1 * CORENUM * workspaceStages * sizeof(int32_t);
    
    PerTokenMatmulKernel<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>
        <<<tilingParams.blockDim, nullptr, stream>>>(fftsAddr, dA, dB, dC, dwQuant, dScale, dPerTokenScale, dTilingParams);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
size_t PerTokenMatmulKernelGetWorkspaceSize(TilingParams &tilingParams)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = static_cast<uint32_t>(tilingParams.m1);
    uint32_t n1 = static_cast<uint32_t>(tilingParams.n1);
    uint32_t k1 = static_cast<uint32_t>(tilingParams.k1);
    uint32_t m0 = static_cast<uint32_t>(tilingParams.m0);
    uint32_t n0 = static_cast<uint32_t>(tilingParams.n0);
    uint32_t k0 = static_cast<uint32_t>(tilingParams.k0);
    size_t sizeWQuant = 0;

    constexpr uint32_t workspaceStages = 2;
    constexpr uint32_t CORENUM = 24;

    // LiTileShape::M * LiTileShape::N * coreNum * workspaceStages * sizeof(int32_t)
    sizeWQuant = m1 * n1 * CORENUM * workspaceStages * sizeof(int32_t);

    return sizeWQuant;
}

#endif  // PER_TOKEN_MATMUL_KERNEL_H