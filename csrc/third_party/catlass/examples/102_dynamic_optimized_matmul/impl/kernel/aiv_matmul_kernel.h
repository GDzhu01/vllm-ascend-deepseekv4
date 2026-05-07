/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_MATMUL_KERNEL_H
#define AIV_MATMUL_KERNEL_H

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/dynamic_aiv_matmul.hpp"
#include "catlass/layout/layout.hpp"

#include "acl/acl.h"
#include "kernel_utils.h"
#include "tiling_params.h"


enum class DispatchPolicyTag {
    DEFAULT = 0,
    MATMUL_AIV_SIMPLE = 1,
    MATMUL_AIV_TRANS = 2
};

template <class ArchTag, class AType, class BType, class CType>
struct TileCopyAiv {
    using CopyGmToUbA = Catlass::Gemm::Tile::CopyGm2Ub<ArchTag, AType>;
    using CopyGmToUbB = Catlass::Gemm::Tile::CopyGm2Ub<ArchTag, BType>;
    using CopyUbToGmC = Catlass::Gemm::Tile::CopyUb2Gm<ArchTag, CType>;
};

template <
    class ArchTag,
    class ElementA,
    class LayoutA,
    class ElementB,
    class LayoutB,
    class ElementC,
    class LayoutC,
    class DispatchPolicy>
CATLASS_DEVICE void AivMatmul(
    Catlass::GemmCoord &problemShape,
    Catlass::MatrixCoord &taskTileShape,
    GM_ADDR gmA,
    LayoutA &layoutA,
    GM_ADDR gmB,
    LayoutB &layoutB,
    GM_ADDR gmC,
    LayoutC &layoutC,
    Catlass::Arch::Resource<ArchTag> &resource
)
{
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using BiasType = void;
    using TileCopy = TileCopyAiv<ArchTag, AType, BType, CType>;
    static constexpr uint32_t COMPUTE_LENGTH = 16 * 1024;
    using TileVmuls = Catlass::Gemm::Tile::TileMuls<ArchTag, AType, COMPUTE_LENGTH>;
    using BlockMmad =
        Catlass::Gemm::Block::BlockMmadAiv<DispatchPolicy, AType, BType, CType, BiasType, TileCopy, TileVmuls>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<1, 0>;
    // kernel level
    using MatmulKernel = Catlass::Gemm::Kernel::AivMatmul<void, void, BlockMmad, BlockEpilogue, BlockScheduler>;
    typename MatmulKernel::Params params{problemShape, taskTileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};
    // call a kernel
    MatmulKernel matmul;
    matmul(params, resource);
}

template <
    class ArchTag,
    class ElementA,
    class LayoutA,
    class ElementB,
    class LayoutB,
    class ElementC,
    class LayoutC,
    DispatchPolicyTag dispatchPolicyTag>
CATLASS_GLOBAL __attribute__((aiv)) void AivMatmulKernel(
    __gm__ uint8_t *__restrict__ gmA,
    __gm__ uint8_t *__restrict__ gmB,
    __gm__ uint8_t *__restrict__ gmC,
    __gm__ uint8_t *__restrict__ tilingData
)
{
    Catlass::Arch::Resource<ArchTag> resource;

    /*
     * Load tiling parameters from global memory (tilingData) to local array tilingParams
     *
     * tilingData memory layout corresponds to tilingParams as follows:
     * --------------------------------------------------------------------------------
     * | Offset | Size | Variable         | Type      | Description                   |
     * |--------|------|------------------|-----------|-------------------------------|
     * | 0-7    | 8    | strideA          | uint64_t  | unused                        |
     * | 8-15   | 8    | strideB          | uint64_t  | unused                        |
     * | 16-23  | 8    | strideC          | uint64_t  | matrix C stride               |
     * | 24-27  | 4    | m                | uint32_t  | matrix M dimension            |
     * | 28-31  | 4    | n                | uint32_t  | matrix N dimension            |
     * | 32-35  | 4    | k                | uint32_t  | matrix K dimension            |
     * | 36-37  | 2    | m1               | uint16_t  | l1 mTile(16-bit to save space)|
     * | 38-39  | 2    | n1               | uint16_t  | l1 nTile(16-bit to save space)|
     * | 40-41  | 2    | k1               | uint16_t  | unused                        |
     * | 42-47  | 4    | (reserved)       | -         | unused                        |
     * --------------------------------------------------------------------------------
     */

    // This kernel only needs to read TILING_PARAMS_BYTES bytes of data.
    constexpr uint32_t TILING_PARAMS_BYTES = 42;
    uint8_t tilingParams[TILING_PARAMS_BYTES];
    ReadTilingParams(tilingParams, tilingData, TILING_PARAMS_BYTES);

    // The byte size of the TilingParams structure may exceed TILING_PARAMS_BYTES.
    // Please avoid using pointers to access data beyond TILING_PARAMS_BYTES !!!
    TilingParams *tiling = (TilingParams *)(tilingParams);

    int64_t strideC = static_cast<int64_t>(tiling->strideC);
    uint32_t m = tiling->m;
    uint32_t n = tiling->n;
    uint32_t k = tiling->k;

    uint32_t m1 = static_cast<uint32_t>(tiling->m1);
    uint32_t n1 = static_cast<uint32_t>(tiling->n1);
    uint32_t k1 = static_cast<uint32_t>(tiling->k1);

    Catlass::GemmCoord problemShape(m, n, k);
    Catlass::MatrixCoord taskTileShape(m1, n1);
    LayoutA layoutA{m};
    LayoutB layoutB{n};
    LayoutC layoutC{m, n, strideC};

    // default impl: m axis as scalar axis
    constexpr uint32_t SCALAR_BUFFER_ELE_NUM = 256;
    if constexpr (dispatchPolicyTag == DispatchPolicyTag::DEFAULT) {
        constexpr uint32_t STAGES = 2;
        using DispatchPolicy = Catlass::Gemm::MmadAtlasA2DynamicAiv<SCALAR_BUFFER_ELE_NUM, STAGES>;
        AivMatmul<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, DispatchPolicy>(
            problemShape, taskTileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, resource
        );
    } else if constexpr (dispatchPolicyTag == DispatchPolicyTag::MATMUL_AIV_SIMPLE) {
        constexpr bool IS_TILE_M = true;
        using DispatchPolicy = Catlass::Gemm::MmadAtlasA2DynamicAivSimple<SCALAR_BUFFER_ELE_NUM, IS_TILE_M>;
        AivMatmul<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, DispatchPolicy>(
            problemShape, taskTileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, resource
        );
    } else if constexpr (dispatchPolicyTag == DispatchPolicyTag::MATMUL_AIV_TRANS) {
        using DispatchPolicy = Catlass::Gemm::MmadAtlasA2DynamicAivTrans<SCALAR_BUFFER_ELE_NUM>;
        AivMatmul<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, DispatchPolicy>(
            problemShape, taskTileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, resource
        );
    }
}

template <
    class ArchTag,
    class ElementA,
    class LayoutA,
    class ElementB,
    class LayoutB,
    class ElementC,
    class LayoutC,
    DispatchPolicyTag dispatchPolicyTag>
void LaunchAivMatmulKernel(
    aclrtStream &stream,
    uint64_t fftsAddr,
    uint8_t *dA,
    uint8_t *dB,
    uint8_t *dC,
    uint8_t *dTilingParams,
    TilingParams &tilingParams
)
{
    AivMatmulKernel<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, dispatchPolicyTag>
        <<<tilingParams.blockDim, nullptr, stream>>>(dA, dB, dC, dTilingParams);
}

template <
    class ArchTag,
    class ElementA,
    class LayoutA,
    class ElementB,
    class LayoutB,
    class ElementC,
    class LayoutC,
    DispatchPolicyTag dispatchPolicyTag>
size_t AivMatmulKernelGetWorkspaceSize(TilingParams &tilingParams)
{
    return 0;
}

#endif // AIV_MATMUL_KERNEL_H