/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOCAL_PADDING_C_PADDING_COMMON_MATMUL_KERNEL_H
#define LOCAL_PADDING_C_PADDING_COMMON_MATMUL_KERNEL_H

#include "kernel_utils.h"
#include "tiling_params.h"
#include "acl/acl.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/dynamic_local_padding_c_padding_common_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"

using PaddingTag = Catlass::Gemm::Kernel::PaddingTag;

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

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    PaddingTag paddingTagA, PaddingTag paddingTagB,  PaddingTag paddingTagC>
[[bisheng::core_ratio(1, 2)]] CATLASS_GLOBAL void LocalPaddingCPaddingCommonMatmulKernel(uint64_t fftsAddr, __gm__ uint8_t *__restrict__ gmA,
    __gm__ uint8_t *__restrict__ gmB, __gm__ uint8_t *__restrict__ gmC, __gm__ uint8_t *__restrict__ gmWA,
    __gm__ uint8_t *__restrict__ gmWB, __gm__ uint8_t *__restrict__ gmWC, __gm__ uint8_t *__restrict__ tilingData)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    Catlass::Arch::Resource<ArchTag> resource;

     /*
    * Load tiling parameters from global memory (tilingData) to local array tilingParams
    * 
    * tilingData memory layout corresponds to tilingParams as follows:
    * ----------------------------------------------------------------------------------------------------
    * | Offset | Size | Variable         | Type      | Description                                       |
    * |--------|------|------------------|-----------|---------------------------------------------------|
    * | 0-7    | 8    | strideA          | uint64_t  | matrix A stride                                   |
    * | 8-15   | 8    | strideB          | uint64_t  | matrix B stride                                   |
    * | 16-23  | 8    | strideC          | uint64_t  | matrix C stride                                   |
    * | 24-27  | 4    | m                | uint32_t  | matrix M dimension                                |
    * | 28-31  | 4    | n                | uint32_t  | matrix N dimension                                |
    * | 32-35  | 4    | k                | uint32_t  | matrix K dimension                                |
    * | 36-37  | 2    | m1               | uint16_t  | l1 mTile(16-bit to save space)                    |
    * | 38-39  | 2    | n1               | uint16_t  | l1 nTile(16-bit to save space)                    |
    * | 40-41  | 2    | k1               | uint16_t  | l1 kTile(16-bit to save space)                    |
    * | 42-42  | 1    | swizzleOffset    | uint8_t   | swizzle offset                                    |
    * | 43-43  | 1    | swizzleDirection | uint8_t   | swizzle direction                                 |
    * | 44-45  | 2    | splitkFactor     | uint16_t  | splitk factor                                     |
    * | 46-47  | 2    | m0               | uint16_t  | l0 mTile(16-bit to save space)                    |
    * | 48-49  | 2    | n0               | uint16_t  | l0 nTile(16-bit to save space)                    |
    * | 50-51  | 2    | k0               | uint16_t  | l0 kTile(16-bit to save space)                    |
    * | 52-53  | 2    | m1Factor         | uint16_t  | (m1*m1Factor, n1*n1Factor) is for C block swizzle |
    * | 54-55  | 2    | n1Factor         | uint16_t  | (m1*m1Factor, n1*n1Factor) is for C block swizzle |
    * ----------------------------------------------------------------------------------------------------
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
    uint32_t m1Factor = static_cast<uint32_t>(tiling->m1Factor);
    uint32_t n1Factor = static_cast<uint32_t>(tiling->n1Factor);

    uint32_t swizzleOffset = static_cast<uint32_t>(tiling->swizzleOffset);
    uint32_t swizzleDirection = static_cast<uint32_t>(tiling->swizzleDirection);

    Catlass::GemmCoord problemShape(m, n, k);
    Catlass::GemmCoord l1TileShape(m1, n1, k1);
    LayoutA layoutA{m, k, strideA};
    LayoutB layoutB{k, n, strideB};
    LayoutC layoutC{m, n, strideC};

    using PaddingBuilderA = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagA, ArchTag, ElementA, LayoutA>;
    using PaddingBuilderB = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagB, ArchTag, ElementB, LayoutB>;
    using RemovePaddingBuilderC = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagC, ArchTag, ElementC, LayoutC>;
    using PaddingA = typename PaddingBuilderA::Padding;
    using PaddingB = typename PaddingBuilderB::Padding;
    using RemovePaddingC = typename RemovePaddingBuilderC::Padding;

    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2DynamicCommon<enableShuffleK, enableShuffleK>;

    using AType = Catlass::Gemm::GemmType<ElementA, typename PaddingBuilderA::LayoutAfterPadding>;
    using BType = Catlass::Gemm::GemmType<ElementB, typename PaddingBuilderB::LayoutAfterPadding>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using TileCopy = TileCopyDynamicOptimized<ArchTag, AType, BType, CType>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, void, void, AType, BType, CType, void, TileCopy>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Catlass::Gemm::Block::DynamicGemmIdentityBlockSwizzle;
    // kernel level
    using MatmulKernel = Catlass::Gemm::Kernel::DynamicLocalPaddingCPaddingCommonMatmul<
        PaddingA, PaddingB, BlockMmad, BlockEpilogue, BlockScheduler, RemovePaddingC>;
    typename MatmulKernel::Params params{problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, 
        gmWA, gmWB, gmWC, swizzleOffset, swizzleDirection, m1Factor, n1Factor};
    // call a kernel
    MatmulKernel matmul;
    matmul(params, resource);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    PaddingTag paddingTagA, PaddingTag paddingTagB,  PaddingTag paddingTagC>
void LaunchLocalPaddingCPaddingCommonMatmulKernel(aclrtStream &stream, uint64_t fftsAddr, uint8_t *dA, uint8_t *dB, uint8_t *dC,
    uint8_t *dW, uint8_t *dTilingParams, TilingParams &tilingParams)
{
    using PaddingBuilderA = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagA, ArchTag, ElementA, LayoutA>;
    using PaddingBuilderB = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagB, ArchTag, ElementB, LayoutB>;
    using RemovePaddingC = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagC, ArchTag, ElementC, LayoutC>;
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = static_cast<uint32_t>(tilingParams.m1);
    uint32_t n1 = static_cast<uint32_t>(tilingParams.n1);
    uint32_t k1 = static_cast<uint32_t>(tilingParams.k1);
    uint8_t *dWA = nullptr;
    uint8_t *dWB = nullptr;
    uint8_t *dWC = nullptr;
    size_t sizeWA = 0, sizeWB = 0;

    dWA = dW;
    if constexpr (paddingTagA == PaddingTag::PADDING_BLOCK_ND) {
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k, m1, k1);
    } else if constexpr (paddingTagA == PaddingTag::PADDING_ND) {
        // Optimal bandwidth for 512 Byte aligned reads
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k, 512 / sizeof(ElementA));
    } else if constexpr (paddingTagA == PaddingTag::PADDING_NZ) {
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k);
    }

    dWB = dW + sizeWA;
    if constexpr (paddingTagB == PaddingTag::PADDING_BLOCK_ND) {
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n, k1, n1);
    } else if constexpr (paddingTagB == PaddingTag::PADDING_ND) {
        // Optimal bandwidth for 512 Byte aligned reads
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n, 512 / sizeof(ElementB));
    } else if constexpr (paddingTagB == PaddingTag::PADDING_NZ) {
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n);
    }

    if constexpr (paddingTagC == PaddingTag::PADDING_ND) {
        dWC = dW + sizeWA + sizeWB;
    }

    LocalPaddingCPaddingCommonMatmulKernel<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, paddingTagA, paddingTagB, paddingTagC>
        <<<tilingParams.blockDim, nullptr, stream>>>(fftsAddr, dA, dB, dC, dWA, dWB, dWC, dTilingParams);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    PaddingTag paddingTagA, PaddingTag paddingTagB,  PaddingTag paddingTagC>
size_t LocalPaddingCPaddingCommonMatmulKernelGetWorkspaceSize(TilingParams &tilingParams)
{
    using PaddingBuilderA = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagA, ArchTag, ElementA, LayoutA>;
    using PaddingBuilderB = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagB, ArchTag, ElementB, LayoutB>;
    using RemovePaddingC = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagC, ArchTag, ElementC, LayoutC>;
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = static_cast<uint32_t>(tilingParams.m1);
    uint32_t n1 = static_cast<uint32_t>(tilingParams.n1);
    uint32_t k1 = static_cast<uint32_t>(tilingParams.k1);
    uint32_t m1Factor = static_cast<uint32_t>(tilingParams.m1Factor);
    uint32_t n1Factor = static_cast<uint32_t>(tilingParams.n1Factor);
    size_t sizeWA = 0, sizeWB = 0, sizeWC = 0;
    if constexpr (paddingTagA == PaddingTag::PADDING_BLOCK_ND) {
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k, m1, k1);
    } else if constexpr (paddingTagA == PaddingTag::PADDING_ND) {
        // Optimal bandwidth for 512 Byte aligned reads
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k, 512 / sizeof(ElementA));
    } else if constexpr (paddingTagA == PaddingTag::PADDING_NZ) {
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k);
    }

    if constexpr (paddingTagB == PaddingTag::PADDING_BLOCK_ND) {
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n, k1, n1);
    } else if constexpr (paddingTagB == PaddingTag::PADDING_ND) {
        // Optimal bandwidth for 512 Byte aligned reads
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n, 512 / sizeof(ElementB));
    } else if constexpr (paddingTagB == PaddingTag::PADDING_NZ) {
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n);
    }

    if constexpr (paddingTagC == PaddingTag::PADDING_ND) {
        size_t minWorkspaceSize = 2 * 1024 * 1024; // 2M
        // 4 buffers per core
        sizeWC = static_cast<size_t>(m1) * n1 * m1Factor * n1Factor * tilingParams.blockDim * sizeof(ElementC) * 4;
        sizeWC = (sizeWC < minWorkspaceSize) ? minWorkspaceSize : sizeWC;
    }
    return sizeWA + sizeWB + sizeWC;
}

#endif  // LOCAL_PADDING_C_PADDING_COMMON_MATMUL_KERNEL_H