/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_k.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "catlass_kernel.h"
#include "common.hpp"
namespace CatlassKernel {
using namespace Catlass;
template <class LayoutA,
          class LayoutB,
          class LayoutC,
          class InDType,
          class OutDType,
          KernelInfo::GMMSplit GMM_SPLIT,
          bool K_GT_N>
void GroupedMatmulImpl(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo) {
    GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};
    uint32_t problemCount = kernelInfo.g;
    uint8_t *deviceA = kernelInfo.inputAddr.at(0);
    uint8_t *deviceB = kernelInfo.inputAddr.at(1);
    uint8_t *deviceGroupList = kernelInfo.inputAddr.at(2);
    uint8_t *deviceC = kernelInfo.outputAddr.at(0);

    constexpr bool SPLITK_OR_SPLITM_K_GT_N = GMM_SPLIT == KernelInfo::GMMSplit::SPLIT_K || !K_GT_N;

    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = SPLITK_OR_SPLITM_K_GT_N ? 4 : 2;
    constexpr uint32_t l0BStages = SPLITK_OR_SPLITM_K_GT_N ? 2 : 4;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<preloadStages, l1Stages, l0AStages, l0BStages, l0CStages,
                                                         enableUnitFlag, enableShuffleK>;
    using L1TileShape = std::conditional_t<SPLITK_OR_SPLITM_K_GT_N, GemmShape<128, 256, 256>, GemmShape<256, 128, 256>>;
    using L0TileShape = std::conditional_t<SPLITK_OR_SPLITM_K_GT_N, GemmShape<128, 256, 64>, GemmShape<256, 128, 64>>;

    using AType = Gemm::GemmType<InDType, LayoutA>;
    using BType = Gemm::GemmType<InDType, LayoutB>;
    using CType = Gemm::GemmType<OutDType, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;
    using BlockScheduler =
        std::conditional_t<SPLITK_OR_SPLITM_K_GT_N, typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>,
                           typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>>;

    // kernel level
    using MatmulKernel = std::conditional_t<
        GMM_SPLIT == KernelInfo::GMMSplit::SPLIT_K,
        typename Gemm::Kernel::GroupedMatmulSliceK<BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>,
        typename Gemm::Kernel::GroupedMatmulSliceM<BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>>;
    using MatmulAdapter = typename Gemm::Device::DeviceGemm<MatmulKernel>;
    typename MatmulKernel::Arguments arguments{problemShape, problemCount, deviceGroupList, deviceA, deviceB, deviceC};
    MatmulAdapter matmulOp;
    RunAdapter<MatmulAdapter>(matmulOp, arguments, stream, blockNum);
}

void GroupedMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo) {
    if (kernelInfo.split == KernelInfo::GMMSplit::SPLIT_K && kernelInfo.transA && !kernelInfo.transB) {
        if (kernelInfo.inputDataType == ACL_FLOAT16 && kernelInfo.outputDataType == ACL_FLOAT16) {
            GroupedMatmulImpl<layout::ColumnMajor, layout::RowMajor, layout::RowMajor, half, half,
                              KernelInfo::GMMSplit::SPLIT_K, false>(blockNum, stream, kernelInfo);
        }
    } else if (kernelInfo.split == KernelInfo::GMMSplit::SPLIT_M) {
        if (!kernelInfo.transA && !kernelInfo.transB) {
            if (kernelInfo.k > kernelInfo.n) {
                if (kernelInfo.inputDataType == ACL_FLOAT16 && kernelInfo.outputDataType == ACL_FLOAT16) {
                    GroupedMatmulImpl<layout::RowMajor, layout::RowMajor, layout::RowMajor, half, half,
                                      KernelInfo::GMMSplit::SPLIT_M, true>(blockNum, stream, kernelInfo);
                }
            } else {
                if (kernelInfo.inputDataType == ACL_FLOAT16 && kernelInfo.outputDataType == ACL_FLOAT16) {
                    GroupedMatmulImpl<layout::RowMajor, layout::RowMajor, layout::RowMajor, half, half,
                                      KernelInfo::GMMSplit::SPLIT_M, false>(blockNum, stream, kernelInfo);
                }
            }
        }
        if (!kernelInfo.transA && kernelInfo.transB) {
            if (kernelInfo.k > kernelInfo.n) {
                if (kernelInfo.inputDataType == ACL_FLOAT16 && kernelInfo.outputDataType == ACL_FLOAT16) {
                    GroupedMatmulImpl<layout::RowMajor, layout::ColumnMajor, layout::RowMajor, half, half,
                                      KernelInfo::GMMSplit::SPLIT_M, true>(blockNum, stream, kernelInfo);
                }
            } else {
                if (kernelInfo.inputDataType == ACL_FLOAT16 && kernelInfo.outputDataType == ACL_FLOAT16) {
                    GroupedMatmulImpl<layout::RowMajor, layout::ColumnMajor, layout::RowMajor, half, half,
                                      KernelInfo::GMMSplit::SPLIT_M, false>(blockNum, stream, kernelInfo);
                }
            }
        }
    }
}
} // namespace CatlassKernel