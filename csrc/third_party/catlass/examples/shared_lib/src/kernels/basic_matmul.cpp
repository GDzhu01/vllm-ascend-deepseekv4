/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "catlass/gemm/kernel/basic_matmul.hpp"

#include <acl/acl.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "catlass_kernel.h"
#include "common.hpp"

namespace CatlassKernel {
using namespace Catlass;

template <class LayoutA, class LayoutB, class LayoutC, class InDType, class OutDType>
void BasicMatmulImpl(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo) {
    GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};
    uint8_t *deviceA = kernelInfo.inputAddr.at(0);
    uint8_t *deviceB = kernelInfo.inputAddr.at(1);
    uint8_t *deviceC = kernelInfo.outputAddr.at(0);

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = Gemm::GemmType<InDType, LayoutA>;
    using BType = Gemm::GemmType<InDType, LayoutB>;
    using CType = Gemm::GemmType<OutDType, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel = typename Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
    using MatmulAdapter = typename Gemm::Device::DeviceGemm<MatmulKernel>;

    typename MatmulKernel::Arguments arguments{problemShape, deviceA, deviceB, deviceC};
    MatmulAdapter matmulOp;
    RunAdapter<MatmulAdapter>(matmulOp, arguments, stream, blockNum);
}

void BasicMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo) {
    if (kernelInfo.inputDataType == ACL_FLOAT16 && kernelInfo.outputDataType == ACL_FLOAT16 && !kernelInfo.transA
        && !kernelInfo.transB) {
        BasicMatmulImpl<layout::RowMajor, layout::RowMajor, layout::RowMajor, half, half>(blockNum, stream, kernelInfo);
    }
    // If more conditions are needed, add branches manually.
}
} // namespace CatlassKernel