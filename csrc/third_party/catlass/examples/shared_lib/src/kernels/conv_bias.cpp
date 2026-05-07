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

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/conv/block/block_conv.hpp"
#include "catlass/conv/block/block_swizzle.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/conv/kernel/conv3d_bias.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/status.hpp"
#include "catlass/conv/device/device_conv.hpp"

#include "catlass_kernel.h"
#include "common.hpp"

namespace CatlassKernel {
using namespace Catlass;

template<class LayoutFmap, class LayoutFilter, class LayoutOut, class FmapDtype, class BiasDType, class OutDType>
void ConvBiasImpl(const uint32_t blockNum, aclrtStream stream, const ConvKernelInfo &kernelInfo)
{
    Conv3dParams problemShape = Conv3dParams::MakeConvCoord(kernelInfo.fmapRelated.data(),
                                                            kernelInfo.filterRelated.data(),
                                                            kernelInfo.padList.data(),
                                                            kernelInfo.strideList.data(),
                                                            kernelInfo.dilationList.data());
    uint8_t *deviceFmap = kernelInfo.inputAddr.at(0);
    uint8_t *deviceFilter = kernelInfo.inputAddr.at(1);
    uint8_t *deviceBias = kernelInfo.inputAddr.at(2);
    uint8_t *deviceOut = kernelInfo.outputAddr.at(0);
    
    constexpr uint32_t l1AStages = 1;
    constexpr uint32_t l1BStages = 1;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Conv::ConvAtlasA2Pingpong<
        l1AStages, l1BStages,
        l0AStages, l0BStages,
        l0CStages, enableUnitFlag
    >;
    using CoreTileShape = ConvCoreShape<2, 2, 2, 2>;
    using FmapL1TileShape = ConvFmapL1Shape<16, 1, 1>;
    using FilterL1TileShape = ConvFilterL1Shape<1, 1, 16>;
    using L0TileShape = ConvL0Shape<16, 16, 16>;

    using FmapType = Gemm::GemmType<FmapDtype, LayoutFmap>;
    using FilterType = Gemm::GemmType<FmapDtype, LayoutFilter>;
    using BiasType = Gemm::GemmType<BiasDType, layout::VectorLayout>;
    using OutType = Gemm::GemmType<OutDType, LayoutOut>;

    using BlockConv = Conv::Block::BlockConv<DispatchPolicy,
                                             CoreTileShape, FmapL1TileShape, FilterL1TileShape, L0TileShape,
                                             FmapType, FilterType, OutType, BiasType>;
    using BlockEpilogue = void;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Conv::Block::Conv3dIdentityBlockSwizzle<3, 0>;

    // kernel level
    using ConvKernel = Conv::Kernel::ConvBias<BlockConv, BlockEpilogue, BlockScheduler>;
    using ConvAdapter = Conv::Device::DeviceConv<ConvKernel>;

    typename ConvKernel::Arguments arguments{problemShape, deviceFmap, deviceFilter, deviceOut, deviceBias};
    ConvAdapter convOp;
    RunAdapter<ConvAdapter>(convOp, arguments, stream, blockNum);
}

void ConvBias(const uint32_t blockNum, aclrtStream stream, ConvKernelInfo kernelInfo)
{
    if (kernelInfo.inputDataType == ACL_FLOAT16) {
        ConvBiasImpl<layout::NDC1HWC0, layout::KDC1KHKWN1N0C0, layout::NDC1HWC0, half, half, half>(blockNum, stream, kernelInfo);
    } else if (kernelInfo.inputDataType == ACL_BF16) {
        ConvBiasImpl<layout::NDC1HWC0, layout::KDC1KHKWN1N0C0, layout::NDC1HWC0, bfloat16_t, float, bfloat16_t>(blockNum, stream, kernelInfo);
    }
}
} // namespace CatlassKernel