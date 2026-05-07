/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/gemm/kernel/single_core_slicek_matmul.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;

using Options = GemmOptions;
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

static void Run(const Options &options) {
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    using ElementA = half;
    using ElementB = half;
    using ElementC = half;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(m, k);
    LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(k, n);
    LayoutC layoutC = LayoutC::template MakeLayout<ElementC>(m, n);

    using ArchTag = Arch::AtlasA2;   

    // Padding objects
    using ElementAccumulator = typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    
    constexpr PaddingTag paddingTagA = (std::is_same_v<LayoutA, layout::zN> || std::is_same_v<LayoutA, layout::nZ>)
                                           ? PaddingTag::NO_PADDING
                                           : PaddingTag::PADDING_NZ;
    constexpr PaddingTag paddingTagB = (std::is_same_v<LayoutB, layout::zN> || std::is_same_v<LayoutB, layout::nZ>)
                                           ? PaddingTag::NO_PADDING
                                           : PaddingTag::PADDING_NZ;
    constexpr PaddingTag paddingTagC = PaddingTag::NO_PADDING;
    using PaddingBuilderA = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagA, ArchTag, ElementA, LayoutA>;
    using PaddingBuilderB = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagB, ArchTag, ElementB, LayoutB>;
    using PaddingA = typename PaddingBuilderA::Padding;
    using PaddingB = typename PaddingBuilderB::Padding;
    using RemovePaddingNDAndCast_ = Catlass::Gemm::Kernel::RemovePaddingNDAndCast<paddingTagC, ArchTag, ElementAccumulator, ElementC, LayoutC>;
    using RemovePaddingNDAndCastC = std::conditional_t<paddingTagC == PaddingTag::PADDING_ND || !std::is_same_v<ElementAccumulator, ElementC>,
        RemovePaddingNDAndCast_, void>;
    
    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

    // Malloc memory on device
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    // AscendC::SetSyncBaseAddr(fftsAddr);
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    using AType = Catlass::Gemm::GemmType<ElementA, typename PaddingBuilderA::LayoutAfterPadding>;
    using BType = Catlass::Gemm::GemmType<ElementB, typename PaddingBuilderB::LayoutAfterPadding>;
    using CType = Catlass::Gemm::GemmType<ElementAccumulator, LayoutC>;

    using TileCopy = TileCopyDynamicOptimized<ArchTag, AType, BType, CType>;
    using BlockEpilogue = void;

    constexpr bool enableUnitFlag = false;
    constexpr uint32_t l0CStages = 2;
    if ( m > n) {
        constexpr uint32_t l1AStages = 2;
        constexpr uint32_t l1BStages = 1;
        using L1TileShape = GemmShape<128, 256, 512>;
        using L0TileShape = GemmShape<128, 128, 128>;

        using DispatchPolicy = Catlass::Gemm::MmadAtlasA2SingleCoreSplitk<l1AStages, l1BStages, l0CStages, enableUnitFlag>;
        using BlockScheduler = typename Catlass::Gemm::Block::SingleCoreSplitkGemmIdentityBlockSwizzle<20, 0>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
        
        using MatmulKernel = Catlass::Gemm::Kernel::SingleCoreSplitkMatmul<
            PaddingA, PaddingB, BlockMmad, BlockEpilogue, BlockScheduler, RemovePaddingNDAndCastC>;
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};

        MatmulAdapter matmulOp;
        RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
    }else {
        constexpr uint32_t l1AStages = 1;
        constexpr uint32_t l1BStages = 2;
        using L1TileShape = GemmShape<256, 128, 512>;
        using L0TileShape = GemmShape<128, 128, 128>;

        using DispatchPolicy = Catlass::Gemm::MmadAtlasA2SingleCoreSplitk<l1AStages, l1BStages, l0CStages, enableUnitFlag>;
        using BlockScheduler = typename Catlass::Gemm::Block::SingleCoreSplitkGemmIdentityBlockSwizzle<20, 1>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;

        using MatmulKernel = Catlass::Gemm::Kernel::SingleCoreSplitkMatmul<
            PaddingA, PaddingB, BlockMmad, BlockEpilogue, BlockScheduler, RemovePaddingNDAndCastC>;
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};

        MatmulAdapter matmulOp;
        RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
    }

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }
    
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}