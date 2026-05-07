/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
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

#include "catlass/gemm/kernel/quant_matmul_full_loadA_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"


#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

using Options = GemmOptions;

static void Run(const Options &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using ElementScale = float;
    using ElementPerTokenScale = float;
    using ElementD = bfloat16_t;
    using ArchTag = Arch::AtlasA2;
    // TODO: support bias
    // if no bias, set ElementBias to void
    using ElementBias = void;

    using ElementBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    using LayoutTagScale = layout::RowMajor;
    using LayoutTagPerTokenScale = layout::RowMajor;
    using LayoutTagD = layout::RowMajor;

    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);
    LayoutTagScale tagScale = LayoutTagScale::MakeLayout<ElementScale>(1, n);
    LayoutTagPerTokenScale tagPerTokenScale = LayoutTagPerTokenScale::MakeLayout<ElementPerTokenScale>(1, m);
    LayoutTagD tagD = LayoutTagD::MakeLayout<ElementD>(m, n);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity();
    size_t lenC = tagC.Capacity();
    size_t lenScale = static_cast<size_t>(n);
    size_t lenPerTokenScale = static_cast<size_t>(m);
    size_t lenD = tagD.Capacity();
    size_t lenBias = static_cast<size_t>(n);

    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeC = lenC * sizeof(ElementC);
    size_t sizeScale = lenScale * sizeof(ElementScale);
    size_t sizePerTokenScale = lenPerTokenScale * sizeof(ElementPerTokenScale);
    size_t sizeD = lenD * sizeof(ElementD);
    size_t sizeBias = lenBias * sizeof(ElementBiasType);
    size_t sizeWorkspace;

    std::vector<ElementA> hostA(lenA);
    std::vector<ElementB> hostB(lenB);
    std::vector<ElementScale> hostScale(lenScale);
    std::vector<ElementPerTokenScale> hostPerTokenScale(lenPerTokenScale);
    std::vector<ElementBiasType> hostBias(lenBias);
    golden::FillRandomData<ElementA>(hostA, -16, 16);
    golden::FillRandomData<ElementB>(hostB, -16, 16);
    golden::FillRandomData<ElementScale>(hostScale, 0.0, 1.0);
    golden::FillRandomData<ElementPerTokenScale>(hostPerTokenScale, 0.0, 1.0);
    if constexpr (!std::is_void_v<ElementBias>) {
        golden::FillRandomData<ElementBiasType>(hostBias, -5.0f, 5.0f);
    }

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceScale), sizeScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceScale, sizeScale, hostScale.data(), sizeScale, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *devicePerTokenScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devicePerTokenScale), sizePerTokenScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(devicePerTokenScale, sizePerTokenScale, hostPerTokenScale.data(), sizePerTokenScale, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceBias{nullptr};
    if constexpr (!std::is_void_v<ElementBias>) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    uint8_t *deviceWorkspace{nullptr};

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    constexpr uint32_t workspaceStages = 2;

    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;
    constexpr bool useHF32 = false;
    constexpr bool enableL1Resident = false;
    constexpr uint32_t l0CStages = 1;
    constexpr uint32_t l1AStages = 1;
    constexpr uint32_t l1BStages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;

    using L1TileShape = Shape<Int<128>, Int<256>, Int<512>>;
    using L0TileShape = Shape<Int<128>, Int<256>, Int<128>>;

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);
    auto layoutScale = MakeLayoutFromTag(tagScale);
    auto layoutPerTokenScale = MakeLayoutFromTag(tagPerTokenScale);
    auto layoutD = MakeLayoutFromTag(tagD);
	
    using DispatchPolicy = Gemm::MmadFullLoadA<
        ArchTag,
        enableUnitFlag, enableShuffleK, useHF32, l0CStages, enableL1Resident,
        l1AStages, l1BStages, l0AStages, l0BStages
    >;

    using TileCopy = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    
    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequantTla<ubStages>;
    using DType = Gemm::GemmType<ElementD, LayoutTagD>;

    using ElementCompute = float;
    using EpilogueTileShape = MatrixShape<32, 256>;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlkTla<ArchTag, ElementCompute, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopyDequantTla<ArchTag, ElementC, LayoutTagC, ElementScale, LayoutTagScale, ElementPerTokenScale, LayoutTagPerTokenScale,
        ElementD, LayoutTagD>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, ElementC, ElementScale, ElementPerTokenScale, ElementD, TileRowBroadcastMul, TileBroadcastOneBlk,
        TileOneBlkColumnBroadcastMul, EpilogueTileCopy, TileScheduler>;

    uint32_t taskNum = CeilDiv(options.problemShape.m(), tla::get<0>(L1TileShape{})) *
                       CeilDiv(options.problemShape.n(), tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = min(aicCoreNum, taskNum);

    // if m > L1TileShape::M, split M for coreLoop.
    // Use GemmIdentityBlockSwizzleL1FullLoad<SwizzleOffset, SwizzleDirection, AicCoreNum>.
    if (m > tla::get<0>(L1TileShape{})) {
        // Swizzle offset is 1 and direction is 0.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzleL1FullLoad<1, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::QuantMatmulFullLoadATla<BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

        MatmulKernel::Arguments arguments{
            options.problemShape, aicCoreNum, 
            {deviceA, layoutA, deviceB, layoutB, deviceBias}, 
            {deviceScale, layoutScale, devicePerTokenScale, layoutPerTokenScale, deviceD, layoutD}
        };

        MatmulAdapter matmulOp;
        Status status = matmulOp.CanImplement(arguments);
        if (status == Status::kInvalid) {
            std::cout << "Not satisfy the constraints of quant_matmul_full_loadA_tla." << std::endl;
            std::cout << "MatA tile cannot be full loaded to L1." << std::endl;
            ACL_CHECK(aclrtDestroyStream(stream));
            ACL_CHECK(aclrtResetDevice(options.deviceId));
            ACL_CHECK(aclFinalize());
            return;
        }
        sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
            );
        }
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreUsed, fftsAddr);
    } else {
         // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::QuantMatmulFullLoadATla<BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

        MatmulKernel::Arguments arguments{
            options.problemShape, aicCoreNum, 
            {deviceA, layoutA, deviceB, layoutB, deviceBias}, 
            {deviceScale, layoutScale, devicePerTokenScale, layoutPerTokenScale, deviceD, layoutD}
        };

        MatmulAdapter matmulOp;
        Status status = matmulOp.CanImplement(arguments);
        if (status == Status::kInvalid) {
            std::cout << "Not satisfy the constraints of quant_matmul_full_loadA_tla." << std::endl;
            std::cout << "MatA tile cannot be full loaded to L1." << std::endl;
            ACL_CHECK(aclrtDestroyStream(stream));
            ACL_CHECK(aclrtResetDevice(options.deviceId));
            ACL_CHECK(aclFinalize());
            return;
        }
        sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
            );
        }
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreUsed, fftsAddr);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<bfloat16> hostD(lenD);
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::QuantMatmul(options.problemShape, hostA, tagA, hostB, tagB, hostScale, tagScale, 
        hostPerTokenScale, tagPerTokenScale, hostGolden, tagD
    );

    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceScale));
    ACL_CHECK(aclrtFree(devicePerTokenScale));
    ACL_CHECK(aclrtFree(deviceD));

    if constexpr (!std::is_void_v<ElementBias>) {
        ACL_CHECK(aclrtFree(deviceBias));
    }
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
