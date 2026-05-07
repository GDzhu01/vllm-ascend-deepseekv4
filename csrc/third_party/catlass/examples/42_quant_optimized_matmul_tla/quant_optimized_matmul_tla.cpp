/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "catlass/gemm/kernel/quant_optimized_matmul_tla.hpp"

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
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

template <class LayoutTag>
auto GetPaddingLayout(LayoutTag layout, uint32_t blockRows, uint32_t blockCols) {
    if constexpr (std::is_same_v<LayoutTag, layout::RowMajor>) {
        auto shape = MakeShape(
            MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols))
        );
        auto stride = MakeStride(
            MakeStride(
                static_cast<int64_t>(blockCols), static_cast<int64_t>(blockRows) * RoundUp(layout.shape(1), blockCols)
            ),
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols)
        );
        return MakeLayout(shape, stride);
    } else {
        auto shape = MakeShape(
            MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols))
        );
        auto stride = MakeStride(
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols),
            MakeStride(
                static_cast<int64_t>(blockRows), RoundUp(layout.shape(0), blockRows) * static_cast<int64_t>(blockCols)
            )
        );
        return MakeLayout(shape, stride);
    }
}

using Options = GemmOptions;

template <class LayoutTag>
size_t GetWorkspaceLen(LayoutTag layout, size_t blockRows, size_t blockCols) {
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows)
           * RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

static void Run(const Options &options) {
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenScale = static_cast<size_t>(n);
    size_t lenPerTokenScale = static_cast<size_t>(m);
    size_t lenD = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(int8_t);
    size_t sizeB = lenB * sizeof(int8_t);
    size_t sizeScale = lenScale * sizeof(float);
    size_t sizePerTokenScale = lenPerTokenScale * sizeof(float);
    size_t sizeD = lenD * sizeof(bfloat16);
    size_t sizeWorkspace;

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using ElementScale = float;
    using ElementPerTokenScale = float;
    using ElementD = bfloat16_t;
    using ArchTag = Arch::AtlasA2;

    const uint32_t align = 256;
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;
    using LayoutTagScale = layout::RowMajor;
    using LayoutTagPerTokenScale = LayoutTagScale;
    using LayoutTagD = LayoutTagC;
    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);
    LayoutTagScale tagScale = LayoutTagScale::MakeLayout<ElementScale>(1, n);
    LayoutTagPerTokenScale tagPerTokenScale = LayoutTagPerTokenScale::MakeLayout<ElementPerTokenScale>(1, m);
    LayoutTagD tagD = LayoutTagD::MakeLayout<ElementD>(m, n);
    
    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);
    auto layoutScale = MakeLayoutFromTag(tagScale);
    auto layoutPerTokenScale = MakeLayoutFromTag(tagPerTokenScale);
    auto layoutD = MakeLayoutFromTag(tagC);
    using TensorA =
        Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorB =
        Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorC =
        Tensor<AscendC::GlobalTensor<ElementC>, decltype(layoutC), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    bool isNeedPaddingA = IsNeedPadding(tagA, align);
    bool isNeedPaddingB = IsNeedPadding(tagB, align);

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 512> can achieve better performance.
    using L1TileShape = std::conditional_t<
        std::is_same_v<LayoutTagA, layout::ColumnMajor> && std::is_same_v<LayoutTagB, layout::ColumnMajor>,
        Shape<_256, _128, _512>, Shape<_128, _256, _512>>;
    using L0TileShape = std::conditional_t<
        std::is_same_v<LayoutTagA, layout::ColumnMajor> && std::is_same_v<LayoutTagB, layout::ColumnMajor>,
        Shape<_256, _128, _128>, Shape<_128, _256, _128>>;
    
    using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using BlockScheduler31 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    size_t sizeWA = GetWorkspaceLen(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{})) * sizeof(int8_t);
    size_t sizeWB = GetWorkspaceLen(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{})) * sizeof(int8_t);

    constexpr const uint32_t computeLengthA = 96 * 1024 / sizeof(ElementA);
    using PaddingA = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorA, computeLengthA>;
    constexpr const uint32_t computeLengthB = 96 * 1024 / sizeof(ElementB);
    using PaddingB = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorB, computeLengthB>;

    std::vector<int8_t> hostA(lenA);
    std::vector<int8_t> hostB(lenB);
    std::vector<float> hostScale(lenScale);
    std::vector<float> hostPerTokenScale(lenPerTokenScale);
    golden::FillRandomData(hostA, -5, 5);              // Fill with random data, ranging from -5 to 5.
    golden::FillRandomData(hostB, -5, 5);              // Fill with random data, ranging from -5 to 5.
    golden::FillRandomData(hostScale, 0.0, 1.0);         // Fill with random data, ranging from 0.0 to 1.0
    golden::FillRandomData(hostPerTokenScale, 0.0, 1.0); // Fill with random data, ranging from 0.0 to 1.0

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
    ACL_CHECK(aclrtMemcpy(
        devicePerTokenScale, sizePerTokenScale, hostPerTokenScale.data(), sizePerTokenScale, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWA{nullptr};
    if (isNeedPaddingA) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
        // no need to padding A
        deviceWA = deviceA;
    }

    uint8_t *deviceWB{nullptr};
    // If layoutWB has the same stride with layoutB, no need to padding B
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
        // no need to padding B
        deviceWB = deviceB;
    }
    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    constexpr uint32_t workspaceStages = 2;

    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;

    // Block level, define BlockEpilogue
    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequantTla<ubStages>;
    using ElementCompute = float;
    using EpilogueTileShape = MatrixShape<32, 256>;

    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlkTla<ArchTag, ElementCompute, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopyDequantTla<
        ArchTag, ElementC, LayoutTagC, ElementScale, LayoutTagScale, ElementPerTokenScale, LayoutTagPerTokenScale,
        ElementD, LayoutTagD>;
    using EpilogueTileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, ElementC, ElementScale, ElementPerTokenScale, ElementD, TileRowBroadcastMul, TileBroadcastOneBlk,
        TileOneBlkColumnBroadcastMul, EpilogueTileCopy, EpilogueTileScheduler>;

    if (!isNeedPaddingA && !isNeedPaddingB) {
        // no need to padding A and B.
        auto layoutWA = MakeLayout(layoutA.shape(), layoutA.stride());
        auto layoutWB = MakeLayout(layoutB.shape(), layoutB.stride());
        using TensorWA = Tensor<
            AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TensorWB = Tensor<
            AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
            ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB, TensorC, LayoutTagC, void, void, false, false>;
        using BlockMmad = Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB, TensorC, void, TileCopy>;
        if (options.problemShape.m() > options.problemShape.n()) {
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::QuantOptimizedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler30, void, void, workspaceStages>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{
                options.problemShape, aicCoreNum,
                {deviceA, layoutA, deviceB, layoutB, deviceWA, layoutWA, deviceWB, layoutWB},
                {deviceScale, layoutScale, devicePerTokenScale, layoutPerTokenScale, deviceD, layoutD}
            };

            MatmulAdapter matmulOp;
            RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
        } else {
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::QuantOptimizedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler31, void, void, workspaceStages>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{
                options.problemShape, aicCoreNum,
                {deviceA, layoutA, deviceB, layoutB, deviceWA, layoutWA, deviceWB, layoutWB},
                {deviceScale, layoutScale, devicePerTokenScale, layoutPerTokenScale, deviceD, layoutD}
            };

            MatmulAdapter matmulOp;
            RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
        }
    } else if (!isNeedPaddingA && isNeedPaddingB) {
        // no need to padding A, but B needs padding.
        auto layoutWA = MakeLayout(layoutA.shape(), layoutA.stride());
        auto layoutWB = GetPaddingLayout(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{}));
        using TensorWA = Tensor<
            AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TensorWB = Tensor<
            AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
            ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB, TensorC, LayoutTagC, void, void, false, true>;
        using BlockMmad = Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB, TensorC, void, TileCopy>;
        if (options.problemShape.m() > options.problemShape.n()) {
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::QuantOptimizedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler30, void, PaddingB, workspaceStages>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{
                options.problemShape, aicCoreNum,
                {deviceA, layoutA, deviceB, layoutB, deviceWA, layoutWA, deviceWB, layoutWB},
                {deviceScale, layoutScale, devicePerTokenScale, layoutPerTokenScale, deviceD, layoutD}
            };

            MatmulAdapter matmulOp;
            RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
        } else {
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::QuantOptimizedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler31, void, PaddingB, workspaceStages>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{
                options.problemShape, aicCoreNum,
                {deviceA, layoutA, deviceB, layoutB, deviceWA, layoutWA, deviceWB, layoutWB},
                {deviceScale, layoutScale, devicePerTokenScale, layoutPerTokenScale, deviceD, layoutD}
            };

            MatmulAdapter matmulOp;
            RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
        }
    } else if (isNeedPaddingA && !isNeedPaddingB) {
        // no need to padding B, but A needs padding.
        auto layoutWA = GetPaddingLayout(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{}));
        auto layoutWB = MakeLayout(layoutB.shape(), layoutB.stride());
        using TensorWA = Tensor<
            AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TensorWB = Tensor<
            AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
            ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB, TensorC, LayoutTagC, void, void, true, false>;
        using BlockMmad = Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB, TensorC, void, TileCopy>;
        if (options.problemShape.m() > options.problemShape.n()) {
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::QuantOptimizedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler30, PaddingA, void, workspaceStages>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{
                options.problemShape, aicCoreNum,
                {deviceA, layoutA, deviceB, layoutB, deviceWA, layoutWA, deviceWB, layoutWB},
                {deviceScale, layoutScale, devicePerTokenScale, layoutPerTokenScale, deviceD, layoutD}
            };

            MatmulAdapter matmulOp;
            RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
        } else {
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::QuantOptimizedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler31, PaddingA, void, workspaceStages>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{
                options.problemShape, aicCoreNum,
                {deviceA, layoutA, deviceB, layoutB, deviceWA, layoutWA, deviceWB, layoutWB},
                {deviceScale, layoutScale, devicePerTokenScale, layoutPerTokenScale, deviceD, layoutD}
            };

            MatmulAdapter matmulOp;
            RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
        }
    } else {
        // Both A and B need padding.
        auto layoutWA = GetPaddingLayout(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{}));
        auto layoutWB = GetPaddingLayout(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{}));
        using TensorWA = Tensor<
            AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TensorWB = Tensor<
            AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
            ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB, TensorC, LayoutTagC, void, void, true, true>;
        using BlockMmad = Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB, TensorC, void, TileCopy>;
        if (options.problemShape.m() > options.problemShape.n()) {
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::QuantOptimizedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler30, PaddingA, PaddingB, workspaceStages>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{
                options.problemShape, aicCoreNum,
                {deviceA, layoutA, deviceB, layoutB, deviceWA, layoutWA, deviceWB, layoutWB},
                {deviceScale, layoutScale, devicePerTokenScale, layoutPerTokenScale, deviceD, layoutD}
            };

            MatmulAdapter matmulOp;
            RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
        } else {
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::QuantOptimizedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler31, PaddingA, PaddingB, workspaceStages>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{
                options.problemShape, aicCoreNum,
                {deviceA, layoutA, deviceB, layoutB, deviceWA, layoutWA, deviceWB, layoutWB},
                {deviceScale, layoutScale, devicePerTokenScale, layoutPerTokenScale, deviceD, layoutD}
            };

            MatmulAdapter matmulOp;
            RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
        }
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // test precision
    std::vector<bfloat16> hostD(lenD);
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenD);
    golden::QuantMatmul(
        options.problemShape, hostA, tagA, hostB, tagB, hostScale, tagScale,
        hostPerTokenScale, tagPerTokenScale, hostGolden, tagD
    );

    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceD));
    if (isNeedPaddingA) {
        ACL_CHECK(aclrtFree(deviceWA));
    }
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtFree(deviceWB));
    }
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
