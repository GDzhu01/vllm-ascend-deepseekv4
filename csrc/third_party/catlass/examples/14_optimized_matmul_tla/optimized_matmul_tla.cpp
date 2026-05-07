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

#include "catlass/gemm/kernel/optimized_matmul_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
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

template <class Layout>
auto GetPaddingLayout(Layout layout, uint32_t blockRows, uint32_t blockCols) {
    if constexpr (std::is_same_v<Layout, layout::RowMajor>) {
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

template <class Layout>
size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols) {
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
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);
    size_t sizeWorkspace;

    const uint32_t align = 256;
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagC tagC{m, n};
    bool isNeedPaddingA = IsNeedPadding(tagA, align);
    bool isNeedPaddingB = IsNeedPadding(tagB, align);

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape = std::conditional_t<
        std::is_same_v<LayoutTagA, layout::ColumnMajor> && std::is_same_v<LayoutTagB, layout::ColumnMajor>,
        Shape<_256, _128, _256>, Shape<_128, _256, _256>>;
    size_t sizeWA = GetWorkspaceLen(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{})) * sizeof(fp16_t);
    size_t sizeWB = GetWorkspaceLen(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{})) * sizeof(fp16_t);

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

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
    uint8_t *deviceWorkspace{nullptr};
    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ElementA = half;
    using ElementB = half;
    using ElementC = half;
    using ArchTag = Arch::AtlasA2;

    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);
    using TensorA =
        Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorB =
        Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorC =
        Tensor<AscendC::GlobalTensor<ElementC>, decltype(layoutC), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape = std::conditional_t<
        std::is_same_v<LayoutTagA, layout::ColumnMajor> && std::is_same_v<LayoutTagB, layout::ColumnMajor>,
        Shape<_256, _128, _256>, Shape<_128, _256, _256>>;
    using L0TileShape = std::conditional_t<
        std::is_same_v<LayoutTagA, layout::ColumnMajor> && std::is_same_v<LayoutTagB, layout::ColumnMajor>,
        Shape<_256, _128, _64>, Shape<_128, _256, _64>>;
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
        using PaddingA = void;
        using PaddingB = void;
        if (options.problemShape.m() > options.problemShape.n()) {
            using TileScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using BlockEpilogue = void;
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::OptimizedMatmulTla<BlockMmad, BlockEpilogue, TileScheduler, PaddingA, PaddingB>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{options.problemShape,
                                              deviceA,
                                              layoutA,
                                              deviceB,
                                              layoutB,
                                              deviceC,
                                              layoutC,
                                              deviceWA,
                                              layoutWA,
                                              deviceWB,
                                              layoutWB};

            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            matmulOp.Initialize(arguments, deviceWorkspace);
            matmulOp(stream, aicCoreNum, fftsAddr);
        } else {
            using TileScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using BlockEpilogue = void;
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::OptimizedMatmulTla<BlockMmad, BlockEpilogue, TileScheduler, PaddingA, PaddingB>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{options.problemShape,
                                              deviceA,
                                              layoutA,
                                              deviceB,
                                              layoutB,
                                              deviceC,
                                              layoutC,
                                              deviceWA,
                                              layoutWA,
                                              deviceWB,
                                              layoutWB};

            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            matmulOp.Initialize(arguments, deviceWorkspace);
            matmulOp(stream, aicCoreNum, fftsAddr);
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
        using PaddingA = void;
        constexpr const uint32_t computeLengthB = 96 * 1024 / sizeof(ElementB);
        using PaddingB = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorB, TensorWB, computeLengthB>;
        if (options.problemShape.m() > options.problemShape.n()) {
            using TileScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using BlockEpilogue = void;
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::OptimizedMatmulTla<BlockMmad, BlockEpilogue, TileScheduler, PaddingA, PaddingB>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{options.problemShape,
                                              deviceA,
                                              layoutA,
                                              deviceB,
                                              layoutB,
                                              deviceC,
                                              layoutC,
                                              deviceWA,
                                              layoutWA,
                                              deviceWB,
                                              layoutWB};

            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            matmulOp.Initialize(arguments, deviceWorkspace);
            matmulOp(stream, aicCoreNum, fftsAddr);
        } else {
            using TileScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using BlockEpilogue = void;
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::OptimizedMatmulTla<BlockMmad, BlockEpilogue, TileScheduler, PaddingA, PaddingB>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{options.problemShape,
                                              deviceA,
                                              layoutA,
                                              deviceB,
                                              layoutB,
                                              deviceC,
                                              layoutC,
                                              deviceWA,
                                              layoutWA,
                                              deviceWB,
                                              layoutWB};

            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            matmulOp.Initialize(arguments, deviceWorkspace);
            matmulOp(stream, aicCoreNum, fftsAddr);
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
        constexpr const uint32_t computeLengthA = 96 * 1024 / sizeof(ElementA);
        using PaddingA = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorA, TensorWA, computeLengthA>;
        using PaddingB = void;
        if (options.problemShape.m() > options.problemShape.n()) {
            using TileScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using BlockEpilogue = void;
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::OptimizedMatmulTla<BlockMmad, BlockEpilogue, TileScheduler, PaddingA, PaddingB>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{options.problemShape,
                                              deviceA,
                                              layoutA,
                                              deviceB,
                                              layoutB,
                                              deviceC,
                                              layoutC,
                                              deviceWA,
                                              layoutWA,
                                              deviceWB,
                                              layoutWB};

            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            matmulOp.Initialize(arguments, deviceWorkspace);
            matmulOp(stream, aicCoreNum, fftsAddr);
        } else {
            using TileScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using BlockEpilogue = void;
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::OptimizedMatmulTla<BlockMmad, BlockEpilogue, TileScheduler, PaddingA, PaddingB>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{options.problemShape,
                                              deviceA,
                                              layoutA,
                                              deviceB,
                                              layoutB,
                                              deviceC,
                                              layoutC,
                                              deviceWA,
                                              layoutWA,
                                              deviceWB,
                                              layoutWB};

            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            matmulOp.Initialize(arguments, deviceWorkspace);
            matmulOp(stream, aicCoreNum, fftsAddr);
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
        constexpr const uint32_t computeLengthA = 96 * 1024 / sizeof(ElementA);
        using PaddingA = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorA, TensorWA, computeLengthA>;
        constexpr const uint32_t computeLengthB = 96 * 1024 / sizeof(ElementB);
        using PaddingB = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorB, TensorWB, computeLengthB>;
        if (options.problemShape.m() > options.problemShape.n()) {
            using TileScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using BlockEpilogue = void;
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::OptimizedMatmulTla<BlockMmad, BlockEpilogue, TileScheduler, PaddingA, PaddingB>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{options.problemShape,
                                              deviceA,
                                              layoutA,
                                              deviceB,
                                              layoutB,
                                              deviceC,
                                              layoutC,
                                              deviceWA,
                                              layoutWA,
                                              deviceWB,
                                              layoutWB};

            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            matmulOp.Initialize(arguments, deviceWorkspace);
            matmulOp(stream, aicCoreNum, fftsAddr);
        } else {
            using TileScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using BlockEpilogue = void;
            // kernel level
            using MatmulKernel =
                Gemm::Kernel::OptimizedMatmulTla<BlockMmad, BlockEpilogue, TileScheduler, PaddingA, PaddingB>;
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{options.problemShape,
                                              deviceA,
                                              layoutA,
                                              deviceB,
                                              layoutB,
                                              deviceC,
                                              layoutC,
                                              deviceWA,
                                              layoutWA,
                                              deviceWB,
                                              layoutWB};

            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            matmulOp.Initialize(arguments, deviceWorkspace);
            matmulOp(stream, aicCoreNum, fftsAddr);
        }
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmul(options.problemShape, hostA, tagA, hostB, tagB, hostGolden, tagC);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
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
