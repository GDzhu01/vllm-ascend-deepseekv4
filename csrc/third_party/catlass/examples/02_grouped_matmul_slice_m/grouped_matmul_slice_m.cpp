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

#include "catlass/gemm/kernel/grouped_matmul_slice_m.hpp"

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

using Options = GroupedGemmOptions;
static void Run(const Options &options) {
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t problemCount = options.problemCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n * problemCount;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData(hostA, -5.0, 5.0);
    golden::FillRandomData(hostB, -5.0, 5.0);
    auto groupList = golden::GenerateGroupList<int64_t>(m, problemCount);

    size_t sizeGroupList = problemCount * sizeof(int64_t);
    uint8_t *deviceGroupList{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceGroupList), sizeGroupList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceGroupList, sizeGroupList, groupList.data(), sizeGroupList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    size_t sizeWorkspace = 0;
    uint8_t *deviceWorkspace{nullptr};

    if (options.problemShape.k() > options.problemShape.n()) {
        constexpr uint32_t preloadStages = 1;
        constexpr uint32_t l1Stages = 2;
        constexpr uint32_t l0AStages = 2;
        constexpr uint32_t l0BStages = 4;
        constexpr uint32_t l0CStages = 1;
        constexpr bool enableUnitFlag = true;
        constexpr bool enableShuffleK = true;

        using ArchTag = Arch::AtlasA2;
        using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<
            preloadStages, l1Stages, l0AStages, l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;
        using L1TileShape = GemmShape<256, 128, 256>;
        using L0TileShape = GemmShape<256, 128, 64>;

        using AType = Gemm::GemmType<half, LayoutA>;
        using BType = Gemm::GemmType<half, LayoutB>;
        using CType = Gemm::GemmType<half, LayoutC>;

        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        using BlockEpilogue = void;
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceM<BlockMmad, BlockEpilogue, BlockScheduler, int64_t>;
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulKernel::Arguments arguments{
            options.problemShape, problemCount, deviceGroupList, deviceA, deviceB, deviceC};

        // call a kernel
        MatmulAdapter matmulOp;
        // judge arguments can run
        matmulOp.CanImplement(arguments);
        // get workspace
        sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST);
            );
        }
        // initalize kernel argument
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreNum);

    } else {
        constexpr uint32_t preloadStages = 1;
        constexpr uint32_t l1Stages = 2;
        constexpr uint32_t l0AStages = 4;
        constexpr uint32_t l0BStages = 2;
        constexpr uint32_t l0CStages = 1;
        constexpr bool enableUnitFlag = true;
        constexpr bool enableShuffleK = true;

        using ArchTag = Arch::AtlasA2;
        using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<
            preloadStages, l1Stages, l0AStages, l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;
        using L1TileShape = GemmShape<128, 256, 256>;
        using L0TileShape = GemmShape<128, 256, 64>;

        using AType = Gemm::GemmType<half, LayoutA>;
        using BType = Gemm::GemmType<half, LayoutB>;
        using CType = Gemm::GemmType<half, LayoutC>;

        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        using BlockEpilogue = void;
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceM<BlockMmad, BlockEpilogue, BlockScheduler, int64_t>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

        MatmulKernel::Arguments arguments{
            options.problemShape, problemCount, deviceGroupList, deviceA, deviceB, deviceC};

        // call a kernel
        MatmulAdapter matmulOp;
        // judge arguments can run
        matmulOp.CanImplement(arguments);
        // get workspace
        sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST);
            );
        }
        // initalize kernel argument
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreNum);
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<GemmCoord> problemShapeList(problemCount);
    std::vector<LayoutA> layoutAList(problemCount);
    std::vector<LayoutB> layoutBList(problemCount);
    std::vector<LayoutC> layoutCList(problemCount);
    for (uint32_t i = 0; i < problemCount; ++i) {
#ifdef CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED
        uint32_t currentM = groupList[i];
#else
        uint32_t currentM = (i == 0) ? groupList[0] : (groupList[i] - groupList[i - 1]);
#endif
        problemShapeList[i] = GemmCoord{currentM, n, k};
        layoutAList[i] = LayoutA{currentM, k};
        layoutBList[i] = LayoutB{k, n};
        layoutCList[i] = LayoutC{currentM, n};
    }

    std::vector<float> hostGolden(lenC);
    golden::ComputeGroupedMatmul(
        problemCount, problemShapeList, hostA, layoutAList, hostB, layoutBList, hostGolden, layoutCList
    );

#ifdef CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED
    uint64_t totalM = 0;
    for (uint32_t i = 0; i < problemCount; ++i) {
        totalM += groupList[i];
    }
    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k, totalM * n);
#else
    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k, groupList[problemCount - 1] * n);
#endif
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceGroupList));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv) {
    Options options;
    if (options.Parse(argc, argv) == 0) {
        Run(options);
    }
    return 0;
}