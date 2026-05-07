/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/gemm/kernel/sparse_matmul_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

using Options = GemmOptions;

struct MatmulShape {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t b;
};

template <class L1Shape_>
static int64_t GetTotalBlockNum(int atomic, const MatmulShape &shape)
{
    const int l1M = tla::get<0>(L1Shape_{});
    const int l1N = tla::get<1>(L1Shape_{});
    int maxCoreNum = atomic;
    int64_t mTotalCnt = CeilDiv(shape.m, l1M);
    int64_t nTotalCnt = CeilDiv(shape.n, l1N);
    int64_t batch = shape.b ? shape.b : 1;
    int64_t blockNum = 0;
    int64_t totalCnt = mTotalCnt * nTotalCnt * batch;
    if (totalCnt < maxCoreNum) {
        blockNum = totalCnt;
    } else {
        int64_t perCoreBlockNum = CeilDiv(totalCnt, maxCoreNum);
        blockNum = CeilDiv(totalCnt, perCoreBlockNum);
    }
    return blockNum;
}

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

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;  // trans B
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k / 2, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity();              // For 4:2 sparse pattern, only half of the elements are stored
    size_t lenC = tagC.Capacity();
    size_t lenIdx = static_cast<size_t>((n * k + 7) / 8); // For index matrix

    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeC = lenC * sizeof(ElementC);
    size_t sizeIdx = lenIdx * sizeof(uint8_t);
    size_t goldenSize = lenC * sizeof(int32_t);
    size_t sizeWorkspace;

    uint8_t *hostA;
    ACL_CHECK(aclrtMallocHost((void **)(&hostA), sizeA));
    ReadFile("./input/x1_gm.bin", hostA, sizeA);
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *hostB;
    ACL_CHECK(aclrtMallocHost((void **)(&hostB), sizeB));
    ReadFile("./input/x2_gm.bin", hostB, sizeB);
    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceBias = nullptr;

    uint8_t *hostIdx;
    ACL_CHECK(aclrtMallocHost((void **)(&hostIdx), sizeIdx));
    ReadFile("./input/index_gm.bin", hostIdx, sizeIdx);
    uint8_t *deviceIdx{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceIdx), sizeIdx, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceIdx, sizeIdx, hostIdx, sizeIdx, ACL_MEMCPY_HOST_TO_DEVICE));


    uint8_t *deviceWorkspace{nullptr};

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::SparseMatmulMultiBlockOnKAxis<ArchTag, false>;
    using L1TileShape = Shape<_128, _256, _128>;
    using L0TileShape = Shape<_128, _256, _64>;

    auto layoutA = tla::MakeLayoutFromTag(tagA);
    auto layoutB = tla::MakeLayoutFromTag(tagB);
    auto layoutC = tla::MakeLayoutFromTag(tagC);

    using ProblemShape = MatmulShape;

    MatmulShape shape = {m, n, k, 1};

    using TileCopy =
        Gemm::Tile::SparseTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
    using BlockMmad = Gemm::Block::BlockMmadSparseTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = void;

    using BlockScheduler = Gemm::Block::BlockSchedulerIterateK<ProblemShape, L1TileShape, L0TileShape>;

    // kernel level
    using MatmulKernel = Gemm::Kernel::KernelSparseMatmul<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Arguments = typename MatmulKernel::Arguments;
    Arguments args = {
        shape,                                               // problem shape
        {deviceA, deviceB, deviceC, deviceBias, deviceIdx, layoutA, layoutB, layoutC},  // mmad args
        aicCoreNum
    };

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    MatmulAdapter matmulOp;
    matmulOp.CanImplement(args);
    sizeWorkspace = matmulOp.GetWorkspaceSize(args);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(args, deviceWorkspace);
    int blockNum = GetTotalBlockNum<L1TileShape>(aicCoreNum, shape);
    matmulOp(stream, blockNum);
    
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int32_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<int32_t> hostGolden(goldenSize);
    std::string expected_path = "./output/golden.bin";
    ReadFile(expected_path, hostGolden.data(), goldenSize);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceIdx));
    ACL_CHECK(aclrtFree(deviceC));

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