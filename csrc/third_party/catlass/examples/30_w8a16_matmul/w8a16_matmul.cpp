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

#include "catlass/gemm/kernel/w8a16_matmul.hpp"

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

    size_t sizeA = lenA * sizeof(half);
    size_t sizeB = lenB * sizeof(int8_t);
    size_t sizeC = lenC * sizeof(half);

    using ElementA = half;
    using ElementPrologueB = int8_t;
    using ElementB = half;
    using ElementC = half;

    using LayoutA = layout::RowMajor;
    using LayoutPrologueB = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutPrologueB layoutPrologueB{k, n};
    LayoutC layoutC{m, n};

    half deqScalar = 1.5;
    half deqZeroPoint = 0.1;
    fp16_t deqScalarFp16 = 1.5;
    fp16_t deqZeroPointFp16 = 0.1;

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape = std::conditional_t<
        std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
        GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;
    using L0TileShape = std::conditional_t<
        std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
        GemmShape<256, 128, 64>, GemmShape<128, 256, 64>>;

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    std::vector<fp16_t> hostA(lenA);
    std::vector<int8_t> hostB(lenB);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<int8_t>(hostB, -8, 8);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    using ArchTag = Arch::AtlasA2;
    constexpr bool ENABLE_UNIT_FLAG = true;

    using PrologueSrcType = Gemm::GemmType<ElementPrologueB, LayoutPrologueB>;
    using PrologueDstType = Gemm::GemmType<ElementB, LayoutB>;
    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = PrologueDstType;
    using CType = Gemm::GemmType<half, LayoutC>;
    using DispatchPolicy = Gemm::MmadAtlasA2PingPongWithPrologue<ENABLE_UNIT_FLAG>;

    using PrologueA = void;
    constexpr uint32_t computeLen = 32 * 1024;
    using PrologueB = Gemm::Tile::TileCastInt8ToFp16Dequant<ArchTag, PrologueSrcType, PrologueDstType, computeLen>;

    using TileCopy = Gemm::Tile::TileCopyWithPrologue<ArchTag, AType, BType, CType, PrologueA, PrologueB>;
    using BlockMmadOpt =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
    using BlockEpilogue = void;

    if (m > n) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using MatmulKernel = Gemm::Kernel::W8A16Matmul<BlockMmadOpt, BlockEpilogue, BlockScheduler>;
        typename MatmulKernel::Arguments arguments{
                options.problemShape,
                deviceA, layoutA,
                deviceB, layoutPrologueB,
                deviceC, layoutC,
                deqScalar, deqZeroPoint,
                aicCoreNum
            };
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using MatmulKernel = Gemm::Kernel::W8A16Matmul<BlockMmadOpt, BlockEpilogue, BlockScheduler>;
        typename MatmulKernel::Arguments arguments{
                options.problemShape,
                deviceA, layoutA,
                deviceB, layoutPrologueB,
                deviceC, layoutC,
                deqScalar, deqZeroPoint,
                aicCoreNum
            };
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
    }

    std::vector<fp16_t> hostBFp16(hostB.begin(), hostB.end());
    for (size_t i = 0; i < hostBFp16.size(); i++) {
        hostBFp16[i] = static_cast<fp16_t>((hostBFp16[i] + deqZeroPointFp16) * deqScalarFp16);
    }

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostBFp16, layoutPrologueB, hostGolden, layoutC);

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
