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

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemv/block/block_gemv.hpp"
#include "catlass/gemv/device/device_gemv.hpp"
#include "catlass/gemv/kernel/kernel_gemv_aiv.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/gemv/tile/tile_vmad.hpp"
#include "catlass/gemv/tile/tile_vmuls.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;

using ScalarType = float;

using Options = GemvOptions;

static uint32_t getSplictNum(bool trans, uint32_t M, uint32_t N, uint32_t M1, uint32_t N1, uint32_t maxSplict) {
    uint32_t CORENUM = 20;
    uint32_t splitNum = 1;
    uint32_t maxOccupancy = 0;
    uint32_t blockNum = (M - 1) / M1 + 1;
    if (!trans) {
        splitNum = 1;
    } else {
        uint32_t splitNum1 = 1, splitNum2 = 1;
        for (uint32_t i = 1; i <= maxSplict; i += 1) {
            uint32_t occupancy = (i * blockNum) % (CORENUM * 2);
            if (!occupancy)
                occupancy = (CORENUM * 2);
            if (occupancy > maxOccupancy) {
                maxOccupancy = occupancy;
                splitNum1 = i;
            }
        }
        maxOccupancy = 0;
        for (uint32_t i = 1; i <= maxSplict; i <<= 1) {
            uint32_t occupancy = (i * blockNum) % (CORENUM * 2);
            if (!occupancy)
                occupancy = (CORENUM * 2);
            if (occupancy > maxOccupancy) {
                maxOccupancy = occupancy;
                splitNum2 = i;
            }
        }
        splitNum = (splitNum1 - splitNum2) > 4 ? splitNum1 : splitNum2;
    }
    return splitNum;
}

template <class ElementRandom>
void FillRandomScalarData(ElementRandom &scalarData, ElementRandom low, ElementRandom high) {
    scalarData = static_cast<ElementRandom>(
        low + (static_cast<ElementRandom>(rand()) / static_cast<ElementRandom>(RAND_MAX)) * (high - low)
    );
}

static void Run(Options options) {
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    using UBTileShape = GemvShape<32, 512>;

    uint32_t maxSplict = 20;
    uint32_t const split = getSplictNum(false, m, n, UBTileShape::M, UBTileShape::N, maxSplict);

    size_t lenA = static_cast<size_t>(m) * n;
    size_t lenX = static_cast<size_t>(n) * 1;
    size_t lenY = static_cast<size_t>(m) * 1;
    size_t scalarLen = 1;

    size_t sizeA = lenA * sizeof(float);
    size_t sizeX = lenX * sizeof(float);
    size_t sizeY = lenY * sizeof(float);

    using LayoutA = layout::RowMajor;
    using LayoutX = layout::VectorLayout;
    using LayoutY = layout::VectorLayout;

    LayoutA layoutA{m, n};
    LayoutX layoutX{n};
    LayoutY layoutY{m};

    ScalarType alpha{0};
    ScalarType beta{0};
    FillRandomScalarData(alpha, -1.0f, 1.0f);
    FillRandomScalarData(beta, -1.0f, 1.0f);

    std::vector<float> hostA(lenA);
    std::vector<float> hostX(lenX);
    std::vector<float> hostY(lenY);
    golden::FillRandomData(hostA, -1.0f, 1.0f);
    golden::FillRandomData(hostX, -1.0f, 1.0f);
    golden::FillRandomData(hostY, -1.0f, 1.0f);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceY{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceY), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceY, sizeY, hostY.data(), sizeY, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceZ{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceZ), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZ, sizeY, hostY.data(), sizeY, ACL_MEMCPY_HOST_TO_DEVICE));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::GemvAtlasA2;

    using AType = Gemm::GemmType<float, LayoutA>;
    using XType = Gemm::GemmType<float, LayoutX>;
    using YType = Gemm::GemmType<float, LayoutY>;
    using BiasType = void;
    using TileCopy = Gemv::Tile::TileCopyGemvAiv<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>;
    using TileVmad = Gemv::Tile::TileVmad<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>;
    using TileVmuls = Gemv::Tile::TileVmuls<typename DispatchPolicy::ArchTag, XType>;

    using GemvBlock = Gemv::Block::BlockGemv<
        DispatchPolicy, UBTileShape, AType, XType, YType, BiasType, TileCopy, TileVmad, TileVmuls>;
    using BlockEpilogue = void;

    // kernel level
    using GemvKernel = Gemv::Kernel::KernelGemvAiv<GemvBlock, BlockEpilogue>;
    typename GemvKernel::Arguments arguments{
        options.problemShape, deviceA, deviceX, deviceY, deviceZ, alpha, beta, split};

    using GemvAdapter = Gemv::Device::DeviceGemv<GemvKernel>;
    GemvAdapter gemv_op;
    gemv_op.CanImplement(arguments);
    RunAdapter(gemv_op, arguments, stream, aicCoreNum);

    std::vector<float> hostRes(lenY);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeY, deviceZ, sizeY, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenY);
    golden::ComputeGemv(
        options.problemShape, alpha, beta, hostA, layoutA, hostX, layoutX, hostY, layoutY, hostGolden, layoutY
    );
    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, m);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceY));
    ACL_CHECK(aclrtFree(deviceZ));

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