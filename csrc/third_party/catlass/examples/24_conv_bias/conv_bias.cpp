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
#include "catlass/conv/block/block_conv.hpp"
#include "catlass/conv/block/block_swizzle.hpp"
#include "catlass/conv/device/device_conv.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/conv/kernel/conv3d_bias.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;

struct Options {
    const std::string HELPER =
        "24_conv_bias batch di cin1 hi wi cin0 cout kd kh kw sD sH sW dD dH dW pD pH pW [device_id]";

    std::vector<uint32_t> fmapRelated = {1, 1, 1, 2, 9, 16}; // {batch, di, cin1, hi, wi, cin0}
    std::vector<uint32_t> filterRelated = {1, 1, 1, 1};      // {kd, kh, kw, cout}
    std::vector<uint32_t> strides = {1, 1, 1};
    std::vector<uint32_t> pads = {0, 0, 0};
    std::vector<uint32_t> dilations = {1, 1, 1};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv) {
        enum class ArgsIndex {
            BATCH_INDEX = 1,
            DI_INDEX,
            CIN1_INDEX,
            HI_INDEX,
            WI_INDEX,
            CIN0_INDEX,
            COUT_INDEX,
            KD_INDEX,
            KH_INDEX,
            KW_INDEX,
            SD_INDEX,
            SH_INDEX,
            SW_INDEX,
            DD_INDEX,
            DH_INDEX,
            DW_INDEX,
            PD_INDEX,
            PH_INDEX,
            PW_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX)
            || argc < static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)) {
            std::cerr << HELPER << std::endl;
            return 0;
        }

        fmapRelated[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::BATCH_INDEX)]);
        fmapRelated[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DI_INDEX)]);
        fmapRelated[2] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::CIN1_INDEX)]);
        fmapRelated[3] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::HI_INDEX)]);
        fmapRelated[4] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::WI_INDEX)]);
        fmapRelated[5] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::CIN0_INDEX)]);
        filterRelated[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::KD_INDEX)]);
        filterRelated[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::KH_INDEX)]);
        filterRelated[2] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::KW_INDEX)]);
        filterRelated[3] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::COUT_INDEX)]);
        strides[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::SD_INDEX)]);
        strides[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::SH_INDEX)]);
        strides[2] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::SW_INDEX)]);
        dilations[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DD_INDEX)]);
        dilations[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DH_INDEX)]);
        dilations[2] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DW_INDEX)]);
        pads[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::PD_INDEX)]);
        pads[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::PH_INDEX)]);
        pads[2] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::PW_INDEX)]);

        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }

        return 0;
    }
};

static void Run(const Options &options) {
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    Conv3dParams problemShape = Conv3dParams::MakeConvCoord(
        options.fmapRelated.data(), options.filterRelated.data(), options.pads.data(), options.strides.data(),
        options.dilations.data()
    );

    uint32_t n = problemShape.batch();
    uint32_t di = problemShape.di();
    uint32_t cin1 = problemShape.cin1();
    uint32_t hi = problemShape.hi();
    uint32_t wi = problemShape.wi();
    uint32_t cin0 = problemShape.cin0();
    uint32_t kdc1khkw = problemShape.kdc1khkw();
    uint32_t n1 = problemShape.n1();
    uint32_t n0 = problemShape.n0();
    uint32_t dout = problemShape.dout();
    uint32_t ho = problemShape.ho();
    uint32_t wo = problemShape.wo();
    uint32_t cout1 = problemShape.cout1();
    uint32_t cout0 = problemShape.cout0();
    uint32_t cout = problemShape.cout();

    size_t lenFmap = static_cast<size_t>(n) * di * cin1 * hi * wi * cin0;
    size_t lenFilter = static_cast<size_t>(kdc1khkw) * n1 * n0 * cin0;
    size_t lenBias = static_cast<size_t>(cout);
    size_t lenOut = static_cast<size_t>(n) * dout * cout1 * ho * wo * cout0;

    size_t sizeFmap = lenFmap * sizeof(fp16_t);
    size_t sizeFilter = lenFilter * sizeof(fp16_t);
    size_t sizeOut = lenOut * sizeof(fp16_t);
    size_t sizeBias = lenBias * sizeof(fp16_t);

    using LayoutFmap = layout::NDC1HWC0;
    using LayoutFilter = layout::KDC1KHKWN1N0C0;
    using LayoutOut = layout::NDC1HWC0;
    using LayoutBias = layout::VectorLayout;
    LayoutFmap layoutFmap{n, di, cin1, hi * wi, cin0};
    LayoutFilter layoutFilter{kdc1khkw, n1, n0, cin0};
    LayoutOut layoutOut{n, dout, cout1, ho * wo, cout0};

    std::vector<fp16_t> hostFmap(lenFmap);
    std::vector<fp16_t> hostFilter(lenFilter);
    std::vector<fp16_t> hostBias(lenBias);
    ReadFile("./data/fmap.bin", hostFmap.data(), sizeFmap);
    ReadFile("./data/weight.bin", hostFilter.data(), sizeFilter);
    ReadFile("./data/bias.bin", hostBias.data(), sizeBias);

    uint8_t *deviceFmap{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceFmap), sizeFmap, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceFmap, sizeFmap, hostFmap.data(), sizeFmap, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceFilter{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceFilter), sizeFilter, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceFilter, sizeFilter, hostFilter.data(), sizeFilter, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceBias{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceOut{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceOut), sizeOut, ACL_MEM_MALLOC_HUGE_FIRST));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    constexpr uint32_t l1AStages = 1;
    constexpr uint32_t l1BStages = 1;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy =
        Conv::ConvAtlasA2Pingpong<l1AStages, l1BStages, l0AStages, l0BStages, l0CStages, enableUnitFlag>;

    using FmapType = Gemm::GemmType<half, LayoutFmap>;
    using FilterType = Gemm::GemmType<half, LayoutFilter>;
    using BiasType = Gemm::GemmType<half, LayoutBias>;
    using OutType = Gemm::GemmType<half, LayoutOut>;

    using CoreTileShape = ConvCoreShape<2, 2, 2, 2>;       // nDim, dDim, c1Dim, hwDim
    using FmapL1TileShape = ConvFmapL1Shape<16, 1, 1>;     // mAL1, kd, c1
    using FilterL1TileShape = ConvFilterL1Shape<1, 1, 16>; // kd, c1, nBL1
    using L0TileShape = ConvL0Shape<16, 16, 16>;           // mL0 kL0 nL0

    using BlockConv = Conv::Block::BlockConv<
        DispatchPolicy, CoreTileShape, FmapL1TileShape, FilterL1TileShape, L0TileShape, FmapType, FilterType, OutType,
        BiasType>;
    using BlockEpilogue = void;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Conv::Block::Conv3dIdentityBlockSwizzle<3, 0>;

    // kernel level
    using ConvKernel = Conv::Kernel::ConvBias<BlockConv, BlockEpilogue, BlockScheduler>;

    using ConvAdapter = Conv::Device::DeviceConv<ConvKernel>;
    ConvKernel::Arguments arguments{problemShape, deviceFmap, deviceFilter, deviceOut, deviceBias};
    ConvAdapter conv_op;
    conv_op.CanImplement(arguments);
    size_t sizeWorkspace = conv_op.GetWorkspaceSize(arguments);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    conv_op.Initialize(arguments, deviceWorkspace);
    conv_op(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    std::vector<fp16_t> hostOut(lenOut);
    ACL_CHECK(aclrtMemcpy(hostOut.data(), sizeOut, deviceOut, sizeOut, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenOut);
    const size_t goldenSize = sizeOut * 2;
    ReadFile("./data/golden.bin", hostGolden.data(), goldenSize);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostOut, hostGolden, kdc1khkw * cin0);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceFmap));
    ACL_CHECK(aclrtFree(deviceFilter));
    ACL_CHECK(aclrtFree(deviceBias));
    ACL_CHECK(aclrtFree(deviceOut));

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