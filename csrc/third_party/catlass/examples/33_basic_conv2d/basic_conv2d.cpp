/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "catlass/conv/kernel/basic_conv2d.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/conv/block/block_conv.hpp"
#include "catlass/conv/block/block_swizzle.hpp"
#include "catlass/conv/device/device_conv.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;

struct Options {
    const std::string HELPER = "33_basic_conv2d batch, hi, wi, cin, cout, kh, kw, padLeft, padRight, padTop, "
                               "padBottom, strideH, strideW, dilationH, dilationW [device_id]";

    uint32_t dataSizes[5] = {2, 33, 43, 112, 80}; // {batch, hi, wi, cin, cout}
    uint8_t filterSizes[2] = {3, 3};              // {kh, kw}
    uint8_t pads[4] = {2, 2, 2, 2};               // {padLeft, padRight, padTop, padBottom}
    uint8_t strides[2] = {2, 2};                  // {strideH, strideW}
    uint8_t dilations[2] = {1, 1};                // {dilationH, dilationW}
    int32_t deviceId{0};

    Catlass::Conv2dParams problemParams{};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum class ArgsIndex {
            BATCH_INDEX = 1,
            HI_INDEX,
            WI_INDEX,
            CIN_INDEX,
            COUT_INDEX,
            KH_INDEX,
            KW_INDEX,
            PADLEFT_INDEX,
            PADRIGHT_INDEX,
            PADTOP_INDEX,
            PADBOTTOM_INDEX,
            STRIDEH_INDEX,
            STRIDEW_INDEX,
            DILATIONH_INDEX,
            DILATIONW_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX)
            || argc <= static_cast<uint32_t>(ArgsIndex::DILATIONW_INDEX)) {
            std::cerr << HELPER << std::endl;
            return 0;
        }

        dataSizes[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::BATCH_INDEX)]);
        dataSizes[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::HI_INDEX)]);
        dataSizes[2] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::WI_INDEX)]);
        dataSizes[3] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::CIN_INDEX)]);
        dataSizes[4] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::COUT_INDEX)]);
        filterSizes[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::KH_INDEX)]);
        filterSizes[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::KW_INDEX)]);
        pads[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::PADLEFT_INDEX)]);
        pads[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::PADRIGHT_INDEX)]);
        pads[2] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::PADTOP_INDEX)]);
        pads[3] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::PADBOTTOM_INDEX)]);
        strides[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::STRIDEH_INDEX)]);
        strides[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::STRIDEW_INDEX)]);
        dilations[0] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DILATIONH_INDEX)]);
        dilations[1] = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DILATIONW_INDEX)]);

        problemParams = Catlass::Conv2dParams::MakeConv2dParams(dataSizes, filterSizes, pads, strides, dilations);

        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        return 0;
    }
};

int32_t Max(int32_t a, int32_t b)
{
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

void Run(Options const &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t c0 = options.problemParams.C0;
    uint32_t batch = options.problemParams.batch();
    uint32_t hi = options.problemParams.hi();
    uint32_t wi = options.problemParams.wi();
    uint32_t cin1 = options.problemParams.cin1();
    uint32_t ho = options.problemParams.ho();
    uint32_t wo = options.problemParams.wo();
    uint32_t cout1 = options.problemParams.cout1();
    uint32_t cout = options.problemParams.cout();
    uint32_t coutRound = options.problemParams.coutRound();
    uint32_t kh = options.problemParams.kh();
    uint32_t kw = options.problemParams.kw();

    size_t lenFmap = batch * cin1 * hi * wi * c0;
    size_t lenFilter = cin1 * kh * kw * cout * c0;
    size_t lenOutput = batch * ho * wo * coutRound;

    size_t sizeFmap = lenFmap * sizeof(fp16_t);
    size_t sizeFilter = lenFilter * sizeof(fp16_t);
    size_t sizeOutput = lenOutput * sizeof(fp16_t);

    using LayoutFmap = layout::NC1HWC0;
    using LayoutFilter = layout::CI1KHKWCOCI0;
    using LayoutOutput = layout::NC1HWC0;
    LayoutFmap layoutFmap{batch, cin1, hi, wi, c0};
    LayoutFilter layoutFilter{cin1, kh, kw, cout, c0};
    LayoutOutput layoutOutput{batch, cout1, ho, wo, c0};

    std::vector<fp16_t> hostFmap(lenFmap);
    std::vector<fp16_t> hostFilter(lenFilter);
    golden::FillRandomData<fp16_t>(hostFmap, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostFilter, -5.0f, 5.0f);

    uint8_t *deviceFmap{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceFmap), sizeFmap, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceFmap, sizeFmap, hostFmap.data(), sizeFmap, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceFilter{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceFilter), sizeFilter, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceFilter, sizeFilter, hostFilter.data(), sizeFilter, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceOutput{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceOutput), sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::AtlasA2;
    constexpr uint32_t L1A_STAGES = 2;
    constexpr uint32_t L1B_STAGES = 2;
    constexpr uint32_t L0A_STAGES = 2;
    constexpr uint32_t L0B_STAGES = 2;
    constexpr uint32_t L0C_STAGES = 1;
    constexpr bool ENABLE_UNIT_FLAG = false;
    using DispatchPolicy =
        Conv::ConvAtlasA2Pingpong<L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES, L0C_STAGES, ENABLE_UNIT_FLAG>;
    using FmapL1TileShape = Catlass::Conv2dFmapL1Shape<8, 12, 8>;  // (hoBlock, woBlock, cin1BlockSmall)
    using FilterL1TileShape = Catlass::Conv2dFilterL1Shape<96, 8>; // (coutBlock, cin1BlockBig)
    using L0TileShape = Catlass::Conv2dL0Shape<16, 96, 16>;        // (mL0, nL0, kL0)

    using FmapType = Gemm::GemmType<half, LayoutFmap>;
    using FilterType = Gemm::GemmType<half, LayoutFilter>;
    using OutputType = Gemm::GemmType<half, LayoutOutput>;

    using BlockConv2d = Conv::Block::BlockConv2d<
        DispatchPolicy, FmapL1TileShape, FilterL1TileShape, L0TileShape, FmapType, FilterType, OutputType>;
    using BlockEpilogue = void;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Conv::Block::Conv2dIdentityBlockSwizzle<3, 0>;

    // kernel level
    using Conv2dKernel = Conv::Kernel::BasicConv2d<BlockConv2d, BlockEpilogue, BlockScheduler>;

    using Conv2dAdapter = Conv::Device::DeviceConv<Conv2dKernel>;
    Conv2dKernel::Arguments arguments{options.problemParams, deviceFmap, deviceFilter, deviceOutput};
    Conv2dAdapter conv2d_op;
    conv2d_op.CanImplement(arguments);
    size_t sizeWorkspace = conv2d_op.GetWorkspaceSize(arguments);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    conv2d_op.Initialize(arguments, deviceWorkspace);
    conv2d_op(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    std::vector<fp16_t> hostOutput(lenOutput);
    ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenOutput);
    golden::ComputeConv2d(
        options.problemParams, hostFmap, layoutFmap, hostFilter, layoutFilter, hostGolden, layoutOutput
    );

    std::vector<uint64_t> errorIndices = golden::CompareData(hostOutput, hostGolden, cin1 * kh * kw * c0);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceFmap));
    ACL_CHECK(aclrtFree(deviceFilter));
    ACL_CHECK(aclrtFree(deviceOutput));

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