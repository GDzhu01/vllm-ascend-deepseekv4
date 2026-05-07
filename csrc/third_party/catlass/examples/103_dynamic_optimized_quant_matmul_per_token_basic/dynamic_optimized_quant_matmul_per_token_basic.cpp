/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include "golden.hpp"
#include "helper.hpp"
#include "catlass/layout/layout.hpp"
#include "dynamic_optimized_matmul_w8a8.h"


void Run(aclrtStream &stream, uint32_t m, uint32_t n, uint32_t k, LayoutTag layoutTagA, LayoutTag layoutTagB,
    PlatformInfo &platformInfo)
{
    uint32_t m_ori = m;
    uint32_t n_ori = n;
    uint32_t k_ori = k;
    
    LayoutTag layoutTagC = LayoutTag::TagRowMajor;
    TilingParams tilingParams{m, n, k, layoutTagA, layoutTagB, layoutTagC};
    DoTilingAndSelectKernel<fp16_t>(tilingParams, platformInfo);
    PrintTilingParams<fp16_t>(tilingParams, platformInfo);

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;
    size_t lenScale = static_cast<size_t>(n);
    size_t lenPerTokenScale = static_cast<size_t>(m);

    size_t sizeA = lenA * sizeof(int8_t);
    size_t sizeB = lenB * sizeof(int8_t);
    size_t sizeC = lenC * sizeof(fp16_t);
    size_t sizeScale = lenScale * sizeof(float);
    size_t sizePerTokenScale = lenPerTokenScale * sizeof(float);

    std::vector<int8_t> hostA(lenA);
    std::vector<int8_t> hostB(lenB);
    std::vector<fp16_t> hostC(lenC);
    std::vector<float> hostScale(lenScale);
    std::vector<float> hostPerTokenScale(lenPerTokenScale);

    Catlass::golden::FillRandomData<int8_t>(hostA, -16, 16);
    Catlass::golden::FillRandomData<int8_t>(hostB, -16, 16);

    Catlass::golden::FillRandomData(hostScale, 0.0, 1.0);
    Catlass::golden::FillRandomData(hostPerTokenScale, 0.0, 1.0);

    uint8_t *dA, *dB, *dC, *dW, *dScale, *dPerTokenScale, *dTilingParams;

    ACL_CHECK(aclrtMalloc((void **)&dA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&dB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&dC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&dScale, sizeScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&dPerTokenScale, sizePerTokenScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&dTilingParams, sizeof(TilingParams), ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(dA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dScale, sizeScale, hostScale.data(), sizeScale, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dPerTokenScale, sizePerTokenScale, hostPerTokenScale.data(),
        sizePerTokenScale, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t workspaceSize = DynamicOptimizedMatmulGetWorkspace(tilingParams);
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc((void **)&dW, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    ACL_CHECK(aclrtMemcpy(
        dTilingParams, sizeof(TilingParams), &tilingParams, sizeof(TilingParams), ACL_MEMCPY_HOST_TO_DEVICE));

    ExecuteDynamicOptimizedMatmul(stream, fftsAddr, dA, dB, dC, dW, dScale, dPerTokenScale, dTilingParams, tilingParams);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::layout::VectorLayout layoutScale{n};
    Catlass::layout::VectorLayout layoutPerTokenScale{m};
    if (layoutTagA == LayoutTag::TagRowMajor && layoutTagB == LayoutTag::TagRowMajor) {
        Catlass::layout::RowMajor layoutA{m, k};
        Catlass::layout::RowMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::QuantMatmul(
            problemShape, hostA, layoutA, hostB, layoutB, 
            hostScale, layoutScale, hostPerTokenScale, layoutPerTokenScale, hostGolden, layoutC
        );
    } else if (layoutTagA == LayoutTag::TagRowMajor && layoutTagB == LayoutTag::TagColumnMajor) {
        Catlass::layout::RowMajor layoutA{m, k};
        Catlass::layout::ColumnMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::QuantMatmul(
            problemShape, hostA, layoutA, hostB, layoutB, 
            hostScale, layoutScale, hostPerTokenScale, layoutPerTokenScale, hostGolden, layoutC
        );
    } else if (layoutTagA == LayoutTag::TagColumnMajor && layoutTagB == LayoutTag::TagRowMajor) {
        Catlass::layout::ColumnMajor layoutA{m, k};
        Catlass::layout::RowMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::QuantMatmul(
            problemShape, hostA, layoutA, hostB, layoutB, 
            hostScale, layoutScale, hostPerTokenScale, layoutPerTokenScale, hostGolden, layoutC
        );
    } else {
        Catlass::layout::ColumnMajor layoutA{m, k};
        Catlass::layout::ColumnMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::QuantMatmul(
            problemShape, hostA, layoutA, hostB, layoutB, 
            hostScale, layoutScale, hostPerTokenScale, layoutPerTokenScale, hostGolden, layoutC
        );
    }
    std::vector<uint64_t> errorIndices = Catlass::golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(dA));
    ACL_CHECK(aclrtFree(dB));
    ACL_CHECK(aclrtFree(dC));
    ACL_CHECK(aclrtFree(dScale));
    ACL_CHECK(aclrtFree(dPerTokenScale));
    ACL_CHECK(aclrtFree(dTilingParams));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(dW));
    }
}

int main(int argc, const char **argv)
{
    const uint32_t deviceId = std::atoi(argv[argc - 1]);
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclInit(nullptr));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    PlatformInfo platformInfo;

    uint32_t m = std::atoi(argv[1]);
    uint32_t n = std::atoi(argv[2]);
    uint32_t k = std::atoi(argv[3]);

    LayoutTag layoutTagA = static_cast<LayoutTag>(std::atoi(argv[4]));
    LayoutTag layoutTagB = static_cast<LayoutTag>(std::atoi(argv[5]));

    Run(stream, m, n, k, layoutTagA, layoutTagB, platformInfo);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
}
