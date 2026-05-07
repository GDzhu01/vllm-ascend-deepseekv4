/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_CATLASS_KERNEL_H
#define SHARED_LIB_CATLASS_KERNEL_H

#include <cstdint>
#include <vector>

#include <acl/acl.h>

namespace CatlassKernel {
using ElementGroupList = int64_t;
struct KernelInfo {
    enum class GMMSplit : uint32_t {
        SPLIT_M = 0,
        SPLIT_K = 1,
        SPLIT_N = 2
    };
    aclDataType inputDataType = aclDataType::ACL_FLOAT16;
    aclDataType outputDataType = aclDataType::ACL_FLOAT16;
    uint32_t g = 1;
    uint32_t b = 1;
    uint32_t m = 1;
    uint32_t n = 1;
    uint32_t k = 1;
    uint32_t M = 1;
    uint32_t K = 1;
    bool transA = false;
    bool transB = false;
    std::vector<ElementGroupList> groupList;
    GMMSplit split = GMMSplit::SPLIT_M;
    std::vector<uint8_t *> inputAddr;
    std::vector<uint8_t *> outputAddr;
};

struct ConvKernelInfo {
    aclDataType inputDataType = aclDataType::ACL_FLOAT16;
    aclDataType biasDataType = aclDataType::ACL_FLOAT;
    aclDataType outputDataType = aclDataType::ACL_FLOAT16;

    std::vector<uint32_t> fmapRelated;
    std::vector<uint32_t> filterRelated;
    
    std::vector<uint32_t> strideList;
    std::vector<uint32_t> padList;
    std::vector<uint32_t> dilationList;

    std::vector<uint8_t *> inputAddr;
    std::vector<uint8_t *> outputAddr;
};

void BasicMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void GroupedMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void OptimizedMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void ConvBias(const uint32_t blockNum, aclrtStream stream, ConvKernelInfo kernelInfo);
} // namespace CatlassKernel

#endif // SHARED_LIB_CATLASS_KERNEL_H