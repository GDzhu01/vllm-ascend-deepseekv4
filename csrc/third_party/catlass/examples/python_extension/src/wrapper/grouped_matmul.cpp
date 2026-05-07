/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/torch.h>

#include "catlass_kernel.h"
#include "wrapper/catlass_kernel_wrapper.h"
#include "wrapper/common.h"

namespace CatlassKernelWrapper::GroupedMatmulLike {
using namespace CatlassKernel;
using OutputType = at::Tensor;
KernelInfo GetKernelInfo(const at::Tensor &mat1, const at::Tensor &mat2, const at::Tensor &groupList,
                         const std::string &outDType, const bool transA, const bool transB, const bool splitK)
{
    KernelInfo kernelInfo;
    // set input addr
    kernelInfo.inputAddr.resize(3);
    kernelInfo.inputAddr[0] = static_cast<uint8_t *>(mat1.data_ptr());
    kernelInfo.inputAddr[1] = static_cast<uint8_t *>(mat2.data_ptr());
    kernelInfo.inputAddr[2] = static_cast<uint8_t *>(groupList.data_ptr());
    // calculate groupList sum
    at::Tensor groupListHost = groupList.to(torch::kCPU).to(torch::kInt64);
    std::vector<int64_t> groupListVec(groupListHost.data_ptr<int64_t>(),
                                      groupListHost.data_ptr<int64_t>() + groupListHost.numel());
    kernelInfo.g = groupListVec.size();
    int64_t groupListSum = groupListVec[kernelInfo.g - 1];

    std::vector<int64_t> matAShape(mat1.sizes().vec());
    std::vector<int64_t> matBShape(mat2.sizes().vec());
    kernelInfo.inputDataType = TorchDtypeToAclDtype(mat1.scalar_type());
    kernelInfo.outputDataType = TypeStrToAclDtype(outDType);
    // set transpose
    TransposeStatus transposeStatus1 = GetTransposeStatus(mat1);
    TransposeStatus transposeStatus2 = GetTransposeStatus(mat2);
    if (transposeStatus1 == TransposeStatus::NON_CONTINUOUS) {
        throw std::runtime_error("mat1 is not contiguous");
    }
    if (transposeStatus2 == TransposeStatus::NON_CONTINUOUS) {
        throw std::runtime_error("mat2 is not contiguous");
    }
    kernelInfo.transA = transA ? true : static_cast<bool>(transposeStatus1);
    kernelInfo.transB = transB ? true : static_cast<bool>(transposeStatus2);

    int64_t m, k1, k2, n, M, K, N, g, K1, K2;
    if (splitK) {
        kernelInfo.split = KernelInfo::GMMSplit::SPLIT_K;
    } else {
        kernelInfo.split = KernelInfo::GMMSplit::SPLIT_M;
    }
    switch (kernelInfo.split) {
        case KernelInfo::GMMSplit::SPLIT_M:
            // [M, k]@[g, k, n]->[M, n]
            if (matAShape.size() != 2) {
                throw std::runtime_error("dim num of mat1 should be 2 (M, k)");
            }
            if (matBShape.size() != 3) {
                throw std::runtime_error("dim num of mat2 should be 3 (G, k, n)");
            }
            if (kernelInfo.transA) {
                throw std::runtime_error("mat1 transpose is not supported");
            }
            if (kernelInfo.transB) {
                matBShape = {matBShape[0], matBShape[2], matBShape[1]};
            }
            M = matAShape[0];
            k1 = matAShape[1];
            g = matBShape[0];
            k2 = matBShape[1];
            n = matBShape[2];
            if (g != kernelInfo.g) {
                throw std::runtime_error("mat2[0](g) should be equal to groupNum");
            }
            if (M != groupListSum) {
                throw std::runtime_error("mat1[0](M) should be equal to groupListSum");
            }
            if (k1 != k2) {
                throw std::runtime_error("k unequal");
            }
            kernelInfo.M = M;
            kernelInfo.k = k1;
            kernelInfo.n = n;
            break;
        case KernelInfo::GMMSplit::SPLIT_K:
            // [m, K]@[K, n]->[g, m, n]
            if (matAShape.size() != 2) {
                throw std::runtime_error("dim num of mat1 should be 2(m, K)");
            }
            if (matBShape.size() != 2) {
                throw std::runtime_error("dim num of mat2 should be 2(K, n)");
            }
            if (!kernelInfo.transA) {
                throw std::runtime_error("mat1 must be transposed");
            } else {
                matAShape = {matAShape[1], matAShape[0]};
            }
            if (kernelInfo.transB) {
                throw std::runtime_error("mat2 transpose is not supported");
            }
            m = matAShape[0];
            K1 = matAShape[1];
            K2 = matBShape[0];
            n = matBShape[1];
            g = kernelInfo.g;
            if (K1 != K2) {
                throw std::runtime_error("K unequal");
            }
            K = K1;
            if (K != groupListSum) {
                throw std::runtime_error("mat1[1](K) should be equal to groupListSum");
            }
            kernelInfo.K = K;
            kernelInfo.m = m;
            kernelInfo.n = n;
            break;
    }
    return kernelInfo;
};
OutputType AllocOutput(KernelInfo &kernelInfo)
{
    OutputType output;
    if (kernelInfo.split == KernelInfo::GMMSplit::SPLIT_M) {
        output = GetOutputTensor({kernelInfo.M, kernelInfo.n}, AclDtypeToTorchDtype(kernelInfo.outputDataType));
    }
    if (kernelInfo.split == KernelInfo::GMMSplit::SPLIT_K) {
        output = GetOutputTensor({kernelInfo.g, kernelInfo.m, kernelInfo.n},
                                 AclDtypeToTorchDtype(kernelInfo.outputDataType));
    }
    kernelInfo.outputAddr.resize(1);
    kernelInfo.outputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(output.storage().data()));
    return output;
};
} // namespace CatlassKernelWrapper::GroupedMatmulLike