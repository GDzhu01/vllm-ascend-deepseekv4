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
namespace CatlassKernelWrapper::MatmulLike {
using namespace CatlassKernel;
using OutputType = at::Tensor;

OutputType AllocOutput(KernelInfo &kernelInfo)
{
    OutputType output = GetOutputTensor({kernelInfo.m, kernelInfo.n}, AclDtypeToTorchDtype(kernelInfo.outputDataType));
    kernelInfo.outputAddr.resize(1);
    kernelInfo.outputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(output.storage().data()));
    return output;
}

KernelInfo GetKernelInfo(const at::Tensor &mat1, const at::Tensor &mat2, const std::string &outDType)
{
    KernelInfo kernelInfo;
    kernelInfo.inputAddr.resize(2);
    kernelInfo.inputAddr[0] = static_cast<uint8_t *>(mat1.data_ptr());
    kernelInfo.inputAddr[1] = static_cast<uint8_t *>(mat2.data_ptr());
    int64_t m = mat1.sizes().at(0);
    int64_t k1 = mat1.sizes().at(1);
    int64_t k2 = mat2.sizes().at(0);
    int64_t n = mat2.sizes().at(1);
    if (k1 != k2) {
        std::stringstream ss;
        ss << "mat1 and mat2 shapes cannot be multiplied";
        ss << "(" << m << "x" << k1 << " and " << k2 << "x" << n << ")";
        throw std::runtime_error(ss.str());
    }
    kernelInfo.m = m;
    kernelInfo.k = k1;
    kernelInfo.n = n;
    kernelInfo.inputDataType = TorchDtypeToAclDtype(mat1.scalar_type());
    kernelInfo.outputDataType = TypeStrToAclDtype(outDType);
    TransposeStatus transposeStatus1 = GetTransposeStatus(mat1);
    TransposeStatus transposeStatus2 = GetTransposeStatus(mat2);
    if (transposeStatus1 == TransposeStatus::NON_CONTINUOUS) {
        throw std::runtime_error("mat1 is not contiguous");
    }
    if (transposeStatus2 == TransposeStatus::NON_CONTINUOUS) {
        throw std::runtime_error("mat2 is not contiguous");
    }
    kernelInfo.transA = static_cast<bool>(transposeStatus1);
    kernelInfo.transB = static_cast<bool>(transposeStatus2);
    return kernelInfo;
}
} // namespace CatlassKernelWrapper::MatmulLike