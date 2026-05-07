/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>

#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "wrapper/common.h"

namespace CatlassKernelWrapper {
torch::Tensor GetOutputTensor(const std::vector<int64_t> &shape, const torch::Dtype dtype)
{
    at::TensorOptions options = at::TensorOptions();
    options =
        options.dtype(dtype).layout(at::kStrided).requires_grad(false).device(torch_npu::utils::get_npu_device_type());
    return at_npu::native::empty_with_format(shape, options, ACL_FORMAT_ND);
}

torch::Dtype TypeStrToTorchDtype(const std::string &typeStr)
{
    static const std::unordered_map<std::string, torch::Dtype> mapper = {{"float32", torch::kFloat32},
                                                                         {"float16", torch::kFloat16},
                                                                         {"int8", torch::kInt8},
                                                                         {"int32", torch::kInt32},
                                                                         {"bf16", torch::kBFloat16}};
    auto iter = mapper.find(typeStr);
    return iter != mapper.end() ? iter->second : torch::kFloat16;
}

aclDataType TorchDtypeToAclDtype(const torch::Dtype torchDtype)
{
    static const std::unordered_map<torch::Dtype, aclDataType> mapper = {{torch::kFloat32, ACL_FLOAT},
                                                                         {torch::kFloat16, ACL_FLOAT16},
                                                                         {torch::kInt8, ACL_INT8},
                                                                         {torch::kInt32, ACL_INT32},
                                                                         {torch::kBFloat16, ACL_BF16}};
    auto iter = mapper.find(torchDtype);
    return iter != mapper.end() ? iter->second : ACL_FLOAT16;
};

aclDataType TypeStrToAclDtype(const std::string &typeStr)
{
    return TorchDtypeToAclDtype(TypeStrToTorchDtype(typeStr));
};

torch::Dtype AclDtypeToTorchDtype(const aclDataType aclDtype)
{
    static const std::map<aclDataType, torch::Dtype> mapper = {{ACL_FLOAT16, torch::kFloat16},
                                                               {ACL_FLOAT, torch::kFloat32},
                                                               {ACL_INT32, torch::kInt32},
                                                               {ACL_INT8, torch::kInt8},
                                                               {ACL_BF16, torch::kBFloat16}};
    auto iter = mapper.find(aclDtype);
    return iter != mapper.end() ? iter->second : torch::kFloat16;
};

TransposeStatus GetTransposeStatus(const at::Tensor &mat)
{
    if (mat.is_contiguous()) {
        return TransposeStatus::NO_TRANSPOSE;
    }
    std::vector<int64_t> strides = mat.strides().vec();
    std::vector<int64_t> shape = mat.sizes().vec();
    int64_t dimA = shape.at(shape.size() - 2);
    int64_t dimB = shape.at(shape.size() - 1);
    int64_t strideA = strides.at(strides.size() - 2);
    int64_t strideB = strides.at(strides.size() - 1);
    if (strideB == dimA && strideA == 1) {
        return TransposeStatus::TRANSPOSE;
    }
    return TransposeStatus::NON_CONTINUOUS;
}
} // namespace CatlassKernelWrapper