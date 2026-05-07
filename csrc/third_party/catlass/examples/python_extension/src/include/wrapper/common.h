/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PY_EXT_COMMON_H
#define PY_EXT_COMMON_H

#include <acl/acl.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace CatlassKernelWrapper {
torch::Tensor GetOutputTensor(const std::vector<int64_t> &shape, const torch::Dtype dtype);
torch::Dtype TypeStrToTorchDtype(const std::string &typeStr);
aclDataType TypeStrToAclDtype(const std::string &typeStr);
torch::Dtype AclDtypeToTorchDtype(const aclDataType aclDtype);
aclDataType TorchDtypeToAclDtype(const torch::Dtype torchDtype);

enum class TransposeStatus : uint32_t {
    NO_TRANSPOSE = 0,
    TRANSPOSE = 1,
    NON_CONTINUOUS = 2
};

TransposeStatus GetTransposeStatus(const at::Tensor &mat);
} // namespace CatlassKernelWrapper

#endif