/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PY_EXT_CONV_H
#define PY_EXT_CONV_H

#include <torch/torch.h>

#include "catlass_kernel.h"
#include "wrapper/catlass_kernel_wrapper.h"
#include "wrapper/common.h"
namespace CatlassKernelWrapper::ConvLike {
using namespace CatlassKernel;
using OutputType = at::Tensor;

OutputType AllocOutput(ConvKernelInfo &kernelInfo);
ConvKernelInfo GetKernelInfo(const at::Tensor &fmap, const at::Tensor &filter, const at::Tensor &bias,
                             const std::vector<int64_t> &strideList, const std::vector<int64_t> &padList,
                             const std::vector<int64_t> &dilationList, const std::string &outDType);
} // namespace CatlassKernelWrapper::ConvLike
#endif