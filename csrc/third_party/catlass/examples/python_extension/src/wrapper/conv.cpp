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
namespace CatlassKernelWrapper::ConvLike {
using namespace CatlassKernel;
using OutputType = at::Tensor;

torch::Tensor GetConvOutputTensor(const std::vector<int64_t> &shape, const torch::Dtype dtype) {
    at::TensorOptions options = at::TensorOptions();
    options = options.dtype(dtype).layout(at::kStrided).requires_grad(false).device(torch_npu::utils::get_npu_device_type());
    return at_npu::native::empty_with_format(shape, options, ACL_FORMAT_NDC1HWC0);
}

OutputType AllocOutput(ConvKernelInfo &kernelInfo)
{
    int64_t n = kernelInfo.fmapRelated[0];
    int64_t di = kernelInfo.fmapRelated[1];
    int64_t hi = kernelInfo.fmapRelated[3];
    int64_t wi = kernelInfo.fmapRelated[4];
    int64_t cout = kernelInfo.filterRelated[3];
    int64_t kd = kernelInfo.filterRelated[0];
    int64_t kh = kernelInfo.filterRelated[1];
    int64_t kw = kernelInfo.filterRelated[2];

    int64_t Do = (di + kernelInfo.padList[0] * 2 - kernelInfo.dilationList[0] * (kd - 1) - 1) / kernelInfo.strideList[0] + 1;
    int64_t Ho = (hi + kernelInfo.padList[1] * 2 - kernelInfo.dilationList[1] * (kh - 1) - 1) / kernelInfo.strideList[1] + 1;
    int64_t Wo = (wi + kernelInfo.padList[2] * 2 - kernelInfo.dilationList[2] * (kw - 1) - 1) / kernelInfo.strideList[2] + 1;

    OutputType output = GetConvOutputTensor({n, cout, Do, Ho, Wo}, AclDtypeToTorchDtype(kernelInfo.outputDataType));
    kernelInfo.outputAddr.resize(1);
    kernelInfo.outputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(output.storage().data()));
    return output;
}

ConvKernelInfo GetKernelInfo(const at::Tensor &fmap, const at::Tensor &filter, const at::Tensor &bias,
                             const std::vector<int64_t> &strideList, const std::vector<int64_t> &padList,
                             const std::vector<int64_t> &dilationList, const std::string &outDType)
{
    ConvKernelInfo kernelInfo;

    kernelInfo.inputAddr.resize(3);
    kernelInfo.inputAddr[0] = static_cast<uint8_t *>(fmap.data_ptr());
    kernelInfo.inputAddr[1] = static_cast<uint8_t *>(filter.data_ptr());
    kernelInfo.inputAddr[2] = static_cast<uint8_t *>(bias.data_ptr());

    kernelInfo.fmapRelated.resize(6);

    kernelInfo.fmapRelated[0] = at_npu::native::get_npu_storage_sizes(fmap)[0];
    kernelInfo.fmapRelated[1] = at_npu::native::get_npu_storage_sizes(fmap)[1];
    kernelInfo.fmapRelated[2] = at_npu::native::get_npu_storage_sizes(fmap)[2];
    kernelInfo.fmapRelated[3] = at_npu::native::get_npu_storage_sizes(fmap)[3];
    kernelInfo.fmapRelated[4] = at_npu::native::get_npu_storage_sizes(fmap)[4];
    kernelInfo.fmapRelated[5] = at_npu::native::get_npu_storage_sizes(fmap)[5];
    kernelInfo.filterRelated.resize(4);
    kernelInfo.filterRelated[0] = filter.sizes().at(2);
    kernelInfo.filterRelated[1] = filter.sizes().at(3);
    kernelInfo.filterRelated[2] = filter.sizes().at(4);
    kernelInfo.filterRelated[3] = filter.sizes().at(0);
    kernelInfo.strideList.resize(3);
    kernelInfo.strideList[0] = strideList[0];
    kernelInfo.strideList[1] = strideList[1];
    kernelInfo.strideList[2] = strideList[2];
    kernelInfo.padList.resize(3);
    kernelInfo.padList[0] = padList[0];
    kernelInfo.padList[1] = padList[1];
    kernelInfo.padList[2] = padList[2];
    kernelInfo.dilationList.resize(3);
    kernelInfo.dilationList[0] = dilationList[0];
    kernelInfo.dilationList[1] = dilationList[1];
    kernelInfo.dilationList[2] = dilationList[2];

    kernelInfo.inputDataType = TorchDtypeToAclDtype(fmap.scalar_type());
    kernelInfo.biasDataType = TorchDtypeToAclDtype(bias.scalar_type());
    kernelInfo.outputDataType = TypeStrToAclDtype(outDType);
    return kernelInfo;
}
} // namespace CatlassKernelWrapper::MatmulLike