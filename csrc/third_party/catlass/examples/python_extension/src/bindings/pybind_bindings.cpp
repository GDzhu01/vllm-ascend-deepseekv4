/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "catlass_kernel.h"
#include "wrapper/catlass_kernel_wrapper.h"

namespace py = pybind11;
using namespace CatlassKernelWrapper;

PYBIND11_MODULE(_C, m)
{
    m.doc() = "Python bindings for CatlassKernel";
    m.def("basic_matmul", &RunBasicMatmul, "")
        .def("grouped_matmul", &RunGroupedMatmul, "")
        .def("optimized_matmul", &RunOptimizedMatmul, "")
        .def("conv_bias", &RunConvBias, "");
}
