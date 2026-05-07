/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#ifndef PYEXT_RUN_NPU_FUNC_H
#define PYEXT_RUN_NPU_FUNC_H

#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#define RUN_NPU_FUNC(func, ...)                                                                                        \
    do {                                                                                                               \
        if ((func) == nullptr) {                                                                                       \
            throw std::runtime_error(                                                                                  \
                std::string("Function pointer is null at ") + __FILE__ + ":" + std::to_string(__LINE__) + " in "       \
                + #func                                                                                                \
            );                                                                                                         \
        }                                                                                                              \
        at_npu::native::OpCommand::RunOpApiV2(#func, [=]() -> int {                                                    \
            func(__VA_ARGS__);                                                                                         \
            return 0;                                                                                                  \
        });                                                                                                            \
    } while (false)
#endif