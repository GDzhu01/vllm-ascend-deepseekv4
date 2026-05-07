/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_OP_RUNNER_H
#define CATLASS_TUNER_OP_RUNNER_H

#include "op_config.h"

namespace Catlass {

enum class OpRunStatus : uint32_t {
    SUCCESS = 0,
    FAILED,
    FATAL,
};

enum class KernelType : uint32_t {
    CACHE_CLEAR = 0,
    OPERATOR,
};

class OpLauncher {
public:
    explicit OpLauncher(const std::shared_ptr<OpConfig>& opConfig, Library::Operation *op, uint32_t aicCoreNum)
        : opConfig_(opConfig), op_(op), aicCoreNum_(aicCoreNum) {}
    OpRunStatus operator()(void* stream, int times = 1, bool sync = true);
    OpRunStatus Init();

private:

    const std::shared_ptr<OpConfig>& opConfig_;
    Library::Operation *op_;
    uint8_t *workspace_{nullptr};
    uint32_t aicCoreNum_;
};

} // namespace Catlass
#endif // CATLASS_TUNER_OP_RUNNER_H
