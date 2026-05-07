/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_launcher.h"
#include <thread>
#include "device_memory_manager.h"

namespace Catlass {

OpRunStatus OpLauncher::operator()(void* stream, int times, bool sync)
{
    for (int i = 0; i < times; ++i) {
        auto status = op_->Run(stream, aicCoreNum_, DeviceMemoryManager::Instance().GetFftsAddr());
        if (status != Status::kSuccess) {
            LOGE("Operator %s run failed", op_->GetDescription().name);
            return OpRunStatus::FATAL;
        }
        if (sync) {
            constexpr int SYNC_TIME = 1000;
            auto err = aclrtSynchronizeStreamWithTimeout(stream, SYNC_TIME);
            if (err != ACL_SUCCESS) {
                LOGE("Operator %s run failed, aclrtSynchronizeStreamWithTimeout ret: %d",
                     op_->GetDescription().name, err);
                return OpRunStatus::FATAL;
            }
        }
    }
    return OpRunStatus::SUCCESS;
}

OpRunStatus OpLauncher::Init()
{
    if (!opConfig_->InitArgument(op_)) {
        LOGE("Initialize device resource failed");
        return OpRunStatus::FAILED;
    }

    auto status = op_->CanImplement(opConfig_->GetArg(), opConfig_->GetConfig());
    if (status != Status::kSuccess) {
        LOGE("Call CanImplement failed");
        return OpRunStatus::FAILED;
    }

    DeviceMemoryParam param{reinterpret_cast<void**>(&workspace_), 0};
    param.size = op_->GetWorkspaceSize(opConfig_->GetArg(), opConfig_->GetConfig());
    if (param.size > 0 && !DeviceMemoryManager::Instance().MallocWorkspace(param)) {
        LOGE("Malloc workspace failed");
        return OpRunStatus::FAILED;
    }

    status = op_->Initialize(opConfig_->GetArg(), opConfig_->GetConfig(), reinterpret_cast<uint8_t*>(workspace_));
    if (status != Status::kSuccess) {
        LOGE("Operator Initialize failed");
        return OpRunStatus::FAILED;
    }
    return OpRunStatus::SUCCESS;
}
} // namespace Catlass
