/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PLATFORM_INFO_H
#define PLATFORM_INFO_H

#include "tiling/platform/platform_ascendc.h"

struct PlatformInfo
{
    uint32_t coreNum{24};
    uint64_t ubSize{192 * 1024};
    uint64_t l1Size{512 * 1024};
    uint64_t l0ASize{64 * 1024};
    uint64_t l0BSize{64 * 1024};
    uint64_t l0CSize{128 * 1024};

    PlatformInfo()
    {
        coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::UB, ubSize);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L1, l1Size);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L0_A, l0ASize);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L0_B, l0BSize);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L0_C, l0CSize);
    }

    ~PlatformInfo() {}
};

#endif  // PLATFORM_INFO_H