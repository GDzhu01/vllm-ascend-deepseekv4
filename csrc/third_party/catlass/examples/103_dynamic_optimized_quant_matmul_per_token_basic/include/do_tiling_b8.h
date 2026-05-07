/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADJUST_TILING_B8_H
#define ADJUST_TILING_B8_H

#include <array>
#include <opdev/bfloat16.h>
#include <opdev/fp16_t.h>
#include "utils.h"
#include "tiling_params.h"
#include "platform_info.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

void DoTilingB8Layout00(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = 128, n1 = 256, k1 = 256;

    if (n >= 256) {
        // n0 = 256 delivers optimal bandwidth performance.
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), platformInfo.coreNum);
        BalanceWorkload(m, n, m1, n1, 32, platformInfo);
        uint32_t blocks = CeilDiv(m, 64) * CeilDiv(n, 512);
        if (blocks <= maxBlocks - platformInfo.coreNum && k <= 128) {
            m1 = 64;
            n1 = 512;
        }
    } else {
        m1 = 128;
        n1 = RoundUp(n, 32);
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), platformInfo.coreNum);
        uint32_t m1t = m1;
        while (JudgeSpace<fp16_t>(m1t + 32, n1, k1, platformInfo)) {
            m1t += 32;
            uint32_t blocks = CeilDiv(m, m1t) * CeilDiv(n, n1);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                m1 = m1t;
            }
        }
        BalanceWorkload(m, n, m1, n1, 32, platformInfo);
    }
    if (k >= 65536 || n >= 65536) {
        m1 = 128;
        n1 = 256;
    }
    k1 = GetMaxK1<fp16_t>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1);
}

void DoTilingB8Layout01(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = 128, n1 = 256, k1 = 256;
    // When LayoutA is RowMajor and LayoutB is ColumnMajor, bandwidth issues can be completely disregarded,
    // simply choose the tiling configureation with the most balanced workload
    double ratio = (double)(m * k + k * n) / (m * n);
    if (m > n && (ratio > 0.1 || n < 256)) {
        m1 = 256;
        n1 = 128;
        BalanceWorkload(m, n, m1, n1, 64, platformInfo);
        BalanceWorkload(n, m, n1, m1, 64, platformInfo); 
    } else {
        BalanceWorkload(n, m, n1, m1, 64, platformInfo);
        BalanceWorkload(m, n, m1, n1, 64, platformInfo); 
    }
    uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), platformInfo.coreNum);
    if (m < n) {
        uint32_t n1t = n1;
        while (JudgeSpace<fp16_t>(m1, n1t + 32, k1, platformInfo)) {
            n1t += 32;
            uint32_t blocks = CeilDiv(m, m1) * CeilDiv(n, n1t);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                n1 = n1t;
            }
        }
        BalanceWorkload(m, n, m1, n1, 64, platformInfo);
        BalanceWorkload(n, m, n1, m1, 64, platformInfo); 
    } else {
        uint32_t m1t = m1;
        while (JudgeSpace<fp16_t>(m1t + 32, n1, k1, platformInfo)) {
            m1t += 32;
            uint32_t blocks = CeilDiv(m, m1t) * CeilDiv(n, n1);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                m1 = m1t;
            }
        }
        BalanceWorkload(n, m, n1, m1, 64, platformInfo);
        BalanceWorkload(m, n, m1, n1, 64, platformInfo); 
    }
    if (k >= 65536) {
        if (m < n || (ratio < 0.1 && n >= 256)) {
            m1 = 128;
            n1 = 256;
        } else {
            m1 = 256;
            n1 = 128;
        }
    }
    k1 = GetMaxK1<fp16_t>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1);
}

void DoTilingB8Layout10(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = 128, n1 = 256, k1 = 256;
    double ratio = (double)(m * k + k * n) / (m * n);
    if (m > n && (ratio > 0.1 || n < 256)) {
        m1 = 256;
        n1 = 128;
        k1 = 256;
    }
    if (m < m1) {
        m1 = RoundUp(m, 32);
    }
    if (n < n1) {
        n1 = RoundUp(n, 32);
    }

    uint32_t blocks = CeilDiv(m, m1) * CeilDiv(n, n1);
    if (blocks <= platformInfo.coreNum / 4) {
        if (n1 > 32) {
            n1 /= 2;
        }
        if (m1 > 32) {
            m1 /= 2;
        }
    } else if (blocks <= platformInfo.coreNum / 2) {
        if (m1 > n1) {
            m1 /= 2;
        } else if (n1 > 32) {
            n1 /= 2;
        }
    }
    if (n >= 65536 || m >= 65536) {
        if (m < n || (ratio < 0.1 && n >= 256)) {
            m1 = 128;
            n1 = 256;
        } else {
            m1 = 256;
            n1 = 128;
        }
    }
    m1 = m1 / 32 * 32;
    n1 = n1 / 32 * 32;
    k1 = GetMaxK1<fp16_t>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1); 
}

void DoTilingB8Layout11(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = 256, n1 = 128, k1 = 256;

    if (m >= 256) {
        // n0 = 256 delivers optimal bandwidth performance.
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), platformInfo.coreNum);
        BalanceWorkload(n, m, n1, m1, 32, platformInfo);
        uint32_t blocks = CeilDiv(n, 64) * CeilDiv(m, 512);
        if (blocks <= maxBlocks - platformInfo.coreNum && k <= 128) {
            n1 = 64;
            m1 = 512;
        }
    } else {
        n1 = 128;
        m1 = RoundUp(m, 32);
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), platformInfo.coreNum);
        uint32_t n1t = n1;
        while (JudgeSpace<fp16_t>(n1t + 32, m1, k1, platformInfo)) {
            n1t += 32;
            uint32_t blocks = CeilDiv(n, n1t) * CeilDiv(m, m1);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                n1 = n1t;
            }
        }
        BalanceWorkload(n, m, n1, m1, 32, platformInfo);
    }
    if (k >= 65536 || m >= 65536) {
        m1 = 256;
        n1 = 128;
    }
    // fixpipe bound
    double ratio = (double)(m * k + k * n) / (m * n);
    if (ratio < 0.1 && n >= 256) {
        m1 = 128;
        n1 = 256;
    }
    k1 = GetMaxK1<fp16_t>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1);
}

using FuncType = void (*)(TilingParams &tilingParams, PlatformInfo& platformInfo);
std::array<std::array<FuncType, 2>, 2> DoTilingB8 = {
    {{{DoTilingB8Layout00, DoTilingB8Layout01}}, {{DoTilingB8Layout10, DoTilingB8Layout11}}}};
#endif  // ADJUST_TILING_B8_H