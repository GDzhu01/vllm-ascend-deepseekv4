/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file COMPRESSOR_tiling_datay.h
 * \brief
 */

#ifndef COMPRESSOR_TILING_DATA_H
#define COMPRESSOR_TILING_DATA_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

const uint32_t CMP_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

namespace optiling {
    // 1. 基础参数结构体
    struct CompressorBaseParams {   
        uint32_t batchSize;             // bastch size（批大小）
        uint32_t seqSize;               // sequence size（kvs大小）  
        uint32_t hiddenSize;            // hidden size（隐藏层大小）
        uint32_t tokenSize;             // token size = batchSize * seqSize(token总数：批大小x序列1长度)
        uint32_t headDim;               // head size of kv
        uint32_t ropeHeadDim;           // dim size per rope head 64（单个带RoPE头的维度）
        uint32_t csSize;                // Compress sequence len
        uint32_t cmpRatio;              // Compress ratio
        uint32_t cgSize;                // Compress group size
        float normEps;                  // RMSNorm eps
        float reciprocalD;              // 1分之D
        uint32_t usedCoreNum;           // 使用核数
        uint32_t nSize;                 // 控制v2积攒的轮数
    };

    struct CompressorPageAttentionParams {
        uint32_t blockNum;
        uint32_t blockSize;
        uint32_t maxBlockNumPerBatch;
    };

    struct CompressorOuterSplitParams {
        uint32_t bEnd[CMP_MAX_AIC_CORE_NUM];
        uint32_t sEnd[CMP_MAX_AIC_CORE_NUM];
        uint32_t dEnd[CMP_MAX_AIC_CORE_NUM];
    };

    struct CompressorInnerSplitParams {
        uint32_t mBaseSize;
        uint32_t dBaseSize;
    };

    struct CompressorWorkspaceParams {
        uint32_t preMm1ResSize;
        uint32_t curMm1ResSize;
        uint32_t vec1ResSize;
    };

    struct CompressorTilingData {
        CompressorBaseParams baseParams;
        CompressorPageAttentionParams pageAttentionParams;
        CompressorOuterSplitParams outerSplitParams;
        CompressorInnerSplitParams innerSplitParams;
        CompressorWorkspaceParams workspaceParams;
    };
} // optiling

#endif  // COMPRESSOR_TILING_DATA_H