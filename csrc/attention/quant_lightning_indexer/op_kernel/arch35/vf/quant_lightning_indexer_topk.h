/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_topk.h
 * \brief
 */
#ifndef quant_lightning_indexer_TOPK_H
#define quant_lightning_indexer_TOPK_H

#include "kernel_operator.h"
#include "vf_topk.h"

namespace topk {

class LITopk {
public:
    static __aicore__ inline uint32_t GetSharedTmpBufferSize(uint32_t topK)
    {
        return 2 * topK * sizeof(uint32_t) + 5 * 256 * sizeof(uint32_t) + 64 * sizeof(uint32_t);
    }

    __aicore__ inline void Init(uint32_t topK, LocalTensor<uint32_t>& sharedTmpBuffer)
    {
        this->topK = topK;
        tmpIdxLocal = sharedTmpBuffer[0];
        tmpValueLocal = tmpIdxLocal[topK];
        histogramsLocal = tmpValueLocal[topK];
        idx0Local = histogramsLocal[256];
        idx1Local = idx0Local[256]; 
        idx2Local = idx1Local[256]; 
        idx3Local = idx2Local[256]; 
        nkValueLocal = idx3Local[256];
    }

    __aicore__ inline void operator()(LocalTensor<uint32_t>& outputIdxLocal,
                                      LocalTensor<uint32_t>& outputValueLocal,
                                      LocalTensor<uint32_t>& inputLocal,
                                      uint32_t s2SeqLen)
    {
        LiTopKVF(outputIdxLocal, // filter阶段使用输出value Buf topK * 4B
                 outputValueLocal, // filter阶段使用输出 Idx Buf topK * 4B
                 inputLocal, // 输入 s2SeqLen * 4B
                 tmpIdxLocal, // filter阶段使用暂存index Buf topK * 4B
                 tmpValueLocal, // filter阶段使用暂存value Buf topK * 4B
                 histogramsLocal, // 直方图的临时Buf 256 * 4B
                 idx0Local, // 输入数据第1个8位Buf 256 * 4B
                 idx1Local, // 输入数据第2个8位Buf 256 * 4B
                 idx2Local, // 输入数据第3个8位Buf 256 * 4B
                 idx3Local, // 输入数据第4个8位Buf 256 * 4B
                 nkValueLocal, // next_k 暂存Buf 64 * 4B
                 topK,       // topk数量
                 s2SeqLen); // 输入元素总数
    }
private:
    LocalTensor<uint32_t> tmpIdxLocal;     // filter阶段使用暂存index Buf topK * 4B
    LocalTensor<uint32_t> tmpValueLocal;   // filter阶段使用暂存value Buf topK * 4B
    LocalTensor<uint32_t> histogramsLocal; // 直方图的临时Buf 256 * 4B
    LocalTensor<uint32_t> idx0Local;       // 输入数据第1个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx1Local;       // 输入数据第2个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx2Local;       // 输入数据第3个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx3Local;       // 输入数据第4个8位Buf 256 * 4B
    LocalTensor<uint32_t> nkValueLocal; // next_k 暂存Buf 64 * 4B
    uint32_t topK;
};
}
#endif