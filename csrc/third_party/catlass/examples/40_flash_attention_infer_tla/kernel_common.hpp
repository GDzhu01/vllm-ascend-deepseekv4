/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_COMMON
#define KERNEL_COMMON

constexpr uint32_t QK_READY_ID = 1;
constexpr uint32_t SOFTMAX_READY_ID = 2;
constexpr uint32_t PV_READY_ID = 3;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
constexpr uint32_t TMP_SIZE_DECODER = 32768;

constexpr int32_t TILING_BATCH = 0;
constexpr int32_t TILING_NUMHEADS = 1;
constexpr int32_t TILING_HEADDIM = 2;
constexpr int32_t TILING_NUMBLOKS = 3;
constexpr int32_t TILING_BLOCKSIZE = 4;
constexpr int32_t TILING_MAXBLOCKS = 5;
constexpr int32_t TILING_TOR = 6;
constexpr int32_t TILING_KVHEADS = 7;
constexpr int32_t TILING_HEADSIZE = 8;
constexpr int32_t TILING_PARASIZE = 9;
constexpr int32_t TILING_HEAD_SPLIT_SIZE = 10;
constexpr int32_t TILING_HEAD_SPLIT_NUM = 11;
constexpr int32_t TILING_HEADDIM_ROPE = 13;
constexpr int32_t TILING_MAX_KVSEQLEN = 14;
constexpr int32_t TILING_KVSPLIT = 15;
constexpr int32_t TILING_KVCORENUM = 16;
constexpr int32_t TILING_TOTAL_QTOKENS = 18;
constexpr int32_t TILING_FORMERTASKNUM = 19;
constexpr int32_t TILING_TAILTASKNUM = 20;
constexpr int32_t TILING_BLOCKSIZE_CALC = 25;
constexpr int32_t TILING_HEADDIM_K_SPLIT = 38;
constexpr int32_t TILING_HEADDIM_V_SPLIT = 39;
constexpr int32_t TILING_HEADDIM_V_SPLIT_VECTOR_FORMER = 40;
constexpr int32_t TILING_HEADDIM_V_SPLIT_VECTOR_TAIL = 41;

constexpr int32_t NUM1 = 1;
constexpr int32_t NUM4 = 4;
constexpr int32_t NUM64 = 64;
constexpr int32_t NUM512 = 512;
constexpr int32_t NUM576 = 576;

constexpr uint32_t FLOAT_VECTOR_SIZE = 64;

constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;

template <typename T>
CATLASS_DEVICE T AlignUp(T a, T b) {
    return (b == 0) ? 0 : (a + b - 1) / b * b;
}

template <typename T>
CATLASS_DEVICE T Min(T a, T b) {
    return (a > b) ? b : a;
}

template <typename T>
CATLASS_DEVICE T Max(T a, T b) {
    return (a > b) ? a : b;
}

enum class cvPipeLineType {
    FAI_COMMON_NORMAL = 0,
    FAI_COMMON_CHUNK_MASK = 1
};

CATLASS_DEVICE
uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize) {
    uint32_t qNBlockTile = (128 / qSeqlen) / 2 * 2;
    qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
    qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
    return qNBlockTile;
}

CATLASS_DEVICE
uint32_t GetQSBlockTile(uint32_t kvSeqlen) {
    uint32_t qSBlockTile = 128;
    return qSBlockTile;
}

struct FATilingData {
    uint32_t numHeads = 0;
    uint32_t embeddingSize = 0;
    uint32_t numBlocks = 0;
    uint32_t blockSize = 0;
    uint32_t maxKvSeqlen = 0;
    uint32_t kvHeads = 0;
    uint32_t batch = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    uint32_t firstBatchTaskNum = 0;
    uint32_t totalTaskNum = 0;
    uint32_t maskType = 0;
    uint64_t mm1OutSize = 0;
    uint64_t smOnlineOutSize = 0;
    uint64_t mm2OutSize = 0;
    uint64_t UpdateSize = 0;
    uint64_t workSpaceSize = 0;
    float scaleValue = 0.0;
};

struct FAIKernelParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR o;
    GM_ADDR s;
    GM_ADDR p;
    GM_ADDR oTemp;
    GM_ADDR oUpdate;
    GM_ADDR tiling;
    // Methods
    CATLASS_DEVICE
    FAIKernelParams() {
    }
    CATLASS_DEVICE
    FAIKernelParams(GM_ADDR q_,
                    GM_ADDR k_,
                    GM_ADDR v_,
                    GM_ADDR mask_,
                    GM_ADDR blockTables_,
                    GM_ADDR actualQseqlen_,
                    GM_ADDR actualKvseqlen_,
                    GM_ADDR o_,
                    GM_ADDR s_,
                    GM_ADDR p_,
                    GM_ADDR oTemp_,
                    GM_ADDR oUpdate_,
                    GM_ADDR tiling_)
        : q(q_)
        , k(k_)
        , v(v_)
        , mask(mask_)
        , blockTables(blockTables_)
        , actualQseqlen(actualQseqlen_)
        , actualKvseqlen(actualKvseqlen_)
        , o(o_)
        , s(s_)
        , p(p_)
        , oTemp(oTemp_)
        , oUpdate(oUpdate_)
        , tiling(tiling_) {
    }
};

#endif