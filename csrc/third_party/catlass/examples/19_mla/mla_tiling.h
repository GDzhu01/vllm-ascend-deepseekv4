/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EXAMPLES_MLA_MLA_TILING_H
#define CATLASS_EXAMPLES_MLA_MLA_TILING_H

#include <array>
#include <cstdint>

namespace MLATiling {
const int32_t TILING_BATCH = 0;
const int32_t TILING_NUMHEADS = 1;
const int32_t TILING_HEADDIM = 2;
const int32_t TILING_NUMBLOKS = 3;
const int32_t TILING_BLOCKSIZE = 4;
const int32_t TILING_MAXBLOCKS = 5;
const int32_t TILING_TOR = 6;
const int32_t TILING_KVHEADS = 7;
const int32_t TILING_HEADSIZE = 8;
const int32_t TILING_PARASIZE = 9;
const int32_t TILING_HEAD_SPLIT_SIZE = 10;
const int32_t TILING_HEAD_SPLIT_NUM = 11;
const int32_t TILING_MASKTYPE = 12;
const int32_t TILING_HEADDIM_ROPE = 13;
const int32_t TILING_MAX_KVSEQLEN = 14;
const int32_t TILING_KVSPLIT = 15;
const int32_t TILING_KVCORENUM = 16;
const int32_t TILING_MAX_QSEQLEN = 17;
const int32_t TILING_TOTAL_QTOKENS = 18;
const int32_t TILING_FORMERTASKNUM = 19;
const int32_t TILING_TAILTASKNUM = 20;

const int32_t TILING_HEAD_SIZE = 24;
const int32_t TILING_PARA_SIZE = 17;

const int32_t PARA_TILING_ELENUM_SPEC = 17;

const int32_t NUM0 = 0;
const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM5 = 5;
const int32_t NUM6 = 6;
const int32_t NUM7 = 7;
const int32_t NUM8 = 8;
const int32_t NUM9 = 9;
const int32_t NUM10 = 10;
const int32_t NUM11 = 11;
const int32_t NUM12 = 12;
const int32_t NUM13 = 13;
const int32_t NUM14 = 14;
const int32_t NUM15 = 15;
const int32_t NUM16 = 16;
const int32_t NUM17 = 17;
const int32_t NUM18 = 18;
const int32_t NUM19 = 19;
const int32_t NUM20 = 20;
const int32_t NUM21 = 21;
const int32_t NUM32 = 32;
const int32_t NUM64 = 64;
const int32_t NUM128 = 128;
const int32_t NUM256 = 256;
const int32_t NUM512 = 512;
const int32_t NUM576 = 576;
const int32_t EMBEDDING_LIMIT = 512;
const int32_t WORKSPACE_BLOCK_SIZE_DB = 65536;

const float SPLITKV_RATION = 0.8;
const int32_t KV_SEQLEN_SLICE = 128;

constexpr std::array<int32_t, NUM6> QN_TILE_LIST = {128, 64, 32, 16, 8, 1};

enum class MaskType {
    NO_MASK = 0,
    MASK_SPEC = 1
};

struct MLAInfo {
    int32_t numTokens = 0;
    int32_t numHeads = 0;
    int32_t embeddingSize = 0;
    int32_t embeddingSizeRope = 0;
    int32_t numBlocks = 0;
    int32_t blockSize = 0;
    int32_t maxKvSeqlen = 0;
    int32_t kvHeads = 0;
    int32_t batch = 0;
    int32_t *kvSeqLen{nullptr};
    int32_t *qSeqLen{nullptr};
    MaskType maskType = MaskType::NO_MASK;
};

int32_t GetMLATilingParam(const MLAInfo &mlaInfo, uint32_t &blockDim, uint32_t *tilingHost);
} // namespace MLATiling
#endif