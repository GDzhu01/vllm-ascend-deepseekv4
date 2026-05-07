/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
namespace FAInferTiling {
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
const int32_t WORKSPACE_BLOCK_SIZE_DB = 131072;

enum class MaskType {
    NO_MASK = 0,
    MASK_SPEC = 1,
    MASK_CAUSUAL = 2
};

struct FAInfo {
    int32_t numTokens = 0;
    int32_t numHeads = 0;
    int32_t embeddingSize = 0;
    int32_t numBlocks = 0;
    int32_t blockSize = 0;
    int32_t kvHeads = 0;
    int32_t batch = 0;
    int64_t *qSeqlenList{nullptr};
    int64_t *kvSeqlenList{nullptr};
    int64_t *qSeqlen{nullptr};
    MaskType maskType = MaskType::MASK_SPEC;
};

void FillBasicTilingData(const FAInfo &faInfo, FATilingData &faTilingData, int64_t maxKvSeqlen) {
    uint32_t maxNumBlocksPerBatch = (maxKvSeqlen + faInfo.blockSize - 1) / faInfo.blockSize;
    float scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * faInfo.embeddingSize));
    faTilingData.batch = static_cast<uint32_t>(faInfo.batch);
    faTilingData.numHeads = static_cast<uint32_t>(faInfo.numHeads);
    faTilingData.kvHeads = static_cast<uint32_t>(faInfo.kvHeads);
    faTilingData.embeddingSize = static_cast<uint32_t>(faInfo.embeddingSize);
    faTilingData.numBlocks = static_cast<uint32_t>(faInfo.numBlocks);
    faTilingData.blockSize = static_cast<uint32_t>(faInfo.blockSize);
    faTilingData.maxKvSeqlen = static_cast<uint32_t>(maxKvSeqlen);
    faTilingData.maxNumBlocksPerBatch = maxNumBlocksPerBatch;
    faTilingData.maskType = static_cast<uint32_t>(faInfo.maskType);
    faTilingData.scaleValue = scaleValue;
}

uint32_t GetQNBlockTile(int64_t qSeqlen, uint32_t groupSize) {
    uint32_t qRowNumCeil = 128;
    // A trick is used to ensure the qN tile is a even number,
    // thus most tasks have balanced workload between two vec cores,
    // and each vec core possess no more than 64 rows when all-rounded row num is no larger than 128,
    // aiding the coding of rescale block
    uint32_t qNBlockTile = (qRowNumCeil / qSeqlen) / 2 * 2;
    qNBlockTile = std::min(qNBlockTile, groupSize);
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    return qNBlockTile;
}

uint32_t GetQSBlockTile(int64_t kvSeqlen) {
    uint32_t qSBlockTile = 128;
    return qSBlockTile;
}

void FillSplitCoreTilingData(const FAInfo &faInfo, FATilingData &faTilingData) {
    uint32_t totalTaskNum = 0;
    uint32_t groupSize = faInfo.numHeads / faInfo.kvHeads;
    for (int32_t batchIdx = 0; batchIdx < faInfo.batch; batchIdx++) {
        int64_t qSeqlen = *(faInfo.qSeqlenList + batchIdx);
        int64_t kvSeqlen = *(faInfo.kvSeqlenList + batchIdx);
        uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
        uint32_t curQNBlockNum = qNBlockNumPerGroup * faInfo.kvHeads;
        uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
        uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
        uint32_t curTaskNum = curQNBlockNum * curQSBlockNum;
        if (batchIdx == 0) {
            faTilingData.firstBatchTaskNum = curTaskNum;
        }
        totalTaskNum += curTaskNum;
    }
    faTilingData.totalTaskNum = totalTaskNum;
}

void FillWorkSpaceTilingData(uint32_t blockDim, FATilingData &faTilingData) {
    uint64_t mm1OutSize = blockDim * WORKSPACE_BLOCK_SIZE_DB * NUM4 * NUM3;
    uint64_t smOnlineOutSize = blockDim * WORKSPACE_BLOCK_SIZE_DB * NUM2 * NUM3;
    uint64_t mm2OutSize = blockDim * WORKSPACE_BLOCK_SIZE_DB * NUM4 * NUM3;
    uint64_t UpdateSize = blockDim * WORKSPACE_BLOCK_SIZE_DB * NUM4 * NUM3;
    uint64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize;
    faTilingData.mm1OutSize = mm1OutSize;
    faTilingData.smOnlineOutSize = smOnlineOutSize;
    faTilingData.mm2OutSize = mm2OutSize;
    faTilingData.UpdateSize = UpdateSize;
    faTilingData.workSpaceSize = workSpaceSize;
}

int32_t GetFATilingParam(const FAInfo &faInfo, uint32_t blockDim, FATilingData &faTilingData) {
    if (faInfo.qSeqlenList == nullptr || faInfo.kvSeqlenList == nullptr) {
        cerr << "[ERROR] pointer tilingData or seq is nullptr." << endl;
        return -1;
    }
    if (faInfo.blockSize != NUM128) {
        cerr << "[ERROR] blockSize != 128 is not supported." << endl;
        return -1;
    }
    int64_t maxKvSeqlen = 0;
    for (int32_t batchIdx = 0; batchIdx < faInfo.batch; batchIdx++) {
        int64_t qSeqlen = *(faInfo.qSeqlenList + batchIdx);
        int64_t kvSeqlen = *(faInfo.kvSeqlenList + batchIdx);
        maxKvSeqlen = std::max(maxKvSeqlen, kvSeqlen);
    }
    FillBasicTilingData(faInfo, faTilingData, maxKvSeqlen);
    FillSplitCoreTilingData(faInfo, faTilingData);
    FillWorkSpaceTilingData(blockDim, faTilingData);
    return 0;
}
} // namespace FAInferTiling