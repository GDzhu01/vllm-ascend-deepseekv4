/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mla_tiling.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "catlass/detail/alignment.hpp"

using namespace std;
namespace MLATiling {
using AddrOffsets = struct AddressOffsetInfo {
    uint64_t addrQSeqOffset = 0;
    uint64_t addrQSeqRopeOffset = 0;
    uint64_t addrMaskBatchOffset = 0;
    uint64_t addrOFdSeqOffset = 0;
    uint64_t addrLSeqOffset = 0;
};

inline uint32_t GetHigh32Bit(uint64_t v) {
    return static_cast<uint32_t>(v >> NUM32);
}
inline uint32_t GetLow32Bit(uint64_t v) {
    return static_cast<uint32_t>(v);
}

void GetAddrOffsetMLA(uint32_t *tilingHost, const AddrOffsets addrOffsets, const int32_t tilingOffset) {
    // Calculate address offset
    tilingHost[tilingOffset + NUM4] = GetHigh32Bit(addrOffsets.addrQSeqOffset);
    tilingHost[tilingOffset + NUM5] = GetLow32Bit(addrOffsets.addrQSeqOffset);
    tilingHost[tilingOffset + NUM6] = GetHigh32Bit(addrOffsets.addrQSeqRopeOffset);
    tilingHost[tilingOffset + NUM7] = GetLow32Bit(addrOffsets.addrQSeqRopeOffset);
    tilingHost[tilingOffset + NUM8] = GetHigh32Bit(addrOffsets.addrMaskBatchOffset);
    tilingHost[tilingOffset + NUM9] = GetLow32Bit(addrOffsets.addrMaskBatchOffset);
}

void GetMLATilingCommon(const MLAInfo &mlaInfo, uint32_t &blockDim, uint32_t *tilingHost) {
    // Calculate the batch-related tiling parameters
    int32_t maxKVSeqlen = 0;
    int32_t maxQSeqlen = 0;
    AddrOffsets addrOffsets{};
    for (int32_t seqIdx = 0; seqIdx < mlaInfo.batch; seqIdx++) {
        int32_t qSeqLen = *(mlaInfo.qSeqLen + seqIdx);
        qSeqLen = (*(mlaInfo.kvSeqLen + seqIdx) == 0) ? 0 : qSeqLen;
        maxQSeqlen = std::max(maxQSeqlen, qSeqLen);
        int32_t kvSeqlen = *(mlaInfo.kvSeqLen + seqIdx);
        maxKVSeqlen = std::max(maxKVSeqlen, kvSeqlen);
        int32_t tilingOffset = TILING_HEAD_SIZE + TILING_PARA_SIZE * seqIdx;
        tilingHost[tilingOffset] = static_cast<uint32_t>(qSeqLen);
        tilingHost[tilingOffset + NUM1] = static_cast<uint32_t>(kvSeqlen);
        tilingHost[tilingOffset + NUM3] = static_cast<uint32_t>(mlaInfo.blockSize);
        GetAddrOffsetMLA(tilingHost, addrOffsets, tilingOffset);
        uint64_t addressOffset = static_cast<uint64_t>(mlaInfo.numHeads * mlaInfo.embeddingSize * qSeqLen);
        uint64_t addressMaskOffset = static_cast<uint64_t>(mlaInfo.maxKvSeqlen * qSeqLen);
        uint64_t addressOffsetRope = static_cast<uint64_t>(mlaInfo.numHeads * mlaInfo.embeddingSizeRope * qSeqLen);
        addrOffsets.addrQSeqOffset += addressOffset;
        addrOffsets.addrQSeqRopeOffset += addressOffsetRope;
        addrOffsets.addrMaskBatchOffset += addressMaskOffset;
    }
    tilingHost[TILING_MAX_KVSEQLEN] = maxKVSeqlen;
    tilingHost[TILING_MAX_QSEQLEN] = maxQSeqlen;
}

void GetMLATilingSpec(const MLAInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingHost) {
    // Tp1 senario specialization
    // Treat every Q token with 128 heads as one process, regardless of the mtp depth
    int32_t prevTaskNum = 0;
    int32_t maxKVSeqlen = 0;
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batch; seqIdx++) {
        int32_t qSeqLen = mmInfo.qSeqLen == nullptr ? 1 : *(mmInfo.qSeqLen + seqIdx);
        int32_t kvSeqlen = *(mmInfo.kvSeqLen + seqIdx);
        maxKVSeqlen = std::max(maxKVSeqlen, kvSeqlen);
        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq++) {
            int32_t tilingOffset = TILING_HEAD_SIZE + PARA_TILING_ELENUM_SPEC * prevTaskNum;
            tilingHost[tilingOffset] = seqIdx;
            tilingHost[tilingOffset + NUM1] = prevTaskNum;
            tilingHost[tilingOffset + NUM2] = kvSeqlen;
            prevTaskNum++;
        }
    }
    tilingHost[TILING_MAX_KVSEQLEN] = maxKVSeqlen;
}

int32_t GetQNBlockTile(const MLAInfo &mlaInfo, int32_t qSeqLen, uint32_t specStrategyFlag) {
    int32_t tokenNum = qSeqLen;
    if (specStrategyFlag) {
        tokenNum = NUM1;
    }
    int32_t tileListIdx = static_cast<int32_t>(std::ceil(std::log2(tokenNum)));
    tileListIdx = (tileListIdx > NUM5) ? NUM5 : tileListIdx;
    int32_t qNBlockTile = QN_TILE_LIST[tileListIdx];
    int32_t group = mlaInfo.numHeads / mlaInfo.kvHeads;
    qNBlockTile = (qNBlockTile > group) ? group : qNBlockTile;
    return qNBlockTile;
}

void GetTilingHead(
    const MLAInfo &mlaInfo,
    uint32_t *tilingHost,
    const uint32_t *torPtr,
    int32_t maxQseqlen,
    uint32_t specStrategyFlag
) {
    // Calculating tiling parameters
    tilingHost[TILING_BATCH] = static_cast<uint32_t>(mlaInfo.batch);
    tilingHost[TILING_HEADSIZE] = static_cast<uint32_t>(TILING_HEAD_SIZE);
    if (specStrategyFlag) {
        tilingHost[TILING_PARASIZE] = static_cast<uint32_t>(PARA_TILING_ELENUM_SPEC);
    } else {
        tilingHost[TILING_PARASIZE] = static_cast<uint32_t>(TILING_PARA_SIZE);
    }
    tilingHost[TILING_NUMHEADS] = static_cast<uint32_t>(mlaInfo.numHeads);
    tilingHost[TILING_HEADDIM] = static_cast<uint32_t>(mlaInfo.embeddingSize);
    tilingHost[TILING_NUMBLOKS] = static_cast<uint32_t>(mlaInfo.numBlocks);
    tilingHost[TILING_BLOCKSIZE] = static_cast<uint32_t>(mlaInfo.blockSize);
    int32_t maxNumBlocksPerQuery = (mlaInfo.maxKvSeqlen + mlaInfo.blockSize - 1) / mlaInfo.blockSize;
    tilingHost[TILING_MAXBLOCKS] = static_cast<uint32_t>(maxNumBlocksPerQuery);
    tilingHost[TILING_TOR] = *torPtr;
    tilingHost[TILING_KVHEADS] = mlaInfo.kvHeads;
    int32_t curQNBlockTile = GetQNBlockTile(mlaInfo, maxQseqlen, specStrategyFlag);
    int32_t curQNBlockNum = (mlaInfo.numHeads + curQNBlockTile - 1) / curQNBlockTile;
    tilingHost[TILING_HEAD_SPLIT_SIZE] = static_cast<uint32_t>(curQNBlockTile);
    tilingHost[TILING_HEAD_SPLIT_NUM] = static_cast<uint32_t>(curQNBlockNum);
    tilingHost[TILING_MASKTYPE] = static_cast<uint32_t>(mlaInfo.maskType);
    tilingHost[TILING_HEADDIM_ROPE] = static_cast<uint32_t>(mlaInfo.embeddingSizeRope);
    tilingHost[TILING_TOTAL_QTOKENS] = static_cast<uint32_t>(mlaInfo.numTokens);
}

uint32_t GetKVSplitParam(const MLAInfo &mlaInfo, uint32_t &blockDim, uint32_t *tilingHost) {
    // Calculate the tiling parameters related to flash decoding
    bool isKVSplit = (tilingHost[TILING_MAX_KVSEQLEN] >= blockDim * KV_SEQLEN_SLICE * NUM2)
                     && (tilingHost[TILING_BATCH] <= blockDim * SPLITKV_RATION && tilingHost[TILING_MAX_QSEQLEN] == 1);
    if (tilingHost[TILING_NUMHEADS] == NUM128 || !isKVSplit) {
        tilingHost[TILING_KVCORENUM] = 1;
        tilingHost[TILING_KVSPLIT] = tilingHost[TILING_MAX_KVSEQLEN];
        std::cout << "TILING_KVSPLIT = " << tilingHost[TILING_KVSPLIT] << std::endl;
        std::cout << "TILING_KVCORENUM = " << tilingHost[TILING_KVCORENUM] << std::endl;
        return tilingHost[TILING_BATCH];
    }

    uint32_t decoderBatch = tilingHost[TILING_BATCH];
    uint32_t process = std::lcm(decoderBatch, blockDim);
    uint32_t kvSplitCoreNum = process / decoderBatch;

    uint32_t kvSeqlenMaxAlign = RoundUp(tilingHost[TILING_MAX_KVSEQLEN], static_cast<uint32_t>(mlaInfo.blockSize));
    uint32_t kvSeqBlockNum = kvSeqlenMaxAlign / mlaInfo.blockSize;
    uint32_t kvBlockPerCore = CeilDiv(kvSeqBlockNum, kvSplitCoreNum);
    uint32_t kvSplitPerCore = kvBlockPerCore * mlaInfo.blockSize;
    kvSplitCoreNum = CeilDiv(tilingHost[TILING_MAX_KVSEQLEN], kvSplitPerCore);

    tilingHost[TILING_KVSPLIT] = kvSplitPerCore;
    tilingHost[TILING_KVCORENUM] = kvSplitCoreNum;
    std::cout << "TILING_KVSPLIT = " << tilingHost[TILING_KVSPLIT] << std::endl;
    std::cout << "TILING_KVCORENUM = " << tilingHost[TILING_KVCORENUM] << std::endl;

    // Set lOffsetInfo and OfdOffsetInfo
    AddrOffsets addrOffsets;
    for (int32_t seqIdx = 0; seqIdx < mlaInfo.batch; seqIdx++) {
        int32_t qSeqlen = 1;
        qSeqlen = (*(mlaInfo.kvSeqLen + seqIdx) == 0) ? 0 : qSeqlen;
        int32_t tilingOffset = seqIdx * TILING_PARA_SIZE + TILING_HEAD_SIZE;
        tilingHost[tilingOffset + NUM11] = GetHigh32Bit(addrOffsets.addrLSeqOffset);
        tilingHost[tilingOffset + NUM12] = GetLow32Bit(addrOffsets.addrLSeqOffset);
        tilingHost[tilingOffset + NUM13] = GetHigh32Bit(addrOffsets.addrOFdSeqOffset);
        tilingHost[tilingOffset + NUM14] = GetLow32Bit(addrOffsets.addrOFdSeqOffset);
        addrOffsets.addrLSeqOffset += static_cast<uint64_t>(mlaInfo.numHeads * qSeqlen * kvSplitCoreNum);
        addrOffsets.addrOFdSeqOffset += static_cast<uint64_t>(mlaInfo.numHeads * qSeqlen * mlaInfo.embeddingSize);
    }

    return decoderBatch * kvSplitCoreNum;
}

uint32_t GetKVSplitParamSpec(const MLAInfo &mlaInfo, uint32_t &blockDim, uint32_t *tilingHost) {
    // Tp1 senario specialization
    // Calculate the tiling parameters related to flash decoding
    uint32_t totalTaskNumSpec = tilingHost[TILING_TOTAL_QTOKENS];

    uint32_t formerTaskNum = totalTaskNumSpec;
    uint32_t tailTaskNum = 0;

    uint32_t processLoop = totalTaskNumSpec / blockDim;
    formerTaskNum = processLoop * blockDim;
    tailTaskNum = totalTaskNumSpec - formerTaskNum;

    if (tailTaskNum >= blockDim * SPLITKV_RATION) {
        formerTaskNum = totalTaskNumSpec;
        tailTaskNum = 0;
    }

    tilingHost[TILING_FORMERTASKNUM] = formerTaskNum;
    tilingHost[TILING_TAILTASKNUM] = tailTaskNum;
    std::cout << "TILING_FORMERTASKNUM = " << tilingHost[TILING_FORMERTASKNUM] << std::endl;
    std::cout << "TILING_TAILTASKNUM = " << tilingHost[TILING_TAILTASKNUM] << std::endl;

    if (tailTaskNum == 0) {
        tilingHost[TILING_KVCORENUM] = 1;
        tilingHost[TILING_KVSPLIT] = tilingHost[TILING_MAX_KVSEQLEN];
        std::cout << "TILING_KVSPLIT = " << tilingHost[TILING_KVSPLIT] << std::endl;
        std::cout << "TILING_KVCORENUM = " << tilingHost[TILING_KVCORENUM] << std::endl;
        return blockDim;
    }

    uint32_t process = std::lcm(tailTaskNum, blockDim);
    uint32_t kvSplitCoreNum = process / tailTaskNum;

    uint32_t kvSeqlenMaxAlign = RoundUp(tilingHost[TILING_MAX_KVSEQLEN], static_cast<uint32_t>(mlaInfo.blockSize));
    uint32_t kvSeqBlockNum = kvSeqlenMaxAlign / mlaInfo.blockSize;
    uint32_t kvBlockPerCore = CeilDiv(kvSeqBlockNum, kvSplitCoreNum);
    uint32_t kvSplitPerCore = kvBlockPerCore * mlaInfo.blockSize;
    kvSplitCoreNum = CeilDiv(tilingHost[TILING_MAX_KVSEQLEN], kvSplitPerCore);

    tilingHost[TILING_KVSPLIT] = kvSplitPerCore;
    tilingHost[TILING_KVCORENUM] = kvSplitCoreNum;
    std::cout << "TILING_KVSPLIT = " << tilingHost[TILING_KVSPLIT] << std::endl;
    std::cout << "TILING_KVCORENUM = " << tilingHost[TILING_KVCORENUM] << std::endl;

    // Set lOffsetInfo and OfdOffsetInfo
    AddrOffsets addrOffsets;
    int32_t prevTaskNum = 0;
    for (int32_t seqIdx = 0; seqIdx < mlaInfo.batch; seqIdx++) {
        int32_t qSeqLen = mlaInfo.qSeqLen == nullptr ? 1 : *(mlaInfo.qSeqLen + seqIdx);
        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq++) {
            int32_t tilingOffset = TILING_HEAD_SIZE + PARA_TILING_ELENUM_SPEC * prevTaskNum;
            tilingHost[tilingOffset + NUM11] = GetHigh32Bit(addrOffsets.addrLSeqOffset);
            tilingHost[tilingOffset + NUM12] = GetLow32Bit(addrOffsets.addrLSeqOffset);
            tilingHost[tilingOffset + NUM13] = GetHigh32Bit(addrOffsets.addrOFdSeqOffset);
            tilingHost[tilingOffset + NUM14] = GetLow32Bit(addrOffsets.addrOFdSeqOffset);
            addrOffsets.addrLSeqOffset += static_cast<uint64_t>(mlaInfo.numHeads * kvSplitCoreNum);
            addrOffsets.addrOFdSeqOffset += static_cast<uint64_t>(mlaInfo.numHeads * mlaInfo.embeddingSize);
            prevTaskNum++;
        }
    }

    return tailTaskNum * kvSplitCoreNum;
}

int32_t GetMLATilingParam(const MLAInfo &mlaInfo, uint32_t &blockDim, uint32_t *tilingHost) {
    if (tilingHost == nullptr || mlaInfo.qSeqLen == nullptr || mlaInfo.kvSeqLen == nullptr) {
        cerr << "[ERROR] pointer tilingHost or seq is nullptr." << endl;
        return -1;
    }
    if (mlaInfo.blockSize != NUM128) {
        cerr << "[ERROR] blockSize != 128 is not supported." << endl;
        return -1;
    }
    int32_t maxQseqlen = 0;
    int32_t totalKvNumtokens = 0;
    for (int32_t seqIdx = 0; seqIdx < mlaInfo.batch; seqIdx++) {
        int32_t qSeqLen = *(mlaInfo.qSeqLen + seqIdx);
        if (qSeqLen > NUM4) {
            cerr << "[ERROR] qSeqLen > 4 is not supported." << endl;
        }
        int32_t kvSeqLen = *(mlaInfo.kvSeqLen + seqIdx);
        qSeqLen = (kvSeqLen == 0) ? 0 : qSeqLen;
        maxQseqlen = std::max(qSeqLen, maxQseqlen);
        totalKvNumtokens += kvSeqLen;
    }
    if (totalKvNumtokens > mlaInfo.numBlocks * mlaInfo.blockSize) {
        cerr << "[ERROR] the number of K and V tokens is too big to fit in the paged cache." << endl;
        return -1;
    }
    float tor = static_cast<float>(1.0 / sqrt(1.0 * (mlaInfo.embeddingSize + mlaInfo.embeddingSizeRope)));
    uint32_t *torPtr = reinterpret_cast<uint32_t *>(&tor);
    uint32_t specStrategyFlag = (mlaInfo.numHeads == NUM128) ? 1 : 0;
    if (specStrategyFlag) {
        GetMLATilingSpec(mlaInfo, blockDim, tilingHost);
    } else {
        GetMLATilingCommon(mlaInfo, blockDim, tilingHost);
    }
    GetTilingHead(mlaInfo, tilingHost, torPtr, maxQseqlen, specStrategyFlag);
    if (specStrategyFlag) {
        GetKVSplitParamSpec(mlaInfo, blockDim, tilingHost);
    } else {
        GetKVSplitParam(mlaInfo, blockDim, tilingHost);
    }
    return 0;
}
} // namespace MLATiling