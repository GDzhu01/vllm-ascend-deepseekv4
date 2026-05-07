/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "kernel_common.hpp"

using namespace Catlass;

/*
This example demonstrates how to compute mla.
*/
template <
    class BlockMmadQK,
    class BlockMmadPV,
    class EpilogueMLASoftmax,
    class EpilogueMLARescaleO,
    class EpilogueMLAFDRescaleO>
class MLAKernel {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;

    using ElementMask = half;

    using ElementO = typename EpilogueMLARescaleO::ElementOutput;
    using LayoutO = typename EpilogueMLARescaleO::LayoutOutput;

    using ElementOTmp = typename EpilogueMLARescaleO::ElementInput;
    using LayoutOTmp = typename EpilogueMLARescaleO::LayoutInput;

    using ElementUpdate = typename EpilogueMLARescaleO::ElementUpdate;
    using LayoutUpdate = typename EpilogueMLARescaleO::LayoutUpdate;

    static constexpr uint32_t KV_SPLIT_MAX = EpilogueMLAFDRescaleO::KV_SPLIT_MAX;
    static constexpr uint32_t HEADS_PROCESS_MAX = EpilogueMLAFDRescaleO::HEADS_PROCESS_MAX;
    static constexpr uint32_t COMPUTE_ELE_NUM = EpilogueMLAFDRescaleO::COMPUTE_ELE_NUM;

    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR q;
        GM_ADDR qRope;
        GM_ADDR k;
        GM_ADDR kRope;
        GM_ADDR blockTables;
        GM_ADDR o;
        GM_ADDR s;
        GM_ADDR p;
        GM_ADDR oTmp;
        GM_ADDR oUpdate;
        GM_ADDR oCoreTmp;
        GM_ADDR l;
        GM_ADDR tiling;

        // Methods
        CATLASS_DEVICE
        Params() {
        }

        CATLASS_DEVICE
        Params(
            GM_ADDR q_,
            GM_ADDR qRope_,
            GM_ADDR k_,
            GM_ADDR kRope_,
            GM_ADDR blockTables_,
            GM_ADDR o_,
            GM_ADDR s_,
            GM_ADDR p_,
            GM_ADDR oTmp_,
            GM_ADDR oUpdate_,
            GM_ADDR oCoreTmp_,
            GM_ADDR l_,
            GM_ADDR tiling_
        )
            : q(q_)
            , qRope(qRope_)
            , k(k_)
            , kRope(kRope_)
            , blockTables(blockTables_)
            , o(o_)
            , s(s_)
            , p(p_)
            , oTmp(oTmp_)
            , oUpdate(oUpdate_)
            , oCoreTmp(oCoreTmp_)
            , l(l_)
            , tiling(tiling_) {
        }
    };

    // Methods
    CATLASS_DEVICE
    MLAKernel() {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);

        // Get the memory offset address of the input on Global Memory
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);

        AscendC::GlobalTensor<ElementQ> gQRope;
        gQRope.SetGlobalBuffer((__gm__ ElementQ *)params.qRope);

        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);

        AscendC::GlobalTensor<ElementK> gKRope;
        gKRope.SetGlobalBuffer((__gm__ ElementK *)params.kRope);

        AscendC::GlobalTensor<int32_t> gblockTable;
        gblockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));

        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTmp);
        AscendC::GlobalTensor<uint32_t> gTiling;
        gTiling.SetGlobalBuffer((__gm__ uint32_t *)params.tiling);

        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource);

        // Get tiling parameters
        uint32_t batch = gTiling.GetValue(TILING_BATCH);
        uint32_t qHeads = gTiling.GetValue(TILING_NUMHEADS);
        uint32_t embed = gTiling.GetValue(TILING_HEADDIM);
        uint32_t embedRope = gTiling.GetValue(TILING_HEADDIM_ROPE);
        uint32_t blockSize = gTiling.GetValue(TILING_BLOCKSIZE);
        uint32_t maxNumBlocksPerQuery = gTiling.GetValue(TILING_MAXBLOCKS);
        uint32_t kvHeads = gTiling.GetValue(TILING_KVHEADS);
        uint32_t tilingHeadSize = gTiling.GetValue(TILING_HEADSIZE);
        uint32_t tilingParaSize = gTiling.GetValue(TILING_PARASIZE);
        uint32_t curQheadSplitSize = gTiling.GetValue(TILING_HEAD_SPLIT_SIZE);
        uint32_t curQheadSplitNum = gTiling.GetValue(TILING_HEAD_SPLIT_NUM);
        uint32_t kvSplitPerCore = gTiling.GetValue(TILING_KVSPLIT);
        uint32_t kvSplitCoreNum = gTiling.GetValue(TILING_KVCORENUM);

        uint32_t strideQO = qHeads * embed;
        uint32_t strideQORope = qHeads * embedRope;
        uint32_t strideKV = static_cast<uint64_t>(kvHeads) * embed;
        uint32_t strideKVRope = static_cast<uint64_t>(kvHeads) * embedRope;
        uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t processNum = batch * curQheadSplitNum * kvSplitCoreNum;
        // Go through each task
        for (uint32_t process = coreIdx; process < processNum; process += uint32_t(coreNum)) {
            // Get the offset of each core on the GM
            uint32_t curBatch = process / (curQheadSplitNum * kvSplitCoreNum);
            uint32_t offsetTiling = tilingHeadSize + tilingParaSize * curBatch;
            uint32_t qSeqlen = gTiling.GetValue(offsetTiling);
            uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 1);
            uint32_t qAddrHigh = gTiling.GetValue(offsetTiling + 4);
            uint32_t qAddrLow = gTiling.GetValue(offsetTiling + 5);
            uint64_t qAddr = (uint64_t)(((uint64_t)qAddrHigh) << 32 | qAddrLow);
            uint32_t qRopeAddrHigh = gTiling.GetValue(offsetTiling + 6);
            uint32_t qRopeAddrLow = gTiling.GetValue(offsetTiling + 7);
            uint64_t qRopeAddr = (uint64_t)(((uint64_t)qRopeAddrHigh) << 32 | qRopeAddrLow);

            if (kvSeqlen == 0) {
                continue;
            }
            uint32_t qHeadSplitIdx = (process % (curQheadSplitNum * kvSplitCoreNum)) / kvSplitCoreNum;
            uint32_t qHeadSplitSizeActual = (qHeadSplitIdx == (curQheadSplitNum - 1))
                                                ? (qHeads - qHeadSplitIdx * curQheadSplitSize)
                                                : curQheadSplitSize;
            uint32_t curStartHeadIdx = qHeadSplitIdx * curQheadSplitSize;
            uint64_t gQOffset = qAddr + curStartHeadIdx * embed;
            uint64_t gQRopeOffset = qRopeAddr + curStartHeadIdx * embedRope;

            uint32_t kvSeqlenAlign = RoundUp(kvSeqlen, blockSize);
            uint32_t curNIdx = process % kvSplitCoreNum;
            uint32_t curKVSeqlen = kvSplitPerCore;
            uint32_t kvLoop = CeilDiv(kvSeqlen, kvSplitPerCore);
            if (curNIdx >= kvLoop) {
                continue;
            }
            if (curNIdx == (kvLoop - 1)) {
                curKVSeqlen = kvSeqlen - curNIdx * kvSplitPerCore;
            }
            uint32_t startKV = curNIdx * kvSplitPerCore;

            uint32_t tokenNumPerHead = qSeqlen;
            uint32_t seqTile = blockSize;
            uint32_t nLoop = (curKVSeqlen + seqTile - 1) / seqTile;
            uint32_t kSeqTile = seqTile;
            uint32_t kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
            uint32_t vSeqTile = seqTile;
            uint32_t vSeqTileRound = RoundUp<BLOCK_SIZE>(vSeqTile);

            uint32_t rowNum = qHeadSplitSizeActual * tokenNumPerHead;
            uint32_t rowNumRound = RoundUp<BLOCK_SIZE>(rowNum);

            // Split k seqlen
            for (uint32_t nIdx = 0; nIdx < nLoop + 1; nIdx++) {
                if (nIdx != nLoop) {
                    if (nIdx == (nLoop - 1)) {
                        kSeqTile = (curKVSeqlen - nIdx * seqTile);
                        kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
                    }
                    LayoutQ layoutQ(rowNum, embed);
                    LayoutQ layoutQRope(rowNum, embedRope);
                    LayoutK layoutK(embed, kSeqTile);
                    LayoutK layoutKRope(embedRope, kSeqTile);
                    LayoutS layoutS(rowNumRound, kSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embed + embedRope};
                    MatrixCoord qShapeSingleNd{qHeadSplitSizeActual, embed};
                    uint32_t qkPingPongFlag = nIdx % 2;
                    // Get blockTableId
                    int32_t blockTableId = gblockTable.GetValue(
                        curBatch * maxNumBlocksPerQuery + startKV / blockSize + nIdx
                    );
                    uint64_t kvOffset = (uint64_t)blockTableId * blockSize * strideKV;
                    uint64_t kvOffsetRope = (uint64_t)blockTableId * blockSize * strideKVRope;
                    uint64_t gSOffset = (uint64_t)coreIdx * TMP_SIZE_DECODER
                                        + (uint64_t)qkPingPongFlag * TMP_SIZE_DECODER / 2;
                    // Calculate a Q * K^T in advance
                    blockMmadQK(
                        gQ[gQOffset], gQRope[gQRopeOffset], gK[kvOffset], gKRope[kvOffsetRope], gS[gSOffset], layoutQ,
                        layoutQRope, layoutK, layoutKRope, layoutS, actualBlockShapeQK, qShapeSingleNd, qHeads, nIdx
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                }
                // Because a Q * K^T is calculated in advance, the first round is skipped.
                if (nIdx != 0) {
                    if (nIdx == nLoop) {
                        vSeqTile = (curKVSeqlen - (nIdx - 1) * seqTile);
                        vSeqTileRound = RoundUp<BLOCK_SIZE>(vSeqTile);
                    }
                    LayoutP layoutP(rowNum, vSeqTile, vSeqTileRound);
                    LayoutV layoutV(embed, vSeqTile);
                    LayoutOTmp layoutOTmp(rowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed, vSeqTile};
                    uint32_t pvPingPongFlag = (nIdx - 1) % 2;
                    uint64_t gPOffset = (uint64_t)coreIdx * TMP_SIZE + (uint64_t)pvPingPongFlag * TMP_SIZE / 2;
                    uint64_t gOTmpOffset = (uint64_t)coreIdx * TMP_SIZE * 2 + (uint64_t)pvPingPongFlag * TMP_SIZE;
                    // Calculate P * V
                    blockMmadPV(
                        gP[gPOffset], gOTmp[gOTmpOffset], layoutP, layoutV, layoutOTmp, actualBlockShapePV, nIdx,
                        softmaxReady
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                }
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID5);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

        // Get the memory offset address of the input on Global Memory
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTmp);
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)params.oUpdate);
        AscendC::GlobalTensor<ElementOTmp> gOCoreTmp;
        gOCoreTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oCoreTmp);
        AscendC::GlobalTensor<ElementOTmp> gl;
        gl.SetGlobalBuffer((__gm__ ElementOTmp *)params.l);
        AscendC::GlobalTensor<uint32_t> gTiling;
        gTiling.SetGlobalBuffer((__gm__ uint32_t *)params.tiling);
        AscendC::GlobalTensor<float> gTilingFp64;
        gTilingFp64.SetGlobalBuffer((__gm__ float *)params.tiling);

        // Get tiling parameters
        uint32_t batch = gTiling.GetValue(TILING_BATCH);
        uint32_t qHeads = gTiling.GetValue(TILING_NUMHEADS);
        uint32_t embed = gTiling.GetValue(TILING_HEADDIM);
        uint32_t blockSize = gTiling.GetValue(TILING_BLOCKSIZE);
        float tor = gTilingFp64.GetValue(TILING_TOR);
        uint32_t kvHeads = gTiling.GetValue(TILING_KVHEADS);
        uint32_t tilingHeadSize = gTiling.GetValue(TILING_HEADSIZE);
        uint32_t tilingParaSize = gTiling.GetValue(TILING_PARASIZE);
        uint32_t curQheadSplitSize = gTiling.GetValue(TILING_HEAD_SPLIT_SIZE);
        uint32_t curQheadSplitNum = gTiling.GetValue(TILING_HEAD_SPLIT_NUM);
        uint32_t kvSplitPerCore = gTiling.GetValue(TILING_KVSPLIT);
        uint32_t kvSplitCoreNum = gTiling.GetValue(TILING_KVCORENUM);

        uint32_t strideQO = qHeads * embed;
        uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);
        uint32_t glFlag = 1;

        EpilogueMLASoftmax epilogueMLASoftmax(resource, tor, kvSplitCoreNum);
        EpilogueMLARescaleO epilogueMLARescaleO(resource, kvSplitCoreNum);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t processNum = batch * curQheadSplitNum * kvSplitCoreNum;
        // Go through each task.
        for (uint32_t process = coreIdx; process < processNum; process += uint32_t(coreNum)) {
            // Get the offset of each core on the GM.
            uint32_t curBatch = process / (curQheadSplitNum * kvSplitCoreNum);
            uint32_t offsetTiling = tilingHeadSize + tilingParaSize * curBatch;
            uint32_t qSeqlen = gTiling.GetValue(offsetTiling);
            uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 1);
            uint32_t oAddrHigh32 = gTiling.GetValue(offsetTiling + 4);
            uint32_t oAddrLow32 = gTiling.GetValue(offsetTiling + 5);
            uint64_t oAddr = (uint64_t)(((uint64_t)oAddrHigh32) << 32 | oAddrLow32);
            if (kvSeqlen == 0) {
                continue;
            }
            uint32_t qHeadSplitIdx = (process % (curQheadSplitNum * kvSplitCoreNum)) / kvSplitCoreNum;
            uint32_t qHeadSplitSizeActual = (qHeadSplitIdx == (curQheadSplitNum - 1))
                                                ? (qHeads - qHeadSplitIdx * curQheadSplitSize)
                                                : curQheadSplitSize;
            uint32_t curStartHeadIdx = qHeadSplitIdx * curQheadSplitSize;
            uint64_t gmOffsetO = oAddr + curStartHeadIdx * embed;

            uint32_t kvSeqlenAlign = RoundUp(kvSeqlen, blockSize);
            uint32_t curNIdx = process % kvSplitCoreNum;
            uint32_t curKVSeqlen = kvSplitPerCore;
            uint32_t kvLoop = CeilDiv(kvSeqlen, kvSplitPerCore);
            if (curNIdx >= kvLoop) {
                continue;
            }
            if (curNIdx == (kvLoop - 1)) {
                curKVSeqlen = kvSeqlen - curNIdx * kvSplitPerCore;
            }

            uint32_t seqTile = blockSize;
            uint32_t tokenNumPerHead = qSeqlen;
            uint32_t nLoop = (curKVSeqlen + seqTile - 1) / seqTile;

            uint32_t kSeqTile = seqTile;
            uint32_t kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
            uint32_t vSeqTile = seqTile;
            uint32_t vSeqTileRound = RoundUp<BLOCK_SIZE>(vSeqTile);

            uint32_t rowNum = tokenNumPerHead * qHeadSplitSizeActual;
            uint32_t rowNumRound = RoundUp<BLOCK_SIZE>(rowNum);

            uint32_t oFdOffset = 0;
            uint32_t lOffset = 0;
            if (kvSplitCoreNum != 1) {
                uint32_t lAddrHigh32 = gTiling.GetValue(offsetTiling + 11);
                uint32_t lAddrLow32 = gTiling.GetValue(offsetTiling + 12);
                uint64_t lAddr = (uint64_t)(((uint64_t)lAddrHigh32) << 32 | lAddrLow32);
                uint32_t oFdAddrHigh32 = gTiling.GetValue(offsetTiling + 13);
                uint32_t oFdAddrLow32 = gTiling.GetValue(offsetTiling + 14);
                uint64_t FdAddr = (uint64_t)(((uint64_t)oFdAddrHigh32) << 32 | oFdAddrLow32);
                uint32_t headIdx = curStartHeadIdx + AscendC::GetSubBlockIdx() * qHeadSplitSizeActual / 2;
                oFdOffset = FdAddr * kvSplitCoreNum + headIdx * embed * kvSplitCoreNum + curNIdx * embed;
                lOffset = lAddr + headIdx * kvSplitCoreNum + curNIdx;
            }

            uint64_t gmOffsetP = 0;
            uint64_t gmOffsetS = 0;
            uint64_t gmOffsetMask = 0;
            // Split k seqlen
            for (uint32_t nIdx = 0; nIdx < nLoop + 1; nIdx++) {
                if (nIdx != nLoop) {
                    if (nIdx == (nLoop - 1)) {
                        // Calculate the size of the tail block
                        kSeqTile = (curKVSeqlen - nIdx * seqTile);
                        kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
                    }
                    // Wait for Q * K calculation to complete
                    Arch::CrossCoreWaitFlag(qkReady);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);

                    LayoutP layoutP(rowNum, kSeqTile, kSeqTileRound);
                    LayoutS layoutS(rowNumRound, kSeqTile, kSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embedRound};
                    uint32_t softmaxPingPongFlag = nIdx % 2;
                    uint64_t gmOffsetP = (uint64_t)coreIdx * TMP_SIZE + softmaxPingPongFlag * TMP_SIZE / 2;
                    uint64_t gmOffsetS = (uint64_t)coreIdx * TMP_SIZE_DECODER
                                         + softmaxPingPongFlag * TMP_SIZE_DECODER / 2;
                    // Softmax one-stage calculation
                    epilogueMLASoftmax(
                        gP[gmOffsetP], gS[gmOffsetS], layoutP, layoutS, actualBlockShapeQK, nIdx, qHeadSplitSizeActual,
                        softmaxPingPongFlag, glFlag
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
                }

                if (nIdx != 0) {
                    // Calculate the size of the tail block
                    if (nIdx == nLoop) {
                        vSeqTile = (curKVSeqlen - (nIdx - 1) * seqTile);
                        vSeqTileRound = RoundUp<BLOCK_SIZE>(vSeqTile);
                    }
                    // Wait for P * V calculation to complete
                    Arch::CrossCoreWaitFlag(pvReady);

                    LayoutO layoutO(tokenNumPerHead, strideQO);
                    LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                    LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed, vSeqTile};
                    uint32_t rescaleOPingPongFlag = (nIdx - 1) % 2;
                    uint64_t gmOffsetOTmp = (uint64_t)(coreIdx * TMP_SIZE * 2 + rescaleOPingPongFlag * TMP_SIZE);
                    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * TMP_SIZE);
                    uint32_t isLastNTile = (nIdx == nLoop) ? 1 : 0;
                    // Softmax two-stage update
                    epilogueMLARescaleO(
                        gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gO[gmOffsetO], gOCoreTmp[oFdOffset], gl[lOffset],
                        layoutOTmp, layoutO, layoutUpdate, actualBlockShapePV, nIdx, isLastNTile, qHeadSplitSizeActual,
                        rescaleOPingPongFlag, glFlag
                    );
                }
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

        // flash decoding
        if (kvSplitCoreNum != 1) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

            AscendC::SetAtomicNone();
            AscendC::SetMaskNorm();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

            EpilogueMLAFDRescaleO epilogueMLAFDRescaleO(resource, kvSplitCoreNum);

            uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
            uint32_t aivId = AscendC::GetBlockIdx();

            uint32_t headsProcess = (COMPUTE_ELE_NUM / embed) > HEADS_PROCESS_MAX ? HEADS_PROCESS_MAX
                                                                                  : (COMPUTE_ELE_NUM / embed);
            uint32_t loopsPerBatch = (qHeads + headsProcess - 1) / headsProcess;
            uint32_t loopsTotal = batch * loopsPerBatch;

            for (uint32_t loopIdx = aivId; loopIdx < loopsTotal; loopIdx += aivNum) {
                uint32_t batchIdx = loopIdx / loopsPerBatch;
                uint32_t loopIdxInBatch = loopIdx % loopsPerBatch;

                uint32_t offsetTiling = tilingHeadSize + tilingParaSize * batchIdx;
                uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 1);

                if (kvSeqlen == 0) {
                    continue;
                }

                uint32_t oAddrHigh32 = gTiling.GetValue(offsetTiling + 4);
                uint32_t oAddrLow32 = gTiling.GetValue(offsetTiling + 5);
                uint64_t oAddr = (uint64_t)(((uint64_t)oAddrHigh32) << 32 | oAddrLow32);
                uint32_t lAddrHigh32 = gTiling.GetValue(offsetTiling + 11);
                uint32_t lAddrLow32 = gTiling.GetValue(offsetTiling + 12);
                uint64_t lOffset = (uint64_t)(((uint64_t)lAddrHigh32) << 32 | lAddrLow32);
                uint32_t oFdAddrHigh32 = gTiling.GetValue(offsetTiling + 13);
                uint32_t oFdAddrLow32 = gTiling.GetValue(offsetTiling + 14);
                uint64_t oFdOffset = (uint64_t)(((uint64_t)oFdAddrHigh32) << 32 | oFdAddrLow32);

                uint32_t actualHeads = headsProcess;
                if (loopIdxInBatch == loopsPerBatch - 1) {
                    actualHeads = qHeads - loopIdxInBatch * headsProcess;
                }

                epilogueMLAFDRescaleO(
                    gO[oAddr + loopIdxInBatch * headsProcess * embed],
                    gOCoreTmp[oFdOffset * kvSplitCoreNum + loopIdxInBatch * headsProcess * kvSplitCoreNum * embed],
                    gl[lOffset + loopIdxInBatch * headsProcess * kvSplitCoreNum], actualHeads, headsProcess, embed
                );
            }
        }
    }

private:
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
};

CATLASS_GLOBAL void MLAFp16(
    uint64_t fftsAddr,
    GM_ADDR q,
    GM_ADDR qRope,
    GM_ADDR k,
    GM_ADDR kRope,
    GM_ADDR blockTables,
    GM_ADDR o,
    GM_ADDR s,
    GM_ADDR p,
    GM_ADDR oTmp,
    GM_ADDR oUpdate,
    GM_ADDR oCoreTmp,
    GM_ADDR l,
    GM_ADDR tiling
) {
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using ElementQ = half;
    using LayoutQ = layout::RowMajor;
    using ElementK = half;
    using LayoutK = layout::ColumnMajor;
    using ElementV = half;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = half;
    using LayoutP = layout::RowMajor;
    using ElementO = half;
    using LayoutO = layout::RowMajor;
    using ElementMask = half;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;

    // L1TileShape::K must be embdding
    using L1TileShape = GemmShape<128, 128, 576>;
    using L0TileShape = L1TileShape;

    // GEMM Block模块，实现Flash MLA的Q * K^T
    using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

    // Epilogue Block模块，实现Flash MLA中当前S基块的softmax
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueMLASoftmax =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLASoftmax, PType, SType, MaskType>;

    // GEMM Block模块，实现Flash MLA的P * V
    using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中当前O基块的更新
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using EpilogueMLARescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLARescaleO, OType, OUpdateType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中flash decoding
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using lType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    constexpr uint32_t ComputeEleNum = 6144;
    using EpilogueMLAFDRescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;

    // Kernel level
    using MLAKernel =
        MLAKernel<BlockMmadQK, BlockMmadPV, EpilogueMLASoftmax, EpilogueMLARescaleO, EpilogueMLAFDRescaleO>;
    typename MLAKernel::Params params{q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling};

    // call kernel
    MLAKernel mla;
    mla(params);
}

CATLASS_GLOBAL void MLABf16(
    uint64_t fftsAddr,
    GM_ADDR q,
    GM_ADDR qRope,
    GM_ADDR k,
    GM_ADDR kRope,
    GM_ADDR blockTables,
    GM_ADDR o,
    GM_ADDR s,
    GM_ADDR p,
    GM_ADDR oTmp,
    GM_ADDR oUpdate,
    GM_ADDR oCoreTmp,
    GM_ADDR l,
    GM_ADDR tiling
) {
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using ElementQ = __bf16;
    using LayoutQ = layout::RowMajor;
    using ElementK = __bf16;
    using LayoutK = layout::ColumnMajor;
    using ElementV = __bf16;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = __bf16;
    using LayoutP = layout::RowMajor;
    using ElementO = __bf16;
    using LayoutO = layout::RowMajor;
    using ElementMask = __bf16;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;

    // L1TileShape::K must be embdding
    using L1TileShape = GemmShape<128, 128, 576>;
    using L0TileShape = L1TileShape;

    // GEMM Block模块，实现Flash MLA的Q * K^T
    using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

    // Epilogue Block模块，实现Flash MLA中当前S基块的softmax
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueMLASoftmax =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLASoftmax, PType, SType, MaskType>;

    // GEMM Block模块，实现Flash MLA的P * V
    using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中当前O基块的更新
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using EpilogueMLARescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLARescaleO, OType, OUpdateType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中flash decoding
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using lType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    constexpr uint32_t ComputeEleNum = 6144;
    using EpilogueMLAFDRescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;

    // Kernel level
    using MLAKernel =
        MLAKernel<BlockMmadQK, BlockMmadPV, EpilogueMLASoftmax, EpilogueMLARescaleO, EpilogueMLAFDRescaleO>;
    typename MLAKernel::Params params{q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling};

    // call kernel
    MLAKernel mla;
    mla(params);
}