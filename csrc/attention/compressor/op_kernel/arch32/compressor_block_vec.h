/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_block_vec.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_VEC_H
#define COMPRESSOR_BLOCK_VEC_H

#include "../compressor_comm.h"
#include "arch32.h"
#include "../compressor_vector_comm.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

namespace Compressor {
template <typename COMP>
class CompressorBlockVector {
public:
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = 32 * 1024; // 32k
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;

    __aicore__ inline CompressorBlockVector(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate, 
                                __gm__ uint8_t *kvState, __gm__ uint8_t *scoreState, 
                                __gm__ uint8_t *ape, __gm__ uint8_t *normWeight,
                                __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos, 
                                __gm__ uint8_t *kvBlockTable, __gm__ uint8_t *scoreBlockTable,
                                __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, 
                                __gm__ uint8_t *startPos, __gm__ uint8_t *cmpKvOut);
                                
    __aicore__ inline void InitV1();
    __aicore__ inline void InitApe();
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
        __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> preMm1ResGm, GlobalTensor<T> curMm1ResGm,
                                                    GlobalTensor<T> vec1ResGm, GlobalTensor<T> vec2InputGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1(const Compressor::RunInfo &info);
    __aicore__ inline uint32_t GetBasicNum();
    __aicore__ inline uint32_t GetScSize();
    __aicore__ inline uint32_t GetScSize(uint32_t bStart, uint32_t sStart, uint32_t bEnd, uint32_t sEnd);
    __aicore__ inline void GetScIdxInfo(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t v2TcStart,
                                        uint32_t v2TcEnd, uint32_t &outputBStart, uint32_t &outputSStart,
                                        uint32_t &outputScSize);
    __aicore__ inline void CalcTcEndIdx(uint32_t bStart, uint32_t sStart, uint32_t dealTcNum, uint32_t &bEnd,
                                        uint32_t &sEnd);
    __aicore__ inline void CalcScEndIdx(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t &bEnd,
                                        uint32_t &scEnd);
    __aicore__ inline void CalcGlobalScStart(uint32_t bStart, uint32_t scStart);
    __aicore__ inline void SetMSplitInfo(const Compressor::RunInfo &info);

    // vec2
    __aicore__ inline void ComputeVec2(const Compressor::RunInfo &info);
    __aicore__ inline void SplitCoreV2(const Compressor::RunInfo& info);
    __aicore__ inline void CopyFinalResultOut(const Compressor::RunInfo& info, LocalTensor<X_T> &ComOutUb);
    __aicore__ inline void DealVec2BaseBlock(const Compressor::RunInfo& info, uint32_t startRow, uint32_t dealRowCount, uint64_t offset);
    __aicore__ inline void RmsNorm(const Compressor::RunInfo& info, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CalRope(const Compressor::RunInfo& info);
    __aicore__ inline void WriteToCacheState(const GlobalTensor<T> &state, GlobalTensor<int32_t> &blockTable,
                                             const LocalTensor<T> &input, uint32_t batchIdx, 
                                             uint32_t startSeqIdx, uint32_t endSeqIdx,
                                             uint32_t stateDOffset, uint32_t dealDSize);
    __aicore__ inline void ReadFromCacheState(const LocalTensor<T> &output, const GlobalTensor<T> &state,
                                              GlobalTensor<int32_t> &blockTable,
                                              uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx,
                                              uint32_t stateDOffset, uint32_t dealDSize);
    __aicore__ inline void ProcessSingleBatch(uint32_t batchIdx, uint32_t batchStartSeqIdx, uint32_t batchEndSeqIdx,
                                              uint32_t dLoop, uint32_t dealDSize, uint32_t &compressorNum,
                                              uint32_t &processNum, const LocalTensor<T> &ape);

protected:
    GlobalTensor<T> vec1ResGm_;
    GlobalTensor<T> vec2InputGm_;
    GlobalTensor<T> preMm1ResGm_;
    GlobalTensor<T> curMm1ResGm_;
    TBuf<TPosition::VECCALC> shareBuffer1_;
    TBuf<TPosition::VECCALC> shareBuffer2_;
    TBuf<TPosition::VECCALC> shareBuffer3_;
    TBuf<TPosition::VECCALC> shareBufferApe_;
    TBuf<TPosition::VECOUT> outputBuffer_;

private:
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 1U;
    uint32_t curStartPos_ = 0;
    uint32_t curActSeqLength_ = 0;
    uint32_t v1SplitSize_ = 0;
    uint32_t v1ScLoopTimes_ = 0;
    uint32_t v1DLoopTimes_ = 0;
    uint32_t dealTcNum_ = 0;
    bool apeIsLoad_ = false;

    // vec2
    uint32_t v2MBaseSize = 16; // Tc块数量：32 * 1024 / (512 * 4)
    uint32_t v2TcStartIdx = 0U;
    uint32_t v2TcEndIdx = 0U;
    uint32_t mmResColSize_ = 128;
    int64_t vec1ResGmStart = 0U;
    uint32_t usedCoreNum = 16;
    uint32_t OutputBStartIdx = 0;
    uint32_t OutputSStartIdx = 0;
    uint32_t OutputSize = 0;
    uint32_t globalScStart = 0;
    uint32_t pingpongFlag = 0U;
    uint32_t mmResBaseOffset_ = 0;

    ConstInfo constInfo_ = {};
    MSplitInfo mSplitInfo = {};
    GlobalTensor<float> apeGm_;
    GlobalTensor<float> kvStateGm_;
    GlobalTensor<float> scoreStateGm_;
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> kvBlockTableGm_;
    GlobalTensor<int32_t> scoreBlockTableGm_;
    GlobalTensor<X_T> normWeightGm_;
    GlobalTensor<X_T> ropeSinGm_;
    GlobalTensor<X_T> ropeCosGm_;
    GlobalTensor<X_T> cmpKvOutGm_;

    LocalTensor<uint32_t> gatherOffsetCast_;
    LocalTensor<T> gammaCastLocal_;
    LocalTensor<X_T> ropeSinLocal_;
    LocalTensor<X_T> ropeCosLocal_;
    LocalTensor<float> kvLocal_;
    LocalTensor<float> scoreLocal_;
    LocalTensor<float> tempLocal_;
    LocalTensor<float> apeLocal_;
    LocalTensor<float> apeLocal1_;
    LocalTensor<float> apeLocal2_;
    LocalTensor<float> outputLocal_;
};

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo_ = constInfo;
    v2MBaseSize = BLOCK_VEC_BASE_BUFFER_SIZE / (constInfo_.headDim * sizeof(float));
    InitV1();
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::Init(
    __gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate, __gm__ uint8_t *kvState, __gm__ uint8_t *scoreState,
    __gm__ uint8_t *ape, __gm__ uint8_t *normWeight, __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *kvBlockTable, __gm__ uint8_t *scoreBlockTable, __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, 
    __gm__ uint8_t *startPos, __gm__ uint8_t *cmpKvOut)
{
    apeGm_.SetGlobalBuffer((__gm__ float *)ape);
    kvStateGm_.SetGlobalBuffer((__gm__ float *)kvState);
    scoreStateGm_.SetGlobalBuffer((__gm__ float *)scoreState);
    startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    kvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)kvBlockTable);
    scoreBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)scoreBlockTable);
    normWeightGm_.SetGlobalBuffer((__gm__ X_T *)normWeight);
    ropeSinGm_.SetGlobalBuffer((__gm__ X_T *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ X_T *)ropeCos);
    cmpKvOutGm_.SetGlobalBuffer((__gm__ X_T *)cmpKvOut);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitV1()
{
    coff_ = static_cast<uint32_t>(COMP::coff);
    v1SplitSize_ = BLOCK_VEC_BASE_BUFFER_SIZE / (constInfo_.cmpRatio * coff_ * sizeof(T));
    // printf("[InitV1] constInfo_.cmpRatio:%d, coff_:%d\n", constInfo_.cmpRatio, coff_);
    // printf("[InitV1] v1SplitSize_:%d, constInfo_.dBaseSize:%d\n", v1SplitSize_, constInfo_.dBaseSize);
    if (v1SplitSize_ < constInfo_.dBaseSize) {
        dealTcNum_ = 1;
        v1DLoopTimes_ = (constInfo_.dBaseSize + (v1SplitSize_ - 1)) / v1SplitSize_;
    } else {
        v1DLoopTimes_ = 1;
        dealTcNum_ = v1SplitSize_ / constInfo_.dBaseSize;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitApe()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    if (v1DLoopTimes_ == 1) {
        DataCopy(apeLocal_, apeGm_[constInfo_.dIdx], {static_cast<uint16_t>(coff_ * constInfo_.cmpRatio),
                     static_cast<uint16_t>(CeilDivT(constInfo_.dBaseSize, FP32_BLOCK_ELEMENT_NUM)),
                     static_cast<uint16_t>(CeilDivT(constInfo_.headDim - constInfo_.dBaseSize, FP32_BLOCK_ELEMENT_NUM)), 0});
    } else {
        DataCopy(apeLocal1_, apeGm_[constInfo_.dIdx], {static_cast<uint16_t>(coff_ * constInfo_.cmpRatio),
                    static_cast<uint16_t>(CeilDivT(constInfo_.dBaseSize / 2, FP32_BLOCK_ELEMENT_NUM)),
                    static_cast<uint16_t>(CeilDivT(constInfo_.headDim - constInfo_.dBaseSize / 2, FP32_BLOCK_ELEMENT_NUM)), 0});
        DataCopy(apeLocal2_, apeGm_[constInfo_.dIdx + constInfo_.dBaseSize / 2],
                    {static_cast<uint16_t>(coff_ * constInfo_.cmpRatio), static_cast<uint16_t>(CeilDivT(constInfo_.dBaseSize / 2, FP32_BLOCK_ELEMENT_NUM)),
                    static_cast<uint16_t>(CeilDivT(constInfo_.headDim - constInfo_.dBaseSize / 2, FP32_BLOCK_ELEMENT_NUM)), 0});
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(shareBuffer1_, BLOCK_VEC_BASE_BUFFER_SIZE);
    pipe->InitBuffer(shareBuffer2_, BLOCK_VEC_BASE_BUFFER_SIZE);
    pipe->InitBuffer(shareBufferApe_, 2 * BLOCK_VEC_BASE_BUFFER_SIZE);
    pipe->InitBuffer(shareBuffer3_, BLOCK_VEC_BASE_BUFFER_SIZE);
    // pipe->InitBuffer(gatherOffsetBuffer, GATHER_OFFSET_BUFFER_SIZE);
    pipe->InitBuffer(outputBuffer_, BLOCK_VEC_BASE_BUFFER_SIZE);
    kvLocal_ = shareBuffer1_.Get<float>();
    scoreLocal_ = shareBuffer2_.Get<float>();
    tempLocal_ = shareBuffer3_.Get<float>();
    outputLocal_ = outputBuffer_.Get<float>();
    // gatherOffsetLocal = gatherOffsetBuffer.Get<int32_t>();
    if (v1DLoopTimes_ == 1) {
        // printf("[vInit] v1DLoopTimes_ == 1\n");
        apeLocal_ = shareBufferApe_.Get<float>();
    } else {
        // printf("[vInit] v1DLoopTimes_ != 1\n");
        apeLocal1_ = shareBufferApe_.Get<float>();
        apeLocal2_ = apeLocal1_[BLOCK_VEC_BASE_BUFFER_SIZE / sizeof(float)];
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitVec1GlobalTensor(GlobalTensor<T> preMm1ResGm,
                                                                         GlobalTensor<T> curMm1ResGm,
                                                                         GlobalTensor<T> vec1ResGm, 
                                                                         GlobalTensor<T> vec2InputGm)
{
    this->preMm1ResGm_ = preMm1ResGm;
    this->curMm1ResGm_ = curMm1ResGm;
    this->vec1ResGm_ = vec1ResGm;
    this->vec2InputGm_ = vec2InputGm;
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::AllocEventID()
{
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::FreeEventID()
{
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetStartPos(uint32_t bIdx)
{
    return startPosGm_.GetValue(bIdx);
}

// TODO 使用这种方式获取seq的约束为顺序访问，随机访问不可用
template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::WriteToCacheState(const GlobalTensor<T> &state, 
                                                                      GlobalTensor<int32_t> &blockTableGm_,
                                                                      const LocalTensor<T> &input, 
                                                                      uint32_t batchIdx, uint32_t startSeqIdx,
                                                                      uint32_t endSeqIdx,
                                                                      uint32_t stateDOffset, uint32_t dealDSize)
{
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    uint16_t ubDGap = CeilDivT(coff_ * dealDSize - dealDSize, FP32_BLOCK_ELEMENT_NUM);
    while (curSeqIdx < endSeqIdx) {
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint32_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint32_t idInBlockTable = blockTableGm_.GetValue(blockTablebaseOffset + blockIdOffset);
        // printf("idInBlockTable:%d\n", idInBlockTable);
        if (idInBlockTable == 0) {
            continue;
        }
        uint32_t copyRowCnt = Std::min(constInfo_.blockSize - remainRowCnt, seqCnt - copyFinishRowCnt);
        uint64_t stateOffset =
            (idInBlockTable * constInfo_.blockSize + remainRowCnt) * coff_ * constInfo_.headDim + stateDOffset;
        uint64_t ubOffset = (curSeqIdx - startSeqIdx) * coff_ * dealDSize;
        // printf("constInfo_.blockSize:%d, remainRowCnt:%d, stateDOffset:%d\n", constInfo_.blockSize, remainRowCnt, stateDOffset);
        // printf("curSeqIdx:%d, startSeqIdx:%d, ubOffset:%d\n", curSeqIdx, startSeqIdx, ubOffset);
        // printf("stateOffset:%d\n", stateOffset);
        DataCopyParams copyParams{static_cast<uint16_t>(copyRowCnt), static_cast<uint16_t>(CeilDivT(dealDSize, FP32_BLOCK_ELEMENT_NUM)), ubDGap, static_cast<uint16_t>(CeilDivT(coff_ * constInfo_.headDim - dealDSize, FP32_BLOCK_ELEMENT_NUM))};
        // printf("W copyRowCnt: %d, copyColCnt: %d, srcGap: %d, dstGap: %d\n", copyRowCnt, static_cast<uint16_t>(CeilDivT(dealDSize, FP32_BLOCK_ELEMENT_NUM)), ubDGap, static_cast<uint16_t>(CeilDivT(coff_ * constInfo_.headDim - dealDSize, FP32_BLOCK_ELEMENT_NUM)));
        DataCopy(state[stateOffset], input[ubOffset], copyParams);
        // DumpTensor(state[stateOffset], 288, 128*128);
        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::ReadFromCacheState(const LocalTensor<T> &output, const GlobalTensor<T> &state, 
                                                                       GlobalTensor<int32_t> &blockTableGm_,
                                                                       uint32_t batchIdx, uint32_t startSeqIdx,
                                                                       uint32_t endSeqIdx, uint32_t stateDOffset, uint32_t dealDSize)
{
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    uint16_t ubDGap = CeilDivT(coff_ * dealDSize - dealDSize, FP32_BLOCK_ELEMENT_NUM);
    while (curSeqIdx < endSeqIdx) {
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint32_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint32_t idInBlockTable = blockTableGm_.GetValue(blockTablebaseOffset + blockIdOffset);
        // printf("idInBlockTable:%d\n", idInBlockTable);
        if (idInBlockTable == 0) {
            break;
        }
        uint32_t copyRowCnt = Std::min(constInfo_.blockSize - remainRowCnt, seqCnt - copyFinishRowCnt);
        uint64_t stateOffset =
            (idInBlockTable * constInfo_.blockSize + remainRowCnt) * coff_ * constInfo_.headDim + stateDOffset;
        uint64_t ubOffset = (curSeqIdx - startSeqIdx) * startSeqIdx;
        // printf("constInfo_.blockSize:%d, remainRowCnt:%d, stateDOffset:%d\n", constInfo_.blockSize, remainRowCnt, stateDOffset);
        // printf("curSeqIdx:%d, startSeqIdx:%d, ubOffset:%d\n", curSeqIdx, startSeqIdx, ubOffset);
        // printf("copyRowCnt:%d\n",copyRowCnt);
        DataCopyParams copyParams{static_cast<uint16_t>(copyRowCnt), static_cast<uint16_t>(CeilDivT(dealDSize, FP32_BLOCK_ELEMENT_NUM)), static_cast<uint16_t>(CeilDivT(coff_ * constInfo_.headDim - dealDSize, FP32_BLOCK_ELEMENT_NUM)), ubDGap};
        DataCopy(output[ubOffset], state[stateOffset], copyParams);
        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::ProcessSingleBatch(uint32_t batchIdx, uint32_t batchStartSeqIdx,
                                                                         uint32_t batchEndSeqIdx, uint32_t dLoop,
                                                                         uint32_t dealDSize, uint32_t& compressorNum, uint32_t& processNum, const LocalTensor<T> &ape)
{
    uint32_t seqIdx = batchStartSeqIdx;
    uint32_t seqUsed = GetSeqLength(batchIdx); // seqused有效的话还需要调整
    uint32_t startPos = GetStartPos(batchIdx);
    uint32_t MemstartSeqIdx = 0;
    uint32_t tempProcessNum = 0;
    uint32_t tempCompressorNum = CeilDivT(Trunc(startPos + batchEndSeqIdx, constInfo_.cmpRatio) - Trunc(startPos + batchStartSeqIdx, constInfo_.cmpRatio), constInfo_.cmpRatio);
    uint16_t srcStride = CeilDivT(2 * constInfo_.dBaseSize, FP32_BLOCK_ELEMENT_NUM);
    uint16_t srcGap = CeilDivT(2 * constInfo_.dBaseSize - dealDSize, FP32_BLOCK_ELEMENT_NUM);
    uint32_t totalDSize = coff_ * dealDSize;
    uint16_t dstStride = CeilDivT(totalDSize, FP32_BLOCK_ELEMENT_NUM);
    uint16_t dstGap = CeilDivT(totalDSize - dealDSize, FP32_BLOCK_ELEMENT_NUM);
    uint32_t cnt = totalDSize * constInfo_.cmpRatio;
    uint32_t globalStartSeqIdx = 0;

    uint32_t startSeqIdx = 0;
    uint32_t endSeqIdx = 0;
    uint32_t stateDOffset = 0;
    uint32_t ubOffset = 0;
    uint64_t srcStartOffset = 0;
    uint64_t dstStartOffset = 0;
    DataCopyParams copyParams{static_cast<uint16_t>(constInfo_.cmpRatio), static_cast<uint16_t>(CeilDivT(dealDSize, FP32_BLOCK_ELEMENT_NUM)), srcGap, dstGap};
    // bool hasHead = coff_ == 2 && batchStartSeqIdx != 0;
    while (seqIdx < batchEndSeqIdx) {
        globalStartSeqIdx = Trunc(startPos + seqIdx, constInfo_.cmpRatio);
        srcStartOffset = mSplitInfo.vec1StartOffset + (processNum * constInfo_.cmpRatio + MemstartSeqIdx) * 2 * constInfo_.dBaseSize;
        dstStartOffset = MemstartSeqIdx * totalDSize;
        if (coff_ == 2) {
            // printf("L Offset:%d\n", dstStartOffset);
            DataCopy(kvLocal_[dstStartOffset], preMm1ResGm_[srcStartOffset + dealDSize * dLoop], copyParams);
            DataCopy(scoreLocal_[dstStartOffset], preMm1ResGm_[srcStartOffset + dealDSize * dLoop + constInfo_.dBaseSize],
                     copyParams);
        }
        // printf("R Offset:%d, kvGmOffset: %d\n", dstStartOffset + dealDSize * (coff_ - 1), srcStartOffset + dealDSize * dLoop);
        // printf("R Offset:%d, scoreGmOffset: %d\n", dstStartOffset + dealDSize * (coff_ - 1), srcStartOffset + dealDSize * dLoop + constInfo_.dBaseSize);
        DataCopy(kvLocal_[dstStartOffset + dealDSize * (coff_ - 1)], curMm1ResGm_[srcStartOffset + dealDSize * dLoop],
                 copyParams);
        DataCopy(scoreLocal_[dstStartOffset + dealDSize * (coff_ - 1)],
                 curMm1ResGm_[srcStartOffset + dealDSize * dLoop + constInfo_.dBaseSize], copyParams);
        seqIdx = Trunc(startPos + seqIdx + constInfo_.cmpRatio, constInfo_.cmpRatio) - startPos;
        MemstartSeqIdx += constInfo_.cmpRatio;
        tempProcessNum += 1;
    }
    // DumpTensor(kvLocal_[127*64], 296, 64);
    // DumpTensor(scoreLocal_[127*64], 297, 64);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    dstStartOffset = 0;
    // printf("dstStartOffset:%d\n", dstStartOffset);
    processNum += tempProcessNum;
    compressorNum += tempCompressorNum;
    // printf("processNum:%d, compressorNum:%d, tempProcessNum:%d, tempCompressorNum:%d\n", processNum, compressorNum, tempProcessNum, tempCompressorNum);
    // 加APE
    
    for (uint32_t i = 0; i < tempProcessNum; i++) {
        Add(scoreLocal_[dstStartOffset + i * cnt], scoreLocal_[dstStartOffset + i * cnt], ape, cnt);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    // DumpTensor(scoreLocal_, 296, 256*128);
    uint64_t ubStartOffset = dstStartOffset;
    if (batchEndSeqIdx == seqUsed) {
        if ((batchEndSeqIdx + startPos) % constInfo_.cmpRatio != 0 && tempProcessNum > 1) {
            // 将左侧倒数第一个矩阵，和右侧倒数两个矩阵刷入state
            // printf("WRD2\n");
            if (coff_ == 2) {
                startSeqIdx = max(Trunc(startPos + seqUsed - constInfo_.cmpRatio, constInfo_.cmpRatio), startPos + batchStartSeqIdx);
                endSeqIdx = Trunc(startPos + seqUsed, constInfo_.cmpRatio);
                stateDOffset = constInfo_.headDim * (coff_ - 1) + constInfo_.dIdx + dealDSize * dLoop;
                ubOffset = ubStartOffset + (tempProcessNum - 2) * constInfo_.cmpRatio * totalDSize + (startSeqIdx - Trunc(startSeqIdx, constInfo_.cmpRatio)) * totalDSize + (coff_ - 1) * dealDSize;
                WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
                WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);

                // printf("WLD1\n");
                stateDOffset = constInfo_.dIdx + dealDSize * dLoop;
                ubOffset = ubStartOffset + (tempProcessNum - 1) * constInfo_.cmpRatio * totalDSize + (startSeqIdx - Trunc(startSeqIdx, constInfo_.cmpRatio)) * totalDSize;
                WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
                WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset,  dealDSize);
            }

            // printf("WRD1\n");
            startSeqIdx = Trunc(startPos + seqUsed - 1, constInfo_.cmpRatio);
            endSeqIdx = startPos + seqUsed;
            stateDOffset = constInfo_.headDim * (coff_ - 1) + constInfo_.dIdx + dealDSize * dLoop;
            ubOffset = ubStartOffset + (tempProcessNum - 1) * constInfo_.cmpRatio * totalDSize + (coff_ - 1) * dealDSize;
            WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset,dealDSize);
            WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
        } else if (coff_ == 2 || (batchEndSeqIdx + startPos) % constInfo_.cmpRatio != 0) {
            // 将右侧倒数第一个矩阵刷入state
            // printf("WRD1\n");
            startSeqIdx = max(Trunc(startPos + seqUsed - 1, constInfo_.cmpRatio), startPos + batchStartSeqIdx);
            endSeqIdx = startPos + seqUsed;
            stateDOffset = constInfo_.headDim * (coff_ - 1) + constInfo_.dIdx + dealDSize * dLoop;
            ubOffset = ubStartOffset + (tempProcessNum - 1) * constInfo_.cmpRatio * totalDSize + (startSeqIdx - Trunc(startSeqIdx, constInfo_.cmpRatio)) * totalDSize + (coff_ - 1) * dealDSize;
            // printf("constInfo_.dIdx: %d, constInfo_.dBaseSize: %d, dealDSize: %d, dLoop: %d\n", constInfo_.dIdx, constInfo_.dBaseSize, dealDSize, dLoop);
            // printf("startSeqIdx:%d, endSeqIdx:%d, stateDOffset:%d, ubOffset:%d\n", startSeqIdx, endSeqIdx, stateDOffset, ubOffset);
            // DumpTensor(kvStateGm_[425920], 286, 128*128);
            WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
            // DumpTensor(kvStateGm_[425920], 287, 128*128);
            // DumpTensor(scoreStateGm_[425920], 289, 128*128);
            WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
            // DumpTensor(scoreStateGm_[425920], 290, 128*128);
        }
    } else if (coff_ == 2 && (startPos + batchEndSeqIdx) == Trunc(startPos + seqUsed, constInfo_.cmpRatio)) {
        // 将右侧倒数第一个矩阵刷入state
        // printf("WRD1\n");
        startSeqIdx = max(startPos + batchEndSeqIdx - constInfo_.cmpRatio, startPos + batchStartSeqIdx);
        endSeqIdx = startPos + batchEndSeqIdx;
        stateDOffset = constInfo_.headDim * (coff_ - 1) + constInfo_.dIdx + dealDSize * dLoop;
        ubOffset = ubStartOffset + (tempProcessNum - 1) * constInfo_.cmpRatio * totalDSize + (startSeqIdx - Trunc(startSeqIdx, constInfo_.cmpRatio)) * totalDSize + (coff_ - 1) * dealDSize;
        WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
        WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
    }
    if (coff_ == 2 && batchStartSeqIdx == 0) {
        // 将左侧第一个矩阵刷入state
        // printf("WLU1\n");
        uint32_t preSeqUsed = batchIdx == 0 ? GetSeqLength(constInfo_.batchSize - 1) : GetSeqLength(batchIdx - 1); 
        uint32_t preStartPos = batchIdx == 0 ? GetStartPos(constInfo_.batchSize - 1) : GetStartPos(batchIdx - 1);
        startSeqIdx = max(Trunc(preStartPos + preSeqUsed - 1, constInfo_.cmpRatio), preStartPos);
        endSeqIdx = preStartPos + preSeqUsed;
        stateDOffset = constInfo_.dIdx + dealDSize * dLoop;
        ubOffset = ubStartOffset + (startSeqIdx - Trunc(startSeqIdx, constInfo_.cmpRatio)) * totalDSize;
        // printf("startSeqIdx:%d, endSeqIdx:%d, stateDOffset:%d, ubOffset:%d\n", startSeqIdx, endSeqIdx, stateDOffset, ubOffset);
        WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
        WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal_[ubOffset], batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    if (tempCompressorNum > 0) {
        if (batchStartSeqIdx == 0) {
            if (coff_ == 2) {
                // 无更前面的矩阵
                if (startPos < constInfo_.cmpRatio) {
                    // 第一个d设成-inf
                    // printf("DLU1\n");
                    Duplicate(kvLocal_[ubStartOffset], float(-2e38), dealDSize, constInfo_.cmpRatio, 1, dstStride);
                    Duplicate(scoreLocal_[ubStartOffset], float(-2e38), dealDSize, constInfo_.cmpRatio, 1, dstStride);
                } else {
                    // 从state补充第一个左侧矩阵
                    // printf("RLU1\n");
                    startSeqIdx = Trunc(startPos + batchStartSeqIdx - constInfo_.cmpRatio, constInfo_.cmpRatio);
                    endSeqIdx = Trunc(startPos + batchStartSeqIdx, constInfo_.cmpRatio);
                    stateDOffset = constInfo_.dIdx + dealDSize * dLoop;
                    ubOffset = ubStartOffset;
                    // printf("startSeqIdx:%d, endSeqIdx:%d, stateDOffset:%d, ubOffset:%d\n", startSeqIdx, endSeqIdx, stateDOffset, ubOffset);
                    ReadFromCacheState(kvLocal_[ubOffset], kvStateGm_, kvBlockTableGm_, batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
                    ReadFromCacheState(scoreLocal_[ubOffset], scoreStateGm_, scoreBlockTableGm_, batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
                }
            }
            if (startPos % constInfo_.cmpRatio != 0) {
                // 从state补充第一个右侧矩阵
                // printf("RRU1\n");
                startSeqIdx = Trunc(startPos + batchStartSeqIdx, constInfo_.cmpRatio);
                endSeqIdx = startPos + batchStartSeqIdx;
                stateDOffset = constInfo_.headDim * (coff_ - 1) + constInfo_.dIdx + dealDSize * dLoop;
                ubOffset = ubStartOffset + (coff_ - 1) * dealDSize;
                // printf("startSeqIdx:%d, endSeqIdx:%d, stateDOffset:%d, ubOffset:%d\n", startSeqIdx, endSeqIdx, stateDOffset, ubOffset);
                ReadFromCacheState(kvLocal_[ubOffset], kvStateGm_, kvBlockTableGm_, batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
                ReadFromCacheState(scoreLocal_[ubOffset], scoreStateGm_, scoreBlockTableGm_, batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);

                if (coff_ == 2 && tempCompressorNum > 1) {
                    // 从state补充第二个左侧矩阵
                    // printf("RLU2\n");
                    startSeqIdx = Trunc(startPos + batchStartSeqIdx, constInfo_.cmpRatio);
                    endSeqIdx = startPos + batchStartSeqIdx;
                    stateDOffset = constInfo_.dIdx + dealDSize * dLoop;
                    ubOffset = ubStartOffset + constInfo_.cmpRatio * totalDSize;
                    ReadFromCacheState(kvLocal_[ubOffset], kvStateGm_, kvBlockTableGm_, batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
                    ReadFromCacheState(scoreLocal_[ubOffset], scoreStateGm_, scoreBlockTableGm_, batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
                }
            }
        } else if (coff_ == 2 && batchStartSeqIdx < constInfo_.cmpRatio) {
            // 从state补充第一个左侧矩阵
            // printf("RLU1\n");
            startSeqIdx = startPos + batchStartSeqIdx - constInfo_.cmpRatio;
            endSeqIdx = startPos;
            stateDOffset = constInfo_.dIdx + dealDSize * dLoop;
            ubOffset = ubStartOffset;
            ReadFromCacheState(kvLocal_[ubOffset], kvStateGm_, kvBlockTableGm_, batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
            ReadFromCacheState(scoreLocal_[ubOffset], scoreStateGm_, scoreBlockTableGm_, batchIdx, startSeqIdx, endSeqIdx, stateDOffset, dealDSize);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetBasicNum()
{
    // 获取 m方向上对应基本单元Tc的个数
    uint32_t curBasicNum = 0;
    uint32_t headSize = 0;
    if (curStartPos_ % constInfo_.cmpRatio != 0) {
        headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
        headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
        curBasicNum++;
    }
    // 加上中间整块及尾块
    curBasicNum += (curActSeqLength_ - headSize + constInfo_.cmpRatio - 1) / constInfo_.cmpRatio;
    return curBasicNum;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetScSize()
{
    // 获取 m方向上对应基本单元Tc的个数
    uint32_t curBasicNum = 0;
    uint32_t headSize = 0;
    if (curStartPos_ % constInfo_.cmpRatio != 0) {
        headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
        headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
        curBasicNum++;
    }
    // 加上中间整块及尾块
    curBasicNum += (curActSeqLength_ - headSize) / constInfo_.cmpRatio;
    return curBasicNum;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetScSize(uint32_t bStart, uint32_t sStart, uint32_t bEnd, uint32_t sEnd)
{
    // 获取 当前batch压缩后的Sc的个数
    uint32_t totalScSize = 0;
    if (bStart == bEnd) {
        if (sStart == 0) {
            uint32_t headSize = 0;
            if (sEnd > 0 && curStartPos_ % constInfo_.cmpRatio != 0) {
                headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
                headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
                totalScSize += 1;
            }
            totalScSize += (sEnd - headSize) / constInfo_.cmpRatio;
        } else {
            totalScSize += (sEnd - sStart) / constInfo_.cmpRatio;
        }
        return totalScSize;
    }
    for (uint32_t bIdx = bStart; bIdx <= bEnd; ++bIdx) {
        curActSeqLength_ = GetSeqLength(bIdx);
        curStartPos_ = GetStartPos(bIdx);
        uint32_t curScSize = GetScSize();
        if (bIdx == bStart) {
            if (sStart == 0) {
                totalScSize += curScSize;
            } else {
                totalScSize += (curActSeqLength_ - sStart) / constInfo_.cmpRatio;
            }
        } else if (bIdx == bEnd) {
            uint32_t headSize = 0;
            if (sEnd > 0 && curStartPos_ % constInfo_.cmpRatio != 0) {
                headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
                headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
                totalScSize += 1;
            }
            totalScSize += (sEnd - headSize) / constInfo_.cmpRatio;
        } else {
            totalScSize += curScSize;
        }
    }
    return totalScSize;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::SetMSplitInfo(const Compressor::RunInfo &info)
{
    // TODO 处理0块需要考虑？
    // VEC0需要处理的大小
    mSplitInfo.dealTcNum = (info.dealTcNum + 1) / 2;
    mSplitInfo.vecStartB = info.bStart;
    mSplitInfo.vecStartS = info.sStart;
    uint32_t bEnd = 0;
    uint32_t sEnd = 0;

    CalcTcEndIdx(mSplitInfo.vecStartB, mSplitInfo.vecStartS, mSplitInfo.dealTcNum, bEnd, sEnd);
    mSplitInfo.vecEndB = bEnd;
    mSplitInfo.vecEndS = sEnd;
    mSplitInfo.vec1StartOffset = 0;
    mSplitInfo.vec1ResOffset = info.vec1ResOffset;
    if (GetBlockIdx() % 2 == 1) {
        mSplitInfo.vec1ResOffset += GetScSize(mSplitInfo.vecStartB, mSplitInfo.vecStartS, mSplitInfo.vecEndB, mSplitInfo.vecEndS) * constInfo_.headDim;
        mSplitInfo.vecStartB = bEnd;
        mSplitInfo.vecStartS = sEnd;
        mSplitInfo.vec1StartOffset = mSplitInfo.dealTcNum * constInfo_.cmpRatio * constInfo_.dBaseSize * 2;
        mSplitInfo.dealTcNum = info.dealTcNum - mSplitInfo.dealTcNum;
        if (sEnd == curActSeqLength_ && mSplitInfo.dealTcNum > 0) {
            mSplitInfo.vecStartB++;
            mSplitInfo.vecStartS = 0;
        }
        mSplitInfo.vecEndB = info.bEnd;
        mSplitInfo.vecEndS = info.sEnd;
    }
}

// 根据计算Tc开始结束索引
template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CalcTcEndIdx(uint32_t bStart, uint32_t sStart, uint32_t dealTcNum,
                                                                 uint32_t &bEnd, uint32_t &sEnd)
{
    uint32_t accBasicNum = 0;
    for (int bIdx = bStart; bIdx < constInfo_.batchSize; ++bIdx) {
        bEnd = bIdx;
        // 计算起始batch的剩余块
        if (bIdx == bStart) {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            uint32_t curRemainTcNum = 0;
            // 计算起始batch的剩余seq长度 起始位置计算头块
            uint32_t headSize = 0;
            if (curStartPos_ % constInfo_.cmpRatio != 0) {
                headSize = (constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio);
                headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
            }
            if (sStart == 0) {
                curRemainTcNum = (curActSeqLength_ - headSize + constInfo_.cmpRatio - 1) / constInfo_.cmpRatio;
                curRemainTcNum = headSize == 0 ? curRemainTcNum : curRemainTcNum + 1;
            } else {
                curRemainTcNum = (curActSeqLength_ - sStart + constInfo_.cmpRatio - 1) / constInfo_.cmpRatio;
            }
            // printf("[GetEndIdx]  bIdx:%u accBasicNum:%u dealTcNum:%u curRemainTcNum:%u headSize:%u curStartPos_:%u
            // curActSeqLength_:%u \n", bIdx, accBasicNum, dealTcNum, curRemainTcNum, headSize, curStartPos_,
            // curActSeqLength_);
            if (curRemainTcNum > dealTcNum) {
                if (sStart == 0) {
                    if (headSize == 0) {
                        sEnd = sStart + dealTcNum * constInfo_.cmpRatio;
                    } else {
                        sEnd = sStart + headSize + (dealTcNum - 1) * constInfo_.cmpRatio;
                    }
                    return;
                } else {
                    sEnd = sStart + dealTcNum * constInfo_.cmpRatio;
                    return;
                }
            } else if (curRemainTcNum == dealTcNum || bIdx == constInfo_.batchSize - 1) {
                sEnd = curActSeqLength_;
                return;
            } else {
                accBasicNum += curRemainTcNum;
            }
        } else {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            uint32_t curBasicNum = GetBasicNum();
            // printf("[GetEndIdx] accBasicNum:%u curBasicNum:%u dealTcNum:%u\n", accBasicNum, curBasicNum, dealTcNum);
            if (accBasicNum + curBasicNum > dealTcNum) {
                uint32_t headSize = 0;
                if (curStartPos_ % constInfo_.cmpRatio != 0) {
                    headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
                    // 处理seq不足head大小的情况
                    headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
                }
                uint32_t curBasicNumEnd = dealTcNum - accBasicNum;
                if (headSize == 0) {
                    sEnd = curBasicNumEnd * constInfo_.cmpRatio;
                } else {
                    sEnd = headSize + (curBasicNumEnd - 1) * constInfo_.cmpRatio;
                }
                sEnd = sEnd > curActSeqLength_ ? curActSeqLength_ : sEnd;
                return;
            } else if (accBasicNum + curBasicNum == dealTcNum) {
                sEnd = curActSeqLength_;
                return;
            }
            accBasicNum += curBasicNum;
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CalcGlobalScStart(uint32_t bStart, uint32_t scStart) {
    globalScStart = 0;
    for (uint32_t bIdx = 0; bIdx < bStart; ++bIdx) {
        curActSeqLength_ = GetSeqLength(bIdx);
        curStartPos_ = GetStartPos(bIdx);
        globalScStart += GetScSize();
    }
    globalScStart += scStart;
}

// 根据计算Tc开始结束索引
template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CalcScEndIdx(uint32_t bStart, uint32_t scStart, uint32_t dealScSize,
                                                                 uint32_t &bEnd, uint32_t &scEnd)
{
    uint32_t accScSize = 0;
    for (int bIdx = bStart; bIdx < constInfo_.batchSize; ++bIdx) {
        bEnd = bIdx;
        curActSeqLength_ = GetSeqLength(bIdx);
        curStartPos_ = GetStartPos(bIdx);
        uint32_t curBasicNum = GetScSize();
        // 计算起始batch的剩余块
        if (bIdx == bStart) {
            // printf("[GetEndIdx]  bIdx:%u accScSize:%u dealScSize:%u headSize:%u curStartPos_:%u curActSeqLength_:%u
            // \n", bIdx, accScSize, dealScSize, headSize, curStartPos_, curActSeqLength_);
            if (accScSize + curBasicNum >= dealScSize) {
                scEnd = scStart + dealScSize;
                return;
            }
        } else {
            uint32_t curBasicNumEnd = dealScSize - accScSize;
            // printf("[GetEndIdx] accScSize:%u curBasicNum:%u dealScSize:%u\n", accScSize, curBasicNum, dealScSize);
            if (accScSize + curBasicNum >= dealScSize) {
                scEnd = curBasicNumEnd;
                return;
            }
            accScSize += curBasicNum;
        }
    }
}

// 根据sc的开始索引计算vec输出时的b、sc的索引
template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::GetScIdxInfo(uint32_t bStart, uint32_t scStart, uint32_t dealScSize,
                                                                 uint32_t v2TcStart, uint32_t v2TcEnd,
                                                                 uint32_t &outputBStart, uint32_t &outputSStart,
                                                                 uint32_t &outputScSize)
{
    outputScSize = v2TcEnd - v2TcStart;
    uint32_t scEnd = 0;
    uint32_t bEnd = 0;
    CalcScEndIdx(bStart, scStart, v2TcStart, bEnd, scEnd);
    outputSStart = scEnd;
    outputBStart = bEnd;
    // 处理跳batch
    curActSeqLength_ = GetSeqLength(bEnd);
    curStartPos_ = GetStartPos(bEnd);
    uint32_t curScSize = GetScSize();
    if (curScSize == scEnd) {
        outputSStart = 0;
        outputBStart++;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::ComputeVec1(const RunInfo &info)
{
    // TODO 1分核
    SetMSplitInfo(info);
    InitApe();
    uint32_t sStart = mSplitInfo.vecStartS;
    uint32_t sEnd = 0;
    uint32_t bStart = mSplitInfo.vecStartB;
    uint32_t bEnd = 0;
    uint32_t ScompressorNum = 0;
    uint32_t remaindTcNum = mSplitInfo.dealTcNum; // 剩余需要处理的Tc块
    uint32_t SprocessTcNum = 0; // 处理了的Tc块
    uint32_t dealDSize = constInfo_.dBaseSize / v1DLoopTimes_;
    uint32_t rCnt = coff_ * constInfo_.cmpRatio * dealDSize;
    uint32_t v1ScLoopTimes_ = CeilDivT(mSplitInfo.dealTcNum, dealTcNum_);
    LocalTensor<float> ape;
    // printf("v1ScLoopTimes_:%d, v1DLoopTimes_:%d, dealTcNum_:%d\n", v1ScLoopTimes_, v1DLoopTimes_, dealTcNum_);
    for (uint32_t i = 0; i < v1ScLoopTimes_; i++) {
        // 计算当前需要处理的splitSize个Tc块的b、s的开始结束索引
        if (remaindTcNum == 0) {
            return;
        }
        uint32_t tmpTcNum = dealTcNum_ <= remaindTcNum ? dealTcNum_ : remaindTcNum;
        remaindTcNum -= tmpTcNum;
        CalcTcEndIdx(bStart, sStart, tmpTcNum, bEnd, sEnd);
        uint32_t tmpBStart = bStart;
        uint32_t tmpSStart = sStart;
        uint32_t tmpBEnd = bEnd;
        uint32_t tmpSEnd = sEnd;
        uint32_t compressorNum = 0;
        uint32_t processTcNum = 0;
        for (uint32_t j = 0; j < v1DLoopTimes_; j++) {
            bStart = tmpBStart;
            sStart = tmpSStart;
            bEnd = tmpBEnd;
            sEnd = tmpSEnd;
            // printf("[CalcTcEndIdx] tmpTcNum:%d, bStart:%u bEnd:%u sStart:%u sEnd:%u\n", tmpTcNum, bStart, bEnd, sStart, sEnd);
            compressorNum = ScompressorNum;
            processTcNum = SprocessTcNum;
            for (uint32_t k = bStart; k <= bEnd; k++) {
                // 计算当前batch的seq 开始结束索引
                curActSeqLength_ = GetSeqLength(k);
                uint32_t curSStart = 0;
                uint32_t curSEnd = curActSeqLength_;
                if (k == bStart) {
                    curSStart = sStart;
                }
                if (k == bEnd) {
                    curSEnd = sEnd;
                }
                // printf("[IDX] b:%u sStart:%u sEnd:%u curSStart:%u curSEnd:%u\n", k, sStart, sEnd, curSStart, curSEnd);

                if (v1DLoopTimes_ == 1) {
                    ape = apeLocal_;
                } else {
                    if (j == 0) {
                        ape = apeLocal1_;
                    } else {
                        ape = apeLocal2_;
                    }
                }
                uint32_t preCompressorNum = compressorNum;
                ProcessSingleBatch(k, sStart, sEnd, j, dealDSize, compressorNum, processTcNum, ape);
                uint32_t curCompressorNum = compressorNum - preCompressorNum;
                // printf("rcnt:%d, curCompressorNum:%d, compressorNum:%d, preCompressorNum:%d\n", rCnt, curCompressorNum, compressorNum, preCompressorNum);
                // DumpTensor(scoreLocal_[preCompressorNum * rCnt], 708, 512);
                // DumpTensor(kvLocal_[preCompressorNum * rCnt], 709, 512);
                PipeBarrier<PIPE_ALL>();
                for (uint32_t r = 0; r < curCompressorNum; r++) {
                    ColumnSoftMax(scoreLocal_[r * rCnt], scoreLocal_[r * rCnt], tempLocal_[r * rCnt], coff_ * constInfo_.cmpRatio, dealDSize);
                }
                PipeBarrier<PIPE_V>();
                // DumpTensor(scoreLocal_[preCompressorNum * rCnt], 714, 512);
                // DumpTensor(kvLocal_[preCompressorNum * rCnt], 674, 64);
                Mul(kvLocal_, kvLocal_, scoreLocal_, curCompressorNum * rCnt);
                // DumpTensor(scoreLocal_[preCompressorNum * rCnt], 676, 64);
                // DumpTensor(kvLocal_[preCompressorNum * rCnt], 719, 512);
                PipeBarrier<PIPE_V>();
                // DumpTensor(outputLocal_[preCompressorNum * dealDSize], 721, dealDSize);
                // DumpTensor(outputLocal_[(preCompressorNum + 16) * dealDSize], 722, dealDSize);
                // DumpTensor(kvLocal_[(preCompressorNum + 16) * rCnt], 723, dealDSize);
                // DumpTensor(tempLocal_[(preCompressorNum + 16) * rCnt], 724, dealDSize);
                // printf("offset-0:%d, offset:%d, len:%d\n", preCompressorNum * dealDSize, (preCompressorNum + 16) * rCnt, BLOCK_VEC_BASE_BUFFER_SIZE / sizeof(float));
                // DumpTensor(outputLocal_[preCompressorNum * dealDSize], 725, dealDSize);
                for (uint32_t r = 0; r < curCompressorNum; r++) {
                    ColumnSum(outputLocal_[r * dealDSize], kvLocal_[r * rCnt], tempLocal_[r  * rCnt], coff_ * constInfo_.cmpRatio, dealDSize);
                    // DumpTensor(outputLocal_[preCompressorNum * dealDSize], 683 + r, dealDSize);
                    // printf("r:%d, preCompressorNum:%d, offset:%d\n", r, preCompressorNum, (preCompressorNum + r) * dealDSize);
                }
                // DumpTensor(outputLocal_[preCompressorNum * dealDSize], 683, dealDSize);
                // DumpTensor(outputLocal_[preCompressorNum * dealDSize], 682, 64);
                PipeBarrier<PIPE_ALL>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                DataCopyParams copyParams{static_cast<uint16_t>(curCompressorNum), static_cast<uint16_t>(CeilDivT(dealDSize, FP32_BLOCK_ELEMENT_NUM)), 0, static_cast<uint16_t>(CeilDivT(constInfo_.headDim - dealDSize, FP32_BLOCK_ELEMENT_NUM))};
                // printf("vec1ResGm_ pos: %d, info.vec1ResOffset: %d\n", info.vec1ResOffset + preCompressorNum * constInfo_.headDim + j * dealDSize, info.vec1ResOffset);
                DataCopy(vec1ResGm_[mSplitInfo.vec1ResOffset + preCompressorNum * constInfo_.headDim + j * dealDSize], outputLocal_, copyParams);
                // DumpTensor(outputLocal_[preCompressorNum * dealDSize], 10011, curCompressorNum*dealDSize);
                // DumpTensor(vec1ResGm_[info.vec1ResOffset + preCompressorNum * constInfo_.headDim + j * dealDSize], 1001, curCompressorNum*dealDSize);
                // DumpTensor(vec1ResGm_, 100111, 256);
            }
            sStart = sEnd;
            bStart = bEnd;
            // 处理刚好是结尾跳batch
            if (sEnd == curActSeqLength_) {
                sStart = 0;
                bStart++;
            }
            
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
            
        }
        ScompressorNum = compressorNum;
        SprocessTcNum = processTcNum;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::ComputeVec2(const Compressor::RunInfo &info)
{
    SplitCoreV2(info);
    PipeBarrier<PIPE_ALL>();
    // TODO 给GM
    // printf("[WorkSpace] size:%u\n",  constInfo_.vec1ResSize * constInfo_.headDim);
    // printf("[WorkSpace] realsize:%lu\n", vec2InputGm_.GetSize());
    // DumpTensor(vec2InputGm_, 100223, 256);
    // for (int32_t i = 0; i < constInfo_.nSize * 256 * constInfo_.ropeHeadDim; ++i) {
    //     ropeSinGm_.SetValue(i, static_cast<bfloat16_t>(i));
    // }
    // for (int32_t i = 0; i < constInfo_.nSize * 256 * constInfo_.ropeHeadDim; ++i) {
    //     ropeSinGm_.SetValue(i, static_cast<bfloat16_t>(i));
    // }
    // normWeight 搬入
    LocalTensor<X_T> gammaLocal = shareBuffer3_.Get<X_T>();
    DataCopy(gammaLocal, normWeightGm_, constInfo_.headDim);
    // cast normWeight
    gammaCastLocal_ = gammaLocal[constInfo_.headDim].template ReinterpretCast<T>();
    
    PipeBarrier<PIPE_ALL>();
    Cast(gammaCastLocal_, gammaLocal, RoundMode::CAST_NONE, constInfo_.headDim);
    // for (int32_t i = 0; i <  constInfo_.headDim; ++i) {
    //     gammaCastLocal_.SetValue(i, static_cast<float>(i));
    // }
    LocalTensor<int32_t> gatherOffsetLocal = gammaCastLocal_[constInfo_.headDim].template ReinterpretCast<int32_t>();
    // 计算SetGatherSrcOffset
    SetGatherSrcOffset<float>(gatherOffsetLocal, constInfo_.headDim);
    gatherOffsetCast_ = gatherOffsetLocal.ReinterpretCast<uint32_t>();
    
    ropeSinLocal_ = shareBufferApe_.Get<X_T>();
    ropeCosLocal_ = ropeSinLocal_[v2MBaseSize * constInfo_.ropeHeadDim].template ReinterpretCast<X_T>();

    // printf("[ComputeVec2] v2TcEndIdx: %d, v2TcStartIdx: %d, dealScSize: %d\n", v2TcEndIdx, v2TcStartIdx, info.dealScSize);
    uint32_t vec2DealM = v2TcEndIdx - v2TcStartIdx;
    uint32_t loopCount = (vec2DealM + v2MBaseSize - 1) / v2MBaseSize;
    CalcGlobalScStart(OutputBStartIdx, OutputSStartIdx);
    uint64_t ropeOffset =  globalScStart * constInfo_.ropeHeadDim;
    uint64_t outOffset =  globalScStart * constInfo_.headDim;
    PipeBarrier<PIPE_ALL>();
    for (uint32_t v2LoopIdx = 0, dealSize = v2MBaseSize; v2LoopIdx < loopCount; ++v2LoopIdx) {
        // printf("[ComputeVec2] vec%d comes in\n", GetBlockIdx());
        if (v2LoopIdx == loopCount - 1) {
            dealSize = vec2DealM - v2LoopIdx * v2MBaseSize;
            // printf("dealSize: %d\n", dealSize);
        }
        // printf("[ComputeVec2] dealSize: %d, v2MBaseSize: %d, loopCount: %d, vec2DealM: %d\n",
            // dealSize, v2MBaseSize, loopCount, vec2DealM);
        // 搬入 cosRop, sinRop
        uint32_t ropeCount = dealSize * constInfo_.ropeHeadDim;
        ropeOffset = ropeOffset + v2LoopIdx * v2MBaseSize * constInfo_.ropeHeadDim;
        outOffset = outOffset + v2LoopIdx * v2MBaseSize * constInfo_.headDim;
        // printf("ropeOffset: %d, ropeCount: %d\n", ropeOffset, ropeCount);
        DataCopy(ropeSinLocal_, ropeSinGm_[ropeOffset], ropeCount);
        DataCopy(ropeCosLocal_, ropeCosGm_[ropeOffset], ropeCount);
        // DumpTensor(ropeSinGm_, 10031, 64);
        // DumpTensor(ropeSinLocal_, 100311, 64);
        // DumpTensor(ropeCosGm_, 10032, 64);
        // DumpTensor(ropeCosLocal_, 100322, 64);
        // DumpTensor(vec2InputGm_, 100222, 256);
        PipeBarrier<PIPE_ALL>();
        DealVec2BaseBlock(info, v2TcStartIdx + v2LoopIdx * v2MBaseSize, dealSize, outOffset);
    }
    v2TcStartIdx = 0;
    v2TcEndIdx = 0;
    // DumpTensor(cmpKvOutGm_, 12, 2*128);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::DealVec2BaseBlock(const Compressor::RunInfo& info, uint32_t startRow, uint32_t dealRowCount, uint64_t offset)
{
    uint32_t computeSize = dealRowCount * constInfo_.headDim;
    int64_t inGmOffset = startRow * constInfo_.headDim;
    // printf("[CopyVec1ResIn] inGmOffset: %ld, vec1ResGmStart:%ld dealRowCount:%u startRow:%u computeSize:%u\n", inGmOffset, vec1ResGmStart, dealRowCount, startRow, computeSize);
    // // CopyIn
    
    LocalTensor<T> vec1ResLocal = kvLocal_[0];
    LocalTensor<T> rmsNormResLocal = scoreLocal_[0];
    // DumpTensor(ropeSinLocal_, 100418, computeSize);
    LocalTensor<T> rotaryResLocal = outputBuffer_.Get<T>();
    // LocalTensor<T> vec1ResUb =  inputBuf1.Get<T>()[BLOCK_VEC_BASE_BUFFER_SIZE];
    // LocalTensor<X_T> outputLocal;
    // WaitFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(vec1ResLocal, vec2InputGm_[inGmOffset], computeSize);
    PipeBarrier<PIPE_ALL>();
    // DumpTensor(ropeSinLocal_, 100419, computeSize);
    // printf("[DataCopy] v2 data in\n");
    // DumpTensor(vec2InputGm_[inGmOffset], 10022, computeSize);
    // DumpTensor(vec1ResLocal, 1002, computeSize);
    // // RmsNorm
    // RmsNorm(info, startRow, dealRowCount);
    RmsNormParam rmsNormParams = {
        constInfo_.reciprocalD,
        // constInfo_.normEps,
        1e-06,
        dealRowCount,
        constInfo_.headDim};
    LocalTensor<T> tempLocal = ropeCosLocal_[2*v2MBaseSize * constInfo_.ropeHeadDim].template ReinterpretCast<T>();
    // DumpTensor(ropeSinLocal_, 100420, computeSize);
    Compressor::RmsNorm(rmsNormResLocal, vec1ResLocal, gammaCastLocal_, tempLocal, rmsNormParams);
    PipeBarrier<PIPE_V>();
    // printf("[RmsNorm] RmsNorm result dump\n");
    // DumpTensor(rmsNormResLocal, 1003, computeSize);
    // // rope
    // CalRope(info);
    LocalTensor<T> ropeCosFp32Local_ = tempLocal[v2MBaseSize * constInfo_.ropeHeadDim].template ReinterpretCast<T>();
    LocalTensor<T> ropeSinFp32Local_ = ropeCosFp32Local_[v2MBaseSize * constInfo_.ropeHeadDim].template ReinterpretCast<T>();
    PipeBarrier<PIPE_ALL>();
    // DumpTensor(ropeSinLocal_, 100421, computeSize);
    Cast(ropeCosFp32Local_, ropeCosLocal_, RoundMode::CAST_NONE, constInfo_.ropeHeadDim * dealRowCount);
    Cast(ropeSinFp32Local_, ropeSinLocal_, RoundMode::CAST_NONE, constInfo_.ropeHeadDim * dealRowCount);
    // for (int32_t i = 0; i < constInfo_.ropeHeadDim * v2MBaseSize; ++i) {
    //     ropeCosFp32Local_.SetValue(i, static_cast<float>(i));
    // }
    // for (int32_t i = 0; i < constInfo_.ropeHeadDim * v2MBaseSize; ++i) {
    //     ropeSinFp32Local_.SetValue(i, static_cast<float>(i));
    // }
    PipeBarrier<PIPE_ALL>();
    // DumpTensor(ropeSinLocal_, 100331, constInfo_.ropeHeadDim * dealRowCount);
    // DumpTensor(ropeSinFp32Local_, 100332, constInfo_.ropeHeadDim * dealRowCount);
    // printf("[COMP] COMP::rotaryMode:%hhu\n", static_cast<uint8_t>(COMP::rotaryMode));
    RotaryPosEmb<COMP::rotaryMode>(rmsNormResLocal, rmsNormResLocal, ropeCosFp32Local_, ropeSinFp32Local_, tempLocal, gatherOffsetCast_, dealRowCount, 
                                    constInfo_.ropeHeadDim, constInfo_.headDim, constInfo_.headDim - constInfo_.ropeHeadDim);
    // if (COMP::rotaryMode == ROTARY_MODE::INTERLEAVE) {
    //     RotaryPosEmb<ROTARY_MODE::INTERLEAVE>(rmsNormResLocal, rmsNormResLocal, ropeCosFp32Local_, ropeSinFp32Local_, tempLocal, gatherOffsetCast_, dealRowCount, 
    //                                 constInfo_.ropeHeadDim, static_cast<uint64_t>(constInfo_.headDim), static_cast<uint64_t>(constInfo_.headDim - constInfo_.ropeHeadDim));
    // } else {
    //     RotaryPosEmb<ROTARY_MODE::HALF>(rmsNormResLocal, rmsNormResLocal, ropeCosFp32Local_, ropeSinFp32Local_, tempLocal, gatherOffsetCast_, dealRowCount, 
    //                                 constInfo_.ropeHeadDim, static_cast<uint64_t>(constInfo_.headDim), static_cast<uint64_t>(constInfo_.headDim - constInfo_.ropeHeadDim));
    // } 
    
    // // CopyOut
    // CopyFinalResultOut(info, outputLocal);
    // printf("[RotaryPosEmb] RotaryPosEmb result dump\n");
    LocalTensor<X_T>  outputLocal = rmsNormResLocal[0].template ReinterpretCast<X_T>();
    // DumpTensor(ropeCosFp32Local_, 10041, computeSize);
    // DumpTensor(ropeSinFp32Local_, 10042, computeSize);
    // DumpTensor(rmsNormResLocal, 1004, computeSize);
    PipeBarrier<PIPE_ALL>();
    Cast(outputLocal, rmsNormResLocal, RoundMode::CAST_RINT, computeSize);
    PipeBarrier<PIPE_ALL>();
    DataCopy(cmpKvOutGm_[offset], outputLocal, computeSize);
    PipeBarrier<PIPE_ALL>();
    // printf("[cmpKvOutGm_] output\n");
    // DumpTensor(outputLocal, 1005, computeSize);
    // DumpTensor(cmpKvOutGm_[offset], 1006, computeSize);
    // SetFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::RmsNorm(const Compressor::RunInfo& info, uint32_t startRow, uint32_t dealRowCount)
{
    struct RmsNormParam rmsNormParams;
    // RmsNormVF(const LocalTensor<T> &outputLocal, const LocalTensor<T> &inputLocal, const LocalTensor<GammaType> &gammaLocal,
    //     rmsNormParams);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::CalRope(const Compressor::RunInfo& info)
{

}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::SplitCoreV2(const Compressor::RunInfo& info)
{   
    // 累积N个基本块数据后做vec2，N=2，传入的RunInfo包含该组核处理的数据块的bStart、bEnd、sStart、sEnd以及dealTcCount；
    // 每组核切M方向将C1/V1后的数据分8 * 2个vec核上进行V2计算
    // 每次进行v2计算都会根据当前情况将workspace中的每组核处理的数据重新分到当前组的vec核

    // Input: syncAll前每组cube核处理的实际数据块在batch及s方向的起止idx及实际数据量(m方向)
    // Output: 每个vec核的处理数据块在m方向的起止位置及输出到Gm上的起始位置
    uint32_t coreNum = constInfo_.dBasicBlockNum * 2; // 组中有多少个vec核:16
    uint32_t currCoreIdx = GetBlockIdx(); // 当前vec核ID
    uint32_t curVecCoreGroupIdx = currCoreIdx / coreNum; // 当前vec核所在组ID
    // vec1ResGmStart = curVecCoreGroupIdx * constInfo_.nSize * constInfo_.tcBaseSize * constInfo_.headDim;
    vec1ResGmStart = 0;
    // 1.计算总vec2基本块数量
    // uint64_t totalBaseNum = (constInfo_.coreGroupNum * constInfo_.nSize * constInfo_.tcBaseSize + v2MBaseSize - 1) / v2MBaseSize; // TODO:不是按照实际数据量计算，暂时按照m方向完整基本块计算数据量 
    uint64_t totalBaseNum = info.dealScSize; // 当前组核累积的实际数据量
    // 2.每个vec核上分到的数据量
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
        // uint32_t remainder = totalBaseNum % coreNum;
        // avgBaseNum = (currCoreIdx % coreNum) < remainder ? avgBaseNum + 1 : avgBaseNum;
    } else {
        usedCoreNum = totalBaseNum;
    }
    if (currCoreIdx % coreNum >= usedCoreNum) {
        return;
    }
    // 3.计算每个vec核的起始结束位置
    uint32_t accumBaseNum = 0; // 当前累积的基本块数
    uint32_t targetBaseNum = (currCoreIdx % coreNum + 1) * avgBaseNum;  // 当前vec核目标要达到的基本块数量
    uint32_t targetStartBaseNum = targetBaseNum - avgBaseNum;           // 分当前vec核时前面已经完成分核的基本块数量
    // printf("[debug] coreNum: %u, curVecCoreGroupIdx: %u, targetBaseNum: %u, targetStartBaseNum: %u\n",
        // coreNum, curVecCoreGroupIdx, targetBaseNum, targetStartBaseNum);
    bool setStart = false;
    for (uint32_t i = 0; i < totalBaseNum; ++i) {
        if (accumBaseNum >= totalBaseNum) {
            return;
        }
        accumBaseNum += 1;
        if (!setStart && (accumBaseNum > targetStartBaseNum)) {
            v2TcStartIdx = i;
            setStart = true;
        } 
        if (setStart && (accumBaseNum >= targetBaseNum || i == (totalBaseNum - 1))) {
            // 更新当前核的End分核信息
            v2TcEndIdx = i + 1;
            GetScIdxInfo(info.bStart, info.scStart, info.dealScSize, v2TcStartIdx, v2TcEndIdx,
                OutputBStartIdx, OutputSStartIdx, OutputSize);
            // printf("[debug] bStart: %d, scStart: %d, dealScSize: %d, v2TcStartIdx: %d, v2TcEndIdx: %d, OutputBStartIdx: %d, OutputSStartIdx: %d, OutputSize: %d\n",
                // info.bStart, info.scStart, info.dealScSize, v2TcStartIdx, v2TcEndIdx, OutputBStartIdx, OutputSStartIdx, OutputSize);
            return;
        }
    }
}

}

#endif // COMPRESSOR_BLOCK_VECTOR_H