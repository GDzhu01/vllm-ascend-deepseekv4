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
 * \file sparse_flash_attention_service_vector_mla.h
 * \brief
 */
#ifndef SPARSE_ATTN_SHAREDKV_SWA_BLOCK_VECTOR_H
#define SPARSE_ATTN_SHAREDKV_SWA_BLOCK_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../sparse_attn_sharedkv_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SAST> class SWAVectorBlock {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename SAST::kvType;
    using OUT_T = typename SAST::outputType;
    using UPDATE_T = T;
    using SINKS_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline SWAVectorBlock(){};
    __aicore__ inline void ProcessVec1L(const RunInfo &info);
    __aicore__ inline void ProcessVec2L(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo,
                                      const SparseAttnSharedkvTilingData *__restrict tilingData);
    __aicore__ inline void InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<T> sinksGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void CopySinksIn();
    __aicore__ inline void SliceAndContactSinksValue(uint32_t nIdx, uint32_t dealRowCount);
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void AmlaVecCompute(const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb,
                                          LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
                                          uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                          uint32_t columnCount);
    __aicore__ inline void ProcessAmlaNupdate(const RunInfo &info, const MSplitInfo &mSplitInfo);
    // __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
    //                                                    LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    // __aicore__ inline void CopyFALseToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
    //                                     LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    __aicore__ inline void SetBmm2FirstSInnerBias(const RunInfo &info, const MSplitInfo &mSplitInfo);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t mStartRow,
                                            uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    static constexpr bool PAGE_ATTENTION = SAST::pageAttention;
    // static constexpr int TEMPLATE_MODE = SAST::templateMode;
    static constexpr bool FLASH_DECODE = SAST::flashDecode;
    static constexpr SAS_LAYOUT LAYOUT_T = SAST::layout;
    static constexpr SAS_LAYOUT KV_LAYOUT_T = SAST::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_INPUT_BUF2_PONG_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_1K;
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);  // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                                // 32/4=8
    static constexpr uint32_t MAX_N1_SIZE = 128U;
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr SINKS_T R0 = 1.0f;

    const SparseAttnSharedkvTilingData *__restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<T> sinksGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blkTableGm_;
    // GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> keyGm_;
    // GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;
    GlobalTensor<KV_T> oriKvGm_;
    GlobalTensor<KV_T> cmpKvGm_;
    GlobalTensor<int32_t> oriBlockTableGm_;
    GlobalTensor<int32_t> cmpBlockTableGm_;

    // ================================Local Buffer区====================================
    TBuf<> inputBuff1;            // 32K
    TBuf<> inputBuff2;            // 16K
    TBuf<> outputBuff1;           // 32K
    TBuf<> outputBuff2;           // 4K

    TBuf<> tmpBuff1;              // 32K
    TBuf<> v0ValidSizeBuff;       // 8K

    TBuf<> sinksBuff;              // 1K
    TBuf<> sinksBrcbBuff;          // 12K

    TBuf<> nValueBuff;
    TBuf<> cofValueBuff;
    TBuf<> aMlaSumBuff;
    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> nValueUb;
    LocalTensor<T> cofValueUb;
    LocalTensor<T> aMlaSumUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<SINKS_T> sinksUb;
    LocalTensor<SINKS_T> sinksBrcbUb;
};

// ============================== init ==============================================
template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_8K * 2);  // 2:pingpong
    pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_4K);

    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);

    // M_MAX = 512/2vector = 256, 256 * sizeof(T) * N_Buffer
    pipe->InitBuffer(nValueBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(cofValueBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(aMlaSumBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_1K);

    pipe->InitBuffer(sinksBuff, MAX_N1_SIZE * sizeof(SINKS_T));
    // pipe->InitBuffer(sinksBrcbBuff, MAX_N1_SIZE * sizeof(SINKS_T) * BLOCK_ELEMENT_NUM);
    pipe->InitBuffer(sinksBrcbBuff, MAX_N1_SIZE * sizeof(SINKS_T) * BLOCK_ELEMENT_NUM * 3U);  // 分配256+N1大小内存，其中256是m轴VEC最大切块

    nValueUb = nValueBuff.Get<T>();
    cofValueUb = cofValueBuff.Get<T>();
    aMlaSumUb = aMlaSumBuff.Get<T>();

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    // v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();

    sinksUb = sinksBuff.Get<SINKS_T>();
    sinksBrcbUb = sinksBrcbBuff.Get<SINKS_T>();
}


template <typename SAST>
__aicore__ inline void
SWAVectorBlock<SAST>::InitParams(const struct ConstInfo &constInfo,
                                                 const SparseAttnSharedkvTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename SAST>
__aicore__ inline void
SWAVectorBlock<SAST>::InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
    GlobalTensor<int32_t> actualSeqLengthsQGm, GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
    GlobalTensor<T> lseSumFdGm, GlobalTensor<SINKS_T> sinksGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->sinksGm = sinksGm;
}

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                    GlobalTensor<UPDATE_T> vec2ResGm,
                                                                    GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                    GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SAST> __aicore__ inline void SWAVectorBlock<SAST>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SAST> __aicore__ inline void SWAVectorBlock<SAST>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SAST> __aicore__ inline void SWAVectorBlock<SAST>::CopySinksIn()
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = constInfo.qHeadNum * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(sinksUb, sinksGm, dataCopyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    uint32_t repeatTimes = (constInfo.qHeadNum + BLOCK_ELEMENT_NUM - 1U) / BLOCK_ELEMENT_NUM;  // 每次处理 8 datablocks
    Brcb(sinksBrcbUb, sinksUb, repeatTimes, {1, BLOCK_ELEMENT_NUM});
    PipeBarrier<PIPE_V>();

    DataCopyParams repeatParams;
    repeatParams.blockCount = 1;  // 搬到有一个块超过单个vec核减分核M轴大小即可，核间切分每个vec256
    repeatParams.blockLen = constInfo.qHeadNum;
    repeatParams.srcStride = 0U;
    repeatParams.dstStride = 0U;
    for (uint32_t i = 1U; i <= 256U / constInfo.qHeadNum; i++) {
        DataCopy(sinksBrcbUb[constInfo.qHeadNum * BLOCK_ELEMENT_NUM * i], sinksBrcbUb, repeatParams);
    }
    PipeBarrier<PIPE_V>();
}

template <typename SAST> __aicore__ inline void SWAVectorBlock<SAST>::SliceAndContactSinksValue(uint32_t nIdx, uint32_t dealRowCount)
{
    uint32_t repeatTimesOnce = 128;  //由于WholeReduceMax接口中repeatTimes支持范围（0,255），因此需要分多次调用WholeReduceMax，这里就使用每次repeatTime=128
    uint32_t loopTimes = (dealRowCount + repeatTimesOnce - 1) / repeatTimesOnce;
    uint32_t repeatTimes = repeatTimesOnce;

    for (uint32_t loop = 0; loop < loopTimes; ++loop) {
        if (loop == loopTimes - 1) {
            repeatTimes = dealRowCount - loop * repeatTimesOnce;
        }
        WholeReduceMax(softmaxMaxDefaultUb[loop * repeatTimesOnce], sinksBrcbUb[(nIdx + loop * repeatTimesOnce) * BLOCK_ELEMENT_NUM],
            BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM, repeatTimes, 1, 0, 1, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
    }
}

template <typename SAST> __aicore__ inline void SWAVectorBlock<SAST>::InitSoftmaxDefaultBuffer()
{
    CopySinksIn();
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
    Duplicate(softmaxSumDefaultUb, R0, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
}

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::ElewiseCompute(const RunInfo &info,
                                                            const LocalTensor<T> &mmResUb,
                                                            uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.softmaxScale), dealRowCount * columnCount);

    if (info.isOri) {
        // SCFA ori_kv 部分不需要mask？ TODO: CFA & SWA 需要
    } else {
        // // v0的无效值判断
        // uint64_t s2ValidSizeFirstPart = v0ValidSizeUb_.GetValue(128 + info.cmpLoop % MERGE_CACHE_GM_BUF_NUM);
        // uint64_t s2ValidSizeSecondPart = v0ValidSizeUb_.GetValue(256 + info.cmpLoop % MERGE_CACHE_GM_BUF_NUM);

        // int64_t s2ProcessSize = info.actualSingleProcessSInnerSize;
        // int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
        // int64_t s2Mid = CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
        // if (s2Mid > s2ProcessSize) {
        //     s2Mid = s2ProcessSize;
        // }
        // if (unlikely(s2ValidSizeFirstPart < s2Mid)) {
        //     int64_t s2StartCeilAlign = CeilAlign(s2ValidSizeFirstPart, 8);
        //     int64_t s2MidFloorAlign = s2Mid / 8 * 8;
        //     // 场景一 s2Mid > s2ValidSizeFirstPart + oneBlk
        //     // 可以推导出s2StartCeilAlign < s2Mid   第一阶段取到s2StartCeilAlign
        //     // s2StartCeilAlign <= s2MidFloorAlign 第二阶段取到s2MidFloorAlign
        //     // 场景二 s2Mid <= s2ValidSizeFirstPart + oneBlk
        //     // 可以推导出 s2StartCeilAlign >= s2Mid 第一阶段取到mid
        //     // s2StartCeilAlign > s2MidFloorAlign 第二阶段取到s2StartCeilAlign
        //     SetInfInBlk(mmResUb, dealRowCount, columnCount, s2ValidSizeFirstPart,
        //                 s2StartCeilAlign >= s2Mid ? s2Mid : s2StartCeilAlign);
        //     SetMidInf(mmResUb, dealRowCount, columnCount, s2StartCeilAlign, s2MidFloorAlign);
        //     SetInfInBlk(mmResUb, dealRowCount, columnCount,
        //                 s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign, s2Mid);
        // }
        // if (unlikely(s2ValidSizeSecondPart < s2ProcessSize - s2Mid)) {
        //     // 场景一 s2Mid + s2ValidSizeSecondPart > s2ProcessSize + oneBlk
        //     // 可以推导出 s2StartCeilAlign < s2ProcessSize 第一阶段取到s2StartCeilAlign
        //     // s2StartCeilAlign <= s2EndFloorAlign 第二阶段取到s2EndFloorAlign
        //     // 场景二 s2Mid + s2ValidSizeSecondPart <= s2ProcessSize + oneBlk
        //     // 可以推导出 s2StartCeilAlign >= s2ProcessSize 第一阶段取到s2ProcessSize
        //     // s2StartCeilAlign > s2EndFloorAlign 第二阶段取到s2StartCeilAlign
        //     int64_t s2StartCeilAlign = CeilAlign(s2Mid + s2ValidSizeSecondPart, 8);
        //     int64_t s2EndFloorAlign = s2ProcessSize / 8 * 8;
        //     SetInfInBlk(mmResUb, dealRowCount, columnCount, s2Mid + s2ValidSizeSecondPart,
        //                 s2StartCeilAlign >= s2ProcessSize ? s2ProcessSize : s2StartCeilAlign);
        //     SetMidInf(mmResUb, dealRowCount, columnCount, s2StartCeilAlign, s2EndFloorAlign);
        //     SetInfInBlk(mmResUb, dealRowCount, columnCount,
        //                 s2StartCeilAlign <= s2EndFloorAlign ? s2EndFloorAlign : s2StartCeilAlign, s2ProcessSize);
        // }
    }

}


template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::SoftmaxFlashV2Compute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb[startRow];
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
    }
    if (actualColumnCount !=0) {
        SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, SAS_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        uint32_t dealRowCountAlign = SASAlign(dealRowCount, FP32_BLOCK_ELEMENT_NUM);
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCountAlign);
        PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCountAlign);
    }
}

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::AmlaVecCompute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t calCount = dealRowCount;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // compute n(i)
    LocalTensor<T> nTmp = softmaxTmpUb.template ReinterpretCast<T>();
    LocalTensor<T> nUpdateTmp = nTmp[SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nTmp, softmaxMaxUb[softmaxOutOffset], ((T)(-1.0)) * RECIP_OF_LN2, calCount);

    PipeBarrier<PIPE_V>();
    Cast(nTmp, nTmp, RoundMode::CAST_ROUND, calCount);
    PipeBarrier<PIPE_V>();

    uint32_t prOutIdx = (info.loop - 1) % (constInfo.preLoadNum);
    uint32_t PreSoftmaxOutOffset = prOutIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // n(i) - n(i-1)
    if (info.isFirstSInnerLoop) {
        Duplicate(nUpdateTmp, ConstInfo::FLOAT_ZERO, calCount); // n1=n0
    } else {
        Sub(nUpdateTmp, nTmp, nValueUb[PreSoftmaxOutOffset], calCount);
    }
    PipeBarrier<PIPE_V>();
    // update n(i), DataCopy not support when calCount is not align 32B, so use Adds
    Adds(nValueUb[softmaxOutOffset], nTmp, ConstInfo::FLOAT_ZERO, calCount);
    PipeBarrier<PIPE_V>();

    // update softmax res
    LocalTensor<T> nUpdateTmp2 = nTmp[2 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<KV_T> nTmp_KvT = nTmp[3 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)].template ReinterpretCast<KV_T>();
    LocalTensor<T> tmpCofUb = nTmp[4 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<T> epsUb = nTmp[5 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nUpdateTmp2, softmaxMaxUb[softmaxOutOffset], RECIP_OF_LN2, calCount);
    PipeBarrier<PIPE_V>();
    Add(nTmp, nUpdateTmp2, nTmp, calCount);
    PipeBarrier<PIPE_V>();
    Muls(nTmp, nTmp, LN2, calCount);
    PipeBarrier<PIPE_V>();
    Exp(nTmp, nTmp, calCount);
    PipeBarrier<PIPE_V>();
    Cast(nTmp_KvT, nTmp, RoundMode::CAST_ROUND, calCount);       // fp32->fp16/bf16
    PipeBarrier<PIPE_V>();
    Cast(nUpdateTmp2, nTmp_KvT, RoundMode::CAST_NONE, calCount); // fp16/bf16->fp32
    PipeBarrier<PIPE_V>();
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        Mul(aMlaSumUb[softmaxOutOffset], softmaxSumUb[softmaxOutOffset], nUpdateTmp2, calCount);
    }
    if (actualColumnCount == 0) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        return;
    }
    LocalTensor<T> nTmp3 = nTmp[6 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Brcb(nTmp3, nUpdateTmp2, (dealRowCount + 7) / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    RowMuls(mmResUb, mmResUb, nTmp3, dealRowCount, columnCount, actualColumnCount);

    Div(tmpCofUb, nTmp, nUpdateTmp2, calCount); // cof(i)=tmpS32/tmpS16
    if (info.isFirstSInnerLoop) {
        Duplicate(cofValueUb[softmaxOutOffset], (T)1.0, calCount);       // cof_0=1
        PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[softmaxOutOffset], tmpCofUb, calCount);    // 1 / cof(i)
    } else {
        PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[PreSoftmaxOutOffset], tmpCofUb, calCount); // cof(i - 1) / cof(i)
    }
    PipeBarrier<PIPE_V>();

    Adds(cofValueUb[softmaxOutOffset], tmpCofUb, ConstInfo::FLOAT_ZERO, calCount); // store cof(i)
    Adds(epsUb, epsUb, (T)(-1.0), calCount); // cof(i - 1) / cof(i) - 1
    PipeBarrier<PIPE_V>();
    Muls(epsUb, epsUb, (T)1.5, calCount);    // (cof(i - 1) - cof(i)) / cof(i) * 1.5

    Maxs(nUpdateTmp, nUpdateTmp, (T)(-30.0), calCount); // N = max(n(i) - n(i-1), -30)
    PipeBarrier<PIPE_V>();
    Adds(epsUb, epsUb, (T)(0.000001), calCount);
    PipeBarrier<PIPE_V>();
    Add(nUpdateTmp, nUpdateTmp, epsUb, calCount);
    PipeBarrier<PIPE_V>();
    Muls(nUpdateTmp, nUpdateTmp, FLOAT_E_SCALAR, calCount); // N = N * pow(2, 23)
    PipeBarrier<PIPE_V>();

    // nUpdate int32 out
    LocalTensor<int32_t> tmQue = outputBuff2.Get<int32_t>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    LocalTensor<int32_t> nInt32Out = tmQue[startRow]; // 缓存nUpdate

    Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, dealRowCount);
    PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t loopId)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    // if (!info.isOri) {
    //     if (loopId == 0) {
    //         WaitFlag<HardEvent::MTE2_S>(0);
    //     }
    // }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, mmResUb, dealRowCount, columnCount);

    PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                            info.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    AmlaVecCompute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                    info.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> tmpMMResCastTensor = outputBuff1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::SetBmm2FirstSInnerBias(const RunInfo &info, const MSplitInfo &mSplitInfo)
{
    uint32_t mSplitSize = 16U;
    uint64_t baseoffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
                            (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) * constInfo.headDim;
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    LocalTensor<int32_t> tmpTensor = outputBuff1.Get<int32_t>();
    Duplicate(tmpTensor, static_cast<int32_t>(394264576), mSplitSize * constInfo.headDim);  // 394264576 : fp32下2^(-80)的二进制表示对应的int32数值
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    for (uint32_t loop = 0; loop < loopCount; loop++) {
        DataCopy(mm2ResInt32Gm[baseoffset + loop * mSplitSize * constInfo.headDim], tmpTensor, mSplitSize * constInfo.headDim);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::ProcessAmlaNupdate(const RunInfo &info, const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    if (info.isFirstSInnerLoop) {
        SetBmm2FirstSInnerBias(info, mSplitInfo);
        return;
    }

    LocalTensor<int32_t> nUpdateTensor = outputBuff2.Get<int32_t>(); // shape:1/2*s1*g
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

    constexpr uint32_t dGroupSize = 128U;
    constexpr uint32_t mSplitSize = 64U;     // tmpQue size 32KB，一次只能处理64个N，最大保存的数据大小：64*128*sizeof(int32)
    constexpr uint32_t ONE_BLOCK_SIZE = 32U; // 32B

    uint32_t subMSize = SASAlign(mSplitInfo.vecDealM, 16U);
    uint16_t elementPerBlock = ONE_BLOCK_SIZE / sizeof(int32_t);      // 单个datablock的元素数，int32_t类型的为32/4=8
    uint32_t loopCount = (subMSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = subMSize - (loopCount - 1) * mSplitSize; // 尾块

    for (uint32_t loop = 0, processMSize = mSplitSize; loop < loopCount; loop++) {
        if (loop == (loopCount - 1)) {
            processMSize = tailSplitSize;
        }
        LocalTensor<int32_t> tmpQue = outputBuff1.Get<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        // (m,1)单次brcb扩充成(m,8), 重复16次, 扩充为(m,128)
        for (uint32_t i = 0; i < dGroupSize / elementPerBlock; i++) {
            Brcb(tmpQue[i * elementPerBlock],
                 nUpdateTensor[loop * mSplitSize],
                 static_cast<uint8_t>((processMSize + elementPerBlock - 1) / elementPerBlock),
                 {static_cast<uint16_t>(dGroupSize / elementPerBlock), // 单次迭代内，目的操作数不同datablock间地址步长,单位为datablock
                  static_cast<uint16_t>(dGroupSize)});                 // 相邻迭代间，目的操作数相同datablock地址步长
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t baseoffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
                              (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + loop * mSplitSize) * constInfo.headDim;

        SetAtomicAdd<int32_t>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(processMSize);
        dataCopyParams.blockLen = dGroupSize * sizeof(int32_t) / ONE_BLOCK_SIZE; // 每个block是128个元素，单位为32B
        dataCopyParams.srcStride = 0;                                            // 前面一个数据块的尾与后面数据块的头的间隔
        dataCopyParams.dstStride = static_cast<uint16_t>((constInfo.headDim - dGroupSize) *
                                                         sizeof(int32_t) / ONE_BLOCK_SIZE); // 单位为32B
        for (uint32_t i = 0; i < constInfo.headDim / dGroupSize; i++) {          // 4=512/128
            DataCopy(mm2ResInt32Gm[baseoffset + i * dGroupSize] ,tmpQue, dataCopyParams);
        }
        SetAtomicNone();
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::ProcessVec1SingleBuf(const RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = info.actualSingleProcessSInnerSize == 0 ?
        16 : BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;

    SliceAndContactSinksValue((mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) % constInfo.qHeadNum, mSplitInfo.vecDealM);

    // if (!info.isOri) {
    //     DataCopyExtParams dataCopyParams;
    //     dataCopyParams.blockCount = 1;
    //     dataCopyParams.blockLen = 256 * sizeof(int32_t);
    //     dataCopyParams.srcStride = 0;
    //     dataCopyParams.dstStride = 0;
    //     DataCopyPadExtParams<int32_t> padParams;
    //     // 额外偏移128个元素，避免不同loop下v0和v1互相影响
    //     DataCopyPad(v0ValidSizeUb_[128], kvValidSizeGm_[info.cmpLoop % MERGE_CACHE_GM_BUF_NUM * (128 * 2)],
    //                 dataCopyParams, padParams);
    //     SetFlag<HardEvent::MTE2_S>(0);
    //     if (unlikely(loopCount == 0)) {
    //         // scalar同步影响较大，挪到循环内部进行
    //         WaitFlag<HardEvent::MTE2_S>(0);
    //     }
    // }

    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign, i);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

// =======================vec1=============================

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::ProcessVec1L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(constInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
        CrossCoreWaitFlag(constInfo.syncC2V1);
        // add nUpdate to mm2ResGm
        if (info.actualSingleProcessSInnerSize != 0) {
            ProcessAmlaNupdate(info, mSplitInfo);
            CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1NupdateC2);
        }
        // move lse for flash decode or FA
        if (info.s2Idx == info.curSInnerLoopTimes - 1 && ( info.tndIsS2SplitCore)) {
            uint32_t outIdx = info.loop % (constInfo.preLoadNum);
            auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
            auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
            // if (info.tndIsS2SplitCore) {
            //     if constexpr (FLASH_DECODE) {
            //         ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
            //     }
            // }
        }
    }
}

// =======================vec2=============================

template <typename SAST>
__aicore__ inline uint64_t SWAVectorBlock<SAST>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    return 0;
}

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::ProcessVec2SingleBuf(const RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (info.s2Idx + 1 != info.curSInnerLoopTimes) {
        return;
    }
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    ProcessVec2Inner(info, mSplitInfo, 0, mSplitInfo.vecDealM);
}

template <typename SAST> __aicore__ inline void SWAVectorBlock<SAST>::ProcessVec2L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(constInfo.syncC2V2);
        ProcessVec2SingleBuf(info, mSplitInfo);
    }
}

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::ProcessVec2Inner(const RunInfo &info,
                                                                              const MSplitInfo &mSplitInfo,
                                                                              uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * mSplitSize + mStartRow, dealSize,
                             constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename SAST>
__aicore__ inline void SWAVectorBlock<SAST>::GetConfusionTransposeTiling(
    int64_t numR, int64_t numC, const uint32_t stackBufferSize, const uint32_t typeSize,
    ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename SAST>
__aicore__ inline void
SWAVectorBlock<SAST>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                        uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                                        uint32_t actualColumnCount)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim +              // taskoffset
                      info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim + // 份数offset
                      wsMStart * actualColumnCount;                                                                 // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    if (info.actualSingleProcessSInnerSize== 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, dealRowCount * actualColumnCount, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SAST>
__aicore__ inline void
SWAVectorBlock<SAST>::Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
                                                           uint32_t wsMStart, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + wsMStart * actualColumnCount], attenOutUb, dataCopyParams);
    return;
}

template <typename SAST>
__aicore__ inline void
SWAVectorBlock<SAST>::Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                         uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SAST>
__aicore__ inline void
SWAVectorBlock<SAST>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename SAST>
__aicore__ inline void
SWAVectorBlock<SAST>::DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                           uint32_t startRow, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint64_t srcGmOffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
                            mStart * columnCount;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    // 将绝对值大于1e10的数置为0
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);
    LocalTensor<T> absBmm2ResUb = bmm2ResUb.template ReinterpretCast<T>();
    Abs(absBmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> cmpMaskUb = absBmm2ResUb.template ReinterpretCast<uint8_t>();
    CompareScalar(cmpMaskUb, absBmm2ResUb, (T)1e10, CMPMODE::LE, vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    Select(tmpBmm2ResUb, cmpMaskUb, tmpBmm2ResUb, ConstInfo::FLOAT_ZERO,
           SELMODE::VSEL_TENSOR_SCALAR_MODE, vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t idx = info.loop % (constInfo.preLoadNum);
    LocalTensor<T> tmpSumUb = v0ValidSizeBuff.Get<T>()[384]; // sumUb用临时内存 16 * 32B  = 512B
    Brcb(tmpSumUb, aMlaSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (dealRowCount + 7) / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    RowDivs(bmm2ResUb, tmpBmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    Bmm2ResCopyOut(info, bmm2ResUb, mStart, dealRowCount, columnCount, actualColumnCount);
}

template <typename SAST>
__aicore__ inline void
SWAVectorBlock<SAST>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

template <typename SAST>
__aicore__ inline void
SWAVectorBlock<SAST>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}

#endif