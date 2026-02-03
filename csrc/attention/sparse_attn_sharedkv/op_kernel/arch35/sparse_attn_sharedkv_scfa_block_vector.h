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
 * \file flash_attention_score_block_vec_base_scfa.h
 * \brief
 */
 // TODO 修改
#ifndef FLASH_ATTENTION_SCORE_BLOCK_VEC_SCFA_H_
#define FLASH_ATTENTION_SCORE_BLOCK_VEC_SCFA_H_
#include "util_regbase.h"
#include "infer_flash_attention_comm.h" 

#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_scfa.h"
#include "vf/vf_flashupdate_new_scfa.h"

using namespace AscendC;
using namespace SCFaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
TEMPLATES_DEF
class SCFABlockVec {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1HalfS1BaseSize = s1BaseSize >> 1;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);

    // ==================== Functions ======================
    __aicore__ inline SCFABlockVec() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const optiling::FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx) {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            tilingData = tiling;
            this->InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer, hasRope> &constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);

    TPipe *tPipe;
    const optiling::FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
    
    // 在CleanOutput中初始化
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;

    /* =====================V侧UB变量==================== */
    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2]; 
    /* =================初始化后不变的信息================= */
    T negativeFloatScalar;

protected:
/* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = INPUT_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    using VEC2_RES_T = float;
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize);

private:
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer, hasRope> &constInfo)
{
    bmm1ResBuf.WaitCrossCore();

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();

    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    if (runInfo.s2LoopCount == 0) {
        if (likely(runInfo.s2RealSize == 128)) {
            ProcessVec1Vf<T, INPUT_T, false, s1BaseSize, s2BaseSize, SCFaVectorApi::EQ_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.scaleValue), negativeFloatScalar);
        }
    } else {
        if (likely(runInfo.s2RealSize == 128)) {
            ProcessVec1Vf<T, INPUT_T, true, s1BaseSize, s2BaseSize, SCFaVectorApi::EQ_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.scaleValue), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, INPUT_T, true, s1BaseSize, s2BaseSize, SCFaVectorApi::GT_0_AND_LTE_64_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.scaleValue), negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

    DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (blockBytes / sizeof(INPUT_T)) * (runInfo.s1RealSize - runInfo.halfS1RealSize)], stage1CastTensor,
        {s2BaseSize / 16, (uint16_t)runInfo.halfS1RealSize,
        (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
        (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    // ======================================================
    if (runInfo.s2LoopCount != 0) {
        SCFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo) {
    if (unlikely(runInfo.vec2S1BaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }
    
    // TOTO:为什么是BaseSize
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize; 
    int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, runInfo.vec2S1RealSize);
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(
                vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize);
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;

    if constexpr (!IsSameType<INPUT_T, VEC2_RES_T>::value) {
        attenOut.SetAddr(vec2ResUb.address_);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4;
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2S1RealSize;

    int64_t attenOutOffset = constInfo.dSizeV;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        attenOutOffset = constInfo.n2GDv;
        if (constInfo.isPfaGS1Merge) {
            attenOutOffset = 0;
            dataCopyParams.blockLen *= constInfo.gSize;
            dataCopyParams.blockCount /= constInfo.gSize;
        } else if (constInfo.isGqa) {
            attenOutOffset = constInfo.dSizeV;
        }
    } else {
        if (constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_BSH)) {
            attenOutOffset = constInfo.n2GDv;
            if (constInfo.isPfaGS1Merge) {
                attenOutOffset = 0;
                dataCopyParams.blockLen *= constInfo.gSize;
                dataCopyParams.blockCount /= constInfo.gSize;
            } else if (constInfo.isGqa) {
                attenOutOffset = constInfo.dSizeV;
            }
        }
    }

    if (constInfo.isPfaGS1Merge && dSizeAligned64 - constInfo.dSizeV != 0 && (constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_BSH) || constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_TND))) {
        for(int64_t i = 0; i < runInfo.vec2S1BaseSize / constInfo.gSize; i++){
            attenOutOffset = i * constInfo.dSizeV * constInfo.gSize * constInfo.n2Size;
            dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
            dataCopyParams.blockCount = constInfo.gSize;
            dataCopyParams.dstStride = 0;
            DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + attenOutOffset],
                attenOut[i * constInfo.gSize * dSizeAligned64], dataCopyParams);
        }
    } else {
        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
            attenOut, dataCopyParams); 
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo)
{
    auto &initParams = this->tilingData->initOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * initParams.singleCoreSize], singleInitOutputSize, 0.0);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo<isInfer, hasRope> &constInfo) 
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (this->tilingData->initOutputParams.needInit == 1) {
            InitOutputSingleCore(constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    tPipe->InitBuffer(softmaxSumBuf[0], 128); // 64/ 2*sizeof(float) = 128
    tPipe->InitBuffer(softmaxSumBuf[1], 128); 
    tPipe->InitBuffer(softmaxMaxBuf[0], 128); 
    tPipe->InitBuffer(softmaxMaxBuf[1], 128); 
    tPipe->InitBuffer(softmaxExpBuf[0], 128); 
    tPipe->InitBuffer(softmaxExpBuf[1], 128); 
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer, hasRope> &constInfo)
{
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);

    SoftmaxInitBuffer();

    tPipe->InitBuffer(stage1OutQue[0], 1, 8448); // （32 + 1） * 128 * 2(bf16)
    tPipe->InitBuffer(stage1OutQue[1], 1, 8448);
    tPipe->InitBuffer(stage2OutBuf, 32 * dTemplateAlign64 * sizeof(T)); //s1Base/cv_ratio * 512 * 4(float)

    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();

    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
    sharedParams.bSize = inputParamsRegbase.bSize;
    sharedParams.n2Size = inputParamsRegbase.n2Size;
    sharedParams.gSize = inputParamsRegbase.gSize;
    sharedParams.s1Size = inputParamsRegbase.s1Size;
    sharedParams.s2Size = inputParamsRegbase.s2Size;
    sharedParams.dSize = inputParamsRegbase.dSize;
    sharedParams.dSizeV = inputParamsRegbase.dSizeV;
    if constexpr (hasRope) {
        sharedParams.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        sharedParams.dSizeRope = 0;
    }
    sharedParams.preTokens = inputParamsRegbase.preTokens;
    sharedParams.nextTokens = inputParamsRegbase.nextTokens;
    sharedParams.s1SparseValidSize = inputParamsRegbase.s1SparseValidSize;
    sharedParams.s2SparseValidSize = inputParamsRegbase.s2SparseValidSize;
    sharedParams.bandIndex = inputParamsRegbase.bandIndex;
    sharedParams.implMode = inputParamsRegbase.implMode;
    sharedParams.layoutType = inputParamsRegbase.layoutType;
    sharedParams.sparseType = inputParamsRegbase.sparseType;
    sharedParams.compressMode = inputParamsRegbase.attenMaskCompressMode;
    sharedParams.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
    sharedParams.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
 
    sharedParams.isBSNDOut = inputParamsRegbase.isBSNDOut;
    sharedParams.fromFused = inputParamsRegbase.fromFused;
    // sharedParams.isRowInvalid = inputParamsRegbase.isRowInvalid;
    sharedParams.headNumRatio = inputParamsRegbase.headNumRatio;
    sharedParams.isGqa = inputParamsRegbase.isGqa;
    sharedParams.isPfaGS1Merge = (inputParamsRegbase.isGqa && sharedParams.s1Size > 1);
    sharedParams.isKvContinuous = inputParamsRegbase.isKvContinuous;
    sharedParams.actualSeqLengthsSize = inputParamsRegbase.actualSeqLengthsSize;
    sharedParams.actualSeqLengthsKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
    sharedParams.isActualSeqLengthsNull = inputParamsRegbase.isActualSeqLengthsNull;
    sharedParams.isActualSeqLengthsKVNull = inputParamsRegbase.isActualSeqLengthsKVNull;
    sharedParams.isQHasLeftPadding = inputParamsRegbase.isQHasLeftPadding;
    sharedParams.isKVHasLeftPadding = inputParamsRegbase.isKVHasLeftPadding;
    // pageAttention
    if constexpr (isPa) {
        sharedParams.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        sharedParams.blockSize = inputParamsRegbase.blockSize;
        sharedParams.paLayoutType = inputParamsRegbase.paLayoutType;
        sharedParams.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    }
    // prefix
    if constexpr (enableKVPrefix) {
        sharedParams.isActualSharedPrefixLenNull = inputParamsRegbase.isActualSharedPrefixLenNull;
        sharedParams.kvPrefixSize = inputParamsRegbase.prefixSeqInnerSize;
    }
    auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    sharedParams.s1OuterSize = multiCoreParamsRegbase.s1OuterSize;
    sharedParams.coreNum = multiCoreParamsRegbase.coreNum;
    /* 多核切分偏移计算 */
    sharedParams.multiCoreInnerOffset = multiCoreParamsRegbase.sparseStartIdx[aicIdx];
    sharedParams.multiCoreInnerLimit = multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    sharedParams.bnStartIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx];
    sharedParams.bnEndIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
    sharedParams.needInit = this->tilingData->initOutputParams.needInit;

    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
            #pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams<isInfer, isPa>) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }

}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
    negativeScalar = *((float *)&tmp1);
}

TEMPLATES_DEF
class SCFABlockVecDummy {
public:
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    // TODO 是否需要补充其他函数
    __aicore__ inline SCFABlockVecDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut,
        ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const optiling::FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo) {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo) {}
};
}
#endif //FLASH_ATTENTION_SCORE_BLOCK_VEC_SCFA_H_

