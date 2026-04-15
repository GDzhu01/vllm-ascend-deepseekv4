/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_pertoken_quant.h
 * \brief
 */
#ifndef EPILOGUE_BLOCK_EPILOGUE_PERTOKEN_QUANT_H
#define EPILOGUE_BLOCK_EPILOGUE_PERTOKEN_QUANT_H

#include "kernel_operator.h"
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"
#include "../tile/tile_copy_policy.h"

namespace Cgmct {
namespace Gemm {
namespace Block {
using namespace AscendC;
namespace {
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr float INT4_MAX_VALUE = 7.0f;
constexpr float FP8_E5M2_OFFSET_VALUE = 114688.0f;
constexpr float FP8_E4M3FN_OFFSET_VALUE = 896.0f;
constexpr float HIFLOAT8_OFFSET_VALUE = 65536.0f;
constexpr float INT8_OFFSET_VALUE = 255.0f;
constexpr float INT4_OFFSET_VALUE = 15.0f;
constexpr float NEGATIVE_ONE = -1.0f;
constexpr uint32_t REG_LEN = 64;
constexpr uint32_t FIFTEEN = 15;
constexpr uint32_t SIXTEEN = 16;
constexpr uint32_t SEVEN = 7;
constexpr uint32_t EIGHT = 8;
constexpr uint32_t THIRTY_ONE = 31;
constexpr uint32_t THIRTY_TWO = 32;
constexpr uint32_t SIXTY_THREE = 63;
constexpr uint32_t SIXTY_FOUR = 64;
constexpr uint32_t USE_BUFFER_NUM = 2;
constexpr uint32_t COMPARE_INT = 255;
} // namespace

constexpr static AscendC::MicroAPI::CastTrait castTraitB16ToB32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToI16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr static AscendC::MicroAPI::CastTrait castTraitI16ToF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
constexpr static AscendC::MicroAPI::CastTrait castTraitF16ToI8 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};
constexpr static AscendC::MicroAPI::CastTrait castTraitF32tofp8 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT};
constexpr static AscendC::MicroAPI::CastTrait castTraitF32toh8 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_ROUND};

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
constexpr float NEG_INFINITY = -INFINITY;
#define QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS template <typename XDtype_, typename YDtype_>
#define QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS XDtype_, YDtype_

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
class BlockEpiloguePertokenQuant {
public:
    __aicore__ inline BlockEpiloguePertokenQuant(TPipe *pipe) : pPipe_(pipe)
    {
    }

    struct Arguments {
        GM_ADDR workSpaceGMAddr{nullptr};
        GM_ADDR smoothScaleGMAddr{nullptr};
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        uint32_t rowLen = 0;
        uint32_t ubAvail = 1;
        bool hasSmooth = false;
        Arguments() = default;
    };

    using Params = Arguments;
    using XDtype = XDtype_;
    using YDtype = YDtype_;
    using YCopyDtype = std::conditional_t<IsSameType<YDtype_, int4b_t>::value, uint8_t, YDtype_>;
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline void operator()(uint32_t realM);

private:
    const Params *params_;
    __aicore__ inline void CalParams(uint32_t realM);
    __aicore__ inline void InitParams();
    __aicore__ inline void InitCommonParams();
    __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR y, GM_ADDR scale);
    __aicore__ inline void LoopProcess(int32_t multiRow, int32_t loopNum);
    __aicore__ inline void CopyIn(int32_t multiRow, int32_t loopNum);
    __aicore__ inline void Compute(int32_t multiRow);
    __aicore__ inline void CopyOut(int32_t multiRow, int32_t loopCount);
    __aicore__ inline void DataCopyInputVF(__local_mem__ XDtype *xAddr, AscendC::MicroAPI::RegTensor<float> &vregRes,
                                           AscendC::MicroAPI::MaskReg pregMask);
    __aicore__ inline void ComputeYVF(__local_mem__ XDtype *xAddr, __local_mem__ YCopyDtype *yAddr,
                                      AscendC::MicroAPI::RegTensor<float> &vregDupScale, int32_t indexRow);
    __aicore__ inline void ComputeScaleVF(__local_mem__ XDtype *xAddr, __local_mem__ float *scaleAddr,
                                          AscendC::MicroAPI::RegTensor<float> &vregDupScale, int32_t indexRow);
    __aicore__ inline void ComputeVF(__local_mem__ XDtype *xAddr, __local_mem__ YCopyDtype *yAddr,
                                     __local_mem__ float *scaleAddr, int32_t multiRow);
    __aicore__ inline void SetMaxValue();

    TQue<QuePosition::VECIN, USE_BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> outQueue;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> scaleQueue;

    GlobalTensor<XDtype> inGm_;
    GlobalTensor<float> yScaleGm_;
    GlobalTensor<YCopyDtype> outGm_;

    int32_t baseRow_ = 0;
    float maxValue_ = 0.0;
    uint32_t tailNum_ = 0;
    uint16_t vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t blockIdx_, sizeFloatLen_, sizeHalfLen_, outAlignLen_, multiRowNum_;
    uint32_t rowPerHeadCore_, rowPerTailCore_;
    uint32_t lenHead_, lenTail_, lenMultiRow_, lenGMMultiRow_, outLen_;
    uint32_t outLenHead_, outLenTail_;
    uint32_t loopCnt_ = 0;
    uint32_t remainRow_ = 0;
    uint8_t rightPadding_ = 0;
    uint32_t headCoreNum_;
    uint32_t coreNum_ = 1;
    uint32_t multiRowNumHeadCore_ = 0;
    uint32_t multiRowNumTailCore_ = 0;
    bool isPad_ = false;
    TPipe *pPipe_ = nullptr;
};

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::Init(Params const &params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    SetMaxValue();
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::operator()(uint32_t realM)
{
    if ASCEND_IS_AIC {
        return;
    }
    CalParams(realM);
    InitParams();
    InitAndSetBuffer(params_->workSpaceGMAddr, params_->yGmAddr, params_->yScaleGmAddr);
    if (blockIdx_ >= coreNum_) {
        return;
    }

    // loopCnt是UB搬运次数,multiRowNum是一次搬运（一次用满UB）的row数量
    for (int32_t i = 0; i < loopCnt_; i++) {
        LoopProcess(multiRowNum_, i);
    }

    // 处理剩余row，最后一个loop
    if (remainRow_ > 0) {
        LoopProcess(remainRow_, loopCnt_);
    }
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::CalParams(uint32_t realM)
{
    uint32_t rowNum = realM;
    uint32_t vectorCoreNum = uint32_t(GetBlockNum() * GetTaskRation());
    headCoreNum_ = rowNum % vectorCoreNum;
    rowPerHeadCore_ = (rowNum + vectorCoreNum - 1U) / vectorCoreNum;
    rowPerTailCore_ = rowNum / vectorCoreNum;
    coreNum_ = vectorCoreNum < rowNum ? vectorCoreNum : rowNum;
    coreNum_ = coreNum_ > 1 ? coreNum_ : 1;
    multiRowNumHeadCore_ = min(params_->ubAvail, min(COMPARE_INT, rowPerHeadCore_));
    multiRowNumTailCore_ = min(params_->ubAvail, min(COMPARE_INT, rowPerTailCore_));
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    // headCore的每次UB搬运处理的row数量multiRowNum，headCore的UB搬运次数loopCnt，headCore常规搬运结束后处理的row数量remainRow
    if (blockIdx_ < headCoreNum_) {
        multiRowNum_ = multiRowNumHeadCore_;
        loopCnt_ = rowPerHeadCore_ / multiRowNum_;
        remainRow_ = rowPerHeadCore_ % multiRowNum_;
        // 如果有tailCore的话
        // tailCore的每次UB搬运处理的row数量multiRowNum，tailCore的UB搬运次数loopCnt，tailCore常规搬运结束后处理的row数量remainRow
    } else if (blockIdx_ >= headCoreNum_ && blockIdx_ < coreNum_) {
        multiRowNum_ = multiRowNumTailCore_;
        loopCnt_ = rowPerTailCore_ / multiRowNum_;
        remainRow_ = rowPerTailCore_ % multiRowNum_;
    }
    if (IsSameType<XDtype, float>::value) {
        sizeHalfLen_ = (params_->rowLen + SEVEN) / EIGHT * EIGHT;
    } else {
        sizeHalfLen_ = (params_->rowLen + FIFTEEN) / SIXTEEN * SIXTEEN;
    }
    // rowLen需要Padding的数量
    rightPadding_ = sizeHalfLen_ - params_->rowLen;
    if (rightPadding_ > 0) {
        isPad_ = true;
    }
    if (IsSameType<YDtype, int4b_t>::value) {
        outAlignLen_ = (params_->rowLen + SIXTY_THREE) / SIXTY_FOUR * SIXTY_FOUR;
    } else {
        outAlignLen_ = (params_->rowLen + THIRTY_ONE) / THIRTY_TWO * THIRTY_TWO;
    }
    sizeFloatLen_ = (multiRowNum_ + SEVEN) / EIGHT * EIGHT;
    lenMultiRow_ = multiRowNum_ * sizeHalfLen_;
    lenGMMultiRow_ = multiRowNum_ * params_->rowLen;
    outLen_ = multiRowNum_ * outAlignLen_;
    InitCommonParams();
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::InitCommonParams()
{
    lenHead_ = rowPerHeadCore_ * params_->rowLen;
    lenTail_ = rowPerTailCore_ * params_->rowLen;
    if (IsSameType<YDtype, int4b_t>::value) {
        outLenHead_ = lenHead_ >> 1;
        outLenTail_ = lenTail_ >> 1;
    } else {
        outLenHead_ = lenHead_;
        outLenTail_ = lenTail_;
    }
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::InitAndSetBuffer(GM_ADDR x, GM_ADDR y,
                                                                                                  GM_ADDR scale)
{
    // headCoreGM申请
    if (blockIdx_ < headCoreNum_) {
        baseRow_ = blockIdx_ * rowPerHeadCore_;
        // headCore处理的元素总数lenHead、outLenHead。如果输出是INT4，这边outLenHead数量是lenHead一半
        inGm_.SetGlobalBuffer((__gm__ XDtype *)x + blockIdx_ * lenHead_, lenHead_);
        outGm_.SetGlobalBuffer((__gm__ YCopyDtype *)y + blockIdx_ * outLenHead_, outLenHead_);
        // scale每次偏移每个核处理的row数量的地址
        yScaleGm_.SetGlobalBuffer((__gm__ float *)scale + blockIdx_ * rowPerHeadCore_, rowPerHeadCore_);
        // tailCoreGM申请
    } else {
        baseRow_ = headCoreNum_ * rowPerHeadCore_ + (blockIdx_ - headCoreNum_) * rowPerTailCore_;
        inGm_.SetGlobalBuffer((__gm__ XDtype *)x + headCoreNum_ * lenHead_ + (blockIdx_ - headCoreNum_) * lenTail_,
                              lenTail_);
        outGm_.SetGlobalBuffer((__gm__ YCopyDtype *)y + headCoreNum_ * outLenHead_ +
                                   (blockIdx_ - headCoreNum_) * outLenTail_,
                               outLenTail_);
        yScaleGm_.SetGlobalBuffer((__gm__ float *)scale + baseRow_, rowPerTailCore_);
    }

    // 申请Buffer大小inQueue，outQueue，scaleQueue
    pPipe_->InitBuffer(inQueue, USE_BUFFER_NUM, lenMultiRow_ * sizeof(XDtype));
    pPipe_->InitBuffer(outQueue, USE_BUFFER_NUM, outLen_ * sizeof(YCopyDtype));
    pPipe_->InitBuffer(scaleQueue, USE_BUFFER_NUM, sizeFloatLen_ * sizeof(float));
    tailNum_ = params_->rowLen % REG_LEN;
    if (tailNum_ == 0) {
        tailNum_ = REG_LEN;
    }
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::LoopProcess(int32_t multiRow,
                                                                                             int32_t loopNum)
{
    CopyIn(multiRow, loopNum);
    Compute(multiRow);
    CopyOut(multiRow, loopNum);
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::CopyIn(int32_t multiRow,
                                                                                        int32_t loopNum)
{
    LocalTensor<XDtype> inLocal = inQueue.template AllocTensor<XDtype>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(multiRow),
                                    static_cast<uint32_t>(params_->rowLen * sizeof(XDtype)), 0, 0, 0};
    DataCopyPadExtParams<XDtype> padParams{true, 0, rightPadding_, 0};
    DataCopyPad(inLocal, inGm_[loopNum * lenGMMultiRow_], copyParams, padParams);
    inQueue.template EnQue(inLocal);
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::Compute(int32_t multiRow)
{
    uint32_t index = 0;
    LocalTensor<float> scaleLocal = scaleQueue.template AllocTensor<float>();
    LocalTensor<YCopyDtype> yLocal = outQueue.template AllocTensor<YCopyDtype>();
    LocalTensor<XDtype> xLocal = inQueue.template DeQue<XDtype>();

    __local_mem__ XDtype *xAddr = (__local_mem__ XDtype *)xLocal.GetPhyAddr();

    __local_mem__ YCopyDtype *yAddr = (__local_mem__ YCopyDtype *)yLocal.GetPhyAddr();
    __local_mem__ float *scaleAddr = (__local_mem__ float *)scaleLocal.GetPhyAddr();


    ComputeVF(xAddr, yAddr, scaleAddr, multiRow);

    outQueue.template EnQue<YCopyDtype>(yLocal);
    scaleQueue.template EnQue<float>(scaleLocal);
    inQueue.FreeTensor(xLocal);
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::CopyOut(int32_t multiRow,
                                                                                         int32_t loopCount)
{
    LocalTensor<YCopyDtype> yLocal = outQueue.template DeQue<YCopyDtype>();
    LocalTensor<float> scaleLocal = scaleQueue.template DeQue<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(multiRow),
                                 static_cast<uint32_t>(params_->rowLen * sizeof(YCopyDtype)), 0, 0, 0};
    if constexpr (IsSameType<YDtype, int4b_t>::value) {
        copyParams.blockLen = copyParams.blockLen >> 1;
        uint32_t index = (loopCount * lenGMMultiRow_) >> 1;
        DataCopyPad(outGm_[index], yLocal, copyParams);
    } else {
        DataCopyPad(outGm_[loopCount * lenGMMultiRow_], yLocal, copyParams);
    }

    DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(multiRow * sizeof(float)), 0, 0, 0};
    DataCopyPad(yScaleGm_[loopCount * multiRowNum_], scaleLocal, scaleCopyParams);

    outQueue.FreeTensor(yLocal);
    scaleQueue.FreeTensor(scaleLocal);
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::DataCopyInputVF(
    __local_mem__ XDtype *xAddr, AscendC::MicroAPI::RegTensor<float> &vregRes, AscendC::MicroAPI::MaskReg pregMask)
{
    AscendC::MicroAPI::RegTensor<XDtype> vregX;

    if constexpr (IsSameType<XDtype, float>::value) {
        AscendC::MicroAPI::DataCopy<XDtype, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregX, xAddr);
        vregRes = vregX;
    } else {
        AscendC::MicroAPI::DataCopy<XDtype, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX, xAddr);
        AscendC::MicroAPI::Cast<float, XDtype, castTraitB16ToB32>(vregRes, vregX, pregMask);
    }
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::ComputeYVF(
    __local_mem__ XDtype *xAddr, __local_mem__ YCopyDtype *yAddr, AscendC::MicroAPI::RegTensor<float> &vregDupScale,
    int32_t indexRow)
{
    AscendC::MicroAPI::RegTensor<float> vregInput;
    AscendC::MicroAPI::RegTensor<float> vregXDivScale;
    AscendC::MicroAPI::RegTensor<float> vregYFp32;
    AscendC::MicroAPI::RegTensor<int16_t> vregYInt16; // cast成最终y之前的int16类型
    AscendC::MicroAPI::RegTensor<half> vregYFp16;     // cast成最终y之前的half类型
    AscendC::MicroAPI::RegTensor<YCopyDtype> vregY;   // 最终y

    AscendC::MicroAPI::MaskReg preg2;
    AscendC::MicroAPI::MaskReg pregHalf = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::H>();

    uint32_t sreg1 = sizeHalfLen_;
    uint16_t vfLoop = (sizeHalfLen_ + vlForHalfNumber_ - 1) / vlForHalfNumber_;
    for (uint16_t j = 0; j < vfLoop; j++) {
        auto addr = yAddr + indexRow * outAlignLen_ + j * vlForHalfNumber_;
        preg2 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
        DataCopyInputVF(xAddr + indexRow * sizeHalfLen_ + j * vlForHalfNumber_, vregInput, preg2);
        AscendC::MicroAPI::Div(vregYFp32, vregInput, vregDupScale, preg2);
        if constexpr (IsSameType<YDtype, int8_t>::value) {
            AscendC::MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(vregYInt16, vregYFp32, preg2);
            AscendC::MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(vregYFp16, vregYInt16, preg2);
            AscendC::MicroAPI::Cast<YDtype, half, castTraitF16ToI8>(vregY, vregYFp16, preg2);
        } else if constexpr (IsSameType<YDtype, hifloat8_t>::value) {
            AscendC::MicroAPI::Cast<YDtype, float, castTraitF32toh8>(vregY, vregYFp32, preg2);
        } else if constexpr (IsSameType<YDtype, fp8_e4m3fn_t>::value || IsSameType<YDtype, fp8_e5m2_t>::value) {
            AscendC::MicroAPI::Cast<YDtype, float, castTraitF32tofp8>(vregY, vregYFp32, preg2);
        } else if constexpr (IsSameType<YDtype, int4b_t>::value) {
            AscendC::MicroAPI::RegTensor<uint16_t> vreg20;
            AscendC::MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(vregYInt16, vregYFp32, preg2);
            AscendC::MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(vregYFp16, vregYInt16, preg2);
            AscendC::MicroAPI::Pack(vreg20, (AscendC::MicroAPI::RegTensor<uint32_t> &)vregYFp16);
            AscendC::MicroAPI::Cast<int4x2_t, half, castTraitF16ToI8>(
                (AscendC::MicroAPI::RegTensor<int4x2_t> &)vregY, (AscendC::MicroAPI::RegTensor<half> &)vreg20, preg2);
            addr = yAddr + (indexRow * outAlignLen_ + j * vlForHalfNumber_) / 2;
        }
        if constexpr (IsSameType<YDtype, int4b_t>::value) {
            AscendC::MicroAPI::DataCopy<YCopyDtype, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregY,
                                                                                                  pregHalf);
        } else {
            AscendC::MicroAPI::DataCopy<YCopyDtype, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregY, preg2);
        }
    }
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::ComputeScaleVF(
    __local_mem__ XDtype *xAddr, __local_mem__ float *scaleAddr, AscendC::MicroAPI::RegTensor<float> &vregDupScale,
    int32_t indexRow)
{
    AscendC::MicroAPI::RegTensor<float> vregInput;
    AscendC::MicroAPI::RegTensor<float> vregAbs;
    AscendC::MicroAPI::RegTensor<float> vregScale;
    AscendC::MicroAPI::RegTensor<float> vregMinX;
    AscendC::MicroAPI::RegTensor<float> vregMaxX;
    AscendC::MicroAPI::RegTensor<float> vregReduceMaxX;
    AscendC::MicroAPI::RegTensor<float> vregReduceMinX;
    AscendC::MicroAPI::RegTensor<float> vregMaxSubMin;
    AscendC::MicroAPI::RegTensor<float> vregMaxDivScale;
    AscendC::MicroAPI::RegTensor<float> vregNegMaxDivScale;
    AscendC::MicroAPI::RegTensor<float> vregReduceMaxXTail;
    AscendC::MicroAPI::RegTensor<float> vregReduceMinXTail;
    AscendC::MicroAPI::RegTensor<float> vregFinalMax;
    AscendC::MicroAPI::RegTensor<float> vregFinalMin;

    AscendC::MicroAPI::MaskReg preg0;
    AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg preg4;
    AscendC::MicroAPI::MaskReg preg5 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();

    AscendC::MicroAPI::UnalignReg ureg0;
    AscendC::MicroAPI::UnalignReg ureg1;

    uint32_t rowCount = sizeHalfLen_;
    uint16_t vfLoop = (rowCount + vlForHalfNumber_ - 1) / vlForHalfNumber_;
    uint32_t sreg0 = rowCount;
    uint32_t sregTail = tailNum_;
    AscendC::MicroAPI::Duplicate(vregMaxX, NEG_INFINITY, preg1);
    // 1. compute max value for every vf loop  2.do reducemax in the end
    for (uint16_t j = 0; j < vfLoop; j++) {
        preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
        DataCopyInputVF(xAddr + indexRow * rowCount + j * vlForHalfNumber_, vregInput, preg0);
        AscendC::MicroAPI::Abs(vregAbs, vregInput, preg0);
        AscendC::MicroAPI::Max(vregMaxX, vregAbs, vregMaxX, preg1);
    }
    AscendC::MicroAPI::ReduceMax(vregReduceMaxX, vregMaxX, preg1);
    AscendC::MicroAPI::Muls(vregScale, vregReduceMaxX, maxValue_, preg1);
    AscendC::MicroAPI::Duplicate(vregDupScale, vregScale, preg1);
    AscendC::MicroAPI::DataCopyUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(scaleAddr, vregScale,
                                                                                                ureg0, 1);
    AscendC::MicroAPI::DataCopyUnAlignPost(scaleAddr, ureg0, 0);
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::ComputeVF(
    __local_mem__ XDtype *xAddr, __local_mem__ YCopyDtype *yAddr, __local_mem__ float *scaleAddr, int32_t multiRow)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vregDupScale;

        for (uint16_t indexRow = 0; indexRow < (uint16_t)multiRow; indexRow++) {
            ComputeScaleVF(xAddr, scaleAddr + indexRow, vregDupScale, indexRow);
            ComputeYVF(xAddr, yAddr, vregDupScale, indexRow);
        }
    }
}

QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertokenQuant<QMM_BLOCK_EPILOGUE_PERTOKEN_QUANT_FUNC_LOCAL_PARAMS>::SetMaxValue()
{
    if constexpr (IsSameType<YDtype, int8_t>::value) {
        maxValue_ = static_cast<float>(1.0) / INT8_MAX_VALUE;
    } else if constexpr (IsSameType<YDtype, int4b_t>::value) {
        maxValue_ = static_cast<float>(1.0) / INT4_MAX_VALUE;
    } else if constexpr (IsSameType<YDtype, fp8_e5m2_t>::value) {
        maxValue_ = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<YDtype, fp8_e4m3fn_t>::value) {
        maxValue_ = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
    } else if constexpr (IsSameType<YDtype, hifloat8_t>::value) {
        maxValue_ = static_cast<float>(1.0) / HIFLOAT8_MAX_VALUE;
    }
}
} // namespace Block
} // namespace Gemm
} // namespace Cgmct
#endif // EPILOGUE_BLOCK_EPILOGUE_PERTOKEN_QUANT_H