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
 * \file block_epilogue_dequant_swiglu.h
 * \brief
 */

#ifndef EPILOGUE_BLOCK_EPILOGUE_DEQUANT_SWIGLU_H
#define EPILOGUE_BLOCK_EPILOGUE_DEQUANT_SWIGLU_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "kernel_operator.h"
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"
#include "../tile/tile_copy_policy.h"

namespace Cgmct {
namespace Gemm {
namespace Block {
namespace {
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t X2SCALE_IDX = 2;
constexpr uint32_t X1SCALE_IDX = 3;
constexpr uint32_t MAX_SINGLE_MN_SIZE = 64 * 256;
constexpr uint32_t MAX_SCALE_NUM = 256;
} // namespace

static constexpr AscendC::MicroAPI::CastTrait ctInt322Fp32S = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

static constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32ZeroS = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32OneS = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODES = {AscendC::MicroAPI::MaskMergeMode::ZEROING, true};

static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_B16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
#define QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS                                                           \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_,           \
              typename DataTypeX1Scale_, bool IsTensorList_>
#define QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS                                                                   \
    L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
class BlockEpilogueDequantSwiglu {
public:
    __aicore__ inline BlockEpilogueDequantSwiglu()
    {
    }

    struct Arguments {
        GM_ADDR workSpaceGMAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        Arguments() = default;
    };

    // params
    using Params = Arguments;

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;

    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t>; // y, yScale, x2Scale, x1Scale, bias
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetFirstL0c2UbTensor();
    __aicore__ inline auto GetSecondL0c2UbTensor();
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

private:
    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset,
                                               AscendC::LocalTensor<DataTypeOut> &src);
    __aicore__ inline void VFDoSwiglu(__ubuf__ float *firstSrc, __ubuf__ float *secondSrc, uint16_t mSize,
                                      uint16_t nSize);
    __aicore__ inline void VFDoDequantWithX1X2Scale(__ubuf__ float *l0cOutUbFirstFloatAddr,
                                                    __ubuf__ float *l0cOutUbSecondFloatAddr, uint16_t mSize);
    __aicore__ inline void VFDoDequant(__ubuf__ float *dst, __ubuf__ DataTypeIn *l0cOut,
                                       __ubuf__ DataTypeX2Scale *scale2, __ubuf__ DataTypeX1Scale *x1Scale,
                                       uint16_t mSize, uint16_t nSize);
    __aicore__ inline void CopyX1ScaleFromGm2Ub(AscendC::LocalTensor<DataTypeX1Scale> &dst, uint64_t blockLen,
                                                uint64_t offset);
    __aicore__ inline void CopyX2ScaleFromGm2Ub(AscendC::LocalTensor<DataTypeX2Scale> &dst, uint64_t offset);
    __aicore__ inline void CopyDataFromGm2Ub(uint64_t halfSingleM, uint64_t singleMInVec);
    __aicore__ inline void VFDoPertokenDequantAndSwiglu(uint16_t mSize);
    // GM ADDR
    AscendC::GlobalTensor<DataTypeOut> gluResGlobal_; // TODO 看下workspaceoffset是否需要
    AscendC::GlobalTensor<DataTypeX1Scale> x1ScaleGlobal_;
    AscendC::GlobalTensor<DataTypeX2Scale> x2ScaleGlobal_;
    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN_SIZE};
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, MAX_SINGLE_MN_SIZE * sizeof(DataTypeIn),
                                                     MAX_SINGLE_MN_SIZE};
    AscendC::LocalTensor<float> l0cOutUbFirstFloat_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN_SIZE};
    AscendC::LocalTensor<float> l0cOutUbSecondFloat_{AscendC::TPosition::VECIN, MAX_SINGLE_MN_SIZE * sizeof(float),
                                                     MAX_SINGLE_MN_SIZE};
    AscendC::LocalTensor<DataTypeOut> gluRes_;
    AscendC::LocalTensor<DataTypeX2Scale> x2ScaleUbFirst_;
    AscendC::LocalTensor<DataTypeX2Scale> x2ScaleUbSecond_;
    AscendC::LocalTensor<DataTypeX1Scale> x1ScaleUb_;
    const Params *params_;
    int64_t n_;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint32_t singleM_; // cur singleShapeM
    uint32_t singleN_;
    BlockCoord blockCoord_{0, 0, 0, 0, 0};
};

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(Params const &params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    constexpr uint32_t afterIn = 2 * MAX_SINGLE_MN_SIZE * sizeof(DataTypeIn); // 2: 2 block in
    gluRes_ = AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECOUT, afterIn, MAX_SINGLE_MN_SIZE);
    constexpr uint32_t afterGlu = afterIn + MAX_SINGLE_MN_SIZE * sizeof(DataTypeOut);
    x2ScaleUbFirst_ = AscendC::LocalTensor<DataTypeX2Scale>(AscendC::TPosition::VECOUT, afterGlu, MAX_SCALE_NUM);
    constexpr uint32_t afterGluAndHalfX2 = afterGlu + MAX_SCALE_NUM * sizeof(DataTypeX2Scale);
    x2ScaleUbSecond_ =
        AscendC::LocalTensor<DataTypeX2Scale>(AscendC::TPosition::VECOUT, afterGluAndHalfX2, MAX_SCALE_NUM);
    constexpr uint32_t afterGluAndX2 = afterGluAndHalfX2 + MAX_SCALE_NUM * sizeof(DataTypeX2Scale);
    x1ScaleUb_ = AscendC::LocalTensor<DataTypeX1Scale>(AscendC::TPosition::VECOUT, afterGluAndX2, MAX_SCALE_NUM);
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(const BlockCoord &baseOffset)
{
    if ASCEND_IS_AIV {
        gluResGlobal_.SetGlobalBuffer((__gm__ DataTypeOut *)params_->workSpaceGMAddr + Get<Y_INDEX>(baseOffset));
        x1ScaleGlobal_.SetGlobalBuffer((__gm__ DataTypeX1Scale *)params_->x1ScaleGmAddr + Get<X1SCALE_IDX>(baseOffset));
        x2ScaleGlobal_.SetGlobalBuffer(GetTensorAddr<DataTypeX2Scale>(0, params_->x2ScaleGmAddr) +
                                       Get<X2SCALE_IDX>(baseOffset));
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    n_ = Get<MNK_N>(problemShape);
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<DataTypeOut> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.blockLen = singleN_ * sizeof(DataTypeOut);
    ub2GmParams.dstStride = (n_ - singleN_) * sizeof(DataTypeOut);
    AscendC::DataCopyPad(gluResGlobal_[offset], src, ub2GmParams);
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwiglu(
    __ubuf__ float *firstSrc, __ubuf__ float *secondSrc, uint16_t mSize, uint16_t nSize)
{
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint16_t OneRowRepeatTimes = CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(sizePerRepeat));
    uint32_t nSrcUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn));
    uint32_t nDstUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE / sizeof(DataTypeOut));
    const float scalarOne = 1.0;
    __ubuf__ DataTypeOut *gluResAddr = (__ubuf__ DataTypeOut *)gluRes_.GetPhyAddr();
    // swiglu
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<DataTypeIn>(elementNum);
                AscendC::MicroAPI::RegTensor<float> l0cOutRegFirst, l0cOutRegSecond;
                AscendC::MicroAPI::RegTensor<DataTypeOut> verg7;
                AscendC::MicroAPI::RegTensor<float> verg1, verg2, verg3, verg4, verg5, verg6, swishOutput;
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat;
                AscendC::MicroAPI::DataCopy(l0cOutRegFirst, firstSrc + l0cOutOffset);
                // swish
                AscendC::MicroAPI::Muls(verg2, l0cOutRegFirst, -(scalarOne), mask);
                AscendC::MicroAPI::Exp(verg3, verg2, mask);
                AscendC::MicroAPI::Adds(verg4, verg3, scalarOne, mask);
                AscendC::MicroAPI::Div<float, &DIV_MODES>(swishOutput, l0cOutRegFirst, verg4, mask);
                // load gate data to regbase
                AscendC::MicroAPI::DataCopy(l0cOutRegSecond, secondSrc + l0cOutOffset);
                // swish * gate
                AscendC::MicroAPI::Mul(verg6, swishOutput, l0cOutRegSecond, mask);
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat;
                if constexpr (IsSameType<DataTypeOut, half>::value) {
                    AscendC::MicroAPI::Cast<half, float, CAST_FP32_TO_B16>(verg7, verg6, mask);
                    AscendC::MicroAPI::DataCopy<DataTypeOut, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        gluResAddr + dstUbOffset, verg7, mask);
                } else if constexpr (IsSameType<DataTypeOut, bfloat16_t>::value) {
                    AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_B16>(verg7, verg6, mask);
                    AscendC::MicroAPI::DataCopy<DataTypeOut, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        gluResAddr + dstUbOffset, verg7, mask);
                } else if constexpr (IsSameType<DataTypeOut, float>::value) {
                    verg7 = verg6;
                    AscendC::MicroAPI::DataCopy<DataTypeOut, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                        gluResAddr + dstUbOffset, verg7, mask);
                }
            }
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoDequantWithX1X2Scale(
    __ubuf__ float *l0cOutUbFirstFloatAddr, __ubuf__ float *l0cOutUbSecondFloatAddr, uint16_t mSize)
{
    __ubuf__ DataTypeIn *l0cOutUbFirstAddr = (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr();
    __ubuf__ DataTypeIn *l0cOutUbSecondAddr = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
    __ubuf__ DataTypeX1Scale *ptScaleUbAddr = (__ubuf__ DataTypeX1Scale *)x1ScaleUb_.GetPhyAddr();
    VFDoDequant(l0cOutUbFirstFloatAddr, l0cOutUbFirstAddr, (__ubuf__ DataTypeX2Scale *)x2ScaleUbFirst_.GetPhyAddr(),
                ptScaleUbAddr, mSize, singleN_);
    VFDoDequant(l0cOutUbSecondFloatAddr, l0cOutUbSecondAddr, (__ubuf__ DataTypeX2Scale *)x2ScaleUbSecond_.GetPhyAddr(),
                ptScaleUbAddr, mSize, singleN_);
}


QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoDequant(
    __ubuf__ float *dst, __ubuf__ DataTypeIn *l0cOut, __ubuf__ DataTypeX2Scale *scale2,
    __ubuf__ DataTypeX1Scale *x1Scale, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint32_t nSrcUbAligned = Align(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(DataTypeIn)));
    uint32_t nDstUbAligned = nSrcUbAligned;
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg maskN4B16 =
            AscendC::MicroAPI::CreateMask<DataTypeX2Scale, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                AscendC::MicroAPI::RegTensor<DataTypeIn> l0cOutReg;
                AscendC::MicroAPI::RegTensor<DataTypeX2Scale> scaleReg;
                AscendC::MicroAPI::RegTensor<DataTypeX1Scale> perTokenScaleReg;
                AscendC::MicroAPI::RegTensor<float> l0cOutRegFloat;
                AscendC::MicroAPI::RegTensor<float> castScaleReg, castScaleOneReg, mulScaleOutReg, mulPtScaleOutReg;
                AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<float>(elementNum);
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
                if constexpr (IsSameType<DataTypeIn, int32_t>::value) {
                    AscendC::MicroAPI::Cast<float, DataTypeIn, ctInt322Fp32S>(l0cOutRegFloat, l0cOutReg, maskN);
                } else {
                    l0cOutRegFloat = l0cOutReg;
                }
                // l0c_out * scale2
                AscendC::MicroAPI::DataCopy(scaleReg, scale2 + vfBlockIdx * eleNumPerVf);
                if constexpr (IsSameType<DataTypeX2Scale, bfloat16_t>::value ||
                              IsSameType<DataTypeX2Scale, half>::value) {
                    AscendC::MicroAPI::Cast<float, DataTypeX2Scale, ctHalf2Fp32ZeroS>(castScaleReg, scaleReg, maskN);
                    AscendC::MicroAPI::Cast<float, DataTypeX2Scale, ctHalf2Fp32OneS>(castScaleOneReg, scaleReg,
                                                                                     maskN4B16);
                    AscendC::MicroAPI::Interleave(castScaleReg, castScaleOneReg, castScaleReg, castScaleOneReg);
                } else if constexpr (IsSameType<DataTypeX2Scale, float>::value) {
                    castScaleReg = scaleReg;
                }
                AscendC::MicroAPI::Mul(mulScaleOutReg, l0cOutRegFloat, castScaleReg, maskN);
                // out * x1Scale
                AscendC::MicroAPI::DataCopy<DataTypeX1Scale, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    perTokenScaleReg, x1Scale + mIdx);
                AscendC::MicroAPI::Mul(mulPtScaleOutReg, mulScaleOutReg, perTokenScaleReg, maskN);
                // copy out from register to ub
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                    dst + dstUbOffset, mulPtScaleOutReg, maskN);
            }
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyX1ScaleFromGm2Ub(
    AscendC::LocalTensor<DataTypeX1Scale> &dst, uint64_t blockLen, uint64_t offset)
{
    DataCopyParams ptScale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    ptScale2UbParams.blockLen = blockLen;
    AscendC::DataCopyPad(dst, x1ScaleGlobal_[Get<X1SCALE_IDX>(blockCoord_) + offset], ptScale2UbParams, padParams);
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyX2ScaleFromGm2Ub(
    AscendC::LocalTensor<DataTypeX2Scale> &dst, uint64_t offset)
{
    DataCopyParams scale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    scale2UbParams.blockLen = singleN_ * sizeof(DataTypeX2Scale);
    AscendC::DataCopyPad(dst, x2ScaleGlobal_[Get<X2SCALE_IDX>(blockCoord_) + offset], scale2UbParams, padParams);
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyDataFromGm2Ub(uint64_t halfSingleM,
                                                                                            uint64_t singleMInVec)
{
    // x2scale: GM -> UB
    CopyX2ScaleFromGm2Ub(x2ScaleUbFirst_, 0);
    CopyX2ScaleFromGm2Ub(x2ScaleUbSecond_, n_);
    uint64_t mOffset = subBlockIdx_ * halfSingleM;
    // x1Scale: GM -> UB
    CopyX1ScaleFromGm2Ub(x1ScaleUb_, singleMInVec * sizeof(DataTypeX1Scale), mOffset);
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetFirstL0c2UbTensor()
{
    return l0cOutUbFirst_;
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetSecondL0c2UbTensor()
{
    return l0cOutUbSecond_;
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoPertokenDequantAndSwiglu(uint16_t mSize)
{
    __ubuf__ float *l0cOutUbFirstFloatAddr = (__ubuf__ float *)l0cOutUbFirstFloat_.GetPhyAddr();
    __ubuf__ float *l0cOutUbSecondFloatAddr = (__ubuf__ float *)l0cOutUbSecondFloat_.GetPhyAddr();
    VFDoDequantWithX1X2Scale(l0cOutUbFirstFloatAddr, l0cOutUbSecondFloatAddr, mSize);
    VFDoSwiglu(l0cOutUbFirstFloatAddr, l0cOutUbSecondFloatAddr, mSize, singleN_);
}

QMM_BLOCK_EPILOGUE_DEQUANT_SWIGLU_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantSwiglu<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(const BlockShape &blockShape,
                                                                                     const BlockCoord &blockCoord)
{
    singleM_ = Get<MNK_M>(blockShape);
    singleN_ = Get<MNK_N>(blockShape);
    blockCoord_ = blockCoord;
    auto halfSingleM = CeilDiv(static_cast<uint64_t>(singleM_), static_cast<uint64_t>(AscendC::GetTaskRation()));
    uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    uint64_t yOffset = Get<Y_INDEX>(blockCoord_) + subBlockIdx_ * halfSingleM * n_;
    if (singleMInVec == 0) {
        return;
    }
    CopyDataFromGm2Ub(halfSingleM, singleMInVec);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    VFDoPertokenDequantAndSwiglu(singleMInVec);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(singleMInVec, yOffset, gluRes_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    return;
}
} // namespace Block
} // namespace Gemm
} // namespace Cgmct

#endif // EPILOGUE_BLOCK_EPILOGUE_DEQUANT_SWIGLU_H
#endif