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
 * \file block_epilogue_dequant_finalize_routing.h
 * \brief
 */

#ifndef BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_H
#define BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "kernel_basic_intf.h"
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"
#include "../tile/tile_copy_policy.h"

namespace Cgmct {
namespace Gemm {
namespace Block {
namespace {
constexpr uint32_t Y_IDXS = 0U;
constexpr uint32_t BIAS_IDXS = 1U;
constexpr uint32_t X2SCALE_IDXS = 2U;
constexpr uint32_t X1SCALE_IDXS = 3U;
constexpr uint32_t LOGIT_INDEXS = 4U;
constexpr uint32_t MAX_SINGLE_MNS = 128 * 256U;
constexpr uint32_t HALF_DB_MAX_SINGLE_MNS = 32 * 256U;
constexpr uint32_t BLOCKS_BYTES = 256U;
constexpr uint64_t MAX_OUTPUT_M_UBS = 32UL;
constexpr uint64_t BLOCK_ELEMENTS_FP32S = 8UL;
} // namespace

static constexpr AscendC::MicroAPI::CastTrait ctInt322Fp32ES = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

static constexpr AscendC::MicroAPI::CastTrait ctInt642Fp32S = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

static constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32ZeroES = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32OneES = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

using namespace AscendC;

#define GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS                                                 \
    template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, typename DataTypeX1Scale_,       \
              typename DataTypeBias_, typename DataTypeRowIndex_>
#define GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS                                                  \
    DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, DataTypeBias_, DataTypeRowIndex_

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
class BlockEpilogueDequantFinalizeRouting {
public:
    __aicore__ inline BlockEpilogueDequantFinalizeRouting();
    __aicore__ inline ~BlockEpilogueDequantFinalizeRouting();

    struct Params {
        GM_ADDR yGMAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR logitGmAddr{nullptr};
        GM_ADDR rowIndexGmAddr{nullptr};
        int32_t baseM = 256;
        int32_t baseN = 256;
        Params() = default;
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;
    using BiasDtype = DataTypeBias_;
    using DataTypeRowIndex = DataTypeRowIndex_;
    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord =
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(const Params &params);
    __aicore__ inline auto GetL0c2UbTensor();
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);

private:
    __aicore__ inline void CopyInLogit(uint32_t curBaseM, uint64_t offsetM, LocalTensor<float> &logitUb);
    __aicore__ inline void VFDoLogitMuls(uint32_t offsetRe, uint32_t offsetLogit, uint16_t repeatTimesLogit,
                                         uint16_t repeatTimesRe, __ubuf__ DataTypeOut *outUbAddr,
                                         LocalTensor<float> &logitUb);
    __aicore__ inline void VectorAtomicProcess(uint32_t curBaseN, uint32_t curVecBaseM, uint64_t offsetM,
                                               uint64_t yOffset, LocalTensor<DataTypeOut> &yLocal);
    __aicore__ inline void VFDoDequantWithX1X2Scale(LocalTensor<DataTypeX2Scale> &x2ScaleUb,
                                                    LocalTensor<DataTypeX1Scale> &x1ScaleUb,
                                                    LocalTensor<BiasDtype> &biasUb, uint16_t mSize);
    template <bool isBiasEpilogue>
    __aicore__ inline void VFDoDequantOnlyX2(__ubuf__ float *dst, __ubuf__ DataTypeIn *l0cOut,
                                             __ubuf__ DataTypeX2Scale *x2Scale, __ubuf__ BiasDtype *bias,
                                             uint16_t mSize, uint16_t nSize);
    template <bool isBiasEpilogue>
    __aicore__ inline void VFDoDequant(__ubuf__ float *dst, __ubuf__ DataTypeIn *l0cOut,
                                       __ubuf__ DataTypeX2Scale *x2Scale, __ubuf__ DataTypeX1Scale *x1Scale,
                                       __ubuf__ BiasDtype *bias, uint16_t mSize, uint16_t nSize);
    __aicore__ inline void CopyX1ScaleFromGm2Ub(LocalTensor<DataTypeX1Scale> &dst, uint64_t blockLen, uint64_t offset);
    __aicore__ inline void CopyX2ScaleFromGm2Ub(LocalTensor<DataTypeX2Scale> &dst, uint64_t offset);
    __aicore__ inline void CopyBiasFromGm2Ub(LocalTensor<BiasDtype> &dst);

    // GM ADDR
    AscendC::GlobalTensor<float> logitGlobal_;
    AscendC::GlobalTensor<DataTypeRowIndex> rowIndexGlobal_;
    AscendC::GlobalTensor<DataTypeX1Scale> x1ScaleGlobal_;
    AscendC::GlobalTensor<DataTypeX2Scale> x2ScaleGlobal_;
    AscendC::GlobalTensor<BiasDtype> biasGlobal_;
    AscendC::GlobalTensor<DataTypeOut> yGlobal_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUb_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MNS};
    AscendC::LocalTensor<float> l0cOutUbFloat_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MNS};
    AscendC::LocalTensor<float> logitUbPing_;
    AscendC::LocalTensor<float> logitUbPong_;
    AscendC::LocalTensor<DataTypeX2Scale> x2ScaleUbPing_;
    AscendC::LocalTensor<DataTypeX2Scale> x2ScaleUbPong_;
    AscendC::LocalTensor<DataTypeX1Scale> x1ScaleUbPing_;
    AscendC::LocalTensor<DataTypeX1Scale> x1ScaleUbPong_;
    AscendC::LocalTensor<BiasDtype> biasUbPing_;
    AscendC::LocalTensor<BiasDtype> biasUbPong_;
    AscendC::LocalTensor<DataTypeOut> outUbPing_;
    AscendC::LocalTensor<DataTypeOut> outUbPong_;

    const Params *params_;
    int64_t n_;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint64_t singleM_;
    uint64_t singleN_;
    uint64_t alignN_;
    uint64_t vlForFloatNumber_;
    uint32_t repeatTimesLine_;
    uint16_t yPingPongID_ = 0;
    uint16_t logitPingPongID_ = 0;
    BlockCoord blockCoord_{0, 0, 0, 0, 0, 0};
};

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::BlockEpilogueDequantFinalizeRouting()
{
    if ASCEND_IS_AIV {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    }
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::~BlockEpilogueDequantFinalizeRouting()
{
    if ASCEND_IS_AIV {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    }
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::Init(
    Params const &params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    yGlobal_.SetGlobalBuffer((__gm__ DataTypeOut *)params_->yGMAddr);
    // input
    uint32_t afterFirstIn = MAX_SINGLE_MNS * sizeof(DataTypeOut);
    logitUbPing_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, afterFirstIn, BLOCKS_BYTES);
    uint32_t afterSecondIn = afterFirstIn + BLOCKS_BYTES * sizeof(float);
    logitUbPong_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, afterSecondIn, BLOCKS_BYTES);
    uint32_t afterLogit = afterSecondIn + BLOCKS_BYTES * sizeof(float);
    x2ScaleUbPing_ = AscendC::LocalTensor<DataTypeX2Scale>(AscendC::TPosition::VECIN, afterLogit, BLOCKS_BYTES);
    uint32_t afterX2ScalePing = afterLogit + BLOCKS_BYTES * sizeof(DataTypeX2Scale);
    x2ScaleUbPong_ = AscendC::LocalTensor<DataTypeX2Scale>(AscendC::TPosition::VECIN, afterX2ScalePing, BLOCKS_BYTES);
    uint32_t afterX2ScalePong = afterX2ScalePing + BLOCKS_BYTES * sizeof(DataTypeX2Scale);
    x1ScaleUbPing_ = AscendC::LocalTensor<DataTypeX1Scale>(AscendC::TPosition::VECIN, afterX2ScalePong, BLOCKS_BYTES);
    uint32_t afterX1ScalePing = afterX2ScalePong + BLOCKS_BYTES * sizeof(DataTypeX1Scale);
    x1ScaleUbPong_ = AscendC::LocalTensor<DataTypeX1Scale>(AscendC::TPosition::VECIN, afterX1ScalePing, BLOCKS_BYTES);
    uint32_t afterX1ScalePong = afterX1ScalePing + BLOCKS_BYTES * sizeof(DataTypeX1Scale);
    biasUbPing_ = AscendC::LocalTensor<BiasDtype>(AscendC::TPosition::VECIN, afterX1ScalePong, BLOCKS_BYTES);
    uint32_t afterBiasPing = afterX1ScalePong + BLOCKS_BYTES * sizeof(BiasDtype);
    biasUbPong_ = AscendC::LocalTensor<BiasDtype>(AscendC::TPosition::VECIN, afterBiasPing, BLOCKS_BYTES);
    uint32_t afterBiasPong = afterBiasPing + BLOCKS_BYTES * sizeof(BiasDtype);
    outUbPing_ = AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECOUT, afterBiasPong, HALF_DB_MAX_SINGLE_MNS);
    outUbPong_ = AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECOUT,
                                                   afterBiasPong + HALF_DB_MAX_SINGLE_MNS * sizeof(DataTypeOut),
                                                   HALF_DB_MAX_SINGLE_MNS);
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline auto
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::GetL0c2UbTensor()
{
    return l0cOutUb_;
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    n_ = Get<MNK_N>(problemShape);
}


GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(
    const BlockCoord &baseOffset)
{
    if ASCEND_IS_AIV {
        logitGlobal_.SetGlobalBuffer((__gm__ float *)params_->logitGmAddr + Get<LOGIT_INDEXS>(baseOffset));
        rowIndexGlobal_.SetGlobalBuffer((__gm__ DataTypeRowIndex *)params_->rowIndexGmAddr +
                                        Get<LOGIT_INDEXS>(baseOffset));
        if (params_->x1ScaleGmAddr != nullptr) {
            x1ScaleGlobal_.SetGlobalBuffer((__gm__ DataTypeX1Scale *)params_->x1ScaleGmAddr +
                                           Get<X1SCALE_IDXS>(baseOffset));
        }
        x2ScaleGlobal_.SetGlobalBuffer((__gm__ DataTypeX2Scale *)params_->x2ScaleGmAddr +
                                       Get<X2SCALE_IDXS>(baseOffset));
        if (params_->biasGmAddr != nullptr) {
            biasGlobal_.SetGlobalBuffer((__gm__ BiasDtype *)params_->biasGmAddr + Get<BIAS_IDXS>(baseOffset));
        }
    }
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::CopyInLogit(
    uint32_t curBaseM, uint64_t offsetM, LocalTensor<float> &logitUb)
{
    DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams;
    DataCopyPad(logitUb, logitGlobal_[offsetM], perTokenScaleParams, padParams);
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::VectorAtomicProcess(
    uint32_t curBaseN, uint32_t curVecBaseM, uint64_t offsetM, uint64_t yOffset, LocalTensor<DataTypeOut> &yLocal)
{
    SetAtomicAdd<float>();
    DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curBaseN * sizeof(DataTypeOut)), 0, 0, 0};
    for (uint32_t i = 0; i < curVecBaseM; i++) {
        auto outRow = static_cast<uint64_t>(rowIndexGlobal_.GetValue(offsetM + i));
        DataCopyPad(yGlobal_[outRow * n_ + yOffset], yLocal[i * alignN_], paramsOut);
    }

    SetAtomicNone();
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::VFDoLogitMuls(
    uint32_t offsetRe, uint32_t offsetLogit, uint16_t repeatTimesLogit, uint16_t repeatTimesRe,
    __ubuf__ DataTypeOut *outUbAddr, LocalTensor<float> &logitUb)
{
    __ubuf__ float *l0cOutUbAddr = (__ubuf__ float *)l0cOutUbFloat_.GetPhyAddr();
    __ubuf__ DataTypeOut *logitUbAddr = (__ubuf__ DataTypeOut *)logitUb.GetPhyAddr();
    l0cOutUbAddr += offsetRe;
    logitUbAddr += offsetLogit;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vregLogit;
        AscendC::MicroAPI::RegTensor<DataTypeOut> vregRe, vDstReg;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimesLogit; ++i) {
            DataCopy<DataTypeOut, MicroAPI::LoadDist::DIST_BRC_B32>(vregLogit, logitUbAddr + i);
            uint32_t elementNum = singleN_;
            for (uint16_t j = 0; j < repeatTimesRe; ++j) {
                mask = AscendC::MicroAPI::UpdateMask<DataTypeOut>(elementNum);
                AscendC::MicroAPI::DataCopy(vregRe, l0cOutUbAddr + i * alignN_ + j * vlForFloatNumber_);
                AscendC::MicroAPI::Mul(vDstReg, vregLogit, vregRe, mask);
                AscendC::MicroAPI::DataCopy(outUbAddr + i * alignN_ + j * vlForFloatNumber_, vDstReg, mask);
            }
        }
    }
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::
    VFDoDequantWithX1X2Scale(LocalTensor<DataTypeX2Scale> &x2ScaleUb, LocalTensor<DataTypeX1Scale> &x1ScaleUb,
                             LocalTensor<BiasDtype> &biasUb, uint16_t mSize)
{
    __ubuf__ DataTypeIn *l0cOutUbAddr = (__ubuf__ DataTypeIn *)l0cOutUb_.GetPhyAddr();
    __ubuf__ float *l0cOutUbFloatAddr = (__ubuf__ float *)l0cOutUbFloat_.GetPhyAddr();
    if (params_->x1ScaleGmAddr != nullptr) {
        if (params_->biasGmAddr != nullptr) {
            VFDoDequant<true>(l0cOutUbFloatAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale *)x2ScaleUb.GetPhyAddr(),
                              (__ubuf__ DataTypeX1Scale *)x1ScaleUb.GetPhyAddr(),
                              (__ubuf__ BiasDtype *)biasUb.GetPhyAddr(), mSize, singleN_);
        } else {
            VFDoDequant<false>(l0cOutUbFloatAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale *)x2ScaleUb.GetPhyAddr(),
                               (__ubuf__ DataTypeX1Scale *)x1ScaleUb.GetPhyAddr(), nullptr, mSize, singleN_);
        }
    } else {
        if (params_->biasGmAddr != nullptr) {
            VFDoDequantOnlyX2<true>(l0cOutUbFloatAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale *)x2ScaleUb.GetPhyAddr(),
                                    (__ubuf__ BiasDtype *)biasUb.GetPhyAddr(), mSize, singleN_);
        } else {
            VFDoDequantOnlyX2<false>(l0cOutUbFloatAddr, l0cOutUbAddr,
                                     (__ubuf__ DataTypeX2Scale *)x2ScaleUb.GetPhyAddr(), nullptr, mSize, singleN_);
        }
    }
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
template <bool isBiasEpilogue>
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::VFDoDequantOnlyX2(
    __ubuf__ float *dst, __ubuf__ DataTypeIn *l0cOut, __ubuf__ DataTypeX2Scale *x2Scale, __ubuf__ BiasDtype *bias,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint32_t nSrcUbAligned = Align(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(DataTypeIn)));
    uint32_t nDstUbAligned = nSrcUbAligned;
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg maskN4B16 =
            AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                AscendC::MicroAPI::RegTensor<DataTypeIn> l0cOutReg;
                AscendC::MicroAPI::RegTensor<DataTypeX2Scale> scaleReg;
                AscendC::MicroAPI::RegTensor<BiasDtype> biasReg;
                AscendC::MicroAPI::RegTensor<float> l0cOutRegFloat;
                AscendC::MicroAPI::RegTensor<float> castScaleReg, castScaleOneReg, mulScaleOutReg, addBiasOutReg,
                    castBiasReg, castBiasOneReg;
                AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<DataTypeIn>(elementNum);
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
                if constexpr (IsSameType<DataTypeIn, int32_t>::value) {
                    AscendC::MicroAPI::Cast<float, DataTypeIn, ctInt322Fp32ES>(l0cOutRegFloat, l0cOutReg, maskN);
                } else {
                    l0cOutRegFloat = l0cOutReg;
                }
                // l0c_out * x2Scale
                AscendC::MicroAPI::DataCopy(scaleReg, x2Scale + vfBlockIdx * eleNumPerVf);
                if constexpr (IsSameType<DataTypeX2Scale, bfloat16_t>::value) {
                    AscendC::MicroAPI::Cast<float, DataTypeX2Scale, ctHalf2Fp32ZeroES>(castScaleReg, scaleReg, maskN);
                    AscendC::MicroAPI::Cast<float, DataTypeX2Scale, ctHalf2Fp32OneES>(castScaleOneReg, scaleReg,
                                                                                      maskN4B16);
                    AscendC::MicroAPI::Interleave(castScaleReg, castScaleOneReg, castScaleReg, castScaleOneReg);
                } else if constexpr (IsSameType<DataTypeX2Scale, float>::value) {
                    castScaleReg = scaleReg;
                }
                AscendC::MicroAPI::Mul(mulScaleOutReg, l0cOutRegFloat, castScaleReg, maskN);
                if constexpr (isBiasEpilogue) {
                    AscendC::MicroAPI::DataCopy(biasReg, bias + vfBlockIdx * eleNumPerVf);
                    // cast bias from bf16/fp16 to float
                    if constexpr (IsSameType<BiasDtype, bfloat16_t>::value) {
                        AscendC::MicroAPI::Cast<float, BiasDtype, ctHalf2Fp32ZeroES>(castBiasReg, biasReg, maskN);
                        AscendC::MicroAPI::Cast<float, BiasDtype, ctHalf2Fp32OneES>(castBiasOneReg, biasReg, maskN4B16);
                        AscendC::MicroAPI::Interleave(castBiasReg, castBiasOneReg, castBiasReg, castBiasOneReg);
                    } else {
                        castBiasReg = biasReg;
                    }
                    AscendC::MicroAPI::Add(addBiasOutReg, mulScaleOutReg, castBiasReg, maskN);
                } else {
                    addBiasOutReg = mulScaleOutReg;
                }
                // copy out from register to ub
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                addBiasOutReg, maskN);
            }
        }
    }
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
template <bool isBiasEpilogue>
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::VFDoDequant(
    __ubuf__ float *dst, __ubuf__ DataTypeIn *l0cOut, __ubuf__ DataTypeX2Scale *x2Scale,
    __ubuf__ DataTypeX1Scale *x1Scale, __ubuf__ BiasDtype *bias, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint32_t nSrcUbAligned = Align(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(DataTypeIn)));
    uint32_t nDstUbAligned = nSrcUbAligned;
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg maskN4B16 =
            AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                AscendC::MicroAPI::RegTensor<DataTypeIn> l0cOutReg;
                AscendC::MicroAPI::RegTensor<DataTypeX2Scale> scaleReg;
                AscendC::MicroAPI::RegTensor<DataTypeX1Scale> perTokenScaleReg;
                AscendC::MicroAPI::RegTensor<BiasDtype> biasReg;
                AscendC::MicroAPI::RegTensor<float> l0cOutRegFloat;
                AscendC::MicroAPI::RegTensor<float> castScaleReg, castScaleOneReg, mulScaleOutReg, mulPtScaleOutReg,
                    addBiasOutReg, castBiasReg, castBiasOneReg;
                AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<DataTypeIn>(elementNum);
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
                if constexpr (IsSameType<DataTypeIn, int32_t>::value) {
                    AscendC::MicroAPI::Cast<float, DataTypeIn, ctInt322Fp32ES>(l0cOutRegFloat, l0cOutReg, maskN);
                } else {
                    l0cOutRegFloat = l0cOutReg;
                }
                // l0c_out * x2Scale
                AscendC::MicroAPI::DataCopy(scaleReg, x2Scale + vfBlockIdx * eleNumPerVf);
                if constexpr (IsSameType<DataTypeX2Scale, bfloat16_t>::value) {
                    AscendC::MicroAPI::Cast<float, DataTypeX2Scale, ctHalf2Fp32ZeroES>(castScaleReg, scaleReg, maskN);
                    AscendC::MicroAPI::Cast<float, DataTypeX2Scale, ctHalf2Fp32OneES>(castScaleOneReg, scaleReg,
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
                if constexpr (isBiasEpilogue) {
                    AscendC::MicroAPI::DataCopy(biasReg, bias + vfBlockIdx * eleNumPerVf);
                    // cast bias from bf16/fp16 to float
                    if constexpr (IsSameType<BiasDtype, bfloat16_t>::value) {
                        AscendC::MicroAPI::Cast<float, BiasDtype, ctHalf2Fp32ZeroES>(castBiasReg, biasReg, maskN);
                        AscendC::MicroAPI::Cast<float, BiasDtype, ctHalf2Fp32OneES>(castBiasOneReg, biasReg, maskN4B16);
                        AscendC::MicroAPI::Interleave(castBiasReg, castBiasOneReg, castBiasReg, castBiasOneReg);
                    } else {
                        castBiasReg = biasReg;
                    }
                    AscendC::MicroAPI::Add(addBiasOutReg, mulPtScaleOutReg, castBiasReg, maskN);
                } else {
                    addBiasOutReg = mulPtScaleOutReg;
                }
                // copy out from register to ub
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                addBiasOutReg, maskN);
            }
        }
    }
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::
    CopyX1ScaleFromGm2Ub(LocalTensor<DataTypeX1Scale> &dst, uint64_t blockLen, uint64_t offset)
{
    DataCopyParams ptScale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    ptScale2UbParams.blockLen = blockLen;
    AscendC::DataCopyPad(dst, x1ScaleGlobal_[Get<X1SCALE_IDXS>(blockCoord_) + offset], ptScale2UbParams, padParams);
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::
    CopyX2ScaleFromGm2Ub(LocalTensor<DataTypeX2Scale> &dst, uint64_t offset)
{
    DataCopyParams scale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    scale2UbParams.blockLen = singleN_ * sizeof(DataTypeX2Scale);
    AscendC::DataCopyPad(dst, x2ScaleGlobal_[Get<X2SCALE_IDXS>(blockCoord_) + offset], scale2UbParams, padParams);
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::CopyBiasFromGm2Ub(
    LocalTensor<BiasDtype> &dst)
{
    DataCopyParams bias2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    bias2UbParams.blockLen = singleN_ * sizeof(BiasDtype);
    AscendC::DataCopyPad(dst, biasGlobal_[Get<BIAS_IDXS>(blockCoord_)], bias2UbParams, padParams);
}

GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequantFinalizeRouting<GMM_BLOCK_EPILOGUE_DEQUANT_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape &blockShape, const BlockCoord &blockCoord)
{
    singleM_ = Get<MNK_M>(blockShape);
    singleN_ = Get<MNK_N>(blockShape);
    uint32_t halfSingleM = CeilDiv(singleM_, static_cast<uint64_t>(AscendC::GetTaskRation()));
    alignN_ = CeilDiv(singleN_, BLOCK_ELEMENTS_FP32S) * BLOCK_ELEMENTS_FP32S;
    uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;
    vlForFloatNumber_ = BLOCKS_BYTES / sizeof(DataTypeOut);
    auto repeatTimesRe = CeilDiv(singleN_, vlForFloatNumber_);
    uint64_t logitOffset = Get<LOGIT_INDEXS>(blockCoord) + mOffset;
    uint64_t yOffset = Get<Y_IDXS>(blockCoord);
    uint16_t remainRepeatTimesLogit =
        (singleMInVec % MAX_OUTPUT_M_UBS != 0) ? singleMInVec % MAX_OUTPUT_M_UBS : MAX_OUTPUT_M_UBS;
    blockCoord_ = blockCoord;
    auto logitUb = logitPingPongID_ == 0 ? logitUbPing_ : logitUbPong_;
    auto x2ScaleUb = logitPingPongID_ == 0 ? x2ScaleUbPing_ : x2ScaleUbPong_;
    auto x1ScaleUb = logitPingPongID_ == 0 ? x1ScaleUbPing_ : x1ScaleUbPong_;
    auto biasUb = logitPingPongID_ == 0 ? biasUbPing_ : biasUbPong_;
    CopyInLogit(singleMInVec, logitOffset, logitUb);
    CopyX2ScaleFromGm2Ub(x2ScaleUb, 0);
    if (params_->x1ScaleGmAddr != nullptr) {
        CopyX1ScaleFromGm2Ub(x1ScaleUb, singleMInVec * sizeof(DataTypeX1Scale), mOffset);
    }
    if (params_->biasGmAddr != nullptr) {
        CopyBiasFromGm2Ub(biasUb);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(logitPingPongID_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(logitPingPongID_);
    VFDoDequantWithX1X2Scale(x2ScaleUb, x1ScaleUb, biasUb, singleMInVec);
    logitPingPongID_ = (logitPingPongID_ + 1) & 1;
    uint32_t loopNumY = CeilDiv(singleMInVec, MAX_OUTPUT_M_UBS);
    for (uint32_t i = 0; i < loopNumY; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(yPingPongID_);
        repeatTimesLine_ = (i == loopNumY - 1) ? remainRepeatTimesLogit : MAX_OUTPUT_M_UBS;
        __ubuf__ DataTypeOut *outUbAddr = yPingPongID_ == 0 ? (__ubuf__ DataTypeOut *)outUbPing_.GetPhyAddr() :
                                                                   (__ubuf__ DataTypeOut *)outUbPong_.GetPhyAddr();
        VFDoLogitMuls(i * MAX_OUTPUT_M_UBS * alignN_, i * MAX_OUTPUT_M_UBS, repeatTimesLine_, repeatTimesRe, outUbAddr,
                      logitUb);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(yPingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(yPingPongID_);
        auto yLocal = yPingPongID_ == 0 ? outUbPing_ : outUbPong_;
        VectorAtomicProcess(singleN_, repeatTimesLine_, logitOffset + i * MAX_OUTPUT_M_UBS, yOffset, yLocal);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(yPingPongID_);
        yPingPongID_ = (yPingPongID_ + 1) & 1;
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(logitPingPongID_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(logitPingPongID_);
}
} // namespace Block
} // namespace Gemm
} // namespace Cgmct
#endif
#endif