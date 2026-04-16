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
 * \file block_prologue_finalize_routing.h
 * \brief
 */

#ifndef BLOCK_PROLOGUE_FINALIZE_ROUTING_H
#define BLOCK_PROLOGUE_FINALIZE_ROUTING_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"

namespace Cgmct {
namespace Gemm {
namespace Block {

namespace {
constexpr uint32_t UB_INIT_REZO_LEN = 1024 * 2;
constexpr uint32_t UB_DOUBLE_BUFFER_LEN = 1024 * 8;
constexpr uint32_t ONE_CORE_ALIGN_LEN = 512;
constexpr uint32_t ONE_CORE_ALIGN_LEN_SMALL = 32;

} // namespace

static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_FP16_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

#define BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS template <typename DataTypeOut_, typename DataTypeIn_>
#define BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS DataTypeOut_, DataTypeIn_

BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
class BlockPrologueFinalizeRouting {
public:
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;

    __aicore__ inline BlockPrologueFinalizeRouting(){};
    struct Arguments {
        GM_ADDR residualGm{nullptr};
        GM_ADDR yGmAddr{nullptr};
        uint32_t sharedInputOffset = 0;
        uint32_t sharedInputLen = 0;
        int32_t n;
        uint32_t batch;
        float residualScale = 1.0;
        Arguments() = default;
    };

    using Params = Arguments;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline void operator()();

private:
    __aicore__ inline void InitAllLocalTensor();
    __aicore__ inline void InitOutputWithZeros(uint64_t offset, uint64_t size);
    __aicore__ inline void Compute(uint64_t singleCount, uint64_t outOffset, uint64_t baseOffset, uint64_t curCount);
    __aicore__ inline void CopyInShareInput(LocalTensor<DataTypeIn> &residualLocal, uint64_t offset, uint64_t size);
    __aicore__ inline void CopyOutShareInput(LocalTensor<DataTypeOut> &yLocal, uint64_t offset, uint64_t size);
    __aicore__ inline void VFDoSharedCastAndMuls(__ubuf__ DataTypeOut *dstPtr, __ubuf__ DataTypeIn *residualLocalUbAddr,
                                                 float sharedWeight, uint64_t curCount);

private:
    const Params *params_;
    AscendC::GlobalTensor<DataTypeIn> residualGm_;
    AscendC::GlobalTensor<DataTypeOut> yGmAddr_;
    // local tensor
    AscendC::LocalTensor<DataTypeOut> initWithZero_;
    AscendC::LocalTensor<DataTypeIn> shareInputUbPing_;
    AscendC::LocalTensor<DataTypeIn> shareInputUbPong_;
    AscendC::LocalTensor<DataTypeOut> UbOutPing_;
    AscendC::LocalTensor<DataTypeOut> UbOutPong_;

    uint32_t vectorCoreNum_ = 0;
    bool isDataBlockInitialized_ = false; // initWithZero_ only needs to be initialized once
};

BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockPrologueFinalizeRouting<BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::Init(Params const &params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    InitAllLocalTensor();
    residualGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeIn *>(params_->residualGm));
    yGmAddr_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut *>(params_->yGmAddr));
}

BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockPrologueFinalizeRouting<BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::CopyInShareInput(
    LocalTensor<DataTypeIn> &residualLocal, uint64_t offset, uint64_t size)
{
    DataCopyExtParams paramsIn{1, static_cast<uint32_t>(size * sizeof(DataTypeIn)), 1, 1, 0};
    DataCopyPadExtParams<DataTypeIn> Padparams;
    DataCopyPad(residualLocal, residualGm_[offset], paramsIn, Padparams);
}

BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockPrologueFinalizeRouting<BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::CopyOutShareInput(
    LocalTensor<DataTypeOut> &yLocal, uint64_t offset, uint64_t size)
{
    DataCopyExtParams paramsOut{1, static_cast<uint32_t>(size * sizeof(DataTypeOut)), 1, 1, 0};
    DataCopyPad(yGmAddr_[offset], yLocal, paramsOut);
}

BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockPrologueFinalizeRouting<BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::InitAllLocalTensor()
{
    initWithZero_ = AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECCALC, 0, UB_INIT_REZO_LEN);
    uint32_t shareInputUbPingOffset = UB_INIT_REZO_LEN * sizeof(DataTypeOut);
    shareInputUbPing_ =
        AscendC::LocalTensor<DataTypeIn>(AscendC::TPosition::VECIN, shareInputUbPingOffset, UB_DOUBLE_BUFFER_LEN);
    uint32_t shareInputUbPongOffset = shareInputUbPingOffset + UB_DOUBLE_BUFFER_LEN * sizeof(DataTypeIn);
    shareInputUbPong_ =
        AscendC::LocalTensor<DataTypeIn>(AscendC::TPosition::VECIN, shareInputUbPongOffset, UB_DOUBLE_BUFFER_LEN);
    uint32_t UbOutPingOffset = shareInputUbPongOffset + UB_DOUBLE_BUFFER_LEN * sizeof(DataTypeIn);
    UbOutPing_ = AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECIN, UbOutPingOffset, UB_DOUBLE_BUFFER_LEN);
    uint32_t UbOutPongOffset = UbOutPingOffset + UB_DOUBLE_BUFFER_LEN * sizeof(DataTypeOut);
    UbOutPong_ = AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECIN, UbOutPongOffset, UB_DOUBLE_BUFFER_LEN);
}

BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockPrologueFinalizeRouting<BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::InitOutputWithZeros(
    uint64_t outOffset, uint64_t totalLen)
{
    // may be sharedInput Offset = 0 or tail = batch
    if (totalLen == 0) {
        return;
    }
    // one core perSize
    uint32_t singleCount = CeilDiv(totalLen, static_cast<uint64_t>(vectorCoreNum_));
    singleCount = CeilDiv(singleCount, static_cast<uint64_t>(ONE_CORE_ALIGN_LEN)) * ONE_CORE_ALIGN_LEN;

    uint64_t baseOffset = GetBlockIdx() * singleCount;
    if (baseOffset > totalLen) {
        return;
    }

    if (baseOffset + singleCount > totalLen) {
        singleCount = totalLen - baseOffset;
    }

    if (!isDataBlockInitialized_) {
        Duplicate<DataTypeOut>(initWithZero_, 0, UB_INIT_REZO_LEN);
        isDataBlockInitialized_ = true;
    }
    // the baseOffset addr of cur vector core, copy out to GM
    baseOffset += outOffset;
    // singleCount smaller than UB_INIT_REZO_LEN, just copy one time
    if (singleCount <= UB_INIT_REZO_LEN) {
        CopyOutShareInput(initWithZero_, baseOffset, singleCount);
        return;
    }
    // once copy size
    uint32_t ubOnceCopyLen = UB_INIT_REZO_LEN;
    for (uint32_t offset = 0; offset < singleCount; offset += ubOnceCopyLen) {
        if (unlikely(offset + ubOnceCopyLen > singleCount)) {
            ubOnceCopyLen = singleCount - offset;
        }
        CopyOutShareInput(initWithZero_, baseOffset + offset, ubOnceCopyLen);
    }
}

BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockPrologueFinalizeRouting<BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::VFDoSharedCastAndMuls(
    __ubuf__ DataTypeOut *dstPtr, __ubuf__ DataTypeIn *residualLocalUbAddr, float sharedWeight, uint64_t curCount)
{
    constexpr uint32_t vectorRegLen = VECTOR_REG_WIDTH / sizeof(DataTypeOut);
    uint16_t loopNum =
        CeilDiv(static_cast<uint64_t>(curCount), static_cast<uint64_t>(vectorRegLen));
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<DataTypeIn> vSrcReg;
        AscendC::MicroAPI::RegTensor<DataTypeOut> vRegCast;
        AscendC::MicroAPI::RegTensor<DataTypeOut> vDstReg;
        uint32_t elementNum = static_cast<uint32_t>(curCount);
        AscendC::MicroAPI::MaskReg mask;
        uint16_t oneRepeatSize = vectorRegLen;
        for (uint16_t i = 0; i < loopNum; ++i) {
            mask = AscendC::MicroAPI::UpdateMask<DataTypeOut>(elementNum);
            AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vSrcReg, residualLocalUbAddr + i * oneRepeatSize);
            AscendC::MicroAPI::Cast<DataTypeOut, DataTypeIn, CAST_BF16_FP16_TO_FP32>(vRegCast, vSrcReg, mask);
            AscendC::MicroAPI::Muls(vDstReg, vRegCast, sharedWeight, mask);
            AscendC::MicroAPI::DataCopy(dstPtr + i * oneRepeatSize, vDstReg, mask);
        }
    }
}

BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockPrologueFinalizeRouting<BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::Compute(
    uint64_t singleCount, uint64_t outOffset, uint64_t baseOffset, uint64_t curCount)
{
    uint32_t pingPongID = 0;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    for (uint32_t offset = 0; offset < singleCount; offset += curCount) {
        if (unlikely(offset + curCount > singleCount)) {
            curCount = singleCount - offset;
        }
        AscendC::LocalTensor<DataTypeIn> *residualLocal = pingPongID == 0 ? &shareInputUbPing_ : &shareInputUbPong_;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingPongID);
        CopyInShareInput(*residualLocal, baseOffset + offset, curCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingPongID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingPongID);
        AscendC::LocalTensor<DataTypeOut> *yLocal = pingPongID == 0 ? &UbOutPing_ : &UbOutPong_;
        __ubuf__ DataTypeIn *residualLocalUbAddr = (__ubuf__ DataTypeIn *)residualLocal->GetPhyAddr();
        __ubuf__ DataTypeOut *outUbAddr = (__ubuf__ DataTypeOut *)yLocal->GetPhyAddr();
        // before vector compute,need wait MTE3 copyout finished
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingPongID);
        VFDoSharedCastAndMuls(outUbAddr, residualLocalUbAddr, params_->residualScale, curCount);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingPongID);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingPongID);
        CopyOutShareInput(*yLocal, outOffset + offset, curCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingPongID);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingPongID);
        pingPongID = (pingPongID + 1) & 1;
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
}

BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockPrologueFinalizeRouting<BLOCK_PROLOGUE_BLOCK_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::operator()()
{
    if ASCEND_IS_AIC {
        return;
    }
    vectorCoreNum_ = uint32_t(GetBlockNum() * GetTaskRation());
    if (GetBlockIdx() >= vectorCoreNum_) {
        return;
    }
    auto sharedInputOffset = params_->sharedInputOffset;
    auto sharedInputLen = params_->sharedInputLen;
    uint64_t firstZeroSize = params_->n * sharedInputOffset;
    // sharedInput is None,only do InitOutputWithZeros, all shape(batch,n) ,then finish
    if (sharedInputLen == 0) {
        InitOutputWithZeros(0, params_->n * params_->batch);
        return;
    }

    InitOutputWithZeros(0, firstZeroSize);
    uint64_t tail = sharedInputOffset + sharedInputLen;
    InitOutputWithZeros(tail * params_->n, params_->n * (params_->batch - tail));

    uint64_t totalOutput = static_cast<uint64_t>(params_->n) * sharedInputLen;
    uint64_t singleCount = CeilDiv(totalOutput, static_cast<uint64_t>(vectorCoreNum_));
    // one vector core perSize
    auto alignLen = totalOutput > ONE_CORE_ALIGN_LEN ? ONE_CORE_ALIGN_LEN : ONE_CORE_ALIGN_LEN_SMALL;
    singleCount = CeilDiv(singleCount, static_cast<uint64_t>(alignLen)) * alignLen;
    uint64_t baseOffset = GetBlockIdx() * singleCount;
    if (baseOffset >= totalOutput) {
        return;
    }
    // last vector core process real size
    if (baseOffset + singleCount > totalOutput) {
        singleCount = totalOutput - baseOffset;
    }
    uint64_t outOffset = baseOffset + firstZeroSize;
    uint64_t curCount = UB_DOUBLE_BUFFER_LEN;
    Compute(singleCount, outOffset, baseOffset, curCount);
}

} // namespace Block
} // namespace Gemm
} // namespace Cgmct
#endif
