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
 * \file fp8_dynamic_quant_pertoken.h
 * \brief
 */

#ifndef FP8_DYNAMIC_QUANT_PERTOKEN_H
#define FP8_DYNAMIC_QUANT_PERTOKEN_H

namespace MC2KernelTemplate {
struct MC2PertokenDQuantContext {
    GM_ADDR quantInputAddr;
    GM_ADDR quantOutputAddr;
    GM_ADDR transposeDstAddr;
    GM_ADDR quantOutputScaleAddr;
    uint64_t rowNum;
    uint64_t colNumPerBlock;
    uint64_t blockNum;
    uint64_t nextBlockDataOffset;
    uint64_t quantInputAddrOffset;
    uint64_t quantOutputAddrOffset;
    uint64_t transposeDstAddrOffset;
    uint64_t quantOutputScaleAddrOffset;
};

template <typename quantInputDataType, typename quantOutputDataType>
class Fp8DynamicQuantPertoken {
public:
    __aicore__ inline Fp8DynamicQuantPertoken(TPipe *tPipe) : tPipe_(tPipe){};
    __aicore__ inline MC2PertokenDQuantContext *GetContextPtr();
    __aicore__ inline void Init() {};
    __aicore__ inline void Process(uint32_t taskIndex);

protected:
    static constexpr uint32_t ALIGN_NUM = 8;
    static constexpr uint32_t TWO_FACTOR = 2;
    static constexpr uint32_t ONE_FACTOR = 1;
    static constexpr uint32_t UB_DATABLOCK = 32;
    static constexpr uint32_t COMPARE_ALIGN_LEN = 256;
    static constexpr uint32_t VECTOR_UB_SIZE = AscendC::TOTAL_UB_SIZE;

    static constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
    static constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;

    constexpr static AscendC::MicroAPI::CastTrait castTraitB16ToB32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToI16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitI16ToF16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF16ToI8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32tofp8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32toh8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    template <typename T>
    __aicore__ static inline T Max(T a, T b)
    {
        return (a > b) ? (a) : (b);
    }

    template <typename T>
    __aicore__ static inline T Min(T a, T b)
    {
        return (a > b) ? (b) : (a);
    }

    template <typename T>
    __aicore__ static inline T Ceil(T x, T y)
    {
        return (x + y - 1) / y;
    }

    template <AscendC::HardEvent event>
    __aicore__ static inline void SyncFunc()
    {
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(eventID);
        AscendC::WaitFlag<event>(eventID);
    }

    TPipe *tPipe_;
    MC2PertokenDQuantContext context_;

    GlobalTensor<quantInputDataType> quantInputGM_;
    GlobalTensor<quantInputDataType> transOutGM_;
    GlobalTensor<quantOutputDataType> quantOutputGM_;
    GlobalTensor<float> quantOutputScaleGM_;

    TQue<QuePosition::VECIN, 1> rawInputQue_;     // 存放原始 float16 数据
    TQue<QuePosition::VECOUT, 1> quantOutputQue_; // 存放量化后的 fp8 数据
    TQue<QuePosition::VECOUT, 1> transOutQue_;    // 存放转置后的原始 float16 数据

    TBuf<TPosition::VECCALC> scaleWorkBuf_;
    TBuf<TPosition::VECOUT> quantScaleBuf_;
    TBuf<TPosition::VECCALC> maxValueBuf_;

    uint64_t usedCoreAivNum_ = 0;   // 使用的aiv核数
    uint64_t rowsThisCore_ = 0;     // 当前核负责的总行数
    uint64_t startRowThisCore_ = 0; // 当前核负责的起始行索引

    float recipFP8MaxLimit_ = 0.0f;
    float fp8MaxLimit_ = 0.0f;
    bool isTransOut_ = false;

    __aicore__ inline void SetMaxValue();
    __aicore__ inline void ProcessOneTokenRegBase();
    __aicore__ inline void CalculateMaxRegBase(__local_mem__ quantInputDataType *xAddr, __local_mem__ float *maxAddr);
    __aicore__ inline void CalculateScale(__local_mem__ float *scaleAddr, float maxValue);
    __aicore__ inline void DoQuantRegBase(__local_mem__ quantInputDataType *xAddr,
                                          __local_mem__ quantOutputDataType *yAddr, uint64_t dataCount, float scale);
    __aicore__ inline void Init(GM_ADDR quantInputAddr, GM_ADDR quantOutputAddr, GM_ADDR quantOutputScaleAddr,
                                GM_ADDR transposeDstAddr);
    __aicore__ inline void Process();
    __aicore__ inline void Destroy();
};

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline MC2PertokenDQuantContext *
Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::GetContextPtr()
{
    return &context_;
}

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::Process(uint32_t taskIndex)
{
    GM_ADDR quantInputAddr = context_.quantInputAddr + taskIndex * context_.quantInputAddrOffset;
    GM_ADDR quantOutputAddr = context_.quantOutputAddr + taskIndex * context_.quantOutputAddrOffset;
    GM_ADDR transposeDstAddr = (context_.transposeDstAddr == nullptr) ?
                                   nullptr :
                                   (context_.transposeDstAddr + taskIndex * context_.transposeDstAddrOffset);
    GM_ADDR quantOutputScaleAddr = context_.quantOutputScaleAddr + taskIndex * context_.quantOutputScaleAddrOffset;
    Init(quantInputAddr, quantOutputAddr, quantOutputScaleAddr, transposeDstAddr);
    Process();
    Destroy();
}

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::Init(
    GM_ADDR quantInputAddr, GM_ADDR quantOutputAddr, GM_ADDR quantOutputScaleAddr, GM_ADDR transposeDstAddr)
{
    if ASCEND_IS_AIC {
        return;
    }

    tPipe_->Reset();

    uint64_t totalCores = static_cast<uint64_t>(GetBlockNum() * TWO_FACTOR);
    this->usedCoreAivNum_ = (context_.rowNum < totalCores) ? context_.rowNum : totalCores;
    // 2. 均匀分配任务到各个核
    uint32_t coreIdx = GetBlockIdx();
    uint64_t avgRows = context_.rowNum / this->usedCoreAivNum_;
    uint32_t tailRows = context_.rowNum % this->usedCoreAivNum_;

    // 每个核计算自己的任务范围，如10行数据，使用4个核（0,1,2,3），第1个aiv核处理3,4,5行，startRowThisCore_的起始索引为3
    this->rowsThisCore_ = avgRows + (coreIdx < tailRows ? 1 : 0);
    this->startRowThisCore_ = coreIdx * avgRows + (coreIdx < tailRows ? coreIdx : tailRows);

    SetMaxValue();

    quantInputGM_.SetGlobalBuffer((__gm__ quantInputDataType *)quantInputAddr);
    quantOutputGM_.SetGlobalBuffer((__gm__ quantOutputDataType *)quantOutputAddr);
    quantOutputScaleGM_.SetGlobalBuffer((__gm__ float *)quantOutputScaleAddr);
    if (transposeDstAddr != nullptr) {
        this->isTransOut_ = true;
        transOutGM_.SetGlobalBuffer((__gm__ quantInputDataType *)transposeDstAddr);
    } else {
        this->isTransOut_ = false;
    }
}

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::Process()
{
    if (GetBlockIdx() >= this->usedCoreAivNum_) {
        return;
    }
    ProcessOneTokenRegBase();
}

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::Destroy()
{
    rawInputQue_.FreeAllEvent();
    quantOutputQue_.FreeAllEvent();
    transOutQue_.FreeAllEvent();
}

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::SetMaxValue()
{
    if constexpr (IsSameType<quantOutputDataType, fp8_e5m2_t>::value) {
        this->recipFP8MaxLimit_ = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
        this->fp8MaxLimit_ = FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<quantOutputDataType, fp8_e4m3fn_t>::value) {
        this->recipFP8MaxLimit_ = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
        this->fp8MaxLimit_ = FP8_E4M3FN_MAX_VALUE;
    }
}

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::ProcessOneTokenRegBase()
{
    uint32_t perBlockDataLength = static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantInputDataType));
    uint32_t inputSizePerBlock = Ceil(perBlockDataLength, UB_DATABLOCK) * UB_DATABLOCK;
    uint32_t inputSize = inputSizePerBlock * context_.blockNum;
    constexpr uint8_t tempRShiftAmountIn = sizeof(quantInputDataType) / 2;
    uint64_t perBlockDataOffsetInUB = inputSizePerBlock >> tempRShiftAmountIn;
    uint32_t outputSizePerBlock =
        Ceil(static_cast<uint32_t>(perBlockDataOffsetInUB * sizeof(quantOutputDataType)), UB_DATABLOCK) * UB_DATABLOCK;
    uint32_t outputSize = perBlockDataOffsetInUB * context_.blockNum;
    constexpr uint8_t tempRShiftAmountOut = sizeof(quantOutputDataType) / 2;
    uint64_t perBlockQuantDataOffsetInUB = outputSizePerBlock >> tempRShiftAmountOut;

    tPipe_->InitBuffer(scaleWorkBuf_, UB_DATABLOCK);
    tPipe_->InitBuffer(maxValueBuf_, UB_DATABLOCK);
    tPipe_->InitBuffer(quantScaleBuf_,
                       Ceil(static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), UB_DATABLOCK) * UB_DATABLOCK);
    tPipe_->InitBuffer(rawInputQue_, ONE_FACTOR, inputSize);
    tPipe_->InitBuffer(quantOutputQue_, ONE_FACTOR, outputSize);

    LocalTensor<float> scaleWorkData = scaleWorkBuf_.Get<float>();
    LocalTensor<float> maxValueData = maxValueBuf_.Get<float>();
    LocalTensor<float> coreQuantScales = quantScaleBuf_.Get<float>();
    LocalTensor<quantInputDataType> rawInputTensor = rawInputQue_.AllocTensor<quantInputDataType>();
    LocalTensor<quantOutputDataType> quantOut = quantOutputQue_.AllocTensor<quantOutputDataType>();
    transOutQue_.EnQue(rawInputTensor);
    quantOutputQue_.EnQue(quantOut);

    // 2.逐token处理
    float scale;
    float maxValue;
    float maxValuePerRank;
    // 申请两个event管理非连续block数据的并行搬入
    AscendC::TEventID rawInputEvent[2];
    rawInputEvent[0] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_S>();
    rawInputEvent[1] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_S>();
    for (uint64_t r = 0; r < this->rowsThisCore_; ++r) {
        uint64_t globalRowIdx = this->startRowThisCore_ + r;
        uint64_t globalInputIdx = globalRowIdx * context_.colNumPerBlock;
        transOutQue_.DeQue();
        // 提前启动数据搬运
        DataCopyPad<quantInputDataType>(rawInputTensor, quantInputGM_[globalInputIdx], {1, perBlockDataLength, 0, 0, 0},
                                        {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[0]);
        DataCopyPad<quantInputDataType>(rawInputTensor[perBlockDataOffsetInUB],
                                        quantInputGM_[globalInputIdx + context_.nextBlockDataOffset],
                                        {1, perBlockDataLength, 0, 0, 0}, {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[1]);

        // 非连续K循环处理
        // regbase循环需要的值
        __local_mem__ quantInputDataType *xAddr = (__local_mem__ quantInputDataType *)rawInputTensor.GetPhyAddr();
        __local_mem__ float *maxValueAddr = (__local_mem__ float *)maxValueData.GetPhyAddr();
        maxValue = 0.0f;
        maxValuePerRank = 0.0f;
        for (uint32_t k = 0; k < context_.blockNum; ++k) {
            // 搬入一块非连续的K
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[k % 2]);
            // 提前启动一块数据搬运
            uint32_t nextK = k + 2;
            if (nextK < context_.blockNum) {
                DataCopyPad<quantInputDataType, PaddingMode::Normal>(
                    rawInputTensor[nextK * perBlockDataOffsetInUB],
                    quantInputGM_[globalInputIdx + nextK * context_.nextBlockDataOffset],
                    {1, perBlockDataLength, 0, 0, 0}, {false, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[k % 2]);
            }

            // 计算最大绝对值
            CalculateMaxRegBase(xAddr + k * perBlockDataOffsetInUB, maxValueAddr);
            SyncFunc<AscendC::HardEvent::V_S>();
            maxValuePerRank = maxValueData.GetValue(0);
            maxValue = Max(maxValue, maxValuePerRank);
        }
        // 转置结果搬出
        uint64_t globalOutIdx = globalRowIdx * context_.colNumPerBlock * context_.blockNum;
        if (this->isTransOut_) {
            DataCopyPad<quantInputDataType, PaddingMode::Normal>(
                transOutGM_[globalOutIdx], rawInputTensor,
                {static_cast<uint16_t>(context_.blockNum), perBlockDataLength, 0, 0, 0});
        }
        transOutQue_.EnQue(rawInputTensor);

        // 计算量化系数
        __local_mem__ float *scaleAddr = (__local_mem__ float *)scaleWorkData.GetPhyAddr();
        CalculateScale(scaleAddr, maxValue);
        SyncFunc<AscendC::HardEvent::V_S>();
        scale = scaleWorkData.GetValue(0);
        coreQuantScales.SetValue(r, scale);
        SyncFunc<AscendC::HardEvent::S_V>();

        // 量化
        quantOutputQue_.DeQue();
        __local_mem__ quantOutputDataType *yAddr = (__local_mem__ quantOutputDataType *)quantOut.GetPhyAddr();
        for (uint32_t k = 0; k < context_.blockNum; ++k) {
            DoQuantRegBase(xAddr + k * perBlockDataOffsetInUB, yAddr + k * perBlockQuantDataOffsetInUB,
                           perBlockDataOffsetInUB, scale);
        }
        SyncFunc<AscendC::HardEvent::V_S>();

        // // 搬出量化结果
        DataCopyPad<quantOutputDataType, PaddingMode::Normal>(
            quantOutputGM_[globalOutIdx], quantOut,
            {static_cast<uint16_t>(context_.blockNum),
             static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantOutputDataType)), 0, 0, 0});
        quantOutputQue_.EnQue(quantOut);
    }

    DataCopyExtParams outScaleParams = {1, static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float, PaddingMode::Normal>(quantOutputScaleGM_[this->startRowThisCore_], coreQuantScales,
                                            outScaleParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    // 释放资源
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_S>(rawInputEvent[0]);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_S>(rawInputEvent[1]);
    transOutQue_.DeQue();
    quantOutputQue_.DeQue();
    rawInputQue_.FreeTensor(rawInputTensor);
    quantOutputQue_.FreeTensor(quantOut);
}


template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::CalculateMaxRegBase(
    __local_mem__ quantInputDataType *xAddr, __local_mem__ float *maxAddr)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<quantInputDataType> vregX;
        AscendC::MicroAPI::RegTensor<float> vregFloatX;
        AscendC::MicroAPI::RegTensor<float> vregAbsX;
        AscendC::MicroAPI::RegTensor<float> vregMaxAbsX;
        AscendC::MicroAPI::RegTensor<float> vregReduceMax;

        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg ureg0;

        AscendC::MicroAPI::Duplicate(vregMaxAbsX, 0.0f, preg1);
        uint32_t dtypeSize = sizeof(float);
        uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize; // 每个向量寄存器能存的元素数
        // colNum对齐到16
        uint32_t colNum = (context_.colNumPerBlock + 15) / 16 * 16;
        uint16_t vfLoop = (colNum + VL - 1) / VL;
        uint32_t sreg0 = colNum;

        // 计算每行的最大值（绝对值）
        for (uint16_t j = 0; j < vfLoop; j++) {
            preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
            // 1. 加载数据到向量寄存器，fp16和bf16用DIST_UNPACK_B16，fp32用DIST_NORM
            AscendC::MicroAPI::DataCopy<quantInputDataType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vregX, xAddr + j * VL);
            // 2. 转换为float
            AscendC::MicroAPI::Cast<float, quantInputDataType, castTraitB16ToB32>(vregFloatX, vregX, preg0);
            // 4. 计算绝对值
            AscendC::MicroAPI::Abs(vregAbsX, vregFloatX, preg0);
            // 5. 更新最大值
            AscendC::MicroAPI::Max(vregMaxAbsX, vregAbsX, vregMaxAbsX, preg1);
        }
        // 6. 归约得到全局最大值
        AscendC::MicroAPI::ReduceMax(vregReduceMax, vregMaxAbsX, preg1);
        // 7. 存储全局最大值到内存
        AscendC::MicroAPI::DataCopyUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            maxAddr, vregReduceMax, ureg0, 1);
        AscendC::MicroAPI::DataCopyUnAlignPost(maxAddr, ureg0, 0);
    }
}

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void
Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::CalculateScale(__local_mem__ float *scaleAddr,
                                                                                 float maxValue)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vregReduceMax;
        AscendC::MicroAPI::RegTensor<float> vregScale;

        AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg ureg0;

        AscendC::MicroAPI::Duplicate(vregReduceMax, maxValue, preg1);
        // 1. 计算scale: reduceMax / fp8MaxLimit
        AscendC::MicroAPI::Muls(vregScale, vregReduceMax, this->recipFP8MaxLimit_, preg1);
        // 2. 存储scale到内存（用于反量化）
        AscendC::MicroAPI::DataCopyUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            scaleAddr, vregScale, ureg0, 1);
        AscendC::MicroAPI::DataCopyUnAlignPost(scaleAddr, ureg0, 0);
    }
}

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::DoQuantRegBase(
    __local_mem__ quantInputDataType *xAddr, __local_mem__ quantOutputDataType *yAddr, uint64_t dataCount, float scale)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<quantInputDataType> vregX;
        AscendC::MicroAPI::RegTensor<float> vregFloatX;
        AscendC::MicroAPI::RegTensor<float> vregScale;
        AscendC::MicroAPI::RegTensor<float> vregScaledX;
        AscendC::MicroAPI::RegTensor<int16_t> vregYInt16;
        AscendC::MicroAPI::RegTensor<half> vregYFp16;
        AscendC::MicroAPI::RegTensor<quantOutputDataType> vregY;

        AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg preg2;
        AscendC::MicroAPI::MaskReg preg3 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::H>();

        AscendC::MicroAPI::Duplicate(vregScale, scale, preg1);

        uint32_t dtypeSize = sizeof(float);
        uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize; // 每个向量寄存器能存的元素数
        // colNum对齐到16
        uint32_t colNum = (dataCount + 15) / 16 * 16;
        uint16_t vfLoop = (colNum + VL - 1) / VL;
        uint32_t sreg1 = colNum;
        for (uint16_t j = 0; j < vfLoop; j++) {
            auto addr = yAddr + j * VL;
            preg2 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
            // 1. 重新加载数据（或可以重用之前加载的）
            AscendC::MicroAPI::DataCopy<quantInputDataType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vregX, xAddr + j * VL);
            // 2. 转换为float并应用平滑缩放
            AscendC::MicroAPI::Cast<float, quantInputDataType, castTraitB16ToB32>(vregFloatX, vregX, preg2);
            // 3. 除以scale进行缩放
            AscendC::MicroAPI::Div(vregScaledX, vregFloatX, vregScale, preg2);
            // 4. 根据目标类型进行量化
            if constexpr (IsSameType<quantOutputDataType, int8_t>::value) {
                AscendC::MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(vregYInt16, vregScaledX, preg2);
                AscendC::MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(vregYFp16, vregYInt16, preg2);
                AscendC::MicroAPI::Cast<quantOutputDataType, half, castTraitF16ToI8>(vregY, vregYFp16, preg2);
            } else if constexpr (IsSameType<quantOutputDataType, hifloat8_t>::value) {
                AscendC::MicroAPI::Cast<quantOutputDataType, float, castTraitF32toh8>(vregY, vregScaledX, preg2);
            } else if constexpr (IsSameType<quantOutputDataType, fp8_e4m3fn_t>::value ||
                                 IsSameType<quantOutputDataType, fp8_e5m2_t>::value) {
                AscendC::MicroAPI::Cast<quantOutputDataType, float, castTraitF32tofp8>(vregY, vregScaledX, preg2);
            } else if constexpr (IsSameType<quantOutputDataType, int4b_t>::value) {
                AscendC::MicroAPI::RegTensor<uint16_t> vregPackedHalf;
                AscendC::MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(vregYInt16, vregScaledX, preg2);
                AscendC::MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(vregYFp16, vregYInt16, preg2);
                AscendC::MicroAPI::Pack(vregPackedHalf, (AscendC::MicroAPI::RegTensor<uint32_t> &)vregYFp16);
                AscendC::MicroAPI::Cast<int4x2_t, half, castTraitF16ToI8>(
                    (AscendC::MicroAPI::RegTensor<int4x2_t> &)vregY,
                    (AscendC::MicroAPI::RegTensor<half> &)vregPackedHalf, preg2);
                addr = yAddr + (j * VL) / 2;
            }
            // 5. 存储量化结果
            if constexpr (IsSameType<quantOutputDataType, int4b_t>::value) {
                AscendC::MicroAPI::DataCopy<quantOutputDataType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    addr, vregY, preg3);
            } else {
                AscendC::MicroAPI::DataCopy<quantOutputDataType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    addr, vregY, preg2);
            }
        }
    }
}

// fp8DynamicQuant的动态实现
#ifndef DEFINE_MC2_FP8_DYNAMIC_QUANT_PERTOKEN
#define DEFINE_MC2_FP8_DYNAMIC_QUANT_PERTOKEN(QuantInputDataType, QuantOutputDataType, DynamicQuantType)               \
    using DynamicQuantType = MC2KernelTemplate::Fp8DynamicQuantPertoken<QuantInputDataType, QuantOutputDataType>
#endif
} // namespace MC2KernelTemplate
#endif
