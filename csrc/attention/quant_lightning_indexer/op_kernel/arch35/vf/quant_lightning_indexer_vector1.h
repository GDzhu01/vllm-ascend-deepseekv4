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
 * \file quant_lightning_indexer_vector1.h
 * \brief
 */
#ifndef quant_lightning_indexer_VECTOR1_H
#define quant_lightning_indexer_VECTOR1_H

#include "kernel_operator.h"
namespace vector1{
constexpr uint32_t MASK_NUM_F    = 0xFFFFFFFF;
constexpr uint32_t MASK_NUM_ZERO = 0x80000000;
constexpr uint32_t MASK_NUM_NAN  = 0x7FC00000;
__aicore__ inline void MulWeightAndReduceSum(LocalTensor<uint32_t> out,//out [1, s2] 128 2
                                             LocalTensor<float> qk,//q*k[1, G, s2] 64* 128 2
                                             LocalTensor<float> weight,//w[1, G] 64 1
                                             LocalTensor<float> kScale,//kScale [1, s2] 128 2 
                                             LocalTensor<float> qScale,//qScale [1, G] 64 1
                                             LocalTensor<float> quantWeight,
                                             const int &gSize){//G 64
    __local_mem__ float* weight_ = (__local_mem__ float*)weight.GetPhyAddr();
    __local_mem__ float* qk_ = (__local_mem__ float*)qk.GetPhyAddr();
    __local_mem__ float* qScale_ = (__local_mem__ float*)qScale.GetPhyAddr();
    __local_mem__ float* kScale_ = (__local_mem__ float*)kScale.GetPhyAddr();
    __local_mem__ uint32_t* out_ = (__local_mem__ uint32_t*)out.GetPhyAddr();
    __local_mem__ float* quantWeight_ = (__local_mem__ float*)quantWeight.GetPhyAddr();

    __VEC_SCOPE__
    {

        AscendC::MicroAPI::RegTensor<float> regQK[2];
        AscendC::MicroAPI::RegTensor<float> regw;
        AscendC::MicroAPI::RegTensor<float> regwBrc;        
        AscendC::MicroAPI::RegTensor<float> regQScale;
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regsum[2];
        AscendC::MicroAPI::RegTensor<uint32_t> regsumuint32[2];
        AscendC::MicroAPI::RegTensor<uint32_t> regConvertMaskF;
        AscendC::MicroAPI::RegTensor<uint32_t> regConvertMask0;
        AscendC::MicroAPI::RegTensor<uint32_t> regAllZero;
        AscendC::MicroAPI::RegTensor<uint32_t> regNan;                
        AscendC::MicroAPI::RegTensor<uint32_t> regTemp[2];
        AscendC::MicroAPI::RegTensor<uint32_t> regMask[2];
        AscendC::MicroAPI::RegTensor<uint32_t> regXor[2];
        AscendC::MicroAPI::MaskReg regSelect[2];
        AscendC::MicroAPI::MaskReg regSelectNan[2];
        AscendC::MicroAPI::MaskReg _fullMask_0 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {AscendC::MicroAPI::RegLayout::ZERO,AscendC::MicroAPI::SatMode::UNKNOWN, 
                                                            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        AscendC::MicroAPI::LoadAlign<float>(regKScale[0], kScale_ );
        AscendC::MicroAPI::LoadAlign<float>(regKScale[1], kScale_ + 64);

        AscendC::MicroAPI::Duplicate(regsum[0], 0.0f, _fullMask_0);
        AscendC::MicroAPI::Duplicate(regsum[1], 0.0f, _fullMask_0);

        AscendC::MicroAPI::Duplicate(regAllZero, 0, _fullMask_0);
        AscendC::MicroAPI::Duplicate(regConvertMaskF, MASK_NUM_F, _fullMask_0);
        AscendC::MicroAPI::Duplicate(regConvertMask0, MASK_NUM_ZERO, _fullMask_0);
        AscendC::MicroAPI::Duplicate(regNan, MASK_NUM_NAN, _fullMask_0);

        AscendC::MicroAPI::LoadAlign<float>(regw, weight_);
        AscendC::MicroAPI::LoadAlign<float>(regQScale, qScale_);
        AscendC::MicroAPI::Mul(regw, regw, regQScale, _fullMask_0);
        AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(quantWeight_, regw, _fullMask_0);

        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); ++i){
            AscendC::MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i);
            AscendC::MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + 64);

            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc, quantWeight_ + i);

            AscendC::MicroAPI::Relu(regQK[0], regQK[0], _fullMask_0);
            AscendC::MicroAPI::Relu(regQK[1], regQK[1], _fullMask_0);

            AscendC::MicroAPI::MulAddDst(regsum[0], regQK[0], regwBrc, _fullMask_0);
            AscendC::MicroAPI::MulAddDst(regsum[1], regQK[1], regwBrc, _fullMask_0);
        }
        AscendC::MicroAPI::Mul(regsum[0], regsum[0], regKScale[0], _fullMask_0);
        AscendC::MicroAPI::Mul(regsum[1], regsum[1], regKScale[1], _fullMask_0);

        AscendC::MicroAPI::Compare<uint32_t, CMPMODE::EQ>(regSelectNan[0] , (AscendC::MicroAPI::RegTensor<uint32_t> &)regsum[0], regNan, _fullMask_0);
        AscendC::MicroAPI::Compare<uint32_t, CMPMODE::EQ>(regSelectNan[1] , (AscendC::MicroAPI::RegTensor<uint32_t> &)regsum[1], regNan, _fullMask_0);

        AscendC::MicroAPI::Select(regsumuint32[0],  regConvertMaskF, (AscendC::MicroAPI::RegTensor<uint32_t> &)regsum[0], regSelectNan[0]);
        AscendC::MicroAPI::Select(regsumuint32[1],  regConvertMaskF, (AscendC::MicroAPI::RegTensor<uint32_t> &)regsum[1], regSelectNan[1]);

        AscendC::MicroAPI::And(regTemp[0], regsumuint32[0], regConvertMask0, _fullMask_0);
        AscendC::MicroAPI::And(regTemp[1], regsumuint32[1], regConvertMask0, _fullMask_0);

        AscendC::MicroAPI::Compare<uint32_t, CMPMODE::GT>(regSelect[0] , regTemp[0], regAllZero, _fullMask_0);
        AscendC::MicroAPI::Compare<uint32_t, CMPMODE::GT>(regSelect[1] , regTemp[1], regAllZero, _fullMask_0);

        AscendC::MicroAPI::Select(regMask[0], regConvertMaskF, regConvertMask0, regSelect[0]);
        AscendC::MicroAPI::Select(regMask[1], regConvertMaskF, regConvertMask0, regSelect[1]);

        AscendC::MicroAPI::Xor(regXor[0], regsumuint32[0], regMask[0], _fullMask_0);
        AscendC::MicroAPI::Xor(regXor[1], regsumuint32[1], regMask[1], _fullMask_0);

        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>((__local_mem__ uint32_t*&)out_, regXor[0], _fullMask_0);
        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>((__local_mem__ uint32_t*&)out_ + 64, regXor[1], _fullMask_0);
    }
}
}
#endif