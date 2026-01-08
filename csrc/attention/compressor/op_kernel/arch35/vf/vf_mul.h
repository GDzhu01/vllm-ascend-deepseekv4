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
 * \file vf_mul.h
 * \brief
 */

#ifndef VF_MUL_H
#define VF_MUL_H
#include "kernel_operator.h"
using namespace AscendC;
// constexpr uint32_t FLOAT_REP_SIZE = 64;
// constexpr uint32_t BTYEALIGNSIZE = 32;
// constexpr uint32_t REGSIZE = 256;
constexpr uint32_t FLOATBYTE = 4;

template<typename T>
__simd_vf__ void MulReduceSumbaseOneVFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
    const uint16_t coff, const uint16_t r, const uint16_t scLoopCnt, const uint16_t baseD) 
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregMul;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    for(uint16_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, mask);
        for(uint16_t rLoop = 0; rLoop < coff*r; rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + offset);
            MicroAPI::LoadAlign(vreg1, scoreAddr + offset);
            MicroAPI::Mul(vregMul, vreg0, vreg1, mask);
            MicroAPI::Add(vregSum, vregSum, vregMul, mask);
            offset += baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop*baseD, vregSum, mask);
    }
}

/*同时处理2个sc，单寄存器存2个sc某行的元素

  scLoopCnt —— sc需要循环的次数
  scLoopLeft —— 循环后遗留的尾块
  loadAlignParam0、loadAlignParam1
*/
template<typename T>
__simd_vf__ void MulReduceSumbaseTwoVFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
    const uint16_t coff, const uint16_t r, const uint16_t scLoopCnt, const uint16_t baseD) 
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregRHalfSum;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskLHalf = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL32>();
    MicroAPI::MaskReg maskRHalf;
    MicroAPI::Not(maskRHalf, maskLHalf, mask);
    uint32_t count = baseD;
    for(uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, mask);
        MicroAPI::Duplicate(vregRHalfSum, 0, mask);
        for(uint32_t rLoop = 0; rLoop < coff * r / 2; rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + rLoop*2*baseD);
            MicroAPI::LoadAlign(vreg1, scoreAddr + rLoop*2*baseD);
            MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
            MicroAPI::Add(vregSum, vregSum, vreg0, mask);
        }
        MicroAPI::Add(vregRHalfSum, vregRHalfSum, vregSum, maskRHalf);
        MicroAPI::Add(vregSum, vregSum, vregRHalfSum, maskLHalf);
        MicroAPI::StoreAlign(outputAddr + scLoop*baseD, vregSum, maskLHalf);
    }
}

/**
 * @brief MulReduceSumbaseVF 包含mul和reducesum
 * @param outputLocal 输出tensor []
 * @param r s方向的最小块
 * @param sc r个sc
 * @param baseD  核内d轴切分大小
 * @param baseS  行数,
 */


template<typename T>
__aicore__ inline void MulReduceSumbaseVF(LocalTensor<T> &kvLocal, LocalTensor<T> &scoreLocal, LocalTensor<T> &outputLocal,
    const uint16_t coff, const uint16_t r, uint32_t outIdx, const uint16_t baseD, const uint32_t baseS) 
{
    uint32_t scLoopNum = (baseD == FLOAT_REP_SIZE)? 1:2;
    uint32_t scNum = baseS / r;
    uint32_t scLoopCnt = scNum / scLoopNum;
    uint32_t scLoopLeft = scNum - scLoopCnt * scLoopNum;

    __ubuf__ T * kvAddr = (__ubuf__ T *)kvLocal.GetPhyAddr();
    __ubuf__ T * scoreAddr = (__ubuf__ T *)scoreLocal.GetPhyAddr();
    __ubuf__ T * outputAddr = (__ubuf__ T *)outputLocal.GetPhyAddr()+outIdx;
    if(scLoopNum ==1) {
        MulReduceSumbaseOneVFImpl(kvAddr, scoreAddr, outputAddr, coff, r, scLoopCnt, baseD);
    } else if(scLoopNum == 2) {
        MulReduceSumbaseTwoVFImpl(kvAddr, scoreAddr, outputAddr, coff, r, scLoopCnt, baseD);
    }
}


#endif