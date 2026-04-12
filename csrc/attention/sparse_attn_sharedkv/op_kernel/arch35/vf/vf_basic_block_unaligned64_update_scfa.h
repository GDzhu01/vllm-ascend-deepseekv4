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
 * \file vf_basic_block_aligned64_update_scfa.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_ALIGNED64_UPDATE_SCFA_H
#define VF_BASIC_BLOCK_ALIGNED64_UPDATE_SCFA_H

#include "vf_basic_block_utils.h"

using namespace regbaseutil;

namespace SCFaVectorApi {
// update, originN <= 64
template <typename T, typename T2, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128>
__simd_vf__ void ProcessVec1UpdateImpl64VF(
    __ubuf__ T2 * expUb,  __ubuf__ T * srcUb, __ubuf__ T * inMaxUb,
    __ubuf__ T * tmpExpSumUb, __ubuf__ T * tmpMaxUb, const uint32_t blockStride, const uint32_t repeatStride,
    const uint16_t m, const T scale, const T minValue, uint32_t pltOriginalN, uint32_t pltSrcN)
{
    RegTensor<float> vreg_input_x;
    RegTensor<float> vreg_max_tmp;
    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_max_new;
    RegTensor<float> vreg_max_brc;
    RegTensor<float> vreg_cur_max;
    RegTensor<float> vreg_exp;
    RegTensor<float> vreg_exp_sum;

    // bfloat16_t
    RegTensor<bfloat16_t> vreg_exp_bf16;
    RegTensor<bfloat16_t> vreg_dst_even_bf16;
    RegTensor<bfloat16_t> vreg_dst_odd_bf16;

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_ori_src_n = UpdateMask<T>(pltOriginalN);
    MaskReg preg_src_n = UpdateMask<T>(pltSrcN);
    MaskReg preg_src_n_b16 = UpdateMask<uint16_t>(pltSrcN);

    // x_max = max(src, axis=-1, keepdims=True)
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        Muls(vreg_input_x, vreg_input_x, scale, preg_ori_src_n);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_src_n);
        Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_cur_max, vreg_input_x, preg_ori_src_n);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpMaxUb), vreg_cur_max, ureg_max, 1);
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
    LoadAlign(vreg_in_max, inMaxUb);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_cur_max, tmpMaxUb);
    Max(vreg_max_new, vreg_cur_max, vreg_in_max, preg_all); // 计算新、旧的最大值
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)tmpMaxUb, vreg_max_new, preg_all);

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg_max_brc, tmpMaxUb + i);
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        ExpSub(vreg_exp, vreg_input_x, vreg_max_brc, preg_ori_src_n);

        // x_sum = sum(x_exp, axis=-1, keepdims=True)
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_exp_sum, vreg_exp, preg_ori_src_n);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum, ureg_exp_sum, 1);

       if constexpr (IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_bf16, vreg_exp, preg_all_b16);
            DeInterleave(vreg_dst_even_bf16, vreg_dst_odd_bf16,
                    vreg_exp_bf16, vreg_exp_bf16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_dst_even_bf16, blockStride, repeatStride, preg_src_n_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);
}


// update, originN <= 64
template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 128>
__aicore__ inline void ProcessVec1UpdateImpl64(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN, const T scale, const T minValue)
{
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    uint32_t pltOriginalN = originN;
    uint32_t pltSrcN = floatRepSize;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;

    ProcessVec1UpdateImpl64VF <T, T2, s1BaseSize, s2BaseSize>(
        expUb, srcUb, inMaxUb, tmpExpSumUb, tmpMaxUb, blockStride, repeatStride, m, scale, minValue, pltOriginalN, pltSrcN);
}
} // namespace

#endif // VF_BASIC_BLOCK_ALIGNED64_UPDATE_SCFA_H
