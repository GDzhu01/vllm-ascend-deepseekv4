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
 * \file quant_lightning_indexer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#if (__CCE_AICORE__ == 310)
    #include "arch35/quant_lightning_indexer_kernel.h"
    #include "arch35/quant_lightning_indexer_metadata.h"
#else
    #include "arch32/quant_lightning_indexer_kernel.h"
#endif
#include "quant_lightning_indexer_template_tiling_key.h"
using namespace QLIKernel;
using namespace optiling::detail;
 	 
template <class T>
__inline__ __attribute__((always_inline)) __aicore__ void InitMetaData(const __gm__ uint8_t *p_metadata, T *metadata)
{
    constexpr uint64_t all_bytes = sizeof(T);
#if defined(ASCENDC_CPU_DEBUG) || defined(__DAV_C220_CUBE__) ||defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__) || defined(__GET_CODE_CHANNEL__)
    copy_data_align64((uint8_t*)metadata, (__gm__ uint8_t *)p_metadata, all_bytes);
#else
    // __ubuf__ uint8_t *metadata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    // constexpr uint32_t len_burst = (all_bytes + 31) / 32;
    // copy_gm_to_ubuf(((__ubuf__ uint8_t *)metadata_in_ub), p_metadata, 0, 1, len_burst, 0, 0);
    // set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    // wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    // copy_data_align64((uint8_t*)metadata, (__ubuf__ uint8_t *)metadata_in_ub, all_bytes);
    copy_data_align64((uint8_t*)metadata, (__gm__ uint8_t *)p_metadata, all_bytes);
#endif
}

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                         \
    do {                                                                                                     \
        templateClass<QLIType<__VA_ARGS__>> op;                                                              \
        GET_TILING_DATA_WITH_STRUCT(QLITilingData, tiling_data_in, tiling);                                  \
        const QLITilingData *__restrict tiling_data = &tiling_data_in;                                       \
        LiqMetaData *__restrict meta_data = nullptr;                                                         \
        LiqMetaData metadataTmp;                                                                             \
        if (metadata != nullptr) {                                                                           \
            InitMetaData<LiqMetaData>(metadata, &metadataTmp);                                               \
            meta_data = &metadataTmp;                                                                        \
        }                                                                                                    \
        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \
                meta_data, sparseIndices, user, tiling_data, &tPipe);                                        \
        op.Process();                                                                                        \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void quant_lightning_indexer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                                   __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                                   __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                                   __gm__ uint8_t *blocktable, __gm__ uint8_t *metadata,
                                                   __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                                   __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    #if (__CCE_AICORE__ == 310)
        INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
                                 PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T));
    #else
        INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, int8_t, int8_t, int32_t,
                                 PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T));
    #endif
}
