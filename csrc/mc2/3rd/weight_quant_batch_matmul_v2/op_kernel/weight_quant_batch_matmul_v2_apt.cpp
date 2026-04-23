/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_batch_matmul_v2_apt.cpp
 * \brief
 */

#define ENABLE_L2_CACHE
#include "weight_quant_batch_matmul_v2_constant.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
#include "arch35/n_first/weight_quant_batch_matmul_v2_basic_block_controller.h"
#endif

#if defined(ORIG_DTYPE_X) && ((defined(DT_BF16) && ORIG_DTYPE_X == DT_BF16)) || \
    ((defined(DT_FLOAT16) && ORIG_DTYPE_X == DT_FLOAT16))
#define A16
#endif
#if defined(ORIG_DTYPE_WEIGHT) && defined(DT_INT32) && ORIG_DTYPE_WEIGHT == DT_INT32
#undef DTYPE_WEIGHT
#define DTYPE_WEIGHT AscendC::int4b_t
#undef ORIG_DTYPE_WEIGHT
#define ORIG_DTYPE_WEIGHT DT_INT4
#endif
#if defined(ORIG_DTYPE_WEIGHT) && (defined(DT_INT4) && ORIG_DTYPE_WEIGHT == DT_INT4)
#define S4
#endif

#if defined(A16) && defined(S4)
#include "arch35/catlass_convertor.h"
using Mc2WeightQuantBatchMatmulV2::Arch35::Catlass::InvokeActKernel;
#endif

#if !defined(__DAV_310R6__) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
#if defined(ORIG_DTYPE_WEIGHT)
#if (                                                                                                               \
    ORIG_DTYPE_WEIGHT == DT_INT8 || ORIG_DTYPE_WEIGHT == DT_FLOAT8_E5M2 || ORIG_DTYPE_WEIGHT == DT_FLOAT8_E4M3FN || \
    ORIG_DTYPE_WEIGHT == DT_HIFLOAT8)
#define WEIGHT_B8_BRANCH
#endif
#if (ORIG_DTYPE_WEIGHT == DT_FLOAT8_E5M2 || ORIG_DTYPE_WEIGHT == DT_FLOAT8_E4M3FN || ORIG_DTYPE_WEIGHT == DT_HIFLOAT8)
#define WEIGHT_F8_INPUT
#endif
#endif
#if defined(ORIG_DTYPE_ANTIQUANT_SCALE) && (ORIG_DTYPE_ANTIQUANT_SCALE == DT_FLOAT8_E8M0)
#define MICROSCALING
#endif
#if (!defined(MICROSCALING) && defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y != DT_INT8) && !defined(WEIGHT_F8_INPUT))
#include "arch35/weight_quant_batch_matmul_v2_reg_base.h"
#endif
#if defined(ORIG_DTYPE_WEIGHT) && defined(DT_FLOAT) && ORIG_DTYPE_WEIGHT == DT_FLOAT
#undef DTYPE_WEIGHT
#define DTYPE_WEIGHT fp4x2_e2m1_t
#endif
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    #include "arch35/weight_quant_batch_matmul_v2_adaptive_sliding_window.h"
#endif

using namespace Mc2WeightQuantBatchMatmulV2;
using namespace Mc2WeightQuantBatchMatmulV2::Arch35;
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_0 = {2, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_1 = {4, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_2 = {2, 1024};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_4 = {2, 256};
#endif

#define INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(templateClass, ...)                                       \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(Mc2WeightQuantBatchMatmulV2ASTilingData, tilingDataIn, tiling);                 \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;        \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(templateClass, ...)                                             \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(Mc2WeightQuantBatchMatmulV2RegBaseTilingData, tilingDataIn, tiling);            \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                               \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(templateClass, ...)                                                      \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2ASWTilingData, tilingDataIn, tiling);                    \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)

extern "C" __global__ __aicore__ void Mc2weight_quant_batch_matmul_v2(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    AscendC::TPipe tPipe;
#if !defined(__DAV_310R6__) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)) 
#undef DTYPE_BIAS
#define DTYPE_BIAS DTYPE_X
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (defined(MICROSCALING))
#if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT == FORMAT_FRACTAL_NZ))
    if (TILING_KEY_IS(2000020003000004101UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         false, Mc2QuantType::MX, false, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::NZ};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000020001000004101UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         false, Mc2QuantType::MX, false, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::NZ};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
    }
#else
    if (TILING_KEY_IS(2000020002000014100UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::MX, false, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_2);
    } else if (TILING_KEY_IS(2000030003000004100UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         false, Mc2QuantType::MX, false, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    }
#endif
#endif
// 当前场景不支持c8输出和fp8输入
#if (!defined(MICROSCALING) && defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y != DT_INT8) & !defined(WEIGHT_F8_INPUT))
    if (TILING_KEY_IS(100300UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(100310UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(100311UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(100301UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(10100300UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, Mc2QuantType::PER_GROUP, true);
    } else if (TILING_KEY_IS(10100310UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, Mc2QuantType::PER_GROUP, true);
    } else if (TILING_KEY_IS(100200UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(100210UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(100211UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(100201UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(100100UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(100110UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(100111UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(100101UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(101300UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(101310UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(101311UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(101301UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(10101300UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, Mc2QuantType::PER_GROUP, true);
    } else if (TILING_KEY_IS(10101310UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, Mc2QuantType::PER_GROUP, true);
    } else if (TILING_KEY_IS(101200UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(101210UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(101211UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(101201UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(101100UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(101110UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(101111UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(101101UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(2000020003000002101UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::NZ};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000020003000002121UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR, CubeFormat::NZ};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
#if defined(A16) && defined(S4)
    } else if (TILING_KEY_IS(2000020002000003101UL)) {
        InvokeActKernel<2000020002000003101UL>(KERNEL_PARAMS);
    } else if (TILING_KEY_IS(2000020002000003121UL)) {
        InvokeActKernel<2000020002000003121UL>(KERNEL_PARAMS);
    } else if (TILING_KEY_IS(2000020001000003101UL)) {
        InvokeActKernel<2000020001000003101UL>(KERNEL_PARAMS);
    } else if (TILING_KEY_IS(2000020001000003121UL)) {
        InvokeActKernel<2000020001000003121UL>(KERNEL_PARAMS);
    } else if (TILING_KEY_IS(2000020006000003101UL)) {
        InvokeActKernel<2000020006000003101UL>(KERNEL_PARAMS);
    } else if (TILING_KEY_IS(2000020006000003121UL)) {
        InvokeActKernel<2000020006000003121UL>(KERNEL_PARAMS);
#endif
    }
#endif
#if (!defined(MICROSCALING) && defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
    if (TILING_KEY_IS(2000030004000012100UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000032100UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000012120UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000032120UL)) {
        static constexpr WqmmConfig wqmmCfg = {true,          true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000011100UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000031100UL)) {
        static constexpr WqmmConfig wqmmCfg = {true,          true, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000011120UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000031120UL)) {
        static constexpr WqmmConfig wqmmCfg = {true,          true, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030003000002100UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000022100UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000002120UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000022120UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001100UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000021100UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001120UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         false, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000021120UL)) {
        static constexpr WqmmConfig wqmmCfg = {true,          false, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    }
// 下列tiling key只在c8场景和非fp8输入下出现
#if (defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y == DT_INT8) && !defined(WEIGHT_F8_INPUT))
    else if (TILING_KEY_IS(2000030004000012200UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000032200UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000012220UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000032220UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000011200UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000031200UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, true, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000011220UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_CHANNEL,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000031220UL)) {
        static constexpr WqmmConfig wqmmCfg = {true,          true, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_CHANNEL,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030003000002200UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000022200UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000002220UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000022220UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001200UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000021200UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001220UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000021220UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    }
#else // 下列tiling key不在c8场景下出现
    else if (TILING_KEY_IS(2000020000000012100UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012100UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
    } else if (TILING_KEY_IS(2000020002000012100UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_2);
    } else if (TILING_KEY_IS(2000020003000012100UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000020000000012120UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012120UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
    } else if (TILING_KEY_IS(2000020002000012120UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_2);
    } else if (TILING_KEY_IS(2000020003000012120UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    }
#endif
#endif
#if defined(ORIG_DTYPE_X) && defined(DT_BF16) && ORIG_DTYPE_X == DT_BF16
#undef DTYPE_BIAS
#define DTYPE_BIAS float
#if (defined(MICROSCALING))
#if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT == FORMAT_FRACTAL_NZ))
    if (TILING_KEY_IS(2000020003000004141UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         false, Mc2QuantType::MX, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::NZ};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000020001000004141UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         false, Mc2QuantType::MX, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::NZ};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
    }
#else
    if (TILING_KEY_IS(2000030003000004140UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         false, Mc2QuantType::MX, false, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000020002000014140UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::MX, false, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_2);
    }
#endif
#endif
// 当前场景不支持c8和fp8输入
#if (!defined(MICROSCALING) && defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y != DT_INT8) && !defined(WEIGHT_F8_INPUT))
    if (TILING_KEY_IS(1100300UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(1100310UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(1100311UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(1100301UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(11100300UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, Mc2QuantType::PER_GROUP, true);
    } else if (TILING_KEY_IS(11100310UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, Mc2QuantType::PER_GROUP, true);
    } else if (TILING_KEY_IS(1100200UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(1100210UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(1100211UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(1100201UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(1100100UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, false, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(1100110UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, false, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(1100111UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, false, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(1100101UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, false, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(1101300UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(1101310UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(1101311UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(1101301UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, Mc2QuantType::PER_GROUP);
    } else if (TILING_KEY_IS(11101300UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, Mc2QuantType::PER_GROUP, true);
    } else if (TILING_KEY_IS(11101310UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, Mc2QuantType::PER_GROUP, true);
    } else if (TILING_KEY_IS(1101200UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(1101210UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(1101211UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(1101201UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, Mc2QuantType::PER_CHANNEL);
    } else if (TILING_KEY_IS(1101100UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, false, true, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(1101110UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, false, true, true, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(1101111UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, true, true, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(1101101UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            Mc2WeightQuantBatchMatmulV2RegBaseKernel, true, false, true, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(2000020003000002141UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::NZ};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000020003000002161UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR, CubeFormat::NZ};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
#if defined(A16) && defined(S4)
    } else if (TILING_KEY_IS(2000020002000003141UL)) {
        InvokeActKernel<2000020002000003141UL>(KERNEL_PARAMS);
    } else if (TILING_KEY_IS(2000020002000003161UL)) {
        InvokeActKernel<2000020002000003161UL>(KERNEL_PARAMS);
    } else if (TILING_KEY_IS(2000020001000003141UL)) {
        InvokeActKernel<2000020001000003141UL>(KERNEL_PARAMS);
    } else if (TILING_KEY_IS(2000020001000003161UL)) {
        InvokeActKernel<2000020001000003161UL>(KERNEL_PARAMS);
    } else if (TILING_KEY_IS(2000020006000003141UL)) {
        InvokeActKernel<2000020006000003141UL>(KERNEL_PARAMS);
    } else if (TILING_KEY_IS(2000020006000003161UL)) {
        InvokeActKernel<2000020006000003161UL>(KERNEL_PARAMS);
#endif
    }
#endif
#if (!defined(MICROSCALING) && defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
    if (TILING_KEY_IS(2000030004000012140UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000032140UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000011140UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000031140UL)) {
        static constexpr WqmmConfig wqmmCfg = {true,          true, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000011160UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000031160UL)) {
        static constexpr WqmmConfig wqmmCfg = {true,          true, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030003000002140UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000022140UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030004000012160UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000032160UL)) {
        static constexpr WqmmConfig wqmmCfg = {true,          true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030003000002160UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000022160UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001140UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000021140UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001160UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         false, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000021160UL)) {
        static constexpr WqmmConfig wqmmCfg = {true,          false, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    }
// 下列tiling key只在c8场景和非fp8输入下出现
#if (defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y == DT_INT8) && !defined(WEIGHT_F8_INPUT))
    else if (TILING_KEY_IS(2000030004000012240UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000032240UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000012260UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000032260UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000011240UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000031240UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, true, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000011260UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_CHANNEL,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030004000031260UL)) {
        static constexpr WqmmConfig wqmmCfg = {true,          true, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_CHANNEL,
                                               CubeFormat::ND};
#if defined(WEIGHT_B8_BRANCH)
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
#else
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
#endif
    } else if (TILING_KEY_IS(2000030003000002240UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000022240UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000002260UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000022260UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001240UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000021240UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_TENSOR, false, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001260UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000021260UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            true, false, Mc2QuantType::PER_TENSOR, true, Mc2QuantType::PER_CHANNEL, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    }
#else // 下列tiling key不在c8场景下出现
    else if (TILING_KEY_IS(2000020000000012140UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012140UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
    } else if (TILING_KEY_IS(2000020002000012140UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_2);
    } else if (TILING_KEY_IS(2000020003000012140UL)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000020000000012160UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012160UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_1);
    } else if (TILING_KEY_IS(2000020002000012160UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_2);
    } else if (TILING_KEY_IS(2000020003000012160UL)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
    }
#endif
#endif
#endif
#elif defined(__DAV_310R6__)
#undef DTYPE_BIAS
#define DTYPE_BIAS DTYPE_X
    // 该分支处理bias与A类型一致的情况
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
    if (TILING_KEY_IS(5000030000000012100)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(5000030000000012120)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(5000030005000002100)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_4);
    } else if (TILING_KEY_IS(5000030005000002120)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_4);
    }
#endif
#if defined(ORIG_DTYPE_X) && defined(DT_BF16) && ORIG_DTYPE_X == DT_BF16
#undef DTYPE_BIAS
#define DTYPE_BIAS float
    // 该分支处理A数据类型为BF16且bias为fp32的情况
    if (TILING_KEY_IS(5000030000000012140)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(5000030000000012160)) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(5000030005000002140)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, false, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_4);
    } else if (TILING_KEY_IS(5000030005000002160)) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, Mc2QuantType::PER_CHANNEL, true, Mc2QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_4);
    }
#endif
#else
#undef DTYPE_BIAS
#define DTYPE_BIAS DTYPE_X
#if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
    if (TILING_KEY_IS(3000240000000001100UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(Mc2WeightQuantBatchMatmulV2ASWKernel, false, false, Mc2QuantType::PER_TENSOR,
                                            false, Mc2QuantType::PER_TENSOR);
    } else if (TILING_KEY_IS(3000240000000002100UL)) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(Mc2WeightQuantBatchMatmulV2ASWKernel, false, false, Mc2QuantType::PER_CHANNEL,
                                            false, Mc2QuantType::PER_TENSOR);
    }
#endif
#endif
}
