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
 * \file mc2_gen_task_ops_utils_arch35.h
 * \brief
 */

#ifndef OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_ARCH35_GEN_TASK_OPS_UTILS_H
#define OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_ARCH35_GEN_TASK_OPS_UTILS_H

#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/kernel_launch_info.h"
#include "graph/arg_desc_info.h"

namespace ops {
const std::string COMM_ALG_FULLMESH_V1 = "fullmesh_v1";
const std::string COMM_ALG_FULLMESH_V2 = "fullmesh_v2";
const std::string COMM_ALG_MTE = "mte";
const std::string COMM_ALG_CCU = "ccu";
class Mc2Arch35GenTaskOpsUtils {
public:
    static const std::string GetCommAlg(const gert::ExeResGenerationContext *context, const size_t commAlgIdx);
    static ge::Status Mc2Arch35GenTaskCallBack(const gert::ExeResGenerationContext *context,
                                               std::vector<std::vector<uint8_t>> &tasks);
    static ge::Status CreateCCUFusionTask(const gert::ExeResGenerationContext *context,
                                          std::vector<std::vector<uint8_t>> &tasks);
};
} // namespace ops

#endif // OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_ARCH35_GEN_TASK_OPS_UTILS_H