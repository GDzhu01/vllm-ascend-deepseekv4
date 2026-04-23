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
 * \file mc2_gen_task_ops_utils_arch35.cpp
 * \brief
 */

#include "mc2_gen_task_ops_utils_arch35.h"
#include "mc2_gen_task_ops_utils.h"
#include "graph/ascend_string.h"
#include "mc2_log.h"

namespace ops {

const std::string MOE_DISTRIBUTE_DISPATCH_OP_TYPE = "MoeDistributeDispatch";
const std::string MOE_DISTRIBUTE_COMBINE_OP_TYPE = "MoeDistributeCombine";
const std::string MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE = "MoeDistributeDispatchV2";
const std::string MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE = "MoeDistributeCombineV2";
const std::string ALL_TO_ALLV_GROUPED_MM_OP_TYPE = "AlltoAllvGroupedMatMul";
const std::string GROUPED_MM_ALL_TO_ALLV_OP_TYPE = "GroupedMatMulAlltoAllv";
const std::string ALL_GATHER_MM_V2_OP_TYPE = "AllGatherMatmulV2";
const std::string MM_REDUCE_SCATTER_V2_OP_TYPE = "MatmulReduceScatterV2";
const std::string MM_ALL_REDUCE_OP_TYPE = "MatmulAllReduce";
const std::string QUANT_ALL_REDUCE_OP_TYPE = "QuantAllReduce";
const std::string QUANT_REDUCE_SCATTER_OP_TYPE = "QuantReduceScatter";
const std::string MM_ALLTO_ALL_OP_TYPE = "MatmulAlltoAll";
const std::string ALLTO_ALL_MATMUL_OP_TYPE = "AlltoAllMatmul";
const std::string ATTR_NAME_GROUP = "group";
const std::string ATTR_NAME_GROUP_EP = "group_ep";
const int32_t MAX_GROUP_CNT = 16;


struct GroupInfo {
    int32_t groupCnt = -1;                         // 算子通信域数量
    std::vector<std::string> groupAttrNames = {}; // 算子通信域属性名
};
// 当前 GetCCuTaskInfo 接口暂不支持多通信域的接口，对于双通信域的算子，暂时不增加 TP 属性名
static const std::map<const std::string, const GroupInfo> GROUP_INFO_MAP_ARCH35{
    {ALL_GATHER_MM_V2_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {MM_REDUCE_SCATTER_V2_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {MM_ALL_REDUCE_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {QUANT_ALL_REDUCE_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {QUANT_REDUCE_SCATTER_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {MM_ALLTO_ALL_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {ALLTO_ALL_MATMUL_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {ALL_TO_ALLV_GROUPED_MM_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {GROUPED_MM_ALL_TO_ALLV_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {MOE_DISTRIBUTE_DISPATCH_OP_TYPE, {2, {ATTR_NAME_GROUP_EP}}},
    {MOE_DISTRIBUTE_COMBINE_OP_TYPE, {2, {ATTR_NAME_GROUP_EP}}},
    {MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE, {2, {ATTR_NAME_GROUP_EP}}},
    {MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE, {2, {ATTR_NAME_GROUP_EP}}},
};

static bool GetGroupInfo(const gert::ExeResGenerationContext *context, GroupInfo &groupInfo)
{
    const char *opType = context->GetNodeType();
    if (opType == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Op type is nullptr.");
        return false;
    }
    const std::string opTypeStr = opType;
    if (GROUP_INFO_MAP_ARCH35.find(opTypeStr) == GROUP_INFO_MAP_ARCH35.end()) {
        OPS_LOG_E(context->GetNodeName(), "Op type [%s] has not registe in group cnt map.", opType);
        return false;
    }
    groupInfo = GROUP_INFO_MAP_ARCH35.at(opTypeStr);
    const int32_t cnt = groupInfo.groupCnt;
    if ((cnt <= 0) || (cnt > MAX_GROUP_CNT)) {
        OPS_LOG_E(context->GetNodeName(), "Group cnt [%d] is invalid, it should in [1, %d].", cnt, MAX_GROUP_CNT);
        return false;
    }

    OPS_LOG_D(context->GetNodeName(), "Op [%s] get group [%d] success.", opType, cnt);
    return true;
}

const std::string Mc2Arch35GenTaskOpsUtils::GetCommAlg(const gert::ExeResGenerationContext *context,
                                                       const size_t commAlgIdx)
{
    auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Attrs pointer is null.");
        return "";
    }
    const char *commAlgPtr = attrs->GetStr(commAlgIdx);
    if (commAlgPtr == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Comm alg pointer is null.");
        return "";
    }
    const std::string commAlg = commAlgPtr;
    if (commAlg.empty()) {
        OPS_LOG_W(context->GetNodeName(), "Comm alg is empty, will use mte alg.");
        return "mte";
    }
    OPS_LOG_D(context->GetNodeName(), "Comm alg is %s.", commAlg.c_str());
    return commAlg;
}

ge::Status Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(const gert::ExeResGenerationContext *context,
                                                              std::vector<std::vector<uint8_t>> &tasks)
{
    if (context == nullptr) {
        OPS_LOG_E("Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack", "Failed to get context.");
        return ge::GRAPH_FAILED;
    }
    
    if (tasks.size() == 0U) {
        OPS_LOG_E(context->GetNodeName(), "Empty task vector when generating task.");
        return ge::GRAPH_FAILED;
    }

    if (CreateCCUFusionTask(context, tasks) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to create CCU fusion task.");
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_I(context->GetNodeName(), "Gen Task for CCU fusion task success.");
    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2Arch35GenTaskOpsUtils::CreateCCUFusionTask(const gert::ExeResGenerationContext *context,
                                                         std::vector<std::vector<uint8_t>> &tasks)
{
    // 填充groupinfo
    GroupInfo groupInfo;
    if (!GetGroupInfo(context, groupInfo)) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get group info.");
        return ge::GRAPH_FAILED;
    }

    // 添加ccugroup
    std::vector<std::string> ccuGroups;
    for (const std::string &groupAttrName : groupInfo.groupAttrNames) {
        ccuGroups.emplace_back(groupAttrName);
        OPS_LOG_D(context->GetNodeName(), "Set group attr [%s] to ccu task info success.", groupAttrName.c_str());
    }

    // 创建 ccu task
    ge::KernelLaunchInfo ccuTask = ge::KernelLaunchInfo::CreateCcuTask(context, ccuGroups);
    OPS_LOG_I(context->GetNodeName(), "Create ccu task successfully.");

    // 获取aicore task
    ge::KernelLaunchInfo prevAicoreTask = ge::KernelLaunchInfo::LoadFromData(context, tasks.back());

    // 设置aicore task的numBlocks
    int64_t numBlocks = -1;
    if (!context->GetIntAttrVal("tvm_blockdim", numBlocks) || numBlocks <= 0) {
        OPS_LOG_E(context->GetNodeName(), "Can't get valid numBlocks, get numBlocks %ld.", numBlocks);
        return ge::GRAPH_FAILED;
    }
    prevAicoreTask.SetBlockDim(numBlocks);
    OPS_LOG_I(context->GetNodeName(), "aicore task set numBlocks successfully, set numBlocks %ld.", numBlocks);

    // 获取aicore task的args format
    const char *prevArgsFormatStr = prevAicoreTask.GetArgsFormat();
    if (prevArgsFormatStr == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get Args Format from aicore task in previous aicore task.");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(context->GetNodeName(), "Before Args Format is : [%s] in previous aicore task", prevArgsFormatStr);
    std::vector<ge::ArgDescInfo> argDescInfos = ge::ArgsFormatSerializer::Deserialize(prevArgsFormatStr);
    // 填充hcom,当前约定Hcom插入于args format的最前面
    for (size_t i = 0; i < groupInfo.groupCnt; ++i) {
        argDescInfos.insert(argDescInfos.begin() + i,
                              ge::ArgDescInfo::CreateHiddenInput(ge::HiddenInputSubType::kHcom));
    }
    auto argsFormatStr = ge::ArgsFormatSerializer::Serialize(argDescInfos);

    // 创建 fusion task
    std::vector<ge::KernelLaunchInfo> fusionTasks = {ccuTask, prevAicoreTask};
    ge::KernelLaunchInfo fusionTask = ge::KernelLaunchInfo::CreateFusionTask(context, fusionTasks);
    // 设置fusion task的args format
    if (fusionTask.SetArgsFormat(argsFormatStr.GetString()) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to set Args Format for aicore task.");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(context->GetNodeName(), "fusion task Args Format: %s", argsFormatStr.GetString());

    // 填充streamid
    int64_t streamId = context->GetStreamId();
    if (streamId < 0) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get context stream id: %ld.", streamId);
        return ge::GRAPH_FAILED;
    }
    fusionTask.SetStreamId(streamId);

    // 序列化fusion task
    tasks.back() = fusionTask.Serialize();

    return ge::GRAPH_SUCCESS;
}

} // namespace ops