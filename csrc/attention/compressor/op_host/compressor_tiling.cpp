/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file compressor_tiling.cpp
* \file compressor_tiling.cpp
* \brief
*/

#include <numeric>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "compressor_tiling.h"

using namespace ge;
using namespace AscendC;
namespace optiling {


void CompressorTiling::ConvertRequiredParams(gert::TilingContext &context, CompressorContext &compressorContext)
{
    compressorContext.x.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    compressorContext.x.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    compressorContext.wkv.desc = context.GetRequiredInputDesc(WEIGHT_KV_INPUT_INDEX);
    compressorContext.wkv.shape = context.GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);
    compressorContext.wgate.desc = context.GetRequiredInputDesc(WEIGHT_WGATE_INPUT_INDEX);
    compressorContext.wgate.shape = context.GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);
    compressorContext.kvState.desc = context.GetRequiredInputDesc(KV_STATE_INPUT_INDEX);
    compressorContext.kvState.shape = context.GetRequiredInputShape(KV_STATE_INPUT_INDEX);
    compressorContext.scoreState.desc = context.GetRequiredInputDesc(SCORE_STATE_INPUT_INDEX);
    compressorContext.scoreState.shape = context.GetRequiredInputShape(SCORE_STATE_INPUT_INDEX);
    compressorContext.ape.desc = context.GetRequiredInputDesc(APE_INPUT_INDEX);
    compressorContext.ape.shape = context.GetRequiredInputShape(APE_INPUT_INDEX);
    compressorContext.normWeight.desc = context.GetRequiredInputDesc(NORM_WEIGHT_INPUT_INDEX);
    compressorContext.normWeight.shape = context.GetRequiredInputShape(NORM_WEIGHT_INPUT_INDEX);
    compressorContext.ropeSin.desc = context.GetRequiredInputDesc(ROPE_SIN_INPUT_INDEX);
    compressorContext.ropeSin.shape = context.GetRequiredInputShape(ROPE_SIN_INPUT_INDEX);
    compressorContext.ropeCos.desc = context.GetRequiredInputDesc(ROPE_COS_INPUT_INDEX);
    compressorContext.ropeCos.shape = context.GetRequiredInputShape(ROPE_COS_INPUT_INDEX);

    compressorContext.cmpKv.desc = context.GetOutputDesc(CMP_KV_OUTPUT_INDEX);
    compressorContext.cmpKv.shape = context.GetOutputShape(CMP_KV_OUTPUT_INDEX);
}

void CompressorTiling::ConvertOptionalParams(gert::TilingContext &context, CompressorContext &compressorContext)
{
    compressorContext.kvBlockTable.desc = context.GetOptionalInputDesc(KV_BLOCK_TABLE_INPUT_INDEX);
    compressorContext.kvBlockTable.shape = context.GetOptionalInputShape(KV_BLOCK_TABLE_INPUT_INDEX);
    compressorContext.scoreBlockTable.desc = context.GetOptionalInputDesc(SCORE_BLOCK_TABLE_INPUT_INDEX);
    compressorContext.scoreBlockTable.shape = context.GetOptionalInputShape(SCORE_BLOCK_TABLE_INPUT_INDEX);
    compressorContext.cuSeqlens.desc = context.GetOptionalInputDesc(CU_SEQ_LEN_INPUT_INDEX);
    compressorContext.cuSeqlens.shape = context.GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
    compressorContext.seqUsed.desc = context.GetOptionalInputDesc(SEQ_USED_INPUT_INDEX);
    compressorContext.seqUsed.shape = context.GetOptionalInputShape(SEQ_USED_INPUT_INDEX);
    compressorContext.startPos.desc = context.GetOptionalInputDesc(START_POS_INPUT_INDEX);
    compressorContext.startPos.shape = context.GetOptionalInputShape(START_POS_INPUT_INDEX);
}

ge::graphStatus CompressorTiling::ConvertContext(gert::TilingContext &context, CompressorContext &compressorContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("Compressor", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI("Getting Context");

    compressorContext.opName = context.GetNodeName();
    compressorContext.opType = context.GetNodeType();
    compressorContext.platformInfo = context.GetPlatformInfo();
    ConvertRequiredParams(context, compressorContext);
    ConvertOptionalParams(context, compressorContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    compressorContext.ropeHeadDim = attrs->GetAttrPointer<int>(ROPE_HEAD_DIM_ATTR_INDEX);
    compressorContext.coff = attrs->GetAttrPointer<int>(COFF_ATTR_INDEX);
    compressorContext.cmpRatio = attrs->GetAttrPointer<int>(CMP_RATIO_ATTR_INDEX);
    compressorContext.normEps = attrs->GetAttrPointer<float>(NORM_EPS_ATTR_INDEX);
    compressorContext.rotaryMode = attrs->GetAttrPointer<int>(ROTARY_MODE_ATTR_INDEX);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    compressorContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetBaseInfo()
{
    if (context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_DIM_NUM_3) {
        baseParams_->batchSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
        baseParams_->seqSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_2);
        baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seqSize;
    } else {
        baseParams_->batchSize = context_->kvBlockTable.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
        baseParams_->tokenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    }
    
    baseParams_->headDim = context_->normWeight.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
    baseParams_->cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    baseParams_->csSize = baseParams_->seqSize - (baseParams_->seqSize %  baseParams_->cmpRatio);
    baseParams_->cgSize = baseParams_->seqSize / baseParams_->cmpRatio;
    baseParams_->ropeHeadDim = static_cast<uint32_t>(*context_->ropeHeadDim);
    baseParams_->normEps = static_cast<float>(*context_->normEps);
    baseParams_->reciprocalD = 1.0 / baseParams_->headDim;
    coff = static_cast<uint8_t>(*context_->coff);
    baseParams_->nSize = 2;

    OP_LOGI(context_->opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u coff:%u", baseParams_->batchSize, baseParams_->tokenSize, baseParams_->cmpRatio, coff);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetPageAttentionInfo()
{
    pageAttentionParams_->blockNum = context_->kvState.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
    pageAttentionParams_->blockSize = context_->kvState.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    pageAttentionParams_->maxBlockNumPerBatch = context_->kvBlockTable.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetWorkSpaceInfo()
{
    workspaceParams_->preMm1ResSize = 0;
    if (coff == 2) {
        workspaceParams_->preMm1ResSize = innerSplitParams_->mBaseSize * innerSplitParams_->dBaseSize * 2;      // 2 wkv和score合一起
    }
    workspaceParams_->curMm1ResSize = innerSplitParams_->mBaseSize * innerSplitParams_->dBaseSize * 2;          // 2 wkv和score合一起
    workspaceParams_->vec1ResSize = innerSplitParams_->mBaseSize / baseParams_->cmpRatio * innerSplitParams_->dBaseSize * baseParams_->nSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetScenarioInfo()
{
    // TODO set mode

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetInnerSplitInfo()
{
    innerSplitParams_->mBaseSize = 256;
    innerSplitParams_->dBaseSize = 128 / coff;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CalcWorkSpace()
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;      // 4: fp32
    constexpr uint32_t V1_RES_ELEM_SIZE = 2;       // 2: fp16/bf16

    workspaceSize_ = libapiSize_;
    workspaceSize_ += aicNum_ * workspaceParams_->preMm1ResSize * MM1_RES_ELEM_SIZE;
    workspaceSize_ += aicNum_ * workspaceParams_->curMm1ResSize * MM1_RES_ELEM_SIZE;
    workspaceSize_ += aicNum_ * workspaceParams_->vec1ResSize * V1_RES_ELEM_SIZE;
    
    // TODO 为后面改动预留
    workspaceSize_ += 1024 * 1024 * 1024;
    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }
    
    OP_LOGI(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::RunBigKernelTiling(CompressorContext &context, CompressorTilingData* tilingData)
{
    this->context_ = &context;
    this->baseParams_ = &tilingData->baseParams;
    this->pageAttentionParams_ = &tilingData->pageAttentionParams;
    this->outerSplitParams_ = &tilingData->outerSplitParams;
    this->innerSplitParams_ = &tilingData->innerSplitParams;
    this->workspaceParams_ = &tilingData->workspaceParams;
    

    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs {
        std::bind(&CompressorTiling::GetNpuInfo, this),
        std::bind(&CompressorTiling::SetBaseInfo, this),
        std::bind(&CompressorTiling::SetPageAttentionInfo, this),
        std::bind(&CompressorTiling::SetInnerSplitInfo, this),
        std::bind(&CompressorTiling::SetWorkSpaceInfo, this),
        std::bind(&CompressorTiling::SetScenarioInfo, this),
    };
    for (const auto &func: requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    std::vector<StatusFunction> optionalTilingFuncs {
        std::bind(&CompressorTiling::CalcWorkSpace, this),
        std::bind(&CompressorTiling::GenTilingKey, this)
    };
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // TODO 使用所有核
    baseParams_->usedCoreNum = aicNum_;

    context_->blockDim = aicNum_;

    OP_LOGI("Run big kernel");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::GenTilingKey() const
{

    // 0:BF16, 1:FP16
    uint8_t dtype = 0;
    // 0: BSH 1:TH
    uint8_t layout = 0;
    uint8_t rotaryMode = static_cast<uint8_t>(*context_->rotaryMode);
    
    auto xDtype = context_->x.desc->GetDataType();
    if (xDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_DIM_NUM_3) {
        layout = 0;
    }else {
        layout = 1;
    }
    
    context_->tilingKey = GET_TPL_TILING_KEY(
        layout,
        dtype,
        coff,
        rotaryMode
    );

    OP_LOGI(context_->opName, "Compressor dtype:%hhu layout:%hhu  coff:%hhu rotary_mode:%hhu", dtype, layout, coff, rotaryMode);
    OP_LOGI(context_->opName, "Compressor tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

CMP_EXTERN_C ge::graphStatus TilingCompressor(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Compressor", "Context is nullptr."),
               return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    CompressorContext compressorContext{};
    if (CompressorTiling::ConvertContext(*context, compressorContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to Compressor context");
        return ge::GRAPH_FAILED;
    }

    CompressorTiling compressorTiling;
    CompressorTilingData* tilingData = context->GetTilingData<CompressorTilingData>();
    OP_CHECK_IF(tilingData == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "TilingData is nullptr."),
            return ge::GRAPH_FAILED);
    if (compressorTiling.RunBigKernelTiling(compressorContext, tilingData) == ge::SUCCESS) {
        // TODO genTilingKey
        context->SetTilingKey(compressorContext.tilingKey);
        context->SetBlockDim(compressorContext.blockDim);
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForCompressor(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Compressor)
    .Tiling(TilingCompressor)
    .TilingParse<CompressorCompileInfo>(TilingPrepareForCompressor);
} // namespace optiling