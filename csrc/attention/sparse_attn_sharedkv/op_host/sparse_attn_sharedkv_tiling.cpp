/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attn_sharedkv_tiling.cpp
 * \brief
 */

#include "sparse_attn_sharedkv_tiling.h"
#include "../op_kernel/sparse_attn_sharedkv_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

// static const std::string QUERY_NAME = "query";
static const std::string ORI_BLOCK_TABLE_NAME = "ori_block_table";
static const std::string CMP_BLOCK_TABLE_NAME = "cmp_block_table";
static const std::string SINKS_NAME = "sinks";

std::string SASLayoutToSerialString(SASLayout layout)
{
    switch (layout) {
        case SASLayout::BSND: return "BSND";
        case SASLayout::TND: return "TND";
        case SASLayout::PA_ND: return "PA_ND";
        default: return "UNKNOWN";
    }
}

struct SASCompileInfo {
    int64_t core_num;
};

static const std::map<SASLayout, std::vector<SASAxis>> SAS_LAYOUT_AXIS_MAP = {
    {SASLayout::BSND, {SASAxis::B, SASAxis::S, SASAxis::N, SASAxis::D}},
    {SASLayout::TND, {SASAxis::T, SASAxis::N, SASAxis::D}},
    {SASLayout::PA_ND, {SASAxis::Bn, SASAxis::Bs, SASAxis::N, SASAxis::D}},
};

static const std::map<SASLayout, size_t> SAS_LAYOUT_DIM_MAP = {
    {SASLayout::BSND, DIM_NUM_FOUR},
    {SASLayout::TND, DIM_NUM_THREE},
    {SASLayout::PA_ND, DIM_NUM_FOUR},
};

// --------------------------SASInfoParser类成员函数定义-------------------------------------
ge::graphStatus SASInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.q.shape == nullptr, OP_LOGE(opName_, "Shape of tensor q is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::CheckRequiredAttrExistence() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("SparseAttnSharedkv", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0, OP_LOGE(opName_, "num of core obtained is 0."), return ge::GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93)) {
        OP_LOGE(opName_, "SOC Version[%d] is not support.", (int32_t)socVersion_);
        return GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void SASInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.oriKv.tensor = context_->GetOptionalInputTensor(ORI_KV_INDEX);
    opParamInfo_.oriKv.desc = context_->GetOptionalInputDesc(ORI_KV_INDEX);
    opParamInfo_.cmpKv.tensor = context_->GetOptionalInputTensor(CMP_KV_INDEX);
    opParamInfo_.cmpKv.desc = context_->GetOptionalInputDesc(CMP_KV_INDEX);
    opParamInfo_.cmpSparseIndices.tensor = context_->GetOptionalInputTensor(CMP_SPARSE_INDICES_INDEX);
    opParamInfo_.cmpSparseIndices.desc = context_->GetOptionalInputDesc(CMP_SPARSE_INDICES_INDEX);
    opParamInfo_.oriBlockTable.tensor = context_->GetOptionalInputTensor(ORI_BLOCK_TABLE_INDEX);
    opParamInfo_.oriBlockTable.desc = context_->GetOptionalInputDesc(ORI_BLOCK_TABLE_INDEX);
    opParamInfo_.cmpBlockTable.tensor = context_->GetOptionalInputTensor(CMP_BLOCK_TABLE_INDEX);
    opParamInfo_.cmpBlockTable.desc = context_->GetOptionalInputDesc(CMP_BLOCK_TABLE_INDEX);
    opParamInfo_.sinks.tensor = context_->GetOptionalInputTensor(SINKS_INDEX);
    opParamInfo_.sinks.desc = context_->GetOptionalInputDesc(SINKS_INDEX);
    opParamInfo_.cuSeqLensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqLensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.sequsedKv.tensor = context_->GetOptionalInputTensor(SEQUSED_KV_INDEX);
    opParamInfo_.sequsedKv.desc = context_->GetOptionalInputDesc(SEQUSED_KV_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
}

void SASInfoParser::GetInputParaInfo()
{
    opParamInfo_.q.desc = context_->GetInputDesc(Q_INDEX);
    opParamInfo_.q.shape = context_->GetInputShape(Q_INDEX);
    GetOptionalInputParaInfo();
}

void SASInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
}

ge::graphStatus SASInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOTFMAX_SCALE_INDEX);
    opParamInfo_.cmpRatio = attrs->GetAttrPointer<uint32_t>(ATTR_CMP_RATIO_INDEX);
    opParamInfo_.oriMaskMode = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_MASK_MODE_INDEX);
    opParamInfo_.cmpMaskMode = attrs->GetAttrPointer<uint32_t>(ATTR_CMP_MASK_MODE_INDEX);
    opParamInfo_.oriWinLeft = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_WIN_LEFT_INDEX);
    opParamInfo_.oriWinRight = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_WIN_RIGHT_INDEX);
    opParamInfo_.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);
    opParamInfo_.layoutKv = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo end");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetInOutDataType()
{
    qType_ = opParamInfo_.q.desc->GetDataType();
    outputType_ = opParamInfo_.attnOut.desc->GetDataType();
    if (opParamInfo_.oriKv.desc != nullptr) {
        oriKvType_ = opParamInfo_.oriKv.desc->GetDataType();
    }
    if (opParamInfo_.cmpKv.desc != nullptr) {
        cmpKvType_ = opParamInfo_.cmpKv.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetSASTemplateMode(SASTilingInfo &sasInfo)
{
    std::string layout(opParamInfo_.layoutKv);
    bool usePaCmpPaButNotPassed = (layout == "PA_ND") && opParamInfo_.cmpBlockTable.desc == nullptr;
    if (opParamInfo_.oriKv.desc != nullptr) {
        if (opParamInfo_.cmpKv.desc != nullptr && opParamInfo_.cmpSparseIndices.tensor != nullptr &&
            !usePaCmpPaButNotPassed) {
            sasInfo.perfMode = SASTemplateMode::SCFA_TEMPLATE_MODE;
        } else if (opParamInfo_.cmpKv.desc != nullptr && !usePaCmpPaButNotPassed) {
            sasInfo.perfMode = SASTemplateMode::CFA_TEMPLATE_MODE;
        } else {
            sasInfo.perfMode = SASTemplateMode::SWA_TEMPLATE_MODE;
        }
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(opName_, "oriKv is nullptr");
        return ge::GRAPH_FAILED;
    }
}

ge::graphStatus SASInfoParser::GetQueryAndOutLayout()
{
    // 获取q和attnOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const map<string, pair<SASLayout, SASLayout>> layoutMap = {
        {"BSND",        {SASLayout::BSND,    SASLayout::BSND}},
        {"TND",         {SASLayout::TND,     SASLayout::TND }},
    };

    std::string layout(opParamInfo_.layoutQ);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OP_LOGE(opName_, "layout of Q is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetKvLayout()
{
    const map<string, SASLayout> layoutKVMap = {
        {"PA_ND",     SASLayout::PA_ND},
    };

    std::string layout(opParamInfo_.layoutKv);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OP_LOGE(opName_, "layoutKV is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// =============Parser function====================

bool SASInfoParser::HasAxis(const SASAxis &axis, const SASLayout &layout, const gert::Shape &shape) const
{
    const auto& layoutIt = SAS_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == SAS_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<SASAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }
    const auto& dimIt = SAS_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == SAS_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t SASInfoParser::GetAxisIdx(const SASAxis &axis, const SASLayout &layout) const
{
    const std::vector<SASAxis>& axes = SAS_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t SASInfoParser::GetAxisNum(const gert::Shape &shape, const SASAxis &axis,const SASLayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

void SASInfoParser::SetSASShape()
{
    qShape_ = opParamInfo_.q.shape->GetStorageShape();
    if (opParamInfo_.oriKv.tensor != nullptr) {
        oriKvShape_ = opParamInfo_.oriKv.tensor->GetStorageShape();
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        cmpKvShape_ = opParamInfo_.cmpKv.tensor->GetStorageShape();
    }
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        cmpSparseIndicesShape_ = opParamInfo_.cmpSparseIndices.tensor->GetStorageShape();
    }
}

ge::graphStatus SASInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(qShape_, SASAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetN2Size()
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        n2Size_ = GetAxisNum(oriKvShape_, SASAxis::N, kvLayout_);
    } else if (opParamInfo_.cmpKv.tensor != nullptr) {
        n2Size_ = GetAxisNum(cmpKvShape_, SASAxis::N, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    SASLayout &layout, const std::string &name) const
{
    if ((tensor == nullptr)) {
        OP_LOGE(opName_, "when layout of q is %s, %s must be provided.",
            SASLayoutToSerialString(layout).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.cuSeqLensQ.tensor, qLayout_, "cuSeqLensQ");
}

ge::graphStatus SASInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == SASLayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(qShape_, SASAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus SASInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == SASLayout::TND) ? GetAxisNum(qShape_, SASAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时, 以query的S维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == SASLayout::TND) {
        s1Size_ = GetAxisNum(qShape_, SASAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(qShape_, SASAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.oriBlockTable.tensor == nullptr) {
        OP_LOGE(opName_, "the layout_kv is %s, blockTable must be provided.", SASLayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    }
    uint32_t oriDimNum = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDimNum();
    if (oriDimNum != DIM_NUM_TWO) {
        OP_LOGE(opName_, "the dim num of ori_block_table is %u, it should be %u.", oriDimNum, DIM_NUM_TWO);
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
        OP_LOGE(opName_, "%s's second dimension(%ld) should be greater than 0",
            ORI_BLOCK_TABLE_NAME.c_str(), opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1));
        return ge::GRAPH_FAILED;
    }
    oriMaxBlockNumPerBatch_ = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1);

    if (opParamInfo_.cmpBlockTable.tensor != nullptr) {
        uint32_t cmpDimNum = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDimNum();
        if (cmpDimNum != DIM_NUM_TWO) {
            OP_LOGE(opName_, "the dim num of cmp_block_table is %u, it should be %u.", cmpDimNum, DIM_NUM_TWO);
            return ge::GRAPH_FAILED;
        }
        if (opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
            OP_LOGE(opName_, "%s's second dimension(%ld) should be greater than 0",
                CMP_BLOCK_TABLE_NAME.c_str(), opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1));
            return ge::GRAPH_FAILED;
        }
        cmpMaxBlockNumPerBatch_ = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetBlockSize()
{
    oriBlockSize_ = GetAxisNum(oriKvShape_, SASAxis::Bs, kvLayout_);
    cmpBlockSize_ = GetAxisNum(cmpKvShape_, SASAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = oriMaxBlockNumPerBatch_ * oriBlockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetS2Size()
{
    // 获取S2基准值:PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    return GetS2SizeForPageAttention();
}

ge::graphStatus SASInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    qkHeadDim_ = GetAxisNum(oriKvShape_, SASAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetValueHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    if (opParamInfo_.oriKv.tensor != nullptr) {
        vHeadDim_ = GetAxisNum(oriKvShape_, SASAxis::D, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetSparseBlockCount()
{
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        sparseBlockCount_ = GetAxisNum(cmpSparseIndicesShape_, SASAxis::K, kvLayout_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetSinks()
{
    if (opParamInfo_.sinks.tensor == nullptr) {
        OP_LOGE(opName_, "%s must be provided!", SINKS_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.sinks.tensor->GetStorageShape().GetDimNum() != DIM_NUM_ONE) {
        OP_LOGE(opName_, "the dim num of %s is %u, it should be %u.", SINKS_NAME.c_str(),
            opParamInfo_.sinks.tensor->GetStorageShape().GetDimNum(), DIM_NUM_ONE);
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.sinks.tensor->GetStorageShape().GetDim(0) != n1Size_) {
        OP_LOGE(opName_, "%s's dimension(%ld) should be equal to query head num(%u).", SINKS_NAME.c_str(),
            opParamInfo_.sinks.tensor->GetStorageShape().GetDim(0), n1Size_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetActualseqInfo()
{
    maxActualseq_ = static_cast<uint32_t>(s2Size_);
    if (opParamInfo_.sequsedKv.tensor != nullptr) {
        actualLenDimsKV_ = opParamInfo_.sequsedKv.tensor->GetShapeSize();
    }
    if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() - 1; // cuSeqLensQ shape is B+1
    }
    return ge::GRAPH_SUCCESS;
}

void SASInfoParser::GenerateInfo(SASTilingInfo &sasInfo)
{
    sasInfo.opName = opName_;
    sasInfo.platformInfo = platformInfo_;
    sasInfo.opParamInfo = opParamInfo_;
    sasInfo.socVersion = socVersion_;

    sasInfo.bSize = bSize_;
    sasInfo.n1Size = n1Size_;
    sasInfo.n2Size = n2Size_;
    sasInfo.s1Size = s1Size_;
    sasInfo.s2Size = s2Size_;
    sasInfo.gSize = gSize_;
    sasInfo.qkHeadDim = qkHeadDim_;
    sasInfo.qTSize = qTSize_;
    sasInfo.sparseBlockCount = sparseBlockCount_;

    sasInfo.qType = qType_;
    sasInfo.oriKvType = oriKvType_;
    sasInfo.cmpKvType = cmpKvType_;
    sasInfo.outputType = outputType_;

    sasInfo.totalBlockNum = (opParamInfo_.oriKv.tensor != nullptr) ?
        opParamInfo_.oriKv.tensor->GetStorageShape().GetDim(0) : 0;
    sasInfo.sparseBlockSize = 1;
    sasInfo.blockSize = oriBlockSize_;
    sasInfo.oriBlockSize = oriBlockSize_;
    sasInfo.cmpBlockSize = cmpBlockSize_;
    sasInfo.blockTypeSize = sizeof(float);
    sasInfo.oriMaxBlockNumPerBatch = oriMaxBlockNumPerBatch_;
    sasInfo.cmpMaxBlockNumPerBatch = cmpMaxBlockNumPerBatch_;

    sasInfo.actualLenDimsQ = actualLenDimsQ_;
    sasInfo.actualLenDimsKV = actualLenDimsKV_;
    sasInfo.maxActualseq = maxActualseq_;
    sasInfo.actualSeqLenFlag = (opParamInfo_.sequsedKv.tensor != nullptr);
    sasInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;

    sasInfo.softmaxScale = *opParamInfo_.softmaxScale;
    sasInfo.cmpRatio = *opParamInfo_.cmpRatio;
    sasInfo.oriMaskMode = *opParamInfo_.oriMaskMode;
    sasInfo.cmpMaskMode = *opParamInfo_.cmpMaskMode;
    sasInfo.oriWinLeft = *opParamInfo_.oriWinLeft;
    sasInfo.oriWinRight = *opParamInfo_.oriWinRight;

    sasInfo.qLayout = qLayout_;
    sasInfo.kvLayout = kvLayout_;
    sasInfo.outLayout = outLayout_;
}

ge::graphStatus SASInfoParser::Parse(SASTilingInfo &sasInfo)
{
    if (context_ == nullptr) {
        OP_LOGE("SparseFlashAttention", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout() ||
        ge::GRAPH_SUCCESS != GetSASTemplateMode(sasInfo)) {
        return ge::GRAPH_FAILED;
    }

    SetSASShape();
    if (
        ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetN2Size() ||
        ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetQkHeadDim() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetSparseBlockCount() ||
        ge::GRAPH_SUCCESS != GetSinks()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(sasInfo);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::Process()
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForSparseAttnSharedkv(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

void SparseAttnSharedkvTiling::CalcUbBmm(SASTilingInfo *tilingInfo)
{
    uint32_t cubeMSize = tilingInfo->gSize * tilingInfo->s1Size;
    uint32_t maxMSize = mBaseSize_;
    if (cubeMSize > maxMSize) {
        cubeMSize = maxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存

    qPreSizeMla_ = tilingInfo->gSize * headDimAlign_ * tilingInfo->s1Size;
}

void SparseAttnSharedkvTiling::SplitBalanced(SASTilingInfo *tilingInfo)
{
    uint32_t s2Size = tilingInfo->s2Size;
    sInnerSize_ = 512; // 512:s2默认切分大小
    sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > s2Size) {
        sInnerSize_ = s2Size;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CalcUbBmm(tilingInfo);

    InnerSplitParams innerSplitParams;
    innerSplitParams.s1GBaseSize = tilingInfo->gSize;
    innerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.baseParams.set_mBaseSize(innerSplitParams.s1GBaseSize);
    tilingData_.baseParams.set_s2BaseSize(innerSplitParams.s2BaseSize);
    tilingData_.baseParams.set_mmResUbSize(mmResUbSize_);
    tilingData_.baseParams.set_bmm2ResUbSize(bmm2ResUbSize_);
}

// --------------------------SparseAttnSharedkvTiling类成员函数定义-----------------------
ge::graphStatus SparseAttnSharedkvTiling::DoOpTiling(SASTilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    OP_LOGI(tilingInfo->opName, "SAS block dim: %u aiv Num: %u aic Num: %u.", blockDim, aivNum, aicNum);

    SplitBalanced(tilingInfo);
    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;             // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 512;             // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 512;            // S2轴基本块大小
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;          // 4: int32
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;          // 保留Index和Value 2种数据
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8; // 8: int64
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;      // Decode参数个数
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;        // Decode每个核需要存储头和尾部两块数据
    constexpr uint32_t S1_BASE_SIZE = 8;              // S1轴基本块的大小
    constexpr uint32_t TOPK_MAX_SIZE = 2048;          // TopK选取个数
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 主流程需Workspace大小
    uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;
    workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
    // Decode流程(LD)需要Workspace大小
    // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
    // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;

    workspaceSize = 120 * 1024 * 1024;
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.baseParams.set_batchSize(tilingInfo->bSize);
    tilingData_.baseParams.set_kvSeqSize(tilingInfo->s2Size);
    tilingData_.baseParams.set_qSeqSize(tilingInfo->s1Size);
    tilingData_.baseParams.set_nNumOfQInOneGroup(tilingInfo->gSize);
    tilingData_.baseParams.set_paBlockSize(tilingInfo->blockSize);
    tilingData_.baseParams.set_oriBlockSize(tilingInfo->oriBlockSize);
    tilingData_.baseParams.set_cmpBlockSize(tilingInfo->cmpBlockSize);
    tilingData_.baseParams.set_oriMaxBlockNumPerBatch(tilingInfo->oriMaxBlockNumPerBatch);
    tilingData_.baseParams.set_actualLenDimsQ(tilingInfo->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(tilingInfo->actualLenDimsKV);

    tilingData_.baseParams.set_softmaxScale(tilingInfo->softmaxScale);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(tilingInfo->outLayout));
    tilingData_.baseParams.set_oriMaskMode(tilingInfo->oriMaskMode);
    tilingData_.baseParams.set_oriWinLeft(tilingInfo->oriWinLeft);
    tilingData_.baseParams.set_oriWinRight(tilingInfo->oriWinRight);
    tilingData_.baseParams.set_sparseBlockSize(tilingInfo->sparseBlockSize);

    tilingData_.cmpParams.set_cmpMaxBlockNumPerBatch(tilingInfo->cmpMaxBlockNumPerBatch);
    tilingData_.cmpParams.set_sparseBlockCount(tilingInfo->sparseBlockCount);
    tilingData_.cmpParams.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.cmpParams.set_cmpMaskMode(tilingInfo->cmpMaskMode);

    usedCoreNum_ = aicNum;
    tilingData_.baseParams.set_usedCoreNum(usedCoreNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE
    uint32_t qLayout = static_cast<uint32_t>(tilingInfo->qLayout);
    uint32_t inputKvLayout = static_cast<uint32_t>(tilingInfo->kvLayout);

    uint32_t tilingKey =
        GET_TPL_TILING_KEY(0U, qLayout, inputKvLayout, static_cast<uint32_t>(tilingInfo->perfMode));
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingSparseAttnSharedkv(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttnSharedkv", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    SASTilingInfo sasInfo;
    SASInfoParser sasInfoParser(context);
    if (sasInfoParser.Parse(sasInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SASTilingCheck sasTilingChecker(sasInfo);
    if (sasTilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SparseAttnSharedkvTiling tiling(context);
    return tiling.DoOpTiling(&sasInfo);
}
// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(SparseAttnSharedkv)
    .Tiling(TilingSparseAttnSharedkv)
    .TilingParse<SASCompileInfo>(TilingPrepareForSparseAttnSharedkv);

} // namespace optiling
