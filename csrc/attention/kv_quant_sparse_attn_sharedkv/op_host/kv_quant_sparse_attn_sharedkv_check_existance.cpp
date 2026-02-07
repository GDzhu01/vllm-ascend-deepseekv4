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
 * \file kv_quant_sparse_attn_sharedkv_check_existance.cpp
 * \brief
 */

#include "kv_quant_sparse_attn_sharedkv_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

ge::graphStatus KvQuantSASTilingCheck::CheckParaExistenceAntiquant() const
{
    if (kvLayout_ == SASLayout::BSND) {
        return ge::GRAPH_SUCCESS;
    }  else if (kvLayout_ == SASLayout::PA_ND) {
        OP_CHECK_IF(opParamInfo_.sequsedKv.tensor == nullptr,
            OP_LOGE(opName_, "when layout_kv is PA_ND, actualSeqLengthsKv must not be null"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((opParamInfo_.oriBlockTable.tensor == nullptr) && (opParamInfo_.cmpBlockTable.tensor == nullptr),
            OP_LOGE(opName_, "when layout_kv is PA_ND, oriBlockTable and cmpBlockTable must be one "),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckCmpSparseIndicesExistence() || 
        ge::GRAPH_SUCCESS != CheckSWAExistence() ||
        ge::GRAPH_SUCCESS != CheckCFAExistence() ||
        ge::GRAPH_SUCCESS != CheckSCFAExistence() ||
        ge::GRAPH_SUCCESS != CheckCmpRatioExistence() ||
        ge::GRAPH_SUCCESS != CheckParaExistenceAntiquant()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckCmpSparseIndicesExistence()
{
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        if (qLayout_ == SASLayout::BSND) {
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(3) != 512) {
                OP_LOGE(opName_, "When qLayout is BNSD, topK should be 512, but got %ld", opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(3));
                return ge::GRAPH_FAILED;
            }
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(1) != s1Size_) {
                OP_LOGE(opName_, "When qLayout is BNSD, cmpSparseIndices's S should be eaque to s1Size, but got %ld", opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(1));
                return ge::GRAPH_FAILED;
            }
        } else {
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(2) != 512) {
                OP_LOGE(opName_, "When qLayout is BNSD, topK should be 512, but got %ld", opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(2));
                return ge::GRAPH_FAILED;
            }
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(0) != qTSize_) {
                OP_LOGE(opName_, "When qLayout is TND, cmpSparseIndices's T should be eaque to qTSize, but got %ld", opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(0));
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSWAExistence()
{
    if (perfMode_ != SASTemplateMode::SWA_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.oriBlockTable.tensor == nullptr,
        OP_LOGE(opName_, "SWA mode, oriBlockTable is lost"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckCFAExistence()
{
    if (perfMode_ != SASTemplateMode::CFA_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr && opParamInfo_.cmpKv.tensor != nullptr,
        OP_LOGE(opName_, "CFA mode, oriKv is lost."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.cmpKv.tensor == nullptr && opParamInfo_.cmpRatio != nullptr,
        OP_LOGE(opName_, "CFA mode, cmpKv is lost."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.cmpKv.tensor != nullptr && opParamInfo_.cmpRatio == nullptr,
        OP_LOGE(opName_, "CFA mode, cmpRatio is lost."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.cmpKv.tensor != nullptr && opParamInfo_.cmpBlockTable.tensor == nullptr,
        OP_LOGE(opName_, "CFA mode, cmpBlockTable is lost."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSCFAExistence()
{
    if (perfMode_ != SASTemplateMode::SCFA_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.cmpKv.tensor == nullptr && opParamInfo_.cmpSparseIndices.tensor != nullptr,
        OP_LOGE(opName_, "SCFA mode, oriKv is lost."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr && opParamInfo_.cmpKv.tensor != nullptr && opParamInfo_.cmpSparseIndices.tensor != nullptr,
        OP_LOGE(opName_, "SCFA mode, cmpKv is lost."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr && opParamInfo_.cmpKv.tensor == nullptr && opParamInfo_.cmpSparseIndices.tensor != nullptr,
        OP_LOGE(opName_, "SCFA mode, oriKv and cmpKv is lost."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckCmpRatioExistence()
{
    if (perfMode_ == SASTemplateMode::SWA_TEMPLATE_MODE) {
        OP_CHECK_IF(*opParamInfo_.cmpRatio != 1 && *opParamInfo_.cmpRatio != 128 && *opParamInfo_.cmpRatio != 4,
            OP_LOGE(opName_, "SWA mode, cmpRatio must be 1, but got %d", *opParamInfo_.cmpRatio),
            return ge::GRAPH_FAILED);
    } else if (perfMode_ == SASTemplateMode::CFA_TEMPLATE_MODE) {
        OP_CHECK_IF(*opParamInfo_.cmpRatio != 128 && *opParamInfo_.cmpRatio != 4,
            OP_LOGE(opName_, "CFA mode, cmpRatio must be 4 or 128, but got %d", *opParamInfo_.cmpRatio),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(*opParamInfo_.cmpRatio != 128 && *opParamInfo_.cmpRatio != 4,
            OP_LOGE(opName_, "SCFA mode, cmpRatio must be 4 or 128, but got %d", *opParamInfo_.cmpRatio),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

}