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
 * \file weight_quant_batch_matmul_v2_checker_for_mmads8s4.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_checker_for_mmads8s4.h"
#include "common/op_host/math_util.h"
#include "common/op_host/op_tiling/debug_tiling.h"

namespace {
constexpr size_t MM_SHAPE_LEN_ND = 2UL;
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr uint64_t MAX_INT32 = 2147483647UL;
}

namespace optiling {
ge::graphStatus Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckContext()
{
    // check Raw TilingData
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    // check the Required input and output desc
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    inputParams_.opName = context_->GetNodeName();
    // OP_LOG_FULL
    OPS_LOG_I(inputParams_.opName, "TilingContext: %s", Ops::Transformer::DebugTilingContext(context_).c_str());
    return ge::GRAPH_SUCCESS;
}

/*
The function is check the dtype limit:
    1. Input x dtype should be fp16, weight should be int8
    2. Output dtype should be same with x dtype without quant
    3. antiquant scale dtype should be uint64
    4. other params should be null
*/
bool Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckDtype()
{
    size_t idx = 0;
    inputParams_.aDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.antiQuantScaleDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_FLOAT16,
                    OP_LOGE(inputParams_.opName, "Input x dtype must be FLOAT16, but is [%s]",
                                                    ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
                    return false);

    OP_TILING_CHECK(inputParams_.bDtype != ge::DT_INT8,
                    OP_LOGE(inputParams_.opName, "Input weight dtype must be INT8, but is [%s]",
                                                    ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                    return false);

    OP_TILING_CHECK(
        inputParams_.antiQuantScaleDtype != ge::DT_UINT64 && inputParams_.antiQuantScaleDtype != ge::DT_INT64,
        OP_LOGE(
            inputParams_.opName, "Input antiquant scale dtype must be UINT64/INT64, but is [%s]",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.antiQuantScaleDtype).c_str()),
        return false);

    OP_TILING_CHECK(inputParams_.cDtype != ge::DT_FLOAT16,
                    OP_LOGE(inputParams_.opName, "Output y dtype must be FLOAT16, but is [%s]",
                                                    ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
                    return false);

    return true;
}

/*
The function is check the attr limit:
    1. transposeX and transposeWeight should be false
    2. Group size should be 0
    3. innerPrecise only support 0 or 1
*/
bool Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckAttr()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto transposeX = attrs->GetAttrPointer<bool>(idx++);
    auto transposeWeight = attrs->GetAttrPointer<bool>(idx++);
    const int64_t *groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    idx++;  // 跳过dtype属性
    const int64_t *innerPrecisePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        innerPrecisePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    inputParams_.transA = transposeX != nullptr && *transposeX;
    inputParams_.transB = transposeWeight != nullptr && *transposeWeight;

    OP_TILING_CHECK(inputParams_.transA,
                    OP_LOGE(inputParams_.opName, "transA should be false, but is true"),
                    return false);
    OP_TILING_CHECK(inputParams_.transB,
                    OP_LOGE(inputParams_.opName, "transB should be false, but is true"),
                    return false);

    if (groupSizePtr != nullptr) {
        OP_TILING_CHECK(
            *groupSizePtr != 0,
            OP_LOGE(inputParams_.opName, "Group size should be 0, but is [%ld]", *groupSizePtr),
            return false);
        inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    }
    if (innerPrecisePtr != nullptr) {
        OP_TILING_CHECK((*innerPrecisePtr != 0) && (*innerPrecisePtr != 1),
                        OP_LOGE(
                            inputParams_.opName, "innerPrecise only support 0 or 1, but is [%ld]", *innerPrecisePtr),
                        return false);
        inputParams_.innerPrecise = static_cast<uint64_t>(*innerPrecisePtr);
    }
    return true;
}

/*
The function is check the shape limit:
    1. the input x shape dims should be 2(ND), weight shape dims should be 2(ND)
    2. the k of x and weight must be same
    3. antiquant shape must be:
        per_channel: not trans_n: (1, n) or (n); trans_b: (n, 1) or (n)
        per_tensor: (1,)
    4. quant shape must be empty tensor
    5. bias shape must be empty tensor
    6. nk must <= 65535, m <= 65535(trans_a) or int32_max(not trans_a);
*/
bool Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckShape()
{
    size_t idx = 0;
    auto xShape = context_->GetInputShape(idx++);
    auto weightShape = context_->GetInputShape(idx++);
    auto antiQuantScaleShape = context_->GetInputShape(idx++);
    auto antiQuantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto quantScaleShape = context_->GetOptionalInputShape(idx++);
    auto quantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto biasShape = context_->GetOptionalInputShape(idx++);
    auto outputShape = context_->GetOutputShape(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, antiQuantScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    // not yet support empty tensor for input
    OP_TILING_CHECK(xShape->GetStorageShape().GetShapeSize() == 0 ||
                        weightShape->GetStorageShape().GetShapeSize() == 0 ||
                        antiQuantScaleShape->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE(inputParams_.opName, "Not yet support empty tensor"), return false);
    // check x, weight
    inputParams_.aFormat = Mc2GetInputStorageFormat(context_, 0);
    inputParams_.bFormat = Mc2GetInputStorageFormat(context_, 1);
    OP_TILING_CHECK(!CheckInputShape(xShape, weightShape),
                    OP_LOGE(inputParams_.opName, "Check input x and weight shape failed"),
                    return false);
    // check antiquant scale, antiquant offset
    OP_TILING_CHECK(!CheckAntiQuantShape(antiQuantScaleShape, antiQuantOffsetShape),
                    OP_LOGE(inputParams_.opName, "Check antiquant shape failed"), return false);
    // check quant scale, quant offset
    OP_TILING_CHECK(quantScaleShape != nullptr,
                    OP_LOGE(inputParams_.opName, "Quant scale should be nullptr"),
                    return false);
    OP_TILING_CHECK(quantOffsetShape != nullptr,
                    OP_LOGE(inputParams_.opName, "Quant offset should be nullptr"),
                    return false);
    // check bias
    OP_TILING_CHECK(biasShape != nullptr,
                    OP_LOGE(inputParams_.opName, "Bias should be nullptr"), return false);
    // check dim value
    OP_TILING_CHECK(inputParams_.kSize > MAX_SHAPE_DIM || inputParams_.nSize > MAX_SHAPE_DIM,
                    OP_LOGE(
                        inputParams_.opName, "Dim of k or n should not more than 65535, but they are [%lu] and [%lu]",
                        inputParams_.kSize, inputParams_.nSize), return false);
    uint64_t batchMax = inputParams_.transA ? MAX_SHAPE_DIM : MAX_INT32;
    OP_TILING_CHECK(
        inputParams_.mSize > batchMax,
        OP_LOGE(inputParams_.opName, "Dim of m should not more than [%lu], but is [%lu]",
                                        batchMax, inputParams_.mSize), return false);
    OP_TILING_CHECK(inputParams_.groupSize >= inputParams_.kSize || inputParams_.groupSize % MIN_GROUP_SIZE != 0,
                    OP_LOGE(
                        inputParams_.opName, "Group sizes should not more than [%lu] and align to 32, but is [%lu]",
                        inputParams_.kSize, inputParams_.groupSize), return false);
    return true;
}

bool Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckInputShape(const gert::StorageShape *xShape,
                                                               const gert::StorageShape *weightShape)
{
    OP_TILING_CHECK(inputParams_.aFormat != ge::Format::FORMAT_ND && inputParams_.aFormat != ge::Format::FORMAT_NCHW,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input x format should be ND , actual format is %s",
                                          ge::TypeUtils::FormatToSerialString(inputParams_.aFormat).c_str()),
                    return false);

    OP_TILING_CHECK(inputParams_.bFormat != ge::Format::FORMAT_ND && inputParams_.bFormat != ge::Format::FORMAT_NCHW,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input weight format should be ND , actual format is %s",
                                          ge::TypeUtils::FormatToSerialString(inputParams_.bFormat).c_str()),
                    return false);

    size_t xOriDimNum = xShape->GetOriginShape().GetDimNum();
    size_t weightOriDimNum = weightShape->GetOriginShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t weightDimNum = weightShape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(
        xOriDimNum != MM_SHAPE_LEN_ND,
        OP_LOGE(inputParams_.opName, "OriginalShape of x must be 2, but is [%zu]", xOriDimNum),
        return false);

    OP_TILING_CHECK(weightOriDimNum != MM_SHAPE_LEN_ND,
                    OP_LOGE(inputParams_.opName,
                                                    "OriginalShape of weight must be 2, but is [%zu]", weightOriDimNum),
                    return false);

    OP_TILING_CHECK(
        xDimNum != MM_SHAPE_LEN_ND,
        OP_LOGE(inputParams_.opName, "StorageShape of x must be 2, but is [%zu]", xDimNum),
        return false);

    OP_TILING_CHECK(weightDimNum != MM_SHAPE_LEN_ND,
                    OP_LOGE(inputParams_.opName,
                                                    "StorageShape of weight must be 2, but is [%zu]", weightDimNum),
                    return false);
    uint64_t weightLastDim = weightShape->GetOriginShape().GetDim(1);
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? xShape->GetOriginShape().GetDim(1)
                                                                   : xShape->GetOriginShape().GetDim(0));
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? xShape->GetOriginShape().GetDim(0)
                                                                   : xShape->GetOriginShape().GetDim(1));
    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? weightLastDim : weightShape->GetOriginShape().GetDim(0));
    inputParams_.nSize =
        static_cast<uint64_t>(inputParams_.transB ? weightShape->GetOriginShape().GetDim(0) : weightLastDim);
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    OP_LOGE(inputParams_.opName,
                                                    "K dim of x and weight must equal, but they are [%lu] and [%lu]",
                                                    inputParams_.kSize, kBSize),
                    return false);
    return true;
}

bool Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckAntiQuantShape(const gert::StorageShape *antiQuantScaleShape,
                                                                   const gert::StorageShape *antiQuantOffsetShape)
{
    size_t antiQuantScaleDimNum = antiQuantScaleShape->GetStorageShape().GetDimNum();
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(antiQuantScaleDimNum > MM_SHAPE_LEN_ND,
                    OP_LOGE(
                        inputParams_.opName, "antiquant scale shape size should not be more than 2, but is [%zu]",
                        antiQuantScaleDimNum),
                    return false);
    if (antiQuantScaleShapeSize != 1) {
        if (antiQuantScaleDimNum == MM_SHAPE_LEN_ND) {
            gert::Shape expectShape;
            uint64_t kNum = inputParams_.groupSize > 0 ? ops::CeilDiv(inputParams_.kSize, inputParams_.groupSize) : 1;
            if (inputParams_.transB) {
                expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
                expectShape.AppendDim(static_cast<int64_t>(kNum));
            } else {
                expectShape.AppendDim(static_cast<int64_t>(kNum));
                expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
            }
            OP_TILING_CHECK(
                expectShape != antiQuantScaleShape->GetStorageShape(),
                OP_LOGE(inputParams_.opName, "Antiquant shape expect %s, but is %s",
                                                Ops::Base::ToString(expectShape).c_str(),
                                                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str()),
                return false);
            inputParams_.antiQuantType = inputParams_.groupSize > 0 ? Mc2QuantType::PER_GROUP : Mc2QuantType::PER_CHANNEL;
        } else {
            OP_TILING_CHECK(antiQuantScaleShapeSize != inputParams_.nSize,
                            OP_LOGE(
                                inputParams_.opName, "Antiquant size should be n size when perchannel, but is %s",
                                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str()),
                            return false);
            inputParams_.antiQuantType = Mc2QuantType::PER_CHANNEL;
        }
    } else {
        inputParams_.antiQuantType = Mc2QuantType::PER_TENSOR;
    }

    OP_TILING_CHECK(antiQuantOffsetShape != nullptr,
                    OP_LOGE(inputParams_.opName, "Antiquant offset should be nullptr"),
                    return false);
    return true;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::Check()
{
    auto ret = CheckContext();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // check the input and output dtype: x, weight, antiquant_scale, y.
    OP_TILING_CHECK(!CheckDtype(),
                    OP_LOGE(inputParams_.opName, "Check input dtype failed"),
                    return ge::GRAPH_FAILED);
    // check attrs: transA, transB, group_size, dtype, innerPrecise
    OP_TILING_CHECK(!CheckAttr(),
                    OP_LOGE(inputParams_.opName, "Check attr failed"), return ge::GRAPH_FAILED);
    // check the input and output shape: all
    OP_TILING_CHECK(!CheckShape(),
                    OP_LOGE(inputParams_.opName, "Check shape failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling