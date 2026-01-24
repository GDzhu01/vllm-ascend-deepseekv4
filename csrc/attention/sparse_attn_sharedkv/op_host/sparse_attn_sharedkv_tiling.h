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
 * \file sparse_attn_sharedkv_tiling.h
 * \brief
 */
#ifndef SPARSE_ATTN_SHAREDKV_TILING_H
#define SPARSE_ATTN_SHAREDKV_TILING_H

#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"

namespace optiling {
// // ------------------公共定义--------------------------
struct SASTilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct SASTilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class SASLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_ND = 2
};

enum class SASAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3,  // sparse_indices的K和key的D枚举值相同，表达相同位置, 最后一维
    T = 5,
    Bn = 6, // block number
    Bs = 7 // block size
};

enum class SASTemplateMode : uint32_t {
    SWA_TEMPLATE_MODE = 0,
    CFA_TEMPLATE_MODE = 1,
    SCFA_TEMPLATE_MODE = 2
};

enum class KvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    TENSOR_LIST = 1,
    PAGE_ATTENTION = 2
};

struct InnerSplitParams {
    uint32_t s1GBaseSize = 1;
    uint32_t s2BaseSize = 1;
};
// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t ORI_KV_INDEX = 1;
constexpr uint32_t CMP_KV_INDEX = 2;
constexpr uint32_t CMP_SPARSE_INDICES_INDEX = 3;
constexpr uint32_t ORI_BLOCK_TABLE_INDEX = 4;
constexpr uint32_t CMP_BLOCK_TABLE_INDEX = 5;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 6;
constexpr uint32_t SEQUSED_KV_INDEX = 7;
constexpr uint32_t SINKS_INDEX = 8;
constexpr uint32_t METADATA_INDEX = 9;
// Outputs Index
constexpr uint32_t ATTN_OUT_INDEX = 0;

// Attributes Index
constexpr uint32_t ATTR_SOTFMAX_SCALE_INDEX = 0;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 1;
constexpr uint32_t ATTR_ORI_MASK_MODE_INDEX = 2;
constexpr uint32_t ATTR_CMP_MASK_MODE_INDEX = 3;
constexpr uint32_t ATTR_ORI_WIN_LEFT_INDEX = 4;
constexpr uint32_t ATTR_ORI_WIN_RIGHT_INDEX = 5;
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 6;
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 7;

// Dim Index
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_IDX_FOUR = 4;

// Dim Num
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

// 常量
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;
// const uint32_t SAS_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_LIMIT = 2048;
constexpr uint32_t SPARSE_MODE_LOWER = 3;

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(SparseAttnSharedkvSwaParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(uint32_t, kvSeqSize)
TILING_DATA_FIELD_DEF(int64_t, paBlockSize)
TILING_DATA_FIELD_DEF(int64_t, oriBlockSize)
TILING_DATA_FIELD_DEF(int64_t, cmpBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, oriMaxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsQ)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsKV)

TILING_DATA_FIELD_DEF(float, softmaxScale) // 即 scaleValue
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint64_t, oriMaskMode)
TILING_DATA_FIELD_DEF(int64_t, oriWinLeft)
TILING_DATA_FIELD_DEF(int64_t, oriWinRight)
TILING_DATA_FIELD_DEF(int64_t, sparseBlockSize)

TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);

TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);

TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseAttnSharedkvSwaParamsOp, SparseAttnSharedkvSwaParams)

BEGIN_TILING_DATA_DEF(SparseAttnSharedkvCmpParams)

TILING_DATA_FIELD_DEF(uint32_t, cmpMaxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockCount)
TILING_DATA_FIELD_DEF(int64_t, cmpRatio)
TILING_DATA_FIELD_DEF(uint64_t, cmpMaskMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseAttnSharedkvCmpParamsOp, SparseAttnSharedkvCmpParams)

// // 内切基本块参数
// BEGIN_TILING_DATA_DEF(SparseAttnSharedkvInnerSplitParams)
// // TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
// // TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(SparseAttnSharedkvInnerSplitParamsOp, SparseAttnSharedkvInnerSplitParams)

BEGIN_TILING_DATA_DEF(SparseAttnSharedkvTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseAttnSharedkvSwaParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(SparseAttnSharedkvCmpParams, cmpParams);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(SparseAttnSharedkv, SparseAttnSharedkvTilingData)

struct SASParaInfo {
    SASTilingRequiredParaInfo q = {nullptr, nullptr};
    SASTilingOptionalParaInfo oriKv = {nullptr, nullptr};
    SASTilingOptionalParaInfo cmpKv = {nullptr, nullptr};
    SASTilingOptionalParaInfo cmpSparseIndices = {nullptr, nullptr};
    SASTilingOptionalParaInfo oriBlockTable = {nullptr, nullptr};
    SASTilingOptionalParaInfo cmpBlockTable = {nullptr, nullptr};
    SASTilingOptionalParaInfo cuSeqLensQ = {nullptr, nullptr};
    SASTilingOptionalParaInfo sequsedKv = {nullptr, nullptr};
    SASTilingOptionalParaInfo sinks = {nullptr, nullptr};
    SASTilingOptionalParaInfo metadata = {nullptr, nullptr};
    SASTilingRequiredParaInfo attnOut = {nullptr, nullptr};


    const float *softmaxScale = nullptr;
    const uint32_t *cmpRatio = nullptr;
    const uint32_t *oriMaskMode = nullptr;
    const uint32_t *cmpMaskMode = nullptr;
    const uint32_t *oriWinLeft = nullptr;
    const uint32_t *oriWinRight = nullptr;
    const char *layoutQ = nullptr;
    const char *layoutKv = nullptr;
};

// -----------算子Tiling入参信息类---------------
class SASTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    SASParaInfo opParamInfo;

    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t gSize = 0;
    uint32_t qkHeadDim = 0;
    uint32_t qTSize = 0; // 仅TND时生效

    uint32_t actualLenDimsQ = 0;
    uint32_t maxActualseq = 0;
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    uint32_t actualLenDimsKV = 0;

    float softmaxScale = 0;
    int64_t cmpRatio = 0;
    uint64_t oriMaskMode = 0;
    uint64_t cmpMaskMode = 0;
    int64_t oriWinLeft = 0;
    int64_t oriWinRight = 0;
    int64_t sparseBlockSize = 0;
    int64_t sparseBlockCount = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    uint32_t sparseCount = 0;
    
    // PageAttention
    uint32_t blockTypeSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;
    int32_t blockSize = 0;
    int32_t oriBlockSize = 0;
    int32_t cmpBlockSize = 0;
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t totalBlockNum = 0;

    // DType
    ge::DataType qType = ge::DT_FLOAT16;
    ge::DataType oriKvType = ge::DT_FLOAT16;
    ge::DataType cmpKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    // Layout
    SASLayout qLayout = SASLayout::TND;
    SASLayout kvLayout = SASLayout::PA_ND;
    SASLayout outLayout = SASLayout::BSND;

    // template mode
    SASTemplateMode perfMode = SASTemplateMode::SWA_TEMPLATE_MODE;

};


// // -----------算子CompileInfo定义-------------------
// struct SASCompileInfo {};


// -----------算子Tiling入参信息解析及Check类---------------
class SASTilingCheck {
public:
    explicit SASTilingCheck(const SASTilingInfo &sasInfo) : sasInfo_(sasInfo) {};
    ~SASTilingCheck() = default;
    virtual ge::graphStatus Process();
private:
    // void Init();
    // void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    //     const ge::DataType &actualDtype, const std::string &name) const;
    // ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    //     const std::string &name) const;
    // template <typename T> void LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    //     const T &actualValue, const std::string &name, const std::string subName) const;
    // template <typename T> void LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    //     const T &actualValue, const std::string &name) const;
    // ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape,
    //     const std::vector<size_t> &expectDimNumList, const std::string &name) const;
    // ge::graphStatus CheckDimNumInLayoutSupport(const SASLayout &layout,
    //     const gert::StorageShape *shape, const std::string &name) const;
    // void LogErrorLayoutSupport(const std::vector<SASLayout> &expectLayoutList,
    //     const SASLayout &actualLayout, const std::string &name) const;
    // ge::graphStatus GetExpectedShape(gert::Shape &shapeExpected,
    // const SASTilingShapeCompareParam &param, const SASLayout &layout) const;
    // ge::graphStatus CompareShape(SASTilingShapeCompareParam &param,
    //     const gert::Shape &shape, const SASLayout &layout, const std::string &name) const;
    // ge::graphStatus CheckLayoutSupport(const SASLayout &actualLayout, const std::string &name) const;
    // ge::graphStatus CheckSingleParaQuery() const;
    // ge::graphStatus CheckSingleParaKey() const;
    // ge::graphStatus CheckSingleParaValue() const;
    // ge::graphStatus CheckSingleParaQueryRope() const;
    // ge::graphStatus CheckSingleParaKeyRope() const;
    // ge::graphStatus CheckSingleParaAttenOut() const;
    // ge::graphStatus CheckSingleParaNumHeads() const;
    // ge::graphStatus CheckSingleParaKvHeadNums() const;
    // ge::graphStatus CheckSingleParaLayout() const;
    // ge::graphStatus CheckSingleParaSparseMode() const;
    // ge::graphStatus CheckSingleParaSparseBlockSize() const;
    // ge::graphStatus CheckSingleParaSparseIndices() const;
    // ge::graphStatus CheckSinglePara() const;
    // ge::graphStatus CheckMultiParaConsistency() const;
    // ge::graphStatus CheckRopeExistence();
    // ge::graphStatus CheckExists(const void *pointer, const std::string &name) const;
    // ge::graphStatus CheckNotExists(const void *pointer, const std::string &name) const;
    // ge::graphStatus CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    // ge::graphStatus CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    // ge::graphStatus CheckExistenceByMap(std::map<std::string, const void *> &existMap,
    //     std::map<std::string, const void *> &notExistMap) const;
    // template <typename T> ge::graphStatus CheckAttrValueByMap(
    //     std::map<std::string, std::pair<const T *, T>> &attrMap) const;
    // ge::graphStatus CheckParaExistenceNoquant() const;
    // ge::graphStatus CheckParaExistenceGqaNoquant() const;
    // ge::graphStatus CheckParaExistence() const;
    // ge::graphStatus CheckParaExistence();
    // ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    //     const SASLayout &layout, const std::string &name) const;
    // void SetSASShapeCompare();
    // ge::graphStatus CheckQRope();
    // ge::graphStatus CheckQRopeShape();
    // ge::graphStatus CheckVAndKRopeShapeForBatchContinuous();
    // uint32_t GetTypeSize(ge::DataType dtype) const;
    // ge::graphStatus CheckVAndKRopeShapeForPageAttention();
    // ge::graphStatus CheckVAndKRopeShape();
    // ge::graphStatus CheckVAndKRope();
    // ge::graphStatus CheckTopK();
    // ge::graphStatus CheckTopkShape();
    // ge::graphStatus CheckBlockTable() const;
    // ge::graphStatus CheckDTypeConsistency(const ge::DataType &actualDtype,
    // const ge::DataType &expectDtype, const std::string &name) const;

    // ge::graphStatus CheckAttenOut();
    // ge::graphStatus CheckAttenOutShape();
    // ge::graphStatus CheckActualSeqLensQ();
    // ge::graphStatus CheckActualSeqLensQShape();
    // ge::graphStatus CheckActualSeqLensQDType();
    // ge::graphStatus CheckActualSeqLens();
    // ge::graphStatus CheckActualSeqLensDType();
    // ge::graphStatus CheckActualSeqLensShape();
    // ge::graphStatus CheckMultiParaConsistency();

    // ge::graphStatus CheckFeatureNoQuantShape() const;
    // ge::graphStatus CheckFeatureNoQuantLayout() const;
    // ge::graphStatus CheckFeatureNoQuantDtype() const;
    // ge::graphStatus CheckFeatureNoquantPa() const;
    // ge::graphStatus CheckFeatureNoquant() const;
    // ge::graphStatus CheckFeature() const;
    // ge::graphStatus CheckFeature() const;

    // ge::graphStatus CheckSingleParaPreTokens() const;
    // ge::graphStatus CheckSingleParaNextTokens() const;

private:
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    SASParaInfo opParamInfo_;
    const SASTilingInfo &sasInfo_;

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t qTSize_ = 0; // 仅TND时生效
    uint32_t kvTSize_ = 0; // 仅TND时生效
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    uint32_t sparseBlockCount_ = 0;

    SASLayout qLayout_ = SASLayout::BSND;
    SASLayout outLayout_ = SASLayout::BSND;
    SASLayout kvLayout_ = SASLayout::PA_ND;

    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    int64_t blockSize_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t l2CacheSize_ = 0;

    bool isSameSeqAllKVTensor_ = true;
    bool isSameActualseq_ = true;
    uint32_t maxActualseq_ = 0;

    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;

};

std::string SASLayoutToSerialString(SASLayout layout);

template <typename T> inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

class SASInfoParser {
public:
    explicit SASInfoParser(gert::TilingContext *context) : context_(context) {}
    ~SASInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
        SASLayout &layout, const std::string &name) const;
    ge::graphStatus GetActualSeqLenQSize(uint32_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetKvCache();
    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetKvLayout();
    ge::graphStatus GetSASTemplateMode(SASTilingInfo &sasInfo);
    void SetSASShape();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetQTSize();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetSparseBlockCount();
    ge::graphStatus GetActualseqInfo();
    ge::graphStatus GetSinks();
    void GenerateInfo(SASTilingInfo &sasInfo);
    ge::graphStatus Parse(SASTilingInfo &sasInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    SASParaInfo opParamInfo_;

    bool HasAxis(const SASAxis &axis, const SASLayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const SASAxis &axis, const SASLayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const SASAxis &axis,const SASLayout &layout) const;
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    uint32_t qTSize_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    int64_t sparseBlockSize_ = 0;
    int64_t sparseBlockCount_ = 0;
    uint32_t maxActualseq_ = 0;
    bool isSameSeqAllKVTensor_ = true;
    uint32_t actualLenDimsKV_ = 0;
    uint32_t actualLenDimsQ_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    // Layout
    SASLayout qLayout_ = SASLayout::BSND;
    SASLayout outLayout_ = SASLayout::BSND;
    SASLayout kvLayout_ = SASLayout::PA_ND;
    // PageAttention
    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    int32_t oriBlockSize_ = 0;
    int32_t cmpBlockSize_ = 0;

    // template mode
    SASTemplateMode perfMode_ = SASTemplateMode::SWA_TEMPLATE_MODE;

    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpSparseIndicesType_ = ge::DT_INT32;
    ge::DataType oriBlockTableType_ = ge::DT_INT32;
    ge::DataType cmpBlockTableType_ = ge::DT_INT32;
    ge::DataType cuSeqLensQType_ = ge::DT_INT32;
    ge::DataType seqsedKvType_ = ge::DT_INT32;
    ge::DataType sinksType_ = ge::DT_INT32;
    ge::DataType metadataType_ = ge::DT_INT32;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    gert::Shape qShape_{};
    gert::Shape oriKvShape_{};
    gert::Shape cmpKvShape_{};
    gert::Shape cmpSparseIndicesShape_{};
};

// ---------------算子Tiling类---------------
class SparseAttnSharedkvTiling {
public:
    explicit SparseAttnSharedkvTiling(gert::TilingContext *context) : context_(context){};
    ge::graphStatus DoOpTiling(SASTilingInfo *tilingInfo);

private:
    void SplitBalanced(SASTilingInfo *tilingInfo);
    void CalcUbBmm(SASTilingInfo *tilingInfo);
    gert::TilingContext *context_ = nullptr;
    SASTemplateMode perfMode_ = SASTemplateMode::SWA_TEMPLATE_MODE;
    SparseAttnSharedkvTilingData tilingData_;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    SASTilingInfo *sasInfo_ = nullptr;

    size_t libapiSize_ = 0;

    uint32_t kvSplitPart_ = 1;
    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    size_t qPreSizeMla_= 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t kvSplit_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t formerCoreNum_ = 0;
    uint32_t blockSplitBn2Range_ = 0;
    uint32_t tailSplitedBatchRange_ = 0;
    
    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 128;
    uint32_t mFdBaseSize_ = 8;
};

}
#endif