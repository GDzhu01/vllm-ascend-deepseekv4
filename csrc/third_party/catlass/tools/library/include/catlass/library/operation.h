/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_LIBRARY_OPERATION_H
#define CATLASS_LIBRARY_OPERATION_H

#include <string>
#include <acl/acl.h>

#include "catlass/gemm_coord.hpp"
#include "catlass/status.hpp"

namespace Catlass {
namespace Library {


using Status = Catlass::Status;

enum class DataType {
    U8,
    Int8,
    Int32,
    Fp16,
    Bf16,
    Fp32,
    Invalid
};

enum class LayoutType {
    RowMajor,
    ColumnMajor,
    nZ,
    zN,
    zZ,
    PaddingRowMajor,
    PaddingColumnMajor,
    nN,
    VectorLayout,
    Invalid
};

enum class OperationKind {
    Gemm,
    Invalid
};

enum class GemmKind {
    BasicMatmul,
    BatchedMatmul,
    GroupedMatmulSliceM,
    MatmulAdd,
    PaddingMatmul,
    GroupedMatmulSliceK,
    OptimizedMatmul,
    GroupedMatmulSliceMPerTokenDequantMoe,
    GroupedMatmul,
    SplitKMatmul,
    GroupedMatmulSliceMPerTokenDequant,
    GroupedMatmulSliceKPerTokenDequant,
    QuantMatmul,
    BasicMatmulTla,
    OptimizedMatmulTla,
    KernelGemm,
    GroupedGemm,
    MatmulBias,
    BasicMatmulPreload,
    PaddingSplitkMatmul,
    MatmulGelu,

    Invalid
};

struct GemmShapeDescription {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    GemmShapeDescription(
        uint32_t m = 0U,
        uint32_t n = 0U,
        uint32_t k = 0U
    ) : m(m), n(n), k(k) {}
};

struct TileDescription {
    GemmShapeDescription L1TileShape;
    GemmShapeDescription L0TileShape;
    TileDescription()
    {
        L1TileShape = GemmShapeDescription(0, 0, 0);
        L0TileShape = GemmShapeDescription(0, 0, 0);
    }
};

struct TensorDescription {
    DataType element;
    LayoutType layout;
    TensorDescription(
        DataType element = DataType::Invalid,
        LayoutType layout = LayoutType::Invalid
    ) : element(element), layout(layout) {}
};

struct OperationDescription {
    char const * name;
    OperationKind kind;
    TileDescription tileDescription;

    OperationDescription(
        char const *name = "",
        OperationKind kind = OperationKind::Invalid,
        TileDescription tileDescription = TileDescription()
    ) : name(name), kind(kind), tileDescription(tileDescription) {}
};

struct GemmOperationDescription : public OperationDescription {
    GemmKind gemmKind;

    TensorDescription A;
    TensorDescription B;
    TensorDescription C;

    DataType epilogueElementType;

    GemmOperationDescription(
        GemmKind gemmKind = GemmKind::Invalid,
        TensorDescription A = TensorDescription(),
        TensorDescription B = TensorDescription(),
        TensorDescription C = TensorDescription(),
        DataType epilogueElementType = DataType::Invalid
    ) : gemmKind(gemmKind), A(A), B(B), C(C), epilogueElementType(epilogueElementType) {}
};


struct QuantMatmulGemmOperationDescription : public GemmOperationDescription {
    TensorDescription Scale;
    TensorDescription PerTokenScale;
    TensorDescription D;  // output
    QuantMatmulGemmOperationDescription(
        GemmKind gemmKind = GemmKind::Invalid,
        TensorDescription A = TensorDescription(),
        TensorDescription B = TensorDescription(),
        TensorDescription C = TensorDescription(),
        TensorDescription D = TensorDescription(),
        TensorDescription Scale = TensorDescription(),
        TensorDescription PerTokenScale = TensorDescription(),
        DataType epilogueElementType = DataType::Invalid
    ) : GemmOperationDescription(gemmKind, A, B, C, epilogueElementType),
        D(D), Scale(Scale), PerTokenScale(PerTokenScale){}
};

class Operation {
public:
    virtual ~Operation() = default;

    virtual Status CanImplement(void *arguments, void *config) = 0;

    virtual size_t GetWorkspaceSize(void *arguments, void *config) = 0;

    virtual Status Initialize(
        void *arguments,
        void *config,
        uint8_t *workspace = nullptr,
        aclrtStream stream = nullptr) = 0;

    virtual Status Run(
        aclrtStream stream, uint32_t blockDim,
        uint64_t fftsAddr = 0) = 0;

    virtual OperationDescription const &GetDescription() const = 0;
};


// Arguments for basic matmul & optimized matmul operations
//
// OperationKind: Gemm
// GemmKind:      BasicMatmul & OptimizedMatmul
//
struct BasicMatmulGemmArguments {
    uint8_t *A;
    uint8_t *B;
    uint8_t *C;
};

struct BasicMatmulGemmConfiguration {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};


// Arguments for grouped matmul operations
//
// OperationKind: Gemm
// GemmKind:      GroupedMatmul
//
struct GroupedMatmulGemmArguments {
    uint8_t *problemShapeList;
    uint8_t *A;
    uint8_t *layoutAList;
    uint8_t *B;
    uint8_t *layoutBList;
    uint8_t *C;
    uint8_t *layoutCList;
};

struct GroupedMatmulGemmConfiguration {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t groupCount;
};

struct QuantMatmulGemmArguments {
    Catlass::GemmCoord problemShape;
    uint32_t aicCoreNum;
    uint8_t *ptrA;
    uint8_t *ptrB;
    uint8_t *ptrD;  // output
    uint8_t *ptrScale;
    uint8_t *ptrPerTokenScale;
};

struct QuantMatmulGemmConfiguration {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};


// Arguments for grouped matmul slice M operations
//
// OperationKind: Gemm
// GemmKind:      GroupedMatmulSliceM
//
struct GroupedMatmulSliceMGemmArguments {
    uint8_t *deviceGroupList;
    uint8_t *A;
    uint8_t *B;
    uint8_t *C;
};

struct GroupedMatmulSliceMGemmConfiguration {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t groupCount;
};

// Arguments for matmul gelu operations
// 
// OperationKind: Gemm
// GemmKind:      MatmulGelu
struct MatmulGeluGemmOperationDescription : public GemmOperationDescription {
    TensorDescription D;  // output
    MatmulGeluGemmOperationDescription(
        GemmKind gemmKind = GemmKind::Invalid,
        TensorDescription A = TensorDescription(),
        TensorDescription B = TensorDescription(),
        TensorDescription C = TensorDescription(),
        TensorDescription D = TensorDescription(),
        DataType epilogueElementType = DataType::Invalid
    ) : GemmOperationDescription(gemmKind, A, B, C, epilogueElementType),
        D(D) {}
};

struct MatmulGeluGemmArguments {
    uint8_t *ptrA;
    uint8_t *ptrB;
    uint8_t *ptrD;
};

struct MatmulGeluGemmConfiguration {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    size_t elementSize;
};

}
}

#endif // CATLASS_LIBRARY_OPERATION_H
