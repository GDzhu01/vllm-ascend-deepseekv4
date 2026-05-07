/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_LIBRARY_GEMM_OPERATION_H
#define CATLASS_LIBRARY_GEMM_OPERATION_H

#include <type_traits>
#include "catlass/library/operation.h"
#include "library_utils.h"

namespace Catlass {
namespace Library {

template <typename Operator_, typename Description_>
class GemmOperationBase : public Operation {
public:
    using Operator = Operator_;
    using OperatorArguments = typename Operator::Arguments;
    using OperatorKernel = typename Operator::Kernel;

    using ElementA = typename OperatorKernel::ElementA;
    using ElementB = typename OperatorKernel::ElementB;
    using ElementC = typename OperatorKernel::ElementC;
    using LayoutA = typename OperatorKernel::LayoutA;
    using LayoutB = typename OperatorKernel::LayoutB;
    using LayoutC = typename OperatorKernel::LayoutC;
    using BlockMmad = typename OperatorKernel::BlockMmad;
    using ArchTag = typename OperatorKernel::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;
    using BlockScheduler = typename OperatorKernel::BlockScheduler;

    GemmOperationBase(char const *name = "")
    {
        this->description_.name = name;
        this->description_.kind = OperationKind::Gemm;

        this->description_.A = MakeTensorDescription<ElementA, LayoutA>();
        this->description_.B = MakeTensorDescription<ElementB, LayoutB>();
        this->description_.C = MakeTensorDescription<ElementC, LayoutC>();

        this->description_.tileDescription.L1TileShape =
            GemmShapeDescription(L1TileShape::M, L1TileShape::N, L1TileShape::K);
        this->description_.tileDescription.L0TileShape =
            GemmShapeDescription(L0TileShape::M, L0TileShape::N, L0TileShape::K);
    }

    virtual OperationDescription const &GetDescription() const override
    {
        return this->description_;
    }

    virtual Status CanImplement(void *argsPtr, void *configPtr) override
    {
        BuildArgs(argsPtr, configPtr);
        return op_.CanImplement(this->args_);
    }

    virtual size_t GetWorkspaceSize(void *argsPtr, void *configPtr) override
    {
        BuildArgs(argsPtr, configPtr);
        return op_.GetWorkspaceSize(this->args_);
    }

    virtual Status Initialize(
        void *argsPtr,
        void *configPtr,
        uint8_t *workspace,
        aclrtStream stream
    ) override
    {
        BuildArgs(argsPtr, configPtr);
        return op_.Initialize(this->args_, workspace, stream);
    }

    virtual Status Run(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr) override
    {
        return op_.Run(stream, blockDim, fftsAddr);
    }

protected:
    virtual void BuildArgs(void *argsPtr, void *configPtr) = 0;

    Description_ description_;
    OperatorArguments args_{};
    Operator op_;
};

/********************* basic matmul *********************/
template <typename Operator_>
class BasicMatmulGemmOperation : public GemmOperationBase<Operator_, GemmOperationDescription> {
public:
    BasicMatmulGemmOperation(char const *name = "") : GemmOperationBase<Operator_, GemmOperationDescription>(name)
    {
        this->description_.gemmKind = GemmKind::BasicMatmul;
    }

private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        BasicMatmulGemmArguments *arguments = (BasicMatmulGemmArguments *)argsPtr;
        BasicMatmulGemmConfiguration *config = (BasicMatmulGemmConfiguration *)configPtr;
        this->args_.problemShape = GemmCoord{config->m, config->n, config->k};
        this->args_.ptrA = arguments->A;
        this->args_.ptrB = arguments->B;
        this->args_.ptrC = arguments->C;
    }
};
/********************* basic matmul end *********************/

/********************* grouped matmul *********************/
template <typename Operator_>
class GroupedMatmulGemmOperation : public GemmOperationBase<Operator_, GemmOperationDescription> {
public:
    GroupedMatmulGemmOperation(char const *name = "") : GemmOperationBase<Operator_, GemmOperationDescription>(name)
    {
        this->description_.gemmKind = GemmKind::GroupedMatmul;
    }

private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        GroupedMatmulGemmArguments *arguments = (GroupedMatmulGemmArguments *)argsPtr;
        GroupedMatmulGemmConfiguration *config = (GroupedMatmulGemmConfiguration *)configPtr;

        this->args_.problemCount = config->groupCount;
        this->args_.ptrProblemShape = arguments->problemShapeList;
        this->args_.ptrA = arguments->A;
        this->args_.ptrLayoutA = arguments->layoutAList;
        this->args_.ptrB = arguments->B;
        this->args_.ptrLayoutB = arguments->layoutBList;
        this->args_.ptrC = arguments->C;
        this->args_.ptrLayoutC = arguments->layoutCList;
    }
};
/********************* grouped matmul end *********************/

/********************* grouped matmul slice M *********************/
template <typename Operator_>
class GroupedMatmulSliceMGemmOperation : public GemmOperationBase<Operator_, GemmOperationDescription> {
public:
    GroupedMatmulSliceMGemmOperation(char const *name = "") : GemmOperationBase<Operator_, GemmOperationDescription>(name)
    {
        this->description_.gemmKind = GemmKind::GroupedMatmulSliceM;
    }

private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        GroupedMatmulSliceMGemmArguments *arguments = (GroupedMatmulSliceMGemmArguments *)argsPtr;
        GroupedMatmulSliceMGemmConfiguration *config = (GroupedMatmulSliceMGemmConfiguration *)configPtr;

        this->args_.problemShape = GemmCoord{config->m, config->n, config->k};
        this->args_.problemCount = config->groupCount;
        this->args_.ptrGroupList = arguments->deviceGroupList;
        this->args_.ptrA = arguments->A;
        this->args_.ptrB = arguments->B;
        this->args_.ptrC = arguments->C;
    }
};
/********************* grouped matmul slice M end *********************/

/********************* optimized matmul *********************/
template <typename Operator_>
class OptimizedMatmulGemmOperation : public GemmOperationBase<Operator_, GemmOperationDescription> {
    using Operator = Operator_;
    using OperatorKernel = typename Operator::Kernel;
    using ElementA = typename OperatorKernel::ElementA;
    using ElementB = typename OperatorKernel::ElementB;
    using LayoutA = typename OperatorKernel::LayoutA;
    using LayoutB = typename OperatorKernel::LayoutB;
    using L1TileShape = typename OperatorKernel::BlockMmad::L1TileShape;
public:
    OptimizedMatmulGemmOperation(char const *name = "") : GemmOperationBase<Operator_, GemmOperationDescription>(name)
    {
        this->description_.gemmKind = GemmKind::OptimizedMatmul;
    }

private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        BasicMatmulGemmArguments *arguments = (BasicMatmulGemmArguments *)argsPtr;
        BasicMatmulGemmConfiguration *config = (BasicMatmulGemmConfiguration *)configPtr;

        constexpr uint32_t alignByByte = 512;

        this->args_.problemShape = GemmCoord{config->m, config->n, config->k};
        this->args_.ptrA = arguments->A;
        this->args_.ptrB = arguments->B;
        this->args_.ptrC = arguments->C;

        constexpr uint32_t alignByElement = 512 / sizeof(half);

        if constexpr (!std::is_same<typename OperatorKernel::LayoutWA, typename OperatorKernel::LayoutA>::value) {
            isThisKernelPaddingA_ = true;
        }
        if constexpr (!std::is_same<typename OperatorKernel::LayoutWB, typename OperatorKernel::LayoutB>::value) {
            isThisKernelPaddingB_ = true;
        }

        LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(config->m, config->k);
        isNeedPaddingA_ = IsNeedPadding(layoutA, alignByElement);
        LayoutA layoutB = LayoutB::template MakeLayout<ElementB>(config->k, config->n);
        isNeedPaddingB_ = IsNeedPadding(layoutB, alignByElement);
    }

    virtual Status CanImplement(void *argsPtr, void *configPtr) override
    {
        // check the operator's padding configuration is correct under the given problem shape
        BuildArgs(argsPtr, configPtr);

        if ((isThisKernelPaddingA_ != isNeedPaddingA_) || (isThisKernelPaddingB_ != isNeedPaddingB_)) {
            return Catlass::Status::kInvalid;
        }

        return this->op_.CanImplement(this->args_);
    }

    inline bool IsNeedPadding(Catlass::layout::RowMajor layout, uint32_t align)
    {
        // If the stride is greater than 65536, padding is required to reduce the stride.
        if (layout.stride(0) < 65536) {
            return layout.stride(0) % align != 0;
        } else {
            return true;
        }
    }

    inline bool IsNeedPadding(Catlass::layout::ColumnMajor layout, uint32_t align)
    {
        // If the stride is greater than 65536, padding is required to reduce the stride.
        if (layout.stride(1) < 65536) {
            return layout.stride(1) % align != 0;
        } else {
            return true;
        }
    }

    inline bool IsNeedPadding(Catlass::layout::zN layout, uint32_t align) {
        return false;
    }

    inline bool IsNeedPadding(Catlass::layout::nZ layout, uint32_t align) {
        return false;
    }

    bool isThisKernelPaddingA_{false};
    bool isThisKernelPaddingB_{false};
    bool isNeedPaddingA_{false};
    bool isNeedPaddingB_{false};
};
/********************* optimized matmul end *********************/


/********************* quant matmul *********************/
template <typename Operator_>
class QuantMatmulGemmOperation : public GemmOperationBase<Operator_, QuantMatmulGemmOperationDescription> {
public:
    using Operator = Operator_;
    using OperatorArguments = typename Operator::Arguments;
    using OperatorKernel = typename Operator::Kernel;

    using ElementD = typename OperatorKernel::ElementD;
    using ElementScale = typename OperatorKernel::ElementScale;
    using ElementPerTokenScale = typename OperatorKernel::ElementPerTokenScale;
    using LayoutD = typename OperatorKernel::LayoutD;
    using LayoutScale = typename OperatorKernel::LayoutScale;
    using LayoutPerTokenScale = typename OperatorKernel::LayoutPerTokenScale;
    QuantMatmulGemmOperation(char const *name = "") : GemmOperationBase<Operator_, QuantMatmulGemmOperationDescription>(name)
    {
        this->description_.gemmKind = GemmKind::QuantMatmul;
        this->description_.D = MakeTensorDescription<ElementD, LayoutD>();
        this->description_.Scale = MakeTensorDescription<ElementScale, LayoutScale>();
        this->description_.PerTokenScale = MakeTensorDescription<ElementPerTokenScale, LayoutPerTokenScale>();
    }
    
private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        QuantMatmulGemmArguments *arguments = (QuantMatmulGemmArguments *)argsPtr;
        QuantMatmulGemmConfiguration *config = (QuantMatmulGemmConfiguration *)configPtr;

        this->args_.problemShape = arguments->problemShape;
        this->args_.aicCoreNum = arguments->aicCoreNum;
        this->args_.ptrA = arguments->ptrA;
        this->args_.ptrB = arguments->ptrB;
        this->args_.ptrD = arguments->ptrD;
        this->args_.ptrScale = arguments->ptrScale;
        this->args_.ptrPerTokenScale = arguments->ptrPerTokenScale;
    }
private:
};
/********************* quant matmul end *********************/

/********************* matmul gelu *********************/
template <typename Operator_> 
class MatmulGeluGemmOperation : public GemmOperationBase<Operator_, MatmulGeluGemmOperationDescription> {
public:
    using Operator = Operator_;
    using OperatorArguments = typename Operator::Arguments;
    using OperatorKernel = typename Operator::Kernel;

    using ElementD = typename OperatorKernel::ElementD;
    using LayoutD = typename OperatorKernel::LayoutD;

    MatmulGeluGemmOperation(char const *name = "") : GemmOperationBase<Operator_, MatmulGeluGemmOperationDescription>(name)
    {
        this->description_.gemmKind = GemmKind::MatmulGelu;
        this->description_.D = MakeTensorDescription<ElementD, LayoutD>();
    }

private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        MatmulGeluGemmArguments *arguments = (MatmulGeluGemmArguments *)argsPtr;
        MatmulGeluGemmConfiguration *config = (MatmulGeluGemmConfiguration *)configPtr;

        this->args_.problemShape = GemmCoord{config->m, config->n, config->k};
        this->args_.elementSize = config->elementSize;
        this->args_.ptrA = arguments->ptrA;
        this->args_.ptrB = arguments->ptrB;
        this->args_.ptrD = arguments->ptrD;
    }
};
/********************* matmul gelu end *********************/
}
}

#endif // CATLASS_LIBRARY_GEMM_OPERATION_H
