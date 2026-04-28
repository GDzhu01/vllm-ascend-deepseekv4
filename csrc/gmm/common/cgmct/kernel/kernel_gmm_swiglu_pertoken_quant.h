/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_gmm_swiglu_pertoken_quant.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_GMM_SWIGLU_PERTOKEN_QUANT_H
#define MATMUL_KERNEL_KERNEL_GMM_SWIGLU_PERTOKEN_QUANT_H
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "./semaphore.h"

#include "../block/block_mmad_builder.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "../epilogue/block_epilogue_dequant_swiglu.h"
#include "../epilogue/block_epilogue_pertoken_quant.h"

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/status_utils.h"

namespace Cgmct {
namespace Gemm {
namespace Kernel {

static constexpr uint64_t M_VALUES = 0UL;
static constexpr uint64_t N_VALUES = 1UL;
static constexpr uint64_t K_VALUES = 2UL;
static constexpr uint64_t INDEX_A_OFFSET = 0UL;
static constexpr uint64_t INDEX_B_OFFSET = 1UL;
static constexpr uint64_t INDEX_X1SCALE_OFFSET = 2UL;
static constexpr uint64_t INDEX_X2SCALE_OFFSET = 3UL;
static constexpr uint64_t INDEX_BIAS_OFFSET = 4UL;
static constexpr uint64_t INDEX_C_OFFSET = 5UL;
static constexpr uint64_t INDEX_C_SCALE_OFFSET = 6UL;
static constexpr uint64_t INDEX_M_TILEIDX = 0UL;
static constexpr uint64_t INDEX_N_TILEIDX = 1UL;
static constexpr uint64_t INDEX_M_TAIL_SPLIT_TILEIDX = 2UL;
static constexpr uint64_t INDEX_N_TAIL_SPLIT_TILEIDX = 3UL;
static constexpr uint8_t SYNC_AIC_AIV_MODES = 4;
static constexpr uint16_t FLAG_ID_MAXS = 16;
static constexpr uint16_t AIC_SYNC_AIV_FLAGS = 4;
static constexpr uint16_t AIV_SYNC_AIC_FLAGS = 6;

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogueDequantAndSwiglu_,
          class BlockEpiloguePertokenQuant_, class BlockScheduler_, typename DataTypeX2Scale_,
          typename DataTypeX1Scale_, typename Enable_ = void>
class KernelGmmSwiGluPertokenQuant {
    static_assert(AscendC::Std::always_false_v<BlockScheduler_>,
                  "KernelGmmSwiGluPertokenQuant is not implemented for this scheduler");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogueDequantAndSwiglu_,
          class BlockEpiloguePertokenQuant_, class BlockScheduler_, typename DataTypeX2Scale_,
          typename DataTypeX1Scale_>
class KernelGmmSwiGluPertokenQuant<
    ProblemShape_, BlockMmadBuilder_, BlockEpilogueDequantAndSwiglu_, BlockEpiloguePertokenQuant_, BlockScheduler_,
    DataTypeX2Scale_, DataTypeX1Scale_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockScheduler_, GroupedMatmulAswtWithTailSplitScheduler>>> {
private:
    TPipe *pPipe_ = nullptr;

public:
    __aicore__ inline KernelGmmSwiGluPertokenQuant(TPipe *pipe) : pPipe_(pipe), epiloguePertokenQuantOp_(pipe)
    {
    }

    __aicore__ inline ~KernelGmmSwiGluPertokenQuant()
    {
    }

    using BlockEpilogueDequantAndSwiglu = BlockEpilogueDequantAndSwiglu_;
    using BlockEpiloguePertokenQuant = BlockEpiloguePertokenQuant_;
    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using DataTypeX2Scale = DataTypeX2Scale_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    static constexpr int64_t l1M = BlockMmadBuilder::l1M;
    static constexpr int64_t l1N = BlockMmadBuilder::l1N;
    static constexpr int64_t l1K = BlockMmadBuilder::l1K;
    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmadBuilder::L1TileShape,
                                               typename BlockMmadBuilder::L0TileShape, BlockScheduler, transA,
                                               transB>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = typename BlockMmadBuilder::BlockMmadOp;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockEpilogueDequantAndSwigluArguments = typename BlockEpilogueDequantAndSwiglu::Arguments;
    using BlockEpilogueQuantArguments = typename BlockEpiloguePertokenQuant::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockEpilogueDequantAndSwigluParams = typename BlockEpilogueDequantAndSwiglu::Params;
    using BlockEpilogueQuantParams = typename BlockEpiloguePertokenQuant::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // a, b, x1scale, x2scale, bias, y, yscale
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // coordinate
    using CoordClass =
        Coordinate<transA, transB, BlockMmadBuilder::formatA, BlockMmadBuilder::formatB, BlockMmadBuilder::formatC>;

    // attribute
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    // shape
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0, 0};
    uint64_t preOffset_ = 0UL;
    uint32_t realM_ = 0UL;
    BlockMmadOp mmadOp_;
    BlockEpiloguePertokenQuant epiloguePertokenQuantOp_;
    BlockEpilogueDequantAndSwiglu epilogueDequantAndSwigluOp_;
    AscendC::LocalTensor<CType> l0cOutUbFirst_;
    AscendC::LocalTensor<CType> l0cOutUbSecond_;
    bool isVecSetSyncCom_ = false;

    struct GMMTiling {
        uint32_t groupNum;
        uint8_t groupListType;
        int32_t baseM;
        int32_t baseN;
        int32_t baseK;
        const TCubeTiling *__restrict matmulTiling;
        __aicore__ GMMTiling()
        {
        }
        __aicore__ GMMTiling(uint32_t groupNum_, uint8_t groupListType_, int32_t baseM_, int32_t baseN_, int32_t baseK_)
            : groupNum(groupNum_), groupListType(groupListType_), baseM(baseM_), baseN(baseN_), baseK(baseK_)
        {
        }
    };

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockEpilogueDequantAndSwigluArguments epilogueDequantSwigluArgs;
        BlockEpilogueQuantArguments epilogueQuantArgs;
        GMMTiling gmmArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueDequantAndSwigluParams epilogueDequantSwigluParams;
        BlockEpilogueQuantParams epilogueQuantParams;
        GMMTiling gmmParams;
        Params() = default;
    };

    __aicore__ inline void NotifyCube()
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODES, PIPE_V>(AIV_SYNC_AIC_FLAGS);
    }
    __aicore__ inline void WaitForVector()
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODES, PIPE_FIX>(AIV_SYNC_AIC_FLAGS);
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODES, PIPE_FIX>(AIV_SYNC_AIC_FLAGS + FLAG_ID_MAXS);
    }
    __aicore__ inline void NotifyVector()
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODES, PIPE_FIX>(AIC_SYNC_AIV_FLAGS);
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODES, PIPE_FIX>(AIC_SYNC_AIV_FLAGS + FLAG_ID_MAXS);
    }
    __aicore__ inline void WaitForCube()
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODES, PIPE_V>(AIC_SYNC_AIV_FLAGS);
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {
                WaitForVector();
            }
        }
    }

    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, uint8_t groupListType)
    {
        int32_t splitValue = 0;
        if (groupListType == 0) {
            int32_t offset = static_cast<int32_t>(groupListGm_.GetValue(groupIdx));
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
        }
        realM_ += splitValue;
        return splitValue;
    }

    __aicore__ inline void UpdateGlobalBuffer(const Params &params)
    {
        if ASCEND_IS_AIC {
            aGlobal_.SetGlobalBuffer((__gm__ AType *)params.mmadParams.aGmAddr + Get<INDEX_A_OFFSET>(baseOffset_));
            bGlobal_.SetGlobalBuffer(GetTensorAddr<BType>(0, params.mmadParams.bGmAddr) +
                                     Get<INDEX_B_OFFSET>(baseOffset_));
        }
        if ASCEND_IS_AIV {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<INDEX_C_OFFSET>(baseOffset_), Get<INDEX_C_SCALE_OFFSET>(baseOffset_),
                Get<INDEX_X2SCALE_OFFSET>(baseOffset_), Get<INDEX_X1SCALE_OFFSET>(baseOffset_), 0L};
            epilogueDequantAndSwigluOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    }

    __aicore__ inline void UpdateOffset(uint32_t groupIdx)
    {
        // baseOffset is 0 when groupIdx = 0
        if (groupIdx == 0) {
            return;
        }
        uint64_t m = Get<M_VALUES>(problemShape_);
        uint64_t n = Get<N_VALUES>(problemShape_);
        uint64_t k = Get<K_VALUES>(problemShape_);
        // aBaseOffset += m * k
        Get<INDEX_A_OFFSET>(baseOffset_) = Get<INDEX_A_OFFSET>(baseOffset_) + m * k;
        // bBaseOffset += n * k
        Get<INDEX_B_OFFSET>(baseOffset_) = Get<INDEX_B_OFFSET>(baseOffset_) + n * k;
        // K-C
        // scaleAAxisBaseOffset = m
        Get<INDEX_X1SCALE_OFFSET>(baseOffset_) += m;
        // scaleBAxisBaseOffset = gi * n
        Get<INDEX_X2SCALE_OFFSET>(baseOffset_) = groupIdx * n;
        // yBaseOffset += (m * n) >> 1
        Get<INDEX_C_OFFSET>(baseOffset_) += (m * n) >> 1;
    }

    __aicore__ inline bool UpdateGroupParams(const Params &params, uint32_t groupIdx)
    {
        UpdateOffset(groupIdx);
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, params.gmmParams.groupListType);
        Get<M_VALUES>(problemShape_) = splitValue;
        // split_m，when m=0, skip
        if (Get<M_VALUES>(problemShape_) == 0) {
            return false;
        }
        return true;
    }

    __aicore__ inline void InitParamsAndTensor(const Params &params)
    {
        Get<N_VALUES>(problemShape_) = params.gmmParams.matmulTiling->N;
        Get<K_VALUES>(problemShape_) = params.gmmParams.matmulTiling->Ka;
        groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(params.mmadParams.groupListGmAddr));
    }

    __aicore__ inline void ComputeOffset(int64_t &bRightOffset, int64_t n, int64_t k)
    {
        int64_t resN = n >> 1; // 2: glu, n->n/2
        if constexpr (transB) {
            bRightOffset += resN * k;
        } else {
            bRightOffset += resN;
        }
    }

    __aicore__ inline void ProcessSingleGroup(const Params &params, BlockSchedulerOp &bs, uint32_t groupIdx)
    {
        int64_t m = Get<M_VALUES>(problemShape_);
        int64_t n = Get<N_VALUES>(problemShape_);
        int64_t k = Get<K_VALUES>(problemShape_);
        TupleShape resProblemShape{Get<M_VALUES>(problemShape_), Get<N_VALUES>(problemShape_) >> 1,
                                   Get<K_VALUES>(problemShape_), 0};
        bs.UpdateNextProblem(resProblemShape);
        epilogueDequantAndSwigluOp_.UpdateNextProblem(resProblemShape);
        UpdateGlobalBuffer(params);
        CoordClass coord(m, n, k, params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        BlockCoord tileIdx;
        while (bs.GetTileIdx(tileIdx)) {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            // isMx = true
            blockOffset_ = coord.template GetQuantIOOffset<GroupedMatmul::QuantMode::PERTOKEN_MODE>(
                Get<INDEX_M_TILEIDX>(tileIdx), Get<INDEX_N_TILEIDX>(tileIdx),
                Get<INDEX_M_TAIL_SPLIT_TILEIDX>(singleShape), Get<INDEX_N_TAIL_SPLIT_TILEIDX>(singleShape));
            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_) {
                    WaitForVector();
                }
                AscendC::Std::tuple<int32_t, int32_t, int32_t> mmSingleShape{Get<M_VALUES>(singleShape),
                                                                             Get<N_VALUES>(singleShape), k};
                // left block
                mmadOp_(aGlobal_[Get<INDEX_A_OFFSET>(blockOffset_)], bGlobal_[Get<INDEX_B_OFFSET>(blockOffset_)],
                        l0cOutUbFirst_, mmSingleShape, transA, transB);
                // right block
                int64_t bRightOffset = Get<INDEX_B_OFFSET>(blockOffset_);
                ComputeOffset(bRightOffset, n, k);
                mmadOp_(aGlobal_[Get<INDEX_A_OFFSET>(blockOffset_)], bGlobal_[bRightOffset], l0cOutUbSecond_,
                        mmSingleShape, transA, transB);
                NotifyVector();
            }
            isVecSetSyncCom_ = true;
            if ASCEND_IS_AIV {
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUES>(singleShape),
                                                                                      Get<N_VALUES>(singleShape), 0, 0};
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    Get<INDEX_C_OFFSET>(blockOffset_),       Get<INDEX_C_SCALE_OFFSET>(blockOffset_),
                    Get<INDEX_X2SCALE_OFFSET>(blockOffset_), Get<INDEX_X1SCALE_OFFSET>(blockOffset_),
                    Get<INDEX_BIAS_OFFSET>(blockOffset_),
                };
                WaitForCube();
                epilogueDequantAndSwigluOp_(epilogueShape, epilogueOffset);
                NotifyCube();
            }
        }
    }

    __aicore__ inline void operator()(const Params &params)
    {
        if ASCEND_IS_AIC {
            mmadOp_.Init(const_cast<TCubeTiling *__restrict>(params.gmmParams.matmulTiling), GetTPipePtr());
            l0cOutUbFirst_ = epilogueDequantAndSwigluOp_.GetFirstL0c2UbTensor();
            l0cOutUbSecond_ = epilogueDequantAndSwigluOp_.GetSecondL0c2UbTensor();
        }
        InitParamsAndTensor(params);
        BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        if ASCEND_IS_AIV {
            epilogueDequantAndSwigluOp_.Init(params.epilogueDequantSwigluParams);
        }
        uint32_t groupNum = params.gmmParams.groupNum;
        for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;
            }
            ProcessSingleGroup(params, bs, groupIdx);
        }
        End();
        SyncAll();
        if ASCEND_IS_AIV {
            epiloguePertokenQuantOp_.Init(params.epilogueQuantParams);
            epiloguePertokenQuantOp_(realM_);
        }
    }
};
} // namespace Kernel
} // namespace Gemm
} // namespace Cgmct
#endif // MATMUL_KERNEL_KERNEL_GMM_SWIGLU_PERTOKEN_QUANT_H