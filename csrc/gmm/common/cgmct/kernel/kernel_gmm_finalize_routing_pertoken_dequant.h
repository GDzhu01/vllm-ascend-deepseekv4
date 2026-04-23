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
 * \file kernel_gmm_finalize_routing_pertoken_dequant.h
 * \brief
 */

#ifndef KERNEL_GMM_FINALIZE_ROUTING_PERTOKEN_DEQUANT_H
#define KERNEL_GMM_FINALIZE_ROUTING_PERTOKEN_DEQUANT_H
#include "kernel_basic_intf.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/status_utils.h"

#include "../block/block_mmad_builder.h"
#include "../epilogue/block_epilogue_dequant_finalize_routing.h"
#include "../prologue/block_prologue_finalize_routing.h"
#include "./semaphore.h"

#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_gmm_aswt_with_tail_split.h"

namespace Cgmct {
namespace Gemm {
namespace Kernel {

namespace {
constexpr uint64_t IDX_A_OFFSETS = 0UL;
constexpr uint64_t IDX_B_OFFSETS = 1UL;
constexpr uint64_t IDX_X1SCALE_OFFSETS = 2UL;
constexpr uint64_t IDX_X2SCALE_OFFSETS = 3UL;
constexpr uint64_t IDX_BIAS_OFFSETS = 4UL;
constexpr uint64_t IDX_C_OFFSETS = 5UL;
constexpr uint64_t IDX_LOGIT_OFFSETS = 6UL;
constexpr uint64_t IDX_M_TILEIDXS = 0UL;
constexpr uint64_t IDX_N_TILEIDXS = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDXS = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDXS = 3UL;
constexpr uint8_t SYNC_AIC_AIV_MODES = 4;
constexpr uint16_t FLAG_ID_MAXS = 16;
constexpr uint16_t AIC_SYNC_AIV_FLAGS = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAGS = 6;
constexpr uint32_t WEIGHT_TILE_K_SMALL = 16;
constexpr uint32_t WEIGHT_TILE_N_SMALL = 16;
constexpr uint32_t WEIGHT_TILE_K_LARGE = 32;
constexpr uint32_t WEIGHT_TILE_N_LARGE = 32;
constexpr uint32_t WEIGHT_TILE_CAPACITY = 512;
} // namespace

using namespace AscendC;
template <class ProblemShape_, class BlockMmadBuilder_, class BlockPrologue_,
          class BlockEpilogueDequantFinalizeRouting_, class BlockScheduler_, typename Enable_ = void>
class KernelGmmFinalizeRoutingPertokenDequant {
    static_assert(AscendC::Std::always_false_v<BlockScheduler_>,
                  "KernelGmmFinalizeRoutingPertokenDequant is not implemented for this scheduler");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockPrologue_,
          class BlockEpilogueDequantFinalizeRouting_, class BlockScheduler_>
class KernelGmmFinalizeRoutingPertokenDequant<
    ProblemShape_, BlockMmadBuilder_, BlockPrologue_, BlockEpilogueDequantFinalizeRouting_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockScheduler_, GroupedMatmulAswtWithTailSplitScheduler>>> {
public:
    __aicore__ inline KernelGmmFinalizeRoutingPertokenDequant()
    {
    }

    __aicore__ inline ~KernelGmmFinalizeRoutingPertokenDequant()
    {
    }

    using BlockPrologue = BlockPrologue_;
    using BlockEpilogueDequantFinalizeRouting = BlockEpilogueDequantFinalizeRouting_;
    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    static constexpr int64_t l1M = BlockMmadBuilder::l1M;
    static constexpr int64_t l1N = BlockMmadBuilder::l1N;
    static constexpr int64_t l1K = BlockMmadBuilder::l1K;
    static constexpr auto formatB = BlockMmadBuilder::formatB;
    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmadBuilder::L1TileShape,
                                               typename BlockMmadBuilder::L0TileShape, BlockScheduler, transA,
                                               transB>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = typename BlockMmadBuilder::BlockMmadOp;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockPrologueArguments = typename BlockPrologue::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockPrologueParams = typename BlockPrologue::Params;
    using BlockEpilogueParams = typename BlockEpilogueDequantFinalizeRouting::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // a, b, x1scale, x2scale, bias, y, logit
    using BaseOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // a, b, x1scale, x2scale, bias, y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // coordinate
    using CoordClass =
        Coordinate<transA, transB, BlockMmadBuilder::formatA, BlockMmadBuilder::formatB, BlockMmadBuilder::formatC>;

    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    // shape
    TupleShape problemShape_{};
    BaseOffset baseOffset_{0, 0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};
    uint64_t preOffset_ = 0;
    BlockMmadOp mmadOp_;

    BlockPrologue prologueOp_;
    BlockEpilogueDequantFinalizeRouting epilogueDequantOp_;
    AscendC::LocalTensor<CType> l0cOutUb_;
    bool isVecSetSyncCom_ = false;

    struct GMMTiling {
        uint32_t groupNum;
        uint8_t groupListType;
        int32_t baseM;
        int32_t baseN;
        int32_t baseK;
        uint8_t hasBias;
        const TCubeTiling *__restrict matmulTiling;
        __aicore__ GMMTiling()
        {
        }
        __aicore__ GMMTiling(uint32_t groupNum_, uint8_t groupListType_, int32_t baseM_, int32_t baseN_, int32_t baseK_,
                             uint8_t hasBias_)
            : groupNum(groupNum_), groupListType(groupListType_), baseM(baseM_), baseN(baseN_), baseK(baseK_),
              hasBias(hasBias_)
        {
        }
    };

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockPrologueArguments prologueArgs;
        BlockEpilogueParams epilogueArgs;
        GMMTiling gmmArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockPrologueParams prologueParams;
        BlockEpilogueParams epilogueParams;
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
        return splitValue;
    }

    __aicore__ inline void UpdateGlobalBuffer(const Params &params)
    {
        if ASCEND_IS_AIC {
            aGlobal_.SetGlobalBuffer((__gm__ AType *)params.mmadParams.aGmAddr + Get<IDX_A_OFFSETS>(baseOffset_));
            bGlobal_.SetGlobalBuffer((__gm__ BType *)params.mmadParams.bGmAddr + Get<IDX_B_OFFSETS>(baseOffset_));
        }
        if ASCEND_IS_AIV {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                0L,
                Get<IDX_BIAS_OFFSETS>(baseOffset_),
                Get<IDX_X2SCALE_OFFSETS>(baseOffset_),
                Get<IDX_X1SCALE_OFFSETS>(baseOffset_),
                Get<IDX_LOGIT_OFFSETS>(baseOffset_),
                Get<IDX_LOGIT_OFFSETS>(baseOffset_)};
            epilogueDequantOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    }

    __aicore__ inline void UpdateOffset(uint32_t groupIdx)
    {
        // baseOffset is 0 when groupIdx = 0
        if (groupIdx == 0) {
            return;
        }
        uint64_t m = Get<MNK_M>(problemShape_);
        uint64_t n = Get<MNK_N>(problemShape_);
        uint64_t k = Get<MNK_K>(problemShape_);
        // aBaseOffset += m * k
        Get<IDX_A_OFFSETS>(baseOffset_) = Get<IDX_A_OFFSETS>(baseOffset_) + m * k;
        // bBaseOffset += n * k
        if constexpr (formatB == CubeFormat::NZ) {
            Get<IDX_B_OFFSETS>(baseOffset_) = Get<IDX_B_OFFSETS>(baseOffset_) +
                                              CeilDiv(n, WEIGHT_TILE_N_LARGE) * CeilDiv(k, WEIGHT_TILE_K_SMALL) * WEIGHT_TILE_CAPACITY;
        } else if constexpr (formatB == CubeFormat::ZN) {
            Get<IDX_B_OFFSETS>(baseOffset_) = Get<IDX_B_OFFSETS>(baseOffset_) +
                                              CeilDiv(k, WEIGHT_TILE_K_LARGE) * CeilDiv(n, WEIGHT_TILE_N_SMALL) * WEIGHT_TILE_CAPACITY;
        } else {
            Get<IDX_B_OFFSETS>(baseOffset_) = Get<IDX_B_OFFSETS>(baseOffset_) + n * k;
        }
        Get<IDX_BIAS_OFFSETS>(baseOffset_) += n;
        // K-C
        // scaleAAxisBaseOffset = m
        Get<IDX_X1SCALE_OFFSETS>(baseOffset_) += m;
        // scaleBAxisBaseOffset = gi * n
        Get<IDX_X2SCALE_OFFSETS>(baseOffset_) = groupIdx * n;
        Get<IDX_LOGIT_OFFSETS>(baseOffset_) += m;
    }

    __aicore__ inline bool UpdateGroupParams(const Params &params, uint32_t groupIdx)
    {
        UpdateOffset(groupIdx);
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, params.gmmParams.groupListType);
        Get<MNK_M>(problemShape_) = splitValue;
        // split_m, when m=0, skip
        if (Get<MNK_M>(problemShape_) == 0) {
            return false;
        }
        return true;
    }

    __aicore__ inline void InitParamsAndTensor(const Params &params)
    {
        Get<MNK_N>(problemShape_) = params.gmmParams.matmulTiling->N;
        Get<MNK_K>(problemShape_) = params.gmmParams.matmulTiling->Ka;
        groupListGm_.SetGlobalBuffer((__gm__ int64_t *)params.mmadParams.groupListGmAddr);
    }

    __aicore__ inline void ProcessSingleGroup(const Params &params, BlockSchedulerOp &bs, uint32_t groupIdx)
    {
        int64_t m = Get<MNK_M>(problemShape_);
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);
        TupleShape resProblemShape{Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_), 0};
        bs.UpdateNextProblem(resProblemShape);
        epilogueDequantOp_.UpdateNextProblem(resProblemShape);
        UpdateGlobalBuffer(params);
        CoordClass coord(m, n, k, params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        BlockCoord tileIdx;
        while (bs.GetTileIdx(tileIdx)) {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            blockOffset_ = coord.template GetQuantOffset<GroupedMatmul::QuantMode::PERTOKEN_MODE>(
                Get<IDX_M_TILEIDXS>(tileIdx), Get<IDX_N_TILEIDXS>(tileIdx), Get<IDX_M_TAIL_SPLIT_TILEIDXS>(singleShape),
                Get<IDX_N_TAIL_SPLIT_TILEIDXS>(singleShape));
            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_) {
                    WaitForVector();
                }
                AscendC::Std::tuple<int32_t, int32_t, int32_t> mmSingleShape{Get<MNK_M>(singleShape),
                                                                             Get<MNK_N>(singleShape), k};
                mmadOp_(aGlobal_[Get<IDX_A_OFFSETS>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSETS>(blockOffset_)],
                        l0cOutUb_, mmSingleShape, transA, transB);
                NotifyVector();
            }
            isVecSetSyncCom_ = true;
            if ASCEND_IS_AIV {
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<MNK_M>(singleShape),
                                                                                      Get<MNK_N>(singleShape), 0, 0};
                int64_t y = Get<IDX_C_OFFSETS>(blockOffset_);
                int64_t mOffset = y / n;
                int64_t nOffset = y - mOffset * n;
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    nOffset,
                    Get<IDX_BIAS_OFFSETS>(blockOffset_),
                    Get<IDX_X2SCALE_OFFSETS>(blockOffset_),
                    Get<IDX_X1SCALE_OFFSETS>(blockOffset_),
                    mOffset,
                    mOffset};
                WaitForCube();
                epilogueDequantOp_(epilogueShape, epilogueOffset);
                NotifyCube();
            }
        }
    }

    __aicore__ inline void operator()(const Params &params)
    {
        if ASCEND_IS_AIV {
            prologueOp_.Init(params.prologueParams);
            prologueOp_();
        }
        if ASCEND_IS_AIC {
            mmadOp_.Init(const_cast<TCubeTiling *__restrict>(params.gmmParams.matmulTiling), GetTPipePtr());
            l0cOutUb_ = epilogueDequantOp_.GetL0c2UbTensor();
        }
        InitParamsAndTensor(params);
        BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        SyncAll<false>();
        if ASCEND_IS_AIV {
            epilogueDequantOp_.Init(params.epilogueParams);
        }
        uint32_t groupNum = params.gmmParams.groupNum;
        if constexpr (formatB == CubeFormat::ZN) {
            bs.SetTailAlign(1, MATMUL_MNK_ALIGN);
        } else if constexpr (formatB == CubeFormat::NZ) {
            bs.SetTailAlign(1, MATMUL_MNK_ALIGN_INT8);
        }
        for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;
            }
            ProcessSingleGroup(params, bs, groupIdx);
        }
        End();
    }
};
} // namespace Kernel
} // namespace Gemm
} // namespace Cgmct
#endif