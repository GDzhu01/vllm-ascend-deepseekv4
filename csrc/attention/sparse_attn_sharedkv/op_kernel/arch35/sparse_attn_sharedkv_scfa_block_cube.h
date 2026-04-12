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
 * \file sparse_attn_sharedkv_scfa_block_cube.h
 * \brief
 */
#ifndef SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H_
#define SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H_
#include "common/offset_calculator.h"
#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"
#include "kernel_operator_list_tensor_intf.h"

#include "util_regbase.h" // todo：以下三个.h需要对应修改
#include "infer_flash_attention_comm.h"
#include "flash_attention_score_common_regbase.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {
template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat() {
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

TEMPLATES_DEF
class FABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64; // todo: FA中来自模板参数 先写固定值，后期根据SCFA情况修改
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dVBaseSize = 512;
    static constexpr bool bmm2Write2Ub = true; // 必为true
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub};
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>; // 必在UB

    __aicore__ inline FABlockCube() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr, __gm__ uint8_t *query);
    __aicore__ inline void InitCubeInput(CVSharedParams<isInfer, isPa> *sharedParams, __gm__ int64_t *actualSeqQlenAddr);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);

private:
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitGmTensor(CVSharedParams<isInfer, isPa> *sharedParams, __gm__ int64_t *actualSeqQlenAddr);
    __aicore__ inline void CalcS1Coord(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void IterateBmm1SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);
    TPipe *tPipe;
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    FaGmTensor<INPUT_T, Q_FORMAT> queryGm;

    /* =====================运行时变量==================== */
    CubeCoordInfo coordInfo[3];

    /* =====================LocalBuffer变量==================== */
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    // D小于等于256 mm1左矩阵Q，GS1循环内左矩阵复用, GS1循环间开pingpong；D大于256使用单块Buffer，S1循环间驻留；fp32场景单块不驻留
    BuffersPolicySingleBuffer<BufferType::L1> l1QBuffers;

    // mm1右矩阵K
    BuffersPolicy3buff<BufferType::L1> l1KBuffers;

    // L0A
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    // L0B
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;
    // L0C
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        l1BufferManagerPtr = l1BuffMgr;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)query);
        InitLocalBuffer();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::InitCubeInput(CVSharedParams<isInfer, isPa> *sharedParams, __gm__ int64_t *actualSeqQlenAddr)
{
    if ASCEND_IS_AIC {
        InitGmTensor(sharedParams, actualSeqQlenAddr);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::InitLocalBuffer() {
    constexpr uint32_t mm1LeftSize = s1BaseSize * dBaseSize * sizeof(INPUT_T); // 64 * 512
    constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(INPUT_T);// 512 * 128
    l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
    l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);

    // L0A B C 当前写死，能否通过基础api获取
    l0aBufferManager.Init(tPipe, 65536);  // 64 * 1024
    l0bBufferManager.Init(tPipe, 65536);  // 64 * 1024
    l0cBufferManager.Init(tPipe, 262144); // 256 * 1024

    mmL0ABuffers.Init(l0aBufferManager, 16 * 1024);  // db类型，填入数值是总大小的一半
    mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
    mmL0CBuffers.Init(l0cBufferManager, 128 * 1024);
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::InitGmTensor(CVSharedParams<isInfer, isPa> *sharedParams,
    __gm__ int64_t *actualSeqQlenAddr)
{
    if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
        this->queryGm.offsetCalculator.Init(sharedParams->bSize, sharedParams->n2Size, sharedParams->gSize,
            sharedParams->s1Size, sharedParams->dSize);
    } else {  // GM_Q_OUT_TND
        GlobalTensor<int64_t> actualSeqQLen;
        actualSeqQLen.SetGlobalBuffer(actualSeqQlenAddr);
        this->queryGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->gSize, sharedParams->dSize,
            actualSeqQLen, sharedParams->actualSeqLengthsSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::CalcS1Coord(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    // 计算s1方向偏移
    coordInfo[runInfo.taskIdMod3].s1Coord = runInfo.s1oIdx * s1BaseSize;
    // 推理无效行场景，s1方向起始跳过无效行
    coordInfo[runInfo.taskIdMod3].s1Coord += (runInfo.nextTokensPerBatch < 0) ? -runInfo.nextTokensPerBatch : 0; // todo
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    CalcS1Coord(runInfo, constInfo);
    // CalcS2Coord(runInfo, constInfo);

    IterateBmm1SCFA(outputBuf,  runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    IterateBmm2SCFA(outputBuf, inputBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm1SCFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    Buffer<BufferType::L1> mm1A;
    Buffer<BufferType::L1> mm1B;
    // 左矩阵复用，S2的第一次循环加载左矩阵
    // 加载左矩阵到L1, 全载
    // query对ori_kv, cmp_kv都一样，无需区分
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本块：搬运Q
        mm1A = l1QBuffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用L1A
        LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();

        uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
            coordInfo[runInfo.taskIdMod3].s1Coord, 0); // todo:确保kernel层传了这些值，或者自己算offset
        CopyToL1Nd2Nz<INPUT_T>(mm1ATensor, this->queryGm.gmTensor[gmOffset], runInfo.s1RealSize, constInfo.dSize,
            constInfo.mm1Ka); // todo:确保kernel层根据layout区分传了constInfo.mm1Ka

        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非S2的第一次循环直接复用Q
        mm1A = l1QBuffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    }

    // 加载当前轮的右矩阵到L1
    mm1B = l1KBuffers.Get(); // TODO：这里使用GetPre()?
    // mm1B.Wait<HardEvent::MTE1_MTE2>(); // 占用L1B
    mm1B.WaitCrossCore();    //核间同步，这里需要根据V0操作处理同步，确保取tensor时，数据已经准备好

    // mm1B.Set<HardEvent::MTE2_MTE1>();  // 通知
    // mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B

    mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)runInfo.s1RealSize,     // singleM
                        (uint32_t)runInfo.s2RealSize,  // singleN
                        (uint32_t)(constInfo.dSize),   // singleK
                        0,    // isLeftTranspose
                        1     // isRightTranspose
    };
    MatmulK<INPUT_T, INPUT_T, T, 64, 128, 128, ABLayout::MK, ABLayout::KN>(  // m,n不切，k切128
        mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),                // mm1B直接用tensor的数据
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        mm1A.Set<HardEvent::MTE1_MTE2>(); // 释放L1A
    }

    // bmm2再释放
    // mm1B.Set<HardEvent::MTE1_MTE2>();  // 释放L1B

    mm1ResL0C.Set<HardEvent::M_FIX>();    // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>();   // 等待L0C

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    fixpipeParams.nSize = (runInfo.s2RealSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.mSize = (runInfo.s1RealSize + 1) >> 1 << 1; // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放L0C
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm2SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get(); // P直接用无需搬运
    mm2A.WaitCrossCore();

    outputBuf.WaitCrossCore(); //占用

    Buffer<BufferType::L1> mm2B = l1KBuffers.GetReused(); // V复用
    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)s1BaseSize,          // singleM 64
                        (uint32_t)constInfo.dSizeV, // singleN 512
                        (uint32_t)s2BaseSize,       // singleK 128
                        0,    // isLeftTranspose    // todo: useDn?
                        1     // isRightTranspose
                    };
    MatmulN<INPUT_T, INPUT_T, T, 64, 128, 128, ABLayout::MK, ABLayout::KN>(
        mm2A.GetTensor<INPUT_T>(),
        mm2B.GetTensor<INPUT_T>(),
        mmL0ABuffers,
        mmL0BBuffers,
        mm2ResL0C.GetTensor<T>(),
        param);

    mm2B.Set<HardEvent::MTE1_MTE2>();   // bmm2才释放KV，在这里释放

    mm2ResL0C.Set<HardEvent::M_FIX>();  // 通知
    mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;   // L0C→UB;FixpipeParamsM300:L0C→UB
    fixpipeParams.nSize = ((uint32_t)constInfo.dSizeV + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    fixpipeParams.mSize = s1BaseSize;                        // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    fixpipeParams.srcStride = (s1BaseSize + 15) >> 4 << 4;   // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF
class FABlockCubeDummy {
public:
    __aicore__ inline FABlockCubeDummy() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr, __gm__ uint8_t *query) {}
    __aicore__ inline void InitCubeInput(CVSharedParams<isInfer, isPa> *sharedParams, __gm__ int64_t *actualSeqQlenAddr) {}
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf,RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo) {}
};

template <typename T>
struct CubeBlockTraits;  // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    };

DEFINE_CUBE_BLOCK_TRAITS(FABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(FABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE)\
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // FLASH_ATTENTION_SCORE_BLOCK_CUBE_H_
