# CATLASS GMM_sliceM_perToken_Dequant
## 原型设计
|名称/Name|类型/Class|数据类型/Dtype|维度/Dims|格式/Format|描述/Description|
|---|---|---|---|---|---|
|matA|inTensor|int8|[m, k]|ND|左矩阵|
|matB|inTensor|int8|[groupCount, n, k]|ND|右矩阵，支持转置|
|groupList|inTensor|int|[groupCount]|ND|m轴方向分组大小，累加和列表|
|scale|inTensor|bf16|[groupCount, n]|ND|perChannel量化系数|
|perTokenScale|inTensor|bf16|[m]|ND|perToken量化系数|
|matD|outTensor|bf16|[m, n]|ND|输出矩阵|

## 样例实现
CATLASS GMM_sliceM_perToken_Dequant样例算子是基于CATLASS Gemm Api实现的亲和昇腾AtlasA2硬件的GMM算子，算子的结构可以分为以下几部分
- **Example组装**，[grouped_matmul_slice_m_per_token_dequant.cpp](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp)；
- **Kernel实现**，[grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp)；
- **Block组件**，包含：
    - 通用的mmad组件[block_mmad_preload_async_with_callback.hpp](../../../include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp)
    - 定制的后处理组件[block_epilogue_per_token_dequant.hpp](../../../include/catlass/epilogue/block/block_epilogue_per_token_dequant.hpp)；
- **Tile组件**，除一些基本Tile_copy和tile_mmad组件外，重点关注组件：
    - `BlockMmad`[CopyGmToL1GMMPTD](../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)
    - 后处理中[`TileRowBroadcastMul`](../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp#L32)
    - 后处理中[`TileBroadcastOneBlk`](../../../include/catlass/epilogue/tile/tile_broadcast_one_blk.hpp#L23)
    - 后处理中[`TileOneBlkColumnBroadcastMul`](../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp#L88)

## Example组装
### 构造输入
- 计算各输入[输入数据量大小(len)](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L82)
- 计算各输入[输入数据尺寸(size)](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L88)
- [host侧生成各输入初始值](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L94)使用[GenerateGroupList()](../common/golden/fill_data.hpp#L70)随机生成`groupList`（M轴前缀和）
- [构造device侧输入](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L105)

### 组装`BlockMmad`
- [定义各输入的layout](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L131)，[layout详见](../../docs/tla/01_layout.md)
- [设置DispatchPolicy](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L152)为[MmadAtlasA2PreloadAsyncWithCallback](../../docs/dispatch_policies.md#L72)，即选取BlockMmad组件
- 设置[L1TileShape和L0TileShape](../../docs/dispatch_policies.md#L154)，用于基本块分核和L1/L0切块。需要注意TileShape和DispatchPolicy的设置有约束关系，例如当前L1::M=128、L1::K=128时，l0AStages不能超过4，以免超出L0A大小、在样例编译时被BlockMmad侧静态校验拦截。TileShape计算可参考[TileShape约束](../../docs/catlass_optimize_guidance.md)
- [设置blockMmad输入输出Type](../../docs/dispatch_policies.md#L157)
- 为了专门使用CopyGmToL1GMMPTD来做CopyGmToL1A的动作，重新[定义](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L49)并[组装](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L161)了新的TileCopyMmad，若不使能CopyGmToL1GMMPTD可以跳过此步骤，BlockMmad会使用默认TileCopyMmad。
- 使用上述模板入参[组装BlockMmad](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L162)

### 组装blockEpilogue
- [设置EpilogueDispatchPolicy](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L166)为[EpilogueAtlasA2PerTokenDequant](../../docs/dispatch_policies.md#L72)，即选取block_epilogue组件
- 定义后处理[相关入参的Type](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L167)：ScaleType、PerTokenScaleType、DType
- 定义后处理[各计算步骤的Type](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L171)：RowBroadcastMulType、BroadcastOneBlkType、OneBlkColumnBroadcastMulType
- 定义后处理[单次计算的Tile大小](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L175)：<m0,n0>
- 定义后处理计算使用的Tile-[TileRowBroadcastMul](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L176)：单个Tile块<m0,n0>和scale片段<1,n0>，先将scale片段<1,n0>Broadcast到<m0,n0>，再做elementwise乘法
- 定义后处理计算使用的Tile-[TileBroadcastOneBlk](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L177)：perTokenScale片段<m0,1>做Broadcast到<m0, blk>
- 定义后处理计算使用的Tile-[TileOneBlkColumnBroadcastMul](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L179)：单个Tile块<m0,n0>和perTokenScale的Broadcast片段<m0,blk>，先将perTokenScale片段<m0,blk>Broadcast到<m0,n0>，再做elementwise乘法。与[TileBroadcastOneBlk](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L177)绑定使用
- 定义后处理搬运使用的Tile-[TileCopy](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L181)，[偏特化位置](../../../include/catlass/epilogue/tile/tile_copy.hpp#L71)
- 定义后处理计算基本块时、按照<m0,n0>切块后的swizzle策略-[EpilogueHorizontalTileSwizzle](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L182)，[源码位置](../../../include/catlass/epilogue/tile/tile_swizzle.hpp#L55)，目前提供EpilogueIdentityTileSwizzle和EpilogueHorizontalTileSwizzle
- 使用上述模板入参[组装BlockEpilogue](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L184)

### 组装和执行Kernel
- 定义Kernel中[基本块的swizzle策略](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L188)，详见[swizzle_explanation](../../docs/swizzle_explanation.md)
- [组装Kernel](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L192)-GroupedMatmulSliceMPerTokenDequantMultiStageWorkspace，需要引用对应[Kernel文件](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L29)
- 将Kernel[组装](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L195)入适配器并[实例化](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L201)
- [构造入参arguments](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L197)
- [校验入参arguments](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L202)，当前Kernel内无实现、直接返回true
- [计算算子需要的workspace大小](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L203)，Kernel内实现
- [申请workspace](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L206)
- [适配器初始化算子](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L208)，特别注意，对于涉及核间同步的算子，需要[初始化fftsAddr](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L138)并在MatmulAdapter执行时调用传入
```
// Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));
    ...
    matmulOp(stream, aicCoreNum, fftsAddr);
```
- [适配器执行算子](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L209)
- [流同步](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L210)

### 精度校验和空间释放
- 将算子输出[搬运回host侧](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L213)
- [计算golden标杆](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L216)
- [精度比对](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L221)
- [释放输入输出和workspace](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L228)

## Kernel实现
### Kernel主要结构体和函数
- [struct Params](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L61)：执行时需要的参数
- [struct Arguments](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L104)：host侧入参
- [static Params ToUnderlyingArguments](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L129)：将host侧入参Arguments解析为Params，由适配器在初始化算子时调用
- [static size_t GetWorkspaceSize](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L121)：根据Arguments计算算子执行时需要的workspace
- [void operator()<AscendC::AIC>](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L167)：AIC执行代码
- [void operator()<AscendC::AIV>](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L277)：AIV执行代码
- [struct AicWaitFunc](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L350)：封装AIC等待AIV的MTE3搬运完成的核间同步。AIC上由于使用MmadAtlasA2PreloadAsyncWithCallback方案，需要将callback传入blockMmad侧，由blockMmad决定调用核间同步的时机。
- [struct AicSetFunc](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L367)：封装AIV等待AIC的FIXPIPE搬运完成的核间同步。
### AIC流程
- 初始化BlockScheduler和BlockMmad，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L170)
- 初始化GlobalTensor：gmA/groupList/gmC，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L173)
- 获取当前AIC序号coreIdx、AIC总数coreNum，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L179)
- 初始化计算参数：A矩阵在GM上地址偏移gmGroupOffsetA、B矩阵在GM上地址偏移gmGroupOffsetB、workspace上的输出数据排布layoutC、WORKSPACE_STAGES操作对应stageId和stageUsed、当前group起始AIC序号startCoreIdx，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L181)
- group循环，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L191)
    - 计算当前group的M，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L192)
    - 判断是否使能A矩阵L2Cache绕过，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L204)
    - 计算当前核在当前group基本块中的起始idx，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L209)
    - 当前group中、当前核的基本块循环，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L211)
        - blockMmad所需入参计算，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L213)
        - 根据是否为异步方案调用不同blockMmad，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L232)，当前样例为异步方案，callback需要传入blockMmad内安排调用
        - 更新stageId，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L252)
    - 更新gmGroupOffsetA、gmGroupOffsetB、startCoreIdx，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L255)
- 异步方案blockMmad完成遗留的计算，[代码](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L261)

AIV计算流程与AIC一致，在调用`blockMmad()`处改为调用`blockEpilogue()`，并在Kernel代码内做核间同步，基于callback组件实现。
### 基本块分核方案
- 各group内(尺寸为`[currentM, N]`)按照`[L1TileShape::M, L1TileShape::N]`的尺寸切分基本块
- group间连续分核，达成不同AIC的负载均衡
- 相关变量：
    - **groupIdx**：当前group序号
    - **coreLoops**：当前group的block数
    - **startCoreIdx**：起始block的aic序号（当前group内）
    - **startLoopIdx**：当前核的起始block序号（当前group内）
    - **loopIdx**：当前核需要处理的block序号（当前group内）
- 示例：

<img src="https://raw.gitcode.com/user-images/assets/7801479/6029234c-39e4-4853-99de-1d4263f4e91f/Block_Partitioning_Scheme.png" width="100%">

### Workspace方案

当前workspaceStages=2的配置下，使用基本块双缓冲，每个AIC分配两个基本块大小的Workspace。对于任意shape的输入，申请的Workspace大小固定。
```cpp
static size_t GetWorkspaceSize(const Arguments &args)
{
    size_t lenWorkspace = static_cast<size_t>(L1TileShape::M) * L1TileShape::N *
        args.aicCoreNum * WORKSPACE_STAGES;
    size_t sizeWorkspace = lenWorkspace * sizeof(uint32_t);
    return sizeWorkspace;
}
```

（库上还有一种方案为申请完整的 `problemShape.m * problemShape.n` 的Workspace，在小shape场景可以节约Workspace，[参考](../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant.hpp)）

<img src="https://raw.gitcode.com/user-images/assets/7801479/8b950918-80c1-4e8d-a0de-63a6a4b6cf56/workspace.png" width="100%">

### AIC/AIV核间同步方案
AIC上对于单个block的Mmad可以拆分为L1Tile搬入、L1Tile计算、L0C搬出三个动作，由于blockMmad有preload预载和async异步的特性，`operator()<AscendC::AIC>`内调用`blockMmad()`时，仅完成当前block的L1Tile搬入和大部分L1Tile计算，而剩余的L1Tile计算和L0C搬出会在下次调用`blockMmad()`时执行。所以需要将AIC/AIV核间同步相关的callback入参传入`blockMmad()`，由`blockMmad()`内决定调用时机。

单个AIC和分配的两个AIV在处理GM上的一个个block的流程图如下：

<img src="https://raw.gitcode.com/user-images/assets/7801479/79aadfa0-3568-4cbc-908f-a90ec7b1cfa3/AIV_AIC同步.png" width="100%">
