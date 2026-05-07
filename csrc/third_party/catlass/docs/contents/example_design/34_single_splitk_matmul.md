# CATLASS Single_core_splitK_Matmul 样例介绍

## 样例实现
CATLASS [`34_single_core_splitk_matmul`样例](../../../examples/34_single_core_splitk_matmul/README.md)算子是基于CATLASS Gemm API实现的昇腾亲和Matmul算子,针对大尺寸矩阵计算场景优化设计，关键算子件包括以下几部分:
 - **Example组装**：[single_core_splitk.cpp](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)
 - **Kernel实现**：
   - 主Kernel文件：[single_core_slicek_matmul.hpp](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp)
   - 复用Padding组件：[padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)

 - **Block组件**： [block_mmad_single_core_splitk.hpp](../../../include/catlass/gemm/block/block_mmad_single_core_splitk.hpp)

## Example组装

```mermaid

graph LR
    S[构造输入] --> GroupT[SingleSplitK算子]

    subgraph GroupT[SingleSplitK算子]
        direction LR
        A[组装`Padding`对象]
        A --> B[组装`blockMmad`]
        B --> C[组装Kernel]
    end

    GroupT --> X[算子执行]
    X --> D[精度校验和空间释放]
```

与通用的模板库开发风格一致，本样例的实现执行过程如上图所示，简要说明如下：

<details>
<summary><strong>构造输入</strong></summary>

*生成要计算的左/右矩阵*

 - 计算各[输入矩阵的尺寸](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L71)
 - 在Host侧生成各[输入矩阵值](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L101)
 - 构造[Device侧输入](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L107)

</details>

<details>
<summary><strong>组装<code>Padding</code>对象</strong></summary>

*为便于数据搬运的对齐，组装Padding对象*

 - 定义不同的[`PaddingTag`](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L90)
 - 通过PaddingBuilder组装出对于矩阵A/B的[Padding对象](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L95)
 - 预定义的[PaddingC对象](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L97)
 - 决定是否[启用PaddingC对象](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L98)
</details>

<details>
<summary><strong>组装<code>blockMmad</code></strong></summary>

*为便于数据搬运的对齐，先组装相关的Padding对象*

 - 定义各输入的[Layout特征](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L78)
 - 声明不同矩阵所对应的[类型情况](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L124)
 - 设置[Dispatch策略](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L139)（用于选取BlockMmad组件）
 - 设置L1和L0的[Tile尺寸](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L136)，用于优化从GM到L1的搬运过程
 - 使用优化后的TileCopy组件，用于...
 - 使用上述模板入参[组装BlockMmad](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L162)
</details>

<details>
<summary><strong>组装并执行<code>Kernel</code></strong></summary>

*组装Kernel并实例化该对象，完成算子计算*

 - 使用新的S型[Swizzle策略](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L140)
 - 使用前述模板[组装Kernel](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L143)
 - 将[Kernel传入适配器](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L160)并[实例化](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L148)
 - 构造[入参参数](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L163)
 - 向Kernel侧[传入参数](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L150)
 - 调用Kernel层[Workspace计算](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L166)
 - 在Device侧[申请Workspace](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L166)(如有必要)
 - [适配器初始化算子](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L166)
 - [执行算子](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp#L166)
</details>

<details>
<summary><strong>精度校验和空间释放</strong></summary>

*完成最后结果的验证与回收处理*

- 将算子输出结果[搬回host侧](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L213)
- [计算golden标杆](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L216)
- [精度比对](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L221)
- [释放输入输出和workspace](../../../examples/10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L228)

</details>

## Kernel实现

以下介绍Kernel层级的结构体与关键函数，以及AIC/AIV部分的简明计算流程，并说明了所使用的优化策略

### Kernel层的主要结构体与函数

以下是在[Kernel层](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp)所实现的结构体与关键函数
 - [`struct Params`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L84)：运行时算子执行所需的参数
 - [`struct Arguments`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L116)：封装Host侧传入的参数
 - [`static size_t GetWorkspaceSize`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L128)：预先计算对齐所需的空间
 - [`static Params ToUnderlyingArguments`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L150)：将Host侧入参解析为算子侧的`Params`结构体
 - [`void operator()<AscendC::AIV>`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L204)：AIV(Vector)部分执行代码
 - [`void operator()<AscendC::AIC>`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L253): AIC(Cube)部分执行代码

## AIV/AIC部分计算流程

以下是Kernel层所进行的AIC/AIV操作

<details>
<summary><strong>AIV上所执行的操作</strong></summary>

- 如果A或B的对齐处理启用：
  - [初始化GlobalTensor](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#L208)：gmA， gmWA(或gmB, gmWB)
  - [实例化PaddingA](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#212)对象，完成资源申请，以便在UB(Unified Buffer)上存储对齐前后的数据
  - 调用PaddingA对象的`operator()`方法，即[执行数据对齐](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#213)操作
  - 进行[核间同步](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#228)，设置标志位通知AIC对齐操作已完成(`CrossCoreSetFlag`)

- 结果矩阵搬出：
  - [等待AIC](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#233)完成计算(`CrossCoreWaitFlag`)
  - [初始化GlobalTensor](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#234)：`gmC`(目的地址), `gmWC`(临时累加地址)
  - 将数据搬出到目的地址，进行[升精度映射](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#240)(`RemovePaddingNDAndCastC`)
</details>

<details>
<summary><strong>AIC上所执行的操作</strong></summary>
 
 - 如果对左/右矩阵有对齐处理：
   - 核间同步，[等待AIV完成标志位](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#255)(`CrossCoreWaitFlag`)
   - [初始化`GlobalTensor`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#270)：`gmWA`, `gmWB`
 
 - [初始化`GlobalTensor`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#270): `gmA`, `gmB`, `gmC`
 - 初始化[`BlockScheduler`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#260)和[`BlockMmad`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#300)对象
 - 获取当前AIC序号`coreIdx`、AIC总数`coreNum`（在[`Swizzle`策略](../../../include/catlass/gemm/block/block_swizzle.hpp)内）以及所需的[`coreLoops`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#262)
 - 进入主循环（循环次`coreLoops`）
   - 计算当前A，B矩阵读入的偏移量[`gmOffsetA`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#333), [`gmOffsetB`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#334)以及下一块的偏移量[`gmOffsetNextA`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#341), [`gmOffsetNextB`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#342)(如果开DoubleBuffer)
   > 注意：优化算法下左矩阵重载，因此`gmOffsetNextA`实际不会“启用”
   - 计算[`needLoadNextA`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#325)、[`needLoadNextB`](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#326)，用以标识是否预加载
   - 调用`blockMmad`进行一次[AIC计算](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp#345)（完成L1A与L1B上相对应的分形矩阵计算）
   > 注意：`blockMmad`会依据K轴方向上切K情况，决定是否启用GM上的原子加
   - 设置标志位，通知AIV 计算已完成
   - 关闭原子加

</details>

### 单核切K算法

<div style="display: flex; justify-content: center;">
    <img src="https://raw.gitcode.com/user-images/assets/7694484/6322a9e2-00e0-449b-8c35-f99fe5883bae/tmp1.jpg" width="85%" height="auto">
</div>


如上图所示，相较于经典的矩阵乘过程，单核切K的`Matmul`模板采取了复用左矩阵的办法，即对于某个AI Core，其L1A上加载的分形矩阵是**驻留**的，进而降低对GM的访问量。

举例而言，对于经典计算过程，单核所进行的数据搬运过程包括：1. 从GM搬运A0到L1A，B0到L1B；2. 从GM搬运A1到L1A，B1到L1B；..., N. 从GM搬运A(n-1)到L1A， B(n-1)到L1B。

在本单核切K策略下，单个AI Core上的流程图如下所示：

<div style="display: flex; justify-content: center;">
<img src="https://raw.gitcode.com/user-images/assets/7694484/5f0dcf49-6c76-47ea-ac2f-70a88d2ca8bc/singleCoreSplitK_Arrange.png" width="85%" height="auto">
</div>

显然，该办法能有效减轻MTE2的搬运负担，但是会加重L0C计算结果搬回到GM的负担，需要在GM上开启原子加。

```cpp
// Atomic set once in mmad level
if (atomicAdd) {
    AscendC::SetAtomicAdd<ElementAccumulator>();
} else {
    AscendC::SetAtomicNone();
}
```

 - 增大L1利用空间
 简单建模可知（见下述补充），从L0C搬运回GM的数据量正比于$MNK/k_{\text{L1}}$，其中$M$, $N$, $K$分别为输入的矩阵尺寸，$k_{\text{L1}}$是K轴上L1Tile的尺寸大小。在符合L1A, L1B物理尺寸限制的条件下，$k_{\text{L1}}$越大可减少写出次数，推荐`L1TileShape`按下表进行配置。

<details>

<summary><strong>数据搬运理论建模</strong></summary>

我们首先分析下基础Matmul（详见本仓[00_basic_matmul](../../../examples/00_basic_matmul/README.md)样例）的内存访问量。设矩阵乘的尺寸为$(M, N, K)$，L1上的Tile尺寸相应为$m$, $n$, $k$。假设二者对齐。则单个分形（结果）矩阵从GM（Global Memory）至L1所需的搬运量是$K(mk+kn)/k$，加之分形矩阵个数为$MN/mn$，因此总的访问量为$MNK(1/m+1/n)$。另一方面，由于在L0C上进行累加，因此搬出的单位数据量即为$MN$。
再考虑使用本优化策略后的内存访问，设复用左矩阵，则单核总的数据搬运量为$mK + KN$，总的访问量是$(mK + KN)M/m$，也即$MNK(1/m+1/N)$，小于基础Matmul的情形。另外由于计算结果并不能在L0C上进行累加，需“随算随搬”，因此搬出的数据量为$MNK/k$， 较基础有一定升高。
| 类别 | 内存访问操作总量 | 
| --- | ------- | 
| 基础矩阵乘 |  $MNK(1/m+1/n) + MN$    | 
| 单核切K    | $MNK(1/m+1/N) + MNK/k$ |


</details>
 
| 数据类型 | `L1TileShape::M` | `L1TileShape::N` | `L1TileShape::K` | 
| --- | --- | --- | --- |
| FP16/BF16 | 256 | 128 | 512 | 
| FP32 | 256 | 128 | 256 | 

 - 数据对齐
 如需对数据进行类型转换，需采取数据对齐，以提高处理流程中的数据搬运带宽。

 ```cpp
// RemovePaddingNDAndCast的核心处理逻辑
uint32_t loopsPerTile = RoundUp(tileLen, COMPUTE_LENGTH);
uint32_t coreLoops = tilesPerAiv * loopsPerTile;

for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
  // 计算gmWC搬运的offset

  // 将GmWC(src)的数据搬运至inputBuffer上
  copyGm2Ub(inputBuffer, src, ubLayout, srcLayout);
  // ...

  // Cast到half
  AscendC::Cast(outputBuffer, inputBuffer,AscendC::RoundMode::CAST_RINT, actualDataNum);
  // ...

  // 将outputBuffer搬出到gmC(dst)
  copyUb2Gm(dst, outputBuffer[bufferIndex], dstLayout, ubLayout);
  // ...
}

```


### Swizzle排布方案


在矩阵C上的swizzle策略采用S型特征，相较于Z型的swizzle排布，在换行处可节约一次搬运。如下图所示，按该S型Swizzle策略，前一组需要加载到L1A, L1B上的分形矩阵为`<A02, <B20, B21, B22, B23>>`，后一次需要加载的是`<A12, <B20, B21, B22, B23>>`(以`SwizzleOffset`为`4`为例)。“换行”过程下，可以固定L1B上的数据，仅从GM上重读入左矩阵，可减轻载入负担。

<div style="display: flex; justify-content: center;">
    <img src="https://raw.gitcode.com/user-images/assets/7694484/45fcbe2f-bcfc-4bde-b11a-43361f09c1d5/singleCoreSplitK_Swizzle.png" width="35%" height="auto">
</div>


> 说明：该Swizzle策略在大尺寸下有效，排布策略会优先将任务分配至不同的AI Core上。

### 性能收益

经过实测，应用上述单核切K算法在大尺寸场景下较基础Matmul有正向收益，且随着K轴的尺寸增大，GM至L1节省的正向收益高过重复写入GM的负向收益，可参考下表。

不过需要说明的是，在M，N过小，或者K偏低的情况下，会出现**分核负载不均衡**的情况。

| M | N | K | 耗时(us) | 耗时[标杆]`(us) | 加速比 | 
| -- | -- | ---- | ----- | ----- | ----- | 
| 2048 |	4096 |	4000 | 261 |	 445 | 1.7049 | 
| 4096 | 4096	| 8000 | 917 |	1231 | 1.3424 |
| 4096 |	4096 |	40000	| 4669 |	7775 | 1.6652 | 
| 2048 | 4096 | 80000 | 16850 |	57144 | 1.7499 | 


说明：
 - 标杆为[`BasicMatmul`](../../../examples/00_basic_matmul/README.md)算子；
 - 统计耗时均为核函数总耗时，使用[`msprof`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/devaids/optool/atlasopdev_16_0082.html)工具得到；
 - 上述测试例中A，B及C矩阵均为`layout::RowMajor`排布方式；
 - 测试环境说明：NPU型号为910B2，CANN包版本为8.2.RC1



