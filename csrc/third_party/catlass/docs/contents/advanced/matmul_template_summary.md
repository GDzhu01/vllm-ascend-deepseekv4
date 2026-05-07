# 矩阵乘模板总结

当前库上`examples`内包含多种矩阵乘的`样例模板`，其来源是不同的matmul`理论模板`与工程实践中发现的`工程优化`点的组合。在充分理解了各个`理论模板`和`工程优化`后，开发者可以基于问题场景选择适合的`样例模板`、甚至进一步自行组合出库上没有的新的`样例模板`，来达成矩阵乘的高性能优化。

注意，本文档仅总结矩阵乘方案相关的样例，其他涉及量化、groupMatmul、后处理等的矩阵乘不在此处总结。

## 样例模板清单
<details>
<summary><strong><font size="4">00_basic_matmul</font></strong></summary>

- 理论模板：`Common模板`
- 工程优化：`流水优化（Multi Buffer）`
- 关键交付件
    - host：[00_basic_matmul](../../../examples/00_basic_matmul/basic_matmul.cpp)
    - kernel：[basic_matmul.hpp](../../../include/catlass/gemm/kernel/basic_matmul.hpp)
    - blockMmad：[block_mmad_pingpong.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`
</details>

<details>
<summary><strong><font size="4">04_padding_matmul</font></strong></summary>

- 理论模板：`Common模板`
- 工程优化：
    - `流水优化（Multi Buffer）`
    - `读取带宽优化（padding）- PaddingMatrixND`
- 关键交付件
    - host：[04_padding_matmul](../../../examples/04_padding_matmul/padding_matmul.cpp)
    - kernel：[padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)
    - blockMmad：[block_mmad_pingpong.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`
</details>

<details>
<summary><strong><font size="4">06_optimized_matmul</font></strong></summary>

- 理论模板：`Common模板`
- 工程优化：
    - `流水优化（Multi Buffer）`
    - `流水优化（Preload）`
    - `读取带宽优化（Padding）- PaddingMatrixNZ`
    - `读取带宽优化（ShuffleK）`
    - `读取带宽优化（小M下指令替换）`（需要修改样例使能）
- 关键交付件
    - host：[06_optimized_matmul](../../../examples/06_optimized_matmul/optimized_matmul.cpp)
    - kernel：[optimized_matmul.hpp](../../../include/catlass/gemm/kernel/optimized_matmul.hpp)
    - Padding前处理组件：[padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)
    - blockMmad：[block_mmad_preload.hpp](../../../include/catlass/gemm/block/block_mmad_preload.hpp)
- dispatchPolicy：`MmadAtlasA2Preload`
- ⚠️ 注意：即使没有使用`PaddingMatrixNZ`前处理，依然会产生MIX算子编译和CV1:0启动的开销（比仅AIC启动的开销大）
</details>

<details>
<summary><strong><font size="4">09_splitk_matmul</font></strong></summary>

- 理论模板：`多核切K模板 MultiCoreSplitK`
- 工程优化：`流水优化（Multi Buffer）`
- 关键交付件
    - host：[09_splitk_matmul](../../../examples/09_splitk_matmul/optimized_matmul.cpp)
    - kernel：[splitk_matmul.hpp](../../../include/catlass/gemm/kernel/splitk_matmul.hpp)
    - blockMmad：[block_mmad_pingpong.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`
</details>

<details>
<summary><strong><font size="4">21_basic_matmul_preload_zN</font></strong></summary>

（此样例主要承载NZ排布输入的适配方法，也可换成ND排布输入，无MIX算子编译启动的开销）
- 理论模板：`Common模板`
- 工程优化：
    - `流水优化（Multi Buffer）`
    - `流水优化（Preload）`
    - `读取带宽优化（ShuffleK）`
- 关键交付件
    - host：[21_basic_matmul_preload_zN](../../../examples/09_splitk_matmul/basic_matmul_preload_zN.cpp)
    - kernel：[basic_matmul_preload.hpp](../../../include/catlass/gemm/kernel/basic_matmul_preload.hpp)
    - blockMmad：[block_mmad_preload.hpp](../../../include/catlass/gemm/block/block_mmad_preload.hpp)
- dispatchPolicy：`MmadAtlasA2Preload`
</details>

<details>
<summary><strong><font size="4">22_padding_splitk_matmul</font></strong></summary>

- 理论模板：`多核切K模板 MultiCoreSplitK`
- 工程优化：
    - `流水优化（Multi Buffer）`
    - `读取带宽优化（padding）- PaddingMatrixND`
- 关键交付件
    - host：[22_padding_splitk_matmul](../../../examples/22_padding_splitk_matmul/padding_splitk_matmul.cpp)
    - kernel：[padding_splitk_matmul.hpp](../../../include/catlass/gemm/kernel/padding_splitk_matmul.hpp)
    - Padding前处理组件：[padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)
    - SplitkReduceAdd后处理组件：[splitk_matmul.hpp](../../../include/catlass/gemm/kernel/splitk_matmul.hpp)
    - blockMmad：[block_mmad_pingpong.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`
</details>

<details>
<summary><strong><font size="4">25_matmul_full_loadA</font></strong></summary>

（此样例及相关组件仅适配了A矩阵全载实现，需要实现B矩阵全载可参考关键交付件自行开发）
- 理论模板：`Common模板`
- 工程优化：
    - `流水优化（Multi Buffer）`(全载的A矩阵在L1上不使用多buffer)
    - `读取带宽优化（L1常驻）`
- 关键交付件
    - host：[25_matmul_full_loadA](../../../examples/09_splitk_matmul/25_matmul_full_loadA.cpp)
    - kernel：[matmul_full_loadA.hpp](../../../include/catlass/gemm/kernel/matmul_full_loadA.hpp)
    - blockMmad：[block_mmad_pingpong_full_loadA.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong_full_loadA.hpp)
- dispatchPolicy：`MmadAtlasA2FullLoadA`
- BlockScheduler：`GemmIdentityBlockSwizzleL1FullLoad`
</details>

<details>
<summary><strong><font size="4">31_small_matmul</font></strong></summary>

- 理论模板：`Common模板`
- 工程优化：
    - `流水优化（Multi Buffer）`
    - `Scalar开销消减`
- 关键交付件
    - host：[31_small_matmul](../../../examples/31_small_matmul/small_matmul.cpp)
    - kernel：[small_matmul.hpp](../../../include/catlass/gemm/kernel/small_matmul.hpp)
    - blockMmad：[block_mmad_small.hpp](../../../include/catlass/gemm/block/block_mmad_small.hpp)
- dispatchPolicy：`MmadAtlasA2Small`
- BlockScheduler：kernel内实际不使用
</details>

<details>
<summary><strong><font size="4">34_single_core_splitk_matmul</font></strong></summary>

- 理论模板：`单核切K模板 SingleCoreSplitK`
- 工程优化：
    - `流水优化（Multi Buffer）`
    - `读取带宽优化（Padding）- PaddingMatrixNZ`
    - `写出带宽优化`
- 关键交付件
    - host：[34_single_core_splitk_matmul](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)
    - kernel：[single_core_slicek_matmul.hpp](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp)
    - Padding前处理组件和RemovePaddingNDAndCast后处理组件：[padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)
    - blockMmad：[block_mmad_single_core_splitk.hpp](../../../include/catlass/gemm/block/block_mmad_single_core_splitk.hpp)
- dispatchPolicy：`MmadAtlasA2SingleCoreSplitk`
- BlockScheduler：`SingleCoreSplitkGemmIdentityBlockSwizzle`
</details>

## 理论模板清单

<details>
<summary><strong><font size="4">Common模板</font></strong></summary>

### Tiling建模

<img src="https://raw.gitcode.com/user-images/assets/7801479/b1cb21ac-af83-4736-8582-4ed7392d766b/1common.png" width="80%">

如图展示一个常规的fp16的矩阵运算（L0C上按照fp32累加），定义相关参数：
- 问题shape：$M$，$N$，$K$
- 搬入L1Cache时的TileShape：$m_1$，$n_1$，$k_1$
- 搬入L0A/L0B、搬出L0C时的TileShape：$m_0$，$n_0$，$k_0$

采用 $M$、$N$ 方向分核，按照$m_1$、$n_1$切分，产生$\frac{MN}{m_1n_1}$个基本任务块、分配给AIC核完成搬运和计算，每个基本任务块需要搬运$m_1K+Kn_1$的数据、计算得到$m_1n_1$的结果并搬出。由此产生约束：
- $m_1k_1*L1Stage_A + n_1k_1*L1Stage_B <= L1Size / 2Byte$
- $m_0k_0*L0AStage <= L0ASize / 2Byte$
- $n_0k_0*L0BStage <= L0BSize / 2Byte$
- $m_0n_0*L0CStage <= L0CSize / 4Byte$
- $m_0 = m_1$
- $n_0 = n_1$

### 读取数据量
每个基本任务块需要搬运$m_1K+Kn_1$的数据，总读取量为：

$2Byte * [m_1K+Kn_1] * \frac{MN}{m_1n_1} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{n_1}]$

### 写出数据量
每个基本任务块计算得到$m_1n_1$的结果并搬出，总写出量为：

$2Byte * MN$

### 计算量
输出矩阵C的每个数据点需要$K$次乘加，总计算量固定为：

$2MNK$

计算耗时多数情况下为刚性时间，仅与参与计算的AIC核数相关，在各理论模板中都一样，后续不再赘述。
</details>

<details>
<summary><strong><font size="4">多核切K模板 MultiCoreSplitK</font></strong></summary>

### Tiling建模

<img src="https://raw.gitcode.com/user-images/assets/7801479/d4b0e2d2-4333-4df4-9af2-44654cc37e54/2multiCoreSplitK.png" width="80%">

如图展示一个常规的fp16的矩阵运算（L0C上按照fp32累加），$MN$方向共切分12个基本任务块，假设有24个AIC物理核，此时负载不均衡，故引入$K$轴切分成2个$k$、产生24个基本任务块，在AIC上负载均衡。定义相关参数：
- 问题shape：$M$，$N$，$K$
- 搬入L1Cache时的TileShape：$m_1$，$n_1$，$k_1$
- 搬入L0A/L0B、搬出L0C时的TileShape：$m_0$，$n_0$，$k_0$
- <font color="red">相较Common模板</font>，新增$K$方向切分长度$k$

在$m_1$、$n_1$较大时，可能存在负载不均问题，即M,N方向切分的任务块远少于AIC核数，导致读取带宽不高（核数不够），因此可以在K方向上分核。产生$\frac{MNK}{m_1n_1k}$个基本任务块、分配给AIC核完成搬运和计算，每个基本任务块需要搬运$m_1k+kn_1$的数据、计算得到$m_1n_1$的结果并搬出。硬件上约束与Common相同。

### 读取数据量
每个基本任务块需要搬运$m_1K+Kn_1$的数据，总读取量与`Common模板`一致：

$2Byte * [m_1k+kn_1] * \frac{MNK}{m_1n_1k} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{n_1}]$

### 写出数据量
每个基本任务块计算得到$m_1n_1$的结果并搬出，需要$\frac{K}{k}$个基本块累加来得到输出矩阵C的$m_1n_1$块的最终输出，总写出量为：

$2Byte * MNK / k$

### 定性分析

相较`Common模板`，搬入数据量不变，写出数据量增加，并产生后处理ReduceAdd的开销（包含MIX算子编译启动的开销），但切分基本块更多、更易负载均衡。

</details>

<details>
<summary><strong><font size="4">单核切K模板 SingleCoreSplitK</font></strong></summary>

### Tiling建模

<img src="https://raw.gitcode.com/user-images/assets/7801479/e16f5a39-2f7b-4a72-9d79-502cc8682e75/3singleCoreSplitK.png" width="80%">

如图展示一个常规的fp16的矩阵运算（L0C上按照fp32累加），定义相关参数：
- 问题shape：$M$，$N$，$K$
- 搬入L1Cache时的TileShape：$m_1$，$n_1$，$k_1$
- 搬入L0A/L0B、搬出L0C时的TileShape：$m_0$，$n_0$，$k_0$

相比`Common模板`，为了减少读取数据量，进一步增大抽象上的$m_1$、$n_1$，考虑将$m_1k_1$的tile块直接与对应的所有$k_1n_1$的tile块完成计算（等同于将$n_1$放大到$N$），此时输出$m_0n_0$的tile块没法在$L0C$常驻累加，需要及时搬出，通过`atomicAdd`在`GM`上累加。硬件上约束如下：
- $m_1k_1*L1Stage_A + n_1k_1*L1Stage_B <= L1Size / 2Byte$
- $m_0k_0*L0AStage <= L0ASize / 2Byte$
- $n_0k_0*L0BStage <= L0BSize / 2Byte$
- $m_0n_0*L0CStage <= L0CSize / 4Byte$
- $m_0 <= m_1$
- $n_0 <= n_1$

### 读取数据量
在`Common模板`的读取数据量公式上，将$n_1$放大到$N$；或者从A矩阵分基本任务块来理解，切分$\frac{MK}{m_1k_1}$个基本块，每个基本块完成搬入此块A矩阵tile块以及对应全部的B矩阵tile块，搬入的数据量为$m_1k_1+k_1N$：

$2Byte * [m_1k_1+k_1N] * \frac{MK}{m_1k_1} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{N}]$

### 写出数据量
切分A矩阵分基本任务块，共$\frac{MK}{m_1k_1}$个基本块，每个基本任务块计算得到$m_1N$的结果并搬出，总写出量为：

$2Byte * MNK / k_1$

### 定性分析

相较`Common模板`，搬入数据量减少，写出数据量增加，与AIV无关。

</details>

## 工程优化清单

<details>
<summary><strong><font size="4">流水优化（Multi Buffer）</font></strong></summary>

### 现象分析

如下图构造一个`Common`模板下的简单场景，对于单个AIC处理一个基本任务块C，需要的A/B矩阵Tile块较小，可以直接全部放入L1，且A/B矩阵从L1搬入L0时需要切4次搬入。

<img src="https://raw.gitcode.com/user-images/assets/7801479/a62726da-f3c2-4c37-8a32-711046cfd239/8pingpong0.png" width="80%">

各PIPE的指令流水图示例如下：

<img src="https://raw.gitcode.com/user-images/assets/7801479/1dca48b1-2b5d-450c-9f66-18dbc1f7e2d1/8pingpong1.png" width="100%">

如果在AIC的L1/L0A/L0B/L0C上，每次载入数据tile块时都尽量填满所有空间，会导致各PIPE的流水串行，整体效率低下

### 优化方案

使用常规优化手段Multi Buffer，即在L1/L0A/L0B/L0C上启用多buffer，使得流水尽可能并行，提升效率，这一优化策略如下图所示：


<img src="https://raw.gitcode.com/user-images/assets/7801479/40315511-85cc-44ac-be3f-9efb6b5c0194/8pingpong2.png" width="80%">

各PIPE的指令流水图示例如下，MTE1上指令的0、1表示pingpong流水：

<img src="https://raw.gitcode.com/user-images/assets/7801479/5917e1ca-dea0-4e00-8322-ef5ba2f32f46/8pingpong3.png" width="100%">

⚠️ 需要注意的是，与`L1常驻`优化结合时，常驻的A/B矩阵的tile块，不启用多buffer。

### 特性承载代码

由于是常规优化手段，所有blockMmad组件均使能。

</details>

<details>
<summary><strong><font size="4">流水优化（Preload）</font></strong></summary>

### 现象分析

通过仿真流水发现pingpong策略的blockMmad存在问题：
- MTE2流水上，“当前C矩阵基本块计算的最后一个A矩阵（B矩阵）的tile块”和“下一个C矩阵基本块计算的第一个A矩阵（B矩阵）的tile块”之间加载的空泡。
### 优化方案

针对GM->L1过程，读取当前轮次的$m_1k_1$（$k_1n_1$）时，计算上一轮读取的数据（假设Preload一轮，PRELOAD_STAGES = 1），步骤伪代码如下：
```cpp
for ... {
    # 搬入当前轮次的数据
    copyGM2L1A
    copyGM2L1B
    preload_count++
    for (preload_count == PRELOAD_STAGES) {
        # 计算前PRELOAD_STAGES轮次的数据
        copyL12L0A
        copyL12L0B
        Mmad
    }
}
```

如下图构造一个`Common`模板下的简单场景，对于单个AIC处理两个基本任务块C1和C2，需要的A/B矩阵Tile块放入L1需要切分2次，且A/B矩阵L1Tile块从L1搬入L0时需要切分4次（可以先学习前文`流水优化（Multi Buffer）`来增强理解）。以下给出`MmadAtlasA2Pingpong`和`MmadAtlasA2Preload`、`MmadAtlasA2PreloadAsync`的指令对比：
- `MmadAtlasA2Pingpong`中，调用两次blockMmad分别完成C1和C2的计算
- `MmadAtlasA2Preload`中，两次blockMmad也是分别完成C1和C2的计算，但A3/B3的GmToL1搬运提前到了第一次blockMmad中执行
- `MmadAtlasA2PreloadAsync`中，调用两次blockMmad和一次收尾的SynchronizeBlock。A2/B2的L1ToL0搬运、tileMmad以及C1的搬出从第一次blockMmad中推迟到第二次blockMmad中；A4/B4的L1ToL0搬运、tileMmad以及C2的搬出从第二次blockMmad中推迟到SynchronizeBlock中

<img src="https://raw.gitcode.com/user-images/assets/7801479/eca1e06a-7c9a-40d6-934b-6843f9229d7c/9preload0.png" width="100%">

各PIPE的指令流水图示例如下，最终达成了A3、B3块的GmToL1搬运提前，减缓了MTE2上的搬运空泡：

<img src="https://raw.gitcode.com/user-images/assets/7801479/1edf8fb8-c8e2-4b84-b742-0e6c507efb64/9preload1.png" width="100%">

### 特性承载代码
- [block_mmad_preload.hpp](../../../include/catlass/gemm/block/block_mmad_preload.hpp)，对应dispatchPolicy：`MmadAtlasA2Preload`，需要在kernel内手动计算传入下一块预载数据的信息。
- [block_mmad_preload_async.hpp](../../../include/catlass/gemm/block/block_mmad_preload_async.hpp)，对应dispatchPolicy：`MmadAtlasA2PreloadAsync`，通过异步控制，无需手动计算下一块预载数据信息，并支持mmad计算完成后的`Callback`传入。
- [block_mmad_preload_async_with_callback.hpp](../../../include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp)，对应dispatchPolicy：`MmadAtlasA2PreloadAsyncWithCallback`，通过异步控制，无需手动计算下一块预载数据信息，并支持blockMmad计算前后的`Callback`传入。
</details>

<details>
<summary><strong><font size="4">读取带宽优化（Padding）</font></strong></summary>

### 现象分析
当数据读取为主流水时，优化读取带宽能带来性能收益，以fp16的A矩阵为例，目前有以下几种低带宽场景：
- **Stride非512B对齐导致的低带宽**。搬运参数srcDValue（详见[昇腾文档：DataCopy-随路转换ND2NZ搬运](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_00127.html)）非512B对齐时，带宽会有明显下降。
- **搬运指令限制导致的低带宽**。ND2NZ的搬运指令，参数srcDValue是uint16类型，最大值65535。当K>65535时，只能通过取ndNum= 1，在m方向循环调用搬运指令，降低了读取带宽。
- 相比ND2ND不转换排布，**ND2NZ随路转换有带宽损失**。
### 优化方案
针对上述情况，可以使用AIV对数据格式进行重排（预处理动作），当重排开销低于带宽损失时，会有性能收益。从复杂度和能应对的场景出发，有下列三种不同的重排方式。
#### PaddingMatrixND
<img src="https://raw.gitcode.com/user-images/assets/7801479/80501346-2ea2-42ba-8cc0-b6f614630606/4paddingND.png" width="100%">

将Stride方向按照512B对齐，实现复杂度最低，可以处理Stride非对齐导致的带宽下降。
#### PaddingMatrixBlockND
<img src="https://raw.gitcode.com/user-images/assets/7801479/9e5e0d37-ef6a-41e0-b54a-1f6d079e1179/4paddingBlockND.png" width="100%">

按$m_1*k_1$作为“block”粒度重排，block内行优先、block间行优先，且$k_1$为512B对齐，实现复杂度适中，可以处理Stride非对齐和Stride超过65535导致的带宽下降。

#### PaddingMatrixNZ
<img src="https://raw.gitcode.com/user-images/assets/7801479/11728de5-073d-481b-a030-b262f425161f/4paddingNZ.png" width="100%">

重排为zN格式，实现复杂度最高（在介绍的几种Padding策略内），因为和L1上数据排布一致，搬运带宽也最高，可以处理Stride非对齐、Stride超过65535、ND2NZ随路转换导致的带宽下降。

<font color="red">实际应用中，不同case适合的padding方式不同，暂无全局最优Padding选择。</font>

### 特性承载代码
- [padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)中包含Padding前处理组件。
- 实际适配可参考[06_optimized_matmul](../../../examples/06_optimized_matmul/optimized_matmul.cpp)，通过`PaddingTag`、`PaddingBuilder`组装出A矩阵/B矩阵的Padding前处理。
</details>

<details>
<summary><strong><font size="4">读取带宽优化（ShuffleK）</font></strong></summary>

### 现象分析
通常，所有AIC核心都从$K$方向的第一个分块开始搬运计算，会存在多个核心同时读取同一片GM上数据的情况，产生数据读取冲突，导致读取带宽降低。
### 优化方案
<img src="https://raw.gitcode.com/user-images/assets/7801479/3b79fdb8-1154-46fa-bde8-057950d16b86/5shuffleK.png" width="100%">

以`Common模板`为例，如图：
- $matC$中的$CoreX$表示该基本块分配给第$X$个AIC进行计算
- $Aj$表示A矩阵该$m_1$下沿$K$轴切分的第$j$个L1Tile基本块
- $Bij$表示B矩阵该$n_1$下沿$K$轴切分的第$j$个L1Tile基本块
- 图中matC基本块分核采用Swizzle<2, 1>，详见[swizzle_explanation](./swizzle_explanation.md)

如图左原始方案，$Core2$和$Core3$搬运A矩阵时均按照“$A0$->$A1$->$A2$->$A3$”的顺序，产生了数据读取冲突。


ShuffleK方案如图所示，根据$CoreIdx$来偏移起始搬运序号$j$，$Core2$搬运A矩阵时按照“$A2$->$A3$->$A0$->$A1$”的顺序，对应B矩阵按照“$B02$->$B03$->$B00$->$B01$”的顺序；$Core3$搬运A矩阵均按照“$A3$->$A0$->$A1$->$A2$”的顺序，对应B矩阵按照“$B13$->$B10$->$B11$->$B12$”的顺序。从时间上错开，避免同地址访问冲突。
### 特性承载代码
- [block_mmad_preload.hpp](../../../include/catlass/gemm/block/block_mmad_preload.hpp)
- [block_mmad_preload_async.hpp](../../../include/catlass/gemm/block/block_mmad_preload_async.hpp)
- [block_mmad_preload_async_with_callback.hpp](../../../include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp)

上述Block层实现代码通过设置起始L1序号为`CoreIdx/kTileCount`来实现错位：
```cpp
kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
startTileIdx = AscendC::GetBlockIdx();
firstTileIdx = startTileIdx % kTileCount;
```

</details>

<details>
<summary><strong><font size="4">读取带宽优化（小M下指令替换）</font></strong></summary>

### 现象分析
<img src="https://raw.gitcode.com/user-images/assets/7801479/e3a90904-677a-4973-b3b5-93e93ea072e2/6smallM.png" width="80%">

矩阵计算中，当$M$很小时（例如$M$ < 8），采用随路ND2NZ的DataCopy（详见[昇腾文档：DataCopy-随路转换ND2NZ搬运](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_00127.html)）效率不高。

### 优化方案
通过`for`循环每次搬运一行，并在每一行调用`DataCopy`进行多次搬运（采用普通间隔搬运）。
### 特性承载代码
- [CopyGmToL1IntervalDataCopy](../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)
- 实际适配可参考[06_optimized_matmul](../../../examples/06_optimized_matmul/optimized_matmul.cpp)，在`struct TileCopyOpt`中手动替换`using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;`，而不是默认的`using CopyGmToL1A = typename Base::CopyGmToL1A;`

```diff
struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    ...

-   // using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;
+   using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;

-   using CopyGmToL1A = typename Base::CopyGmToL1A;
+   // using CopyGmToL1A = typename Base::CopyGmToL1A;
    ...
};
```

</details>

<details>
<summary><strong><font size="4">读取带宽优化（L1常驻）</font></strong></summary>

### 优化方案
实际开发时，可采用tile块常驻L1的方式，减少tile块数据的重复读取，等效提升了读取带宽，该特性需要结合不同理论模板来考虑实现方法。
### 特性承载代码
- `Common`模板可参考[25_matmul_full_loadA](../../../examples/09_splitk_matmul/25_matmul_full_loadA.cpp)及相关交付件，该样例通过$M$轴上的单核全载或多核全载来在特定场景下优化性能，并配合专门的swizzle策略来提高L1上全载的A矩阵块的复用频率。
    - kernel：[matmul_full_loadA.hpp](../../../include/catlass/gemm/kernel/matmul_full_loadA.hpp)
    - blockMmad：[block_mmad_pingpong_full_loadA.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong_full_loadA.hpp)
    - dispatchPolicy：`MmadAtlasA2FullLoadA`
    - BlockScheduler：`GemmIdentityBlockSwizzleL1FullLoad`
- `单核切K模板`[34_single_core_splitk_matmul](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)在理论设计上就考虑了L1Tile块常驻的优化点。
    - kernel：[single_core_slicek_matmul.hpp](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp)
    - blockMmad：[block_mmad_single_core_splitk.hpp](../../../include/catlass/gemm/block/block_mmad_single_core_splitk.hpp)
    - dispatchPolicy：`MmadAtlasA2SingleCoreSplitk`
    - BlockScheduler：`SingleCoreSplitkGemmIdentityBlockSwizzle`

</details>

<details>
<summary><strong><font size="4">Scalar开销消减</font></strong></summary>

### 现象分析
对于小Shape场景，如`Common模板`中：
- $M$、$N$方向分核数小于实际物理核数，每个AIC物理核最多仅处理一个基本任务块
- $k_1$ >= $K$，$K$方向无需切分后从GM搬入L1

此时kernel总耗时较小，scalar开销对性能影响显著。
### 优化方案
消减冗余的scalar计算
- kernel内不使用 BlockScheduler 来将任务块分配给物理核，手动计算每个物理核对应的任务块
- kernel内消除基本块循环（每个AIC仅处理1个任务块）
- kernel内简化offset相关计算
- blockMmad内消除L1A/L1B的$m$、$n$的循环
- blockMmad内简化offset相关计算
### 特性承载代码
- 参考[31_small_matmul](../../../examples/31_small_matmul/small_matmul.cpp)，可以和[00_basic_matmul](../../../examples/00_basic_matmul/basic_matmul.cpp)的相关交付件对比来加深理解
    - kernel：[small_matmul.hpp](../../../include/catlass/gemm/kernel/small_matmul.hpp)
    - blockMmad：[block_mmad_small.hpp](../../../include/catlass/gemm/block/block_mmad_small.hpp)

</details>

<details>
<summary><strong><font size="4">写出带宽优化</font></strong></summary>

### 现象分析
当数据写出为主流水时，优化写出带宽能够带来性能收益。
- 当写出时`dstStride`未进行512B对齐，带宽有明显下降
- 写出时用NZ2ND随路格式转换，产生带宽损失
### 优化方案
针对上述情况，可使用AIV对数据格式进行重排，在重排开销低于带宽损失时，会有性能收益。以下提供四种重排方式。
<img src="https://raw.gitcode.com/user-images/assets/7801479/89afcf74-a193-431b-aea3-a8de2abcb4f9/7rmPadding1.png" width="80%">

（↑）**方式一**：使用局部workSpace，ND写出到GM时512B对齐，随后按block块粒度在UB上重排，再写出到GM上。

<img src="https://raw.gitcode.com/user-images/assets/7801479/12665171-d86f-4208-8c11-3bef2359cfa1/7rmPadding2.png" width="80%">

（↑）**方式二**：使用全量workSpace，ND写出到GM时512B对齐，等全量结果写完后，再启动UB上数据重排，写入到GM上。

<img src="https://raw.gitcode.com/user-images/assets/7801479/13bd80b2-d6ab-4520-839c-cf5e700db0d5/7rmPadding3.png" width="80%">

（↑）**方式三**：使用局部workSpace，NZ写出到GM时512B对齐，再按block块粒度在UB上重排，ND写出到GM上。

<img src="https://raw.gitcode.com/user-images/assets/7801479/204d24e9-5dc8-4318-bfe1-7f958598c514/7rmPadding4.png" width="80%">

（↑）**方式四**：使用全量workSpace，NZ写出到GM时512B对齐，等全量结果写完后，再启动UB上数据重排，ND写出到GM上。

### 特性承载代码
- [padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)中实现了包含**方式二**的`RemovePaddingNDAndCast`后处理组件
- 实际适配可参考[34_single_core_splitk_matmul](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)
</details>

## 模板应用浅述

参考[102_dynamic_optimized_matmul的select_kernel策略](../../../examples/102_dynamic_optimized_matmul/include/select_kernel_b16.h)


<details>
<summary><strong><font size="4">模板选择</font></strong></summary>

先尝试基于[00_basic_matmul](../../../examples/00_basic_matmul/basic_matmul.cpp)进行TileShape调优并获得**性能基线**，可参考[模板库优化指引
](./catlass_optimize_guidance.md)


依次识别是否满足各模板适合场景，并和性能基线作比较：
- [31_small_matmul](../../../examples/31_small_matmul/small_matmul.cpp)：
    - 计算当前基本任务块 taskBlocks
        ```cpp
        taskBlocks = CeilDiv(M, m1) * CeilDiv(N, n1);
        ```
    - 基本任务块小于AIC数： $taskBlocks < aicCoreNum$
    - $K$轴较小，$K <= k_1$
- [09_splitk_matmul](../../../examples/09_splitk_matmul/optimized_matmul.cpp)或[22_padding_splitk_matmul](../../../examples/22_padding_splitk_matmul/padding_splitk_matmul.cpp)（带Padding前处理）
    - 选择$m_1$、$n_1$、$k_1$
        - 先设置$m_1 = 128$、$n_1 = 256$、$k_1 = 256$
        - 满足下列场景其一，修改$m_1 = 256$、$n_1 = 128$
            - A矩阵和B矩阵均为ColumnMajor
            - A矩阵为ColumnMajor，B矩阵为RowMajor，且$M > N$
    - 计算当前基本任务块 taskBlocks
        ```cpp
        taskBlocks = CeilDiv(M, m1) * CeilDiv(N, n1);
        ```
    - 满足下列两种场景：
        - 基本任务块小于一半AIC数，且$K$轴够大：$taskBlocks < aicCoreNum / 2, K > 5120$
        - 基本任务块小于3块，且$K$轴不会小：$taskBlocks <= 2, K > 1024$
- [06_optimized_matmul](../../../examples/06_optimized_matmul/optimized_matmul.cpp)（带Padding前处理）和[21_basic_matmul_preload_zN](../../../examples/09_splitk_matmul/basic_matmul_preload_zN.cpp)（手动改为ND输入）
    - 泛化性更强，适用于剩余场景
    - 不需要使用Padding时，为了节约MIX算子编译启动的开销，建议使用[21_basic_matmul_preload_zN](../../../examples/09_splitk_matmul/basic_matmul_preload_zN.cpp)模板

⚠️ 全载特性的使用[25_matmul_full_loadA](../../../examples/09_splitk_matmul/25_matmul_full_loadA.cpp) 和 单核切K方案[34_single_core_splitk_matmul](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)的适用场景待完善


</details>

<details>
<summary><strong><font size="4">Padding选择</font></strong></summary>

当Stride非512B对齐时可以考虑使用Padding前处理，但需要考虑Padding带来的开销以及MIX算子编译启动的开销（小shape[31_small_matmul](../../../examples/31_small_matmul/small_matmul.cpp)方法不推荐额外适配Padding）

`PaddingMatrixND`、`PaddingMatrixBlockND`和`PaddingMatrixNZ`各自的适用场景待完善，泛化上`PaddingMatrixNZ`更具有优势。

</details>