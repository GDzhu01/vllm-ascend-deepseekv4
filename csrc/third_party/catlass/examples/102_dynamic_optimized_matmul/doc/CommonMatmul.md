# CommonMatmul
## 1 模板说明
在泛化Matmul中存在两个CommonMatmul模板，一个是纯CUBE类型，名为CommonMatmulKernel，一个是MIX类型，名为PaddingCommonMatmulKernel，如果需要用到AIV进行数据格式转换（参考2.3节），则会进入PaddingCommonMatmulKernel，之所以区分这两个模板，是因为编译一个Kernel的时候，要么指定这个Kernel是MIX类型的，要么指定这个Kernel是纯CUBE或纯AIV的，而MIX算子由于需要同时启动AIC和AIV，启动开销会比纯CUBE或纯AIV大，尤其在小shape场景下性能影响显著，如果不需要用到AIV，就不需要启动AIV，进入纯Cube类型的CommonMatmulKernel会有更好的性能。

![image-20251209154303323](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251209154303323.png)

CommonMatmul的典型特征是使用L1上的基本块直接切分矩阵A、B、C，先将C矩阵划分未若干个大小为`L1TileM x L1TileN`的基本任务块，然后以`L1TileM x L1TileK`为粒度读取A矩阵到L1，以`L1TileK x L1TileN`为粒度读取B矩阵到L1。每个任务块的完整计算结果在一个AI Core上产生。为了避免不必要的L1->L0的重复数据搬运，在CommonMatmul中，L0TileM=L1TileM，L0TileN=L1TileN，L0TileK在L0的空间约束下取最大值。

## 2 优化点

此处介绍的优化点是[00_basic_matmul](../../example/00_basic_matmul/README.md)里面没有，但是CommonMatmul中有的优化，至于00_basic_matmul中已有的基础优化在此不做介绍。00_basic_matmul用到的优化点为CommonMatmul的子集。

### 2.1 Preload

CommonMatmul中的Preload实现伪代码如下：

```c++
...
uint32_t kTileCount = CeilDiv(actualShape.k(), l1TileShape.k());
...
// block层的k循环
for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
    // 如果是当前核心计算的第一个C Block块，而且kLoopIdx = 0的时候，需要加载当前计算的Tile块的数据。
    // 如果当前计算的不是第一个C Block块或者kLoopIdx > 0的时候，就不需要加载当前计算的Tile块数据，因为数据已经在上一轮循环加载好了。
    if (isFirstBlock && kLoopIdx == 0) {
        // 根据kLoopIdx计算需要载入GM块的地址
        copyGmToL1A;
        copyGmToL1B;
    }

    // 块内的Preload，预先加载当前计算的C Block的下一个A Tile块和B Tile块。
    if ((kLoopIdx != kTileCount - 1)) {
        // 根据kLoopIdx + 1计算需要载入GM块的地址，因为载入的是下一个需要计算的块的数据。
        copyGmToL1A;
        copyGmToL1B;
    }

    // 块间的Preload，预先加载下一个需要计算的C Block对应的第一个A Tile块和B Tile块。
    if ((kLoopIdx != kTileCount - 1) && hasNextBlock) {
        // 加载下一个需要计算的C Block的对应第一个A Tile块和B Tile块
        copyGmToL1A;
        copyGmToL1B;
    }

    ...
    tileMmad;
    ...
}
...
copyL0CToGm;
...
```

实际代码实现见[block_mmad_dynamic_common.h](../../../include/catlass/gemm/block/block_mmad_dynamic_common.hpp)。

`isFirstBlock`：是否为当前核心计算的第一个C Block。如果是则为`true`。

`hasNextBlock`：当前核心是否存在下一个需要计算的C Block，如果存在则为`true`。

`isFirstBlock`和`hasNextBlock`参数由kernel层传到Block层。

Preload的核心思想为，在计算当前Tile的Matmul前，预先加载下一个需要计算的Tile块的数据，这样就做到了搬运指令的提前发射，可减少MTE2流水的空泡。

关于Preload更细节的描述可以参考[矩阵乘模板总结-流水优化(Preload)](../../../docs/contents/advanced/matmul_template_summary.md)。

### 2.2 ShuffleK

![image-20251209170344805](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251209170344805.png)

多核心同时读取同一片GM数据的时候，存在数据读取冲突，导致读取带宽降低。

通常，所有核心都从K方向第一个分块开始计算，为了缓解上面的问题，让不同的核心从不同的分块开始计算，这样各个核心不再同时读取同一片GM数据。

图中C矩阵的中的数字为核心序号，AB矩阵中的数字为任务块序号，左图为通常的计算方式，图中2号核心和3号核心同时按0 1 2 3的顺序访问A矩阵的同一片GM数据。

右图为采用ShuffleK优化后的计算方式，右图中2号核心按2 3 0 1的顺序访问A矩阵基本块，而3号核心按3 0 1 2的顺序访问A矩阵基本块，在时间上错开，从而避免同地址访问冲突。

图中的分块计算顺序采用GemmIdentityBlockSwizzle<2,1>，请参考[swizzle_explanation](../../../docs/contents/advanced/swizzle_explanation.md)
### 2.3 Padding

在A2或A3上，当A或B矩阵为ND（RowMajor或ColumnMajor）格式时，如果矩阵的Stride为非512B对齐，则ND2NZ搬运接口的带宽会显著下降。为了规避这个问题，采用AIV提前对A或B矩阵进行数据格式转换（或数据填充），目的是让GM2L1搬运时候，避免非用512B对齐的Stride访问GM数据。

#### 2.3.1 矩阵A或B的Padding模式说明

当前泛化Matmul支持三种Padding方式，在[Padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)中定义了枚举值：

```
enum class PaddingTag {
    NO_PADDING,
    PADDING_ND,
    PADDING_BLOCK_ND,
    PADDING_NZ    
};
```

以上的Padding操作都是全局的操作，即针对整个A矩阵或B矩阵，需要开辟A矩阵和B矩阵（对齐后）同等大小的workspace。由AIV进行Padding处理，AIC等待处理完成后开始Matmul计算。

- PADDING_ND

  ![image-20251209193549117](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251209193549117.png)

  PADDING_ND只是简单把Stride方向填充对齐到512B对齐，例如图中的512 x 511的shape，Stride为511，经过PADDING_ND后Stride变为512，对齐到了512B（对于half类型，也就是要求Stride为256倍数）。

- PADDING_BLOCK_ND

  ![image-20251209193625747](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251209193625747.png)

  PADDING_BLOCK_ND会重排矩阵数据，以L1Tile块的粒度对数据进行重组，如果原矩阵为RowMajor，如图所示，重组后每个Tile块内RowMajor排布，然后Tile块间也是RowMajor排布，是一个嵌套的数据格式。这样做的好处是不管原矩阵的Stride为多少，通过这个变换后，每个Tile块内的Stride都变为L1TileK（一般是512B对齐的），规避了Stride过大（超过65536）或者非对齐造成的带宽劣化。

- PADDING_NZ

  ![image-20251209193718695](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251209193718695.png)

PADDING_NZ直接将RowMajor的矩阵转换为zN的格式（如果是ColumMajor的话就是nZ），这样GM2L1读取的时候不再需要ND2NZ的随路转换，也就规避了ND2NZ存在的所有劣化问题。

PADDING_NZ有两种实现，分别用于不同的场景：

- 矩阵的Stride方向（矩阵的内轴）较大的时候，AIV一次读取16行数据（repeat读取，虽然repeat的stride非对齐，但是每次repeat读取的数据量大，非对齐对带宽影响不大），在UB上通过Copy指令将数据重排为zN，然后写出到workspace。
- 矩阵的Stride方向（矩阵的内轴）较小的时候（一般是小于96），如果按第一种方案读取数据，AIV的读取效率也很低。这时候将非对齐的矩阵数据连续读取到UB，采用[TransDataTo5HD](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_0200.html)指令经过两次转置操作完成数据的Padding，同时顺带完成ND数据转NZ。

由于PADDING_NZ有较优的总体性能，所以泛化Matmul中实际采用的是PADDING_NZ，至于其他两种Padding方式，在某些特定的Shape上可能有比PADDING_NZ更好的性能，可以多尝试进行调优。

#### 2.3.2 C矩阵的Padding模式说明

由于基本块的计算结果在L0C上是zN排布的，L0C到UB也需要经过NZ2ND的数据转换，如果C矩阵的Stride非512B对齐，同样也存在带宽明显劣化的问题。

![image-20251210001807177](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251210001807177.png)

为了规避上面问题，当前CommonMatmul的做法是申请与C矩阵同样大小（对齐后）的workspace，当L0C上结果计算完成后，Fixpipe按对齐的Stride先写出到workspace，然后用AIV进行去Padding操作，将最终结果写回到GM C。

当前在CommonMatmul中是否对C矩阵进行Padding的判断条件如下：

```c++
PaddingTag paddingTagC = PaddingTag::PADDING_NONE;
// 要求C矩阵不能太小，且要求非256B对齐
if (static_cast<size_t>(m) * n > 2048 * 2048 && n > 256 && (n % 128 != 0)) {
    size_t totalDataSize = static_cast<size_t>(m) * k * CeilDiv(n, n1) * 2
        + static_cast<size_t>(k) * n * CeilDiv(m, m1) * 2 + static_cast<size_t>(m) * n * 2;
    // 要求读写总数据量小于L2 Cache大小
    if (totalDataSize < 192 * 1024 * 1024) { // L2 cache size
        paddingTagC = PaddingTag::PADDING_ND;
    }
}
```

如果总数据量大小超过了L2 Cache大小，那么AIV进行去Padding的时候，就可能从GM读取数据，这时候去Padding开销就很大，以至于开销大于收益。

#### 2.3.3 Padding建模（决定A矩阵或B矩阵是否需要Padding）

是否Padding影响Matmul的性能，Padding有额外开销，如果Padding后带来的带宽收益无法抵消Padding的额外开销，那么Padding就会是负收益。Padding的开销主要来自下面两个方面：

- AIV的启动开销，只要启动了mix kernel，就会有1~5us左右的额外开销。
- Padding操作的开销，即AIV的读写计算耗时。

在实际的测试中，有下面的规律：

- 一般在非对齐的时候，需要进行Padding优化非对齐矩阵的读取带宽，但是如果该非对齐矩阵在Matmul计算过程中没有重复的数据读取，就不应该进行Padding。
- 如果矩阵的Stride方向长度小于64，nd2nz的随路转换读取带宽会非常差，此时应该进行Padding，即使该矩阵没有重复的数据读取。
- 如果矩阵过于小，即矩阵的rows和cols都小于某个数，此时也不需要进行Padding，一方面小矩阵可以常驻L1不用重复读取，一方面Padding的收益无法抵消额外开销。
- 在矩阵重复读取量不大的情况下，如果矩阵的Stride方向为256B对齐，不Padding会更好。

但是很难通过上面的规律总结出一个确定的Padding策略，有很多的阈值难以确定。所以需要对Padding过程进行建模。

设置以下量：

- B_aiv，AIV的单核读取带宽。
- B_aic512，AIC nd2nz stride 512B对齐情况下的单核读取带宽。
- B_aicunalign AIC stride非512B对齐情况下的单核读取带宽。
- T_headcost，启动mix kernel带来的额外开销。
- M,N,K,m1,n1,k1，矩阵shape和L1 Tile块大小。
- C，CUBE核心数量，V，VECTOR核心数量。

单核最大的计算轮次为：
<center>R_max = RoundUp(CeilDiv(M, m1) * CeilDiv(N, n1) / C)</center>

单核AIC A和B矩阵最大的数据读取量分别为：
<center>CRead_a = R_max * (m1 * K)，CRead_b = R_max * (n1 * K)</center>

假设A矩阵和B矩阵都进行Padding，那么AIV的单核读取数据量为：

<center>VRead_a = M * K / V，VRead_b = K * N / V</center>

那么AB都Padding的AIC和AIV总读取耗时为：

<center>T_11 = (CRead_a + CRead_b) / B_aic512 + (VRead_a + VRead_b) / B_aiv + T_headcost</center>

如果不进行Padding，那么总读取耗时为：

<center>T_00 = (CRead_a + CRead_b) / B_aicunalign</center>

如果Padding A矩阵但是不Padding B矩阵，总读取耗时为：

<center>T_10 = CRead_a / B_aic512 + VRead_a / B_aiv + CRead_b / B_aicunalign + T_headcost</center>

如果Padding B矩阵但是不Padding A矩阵，总读取耗时为：

<center>T_01 = CRead_b / B_aic512 + VRead_b / B_aiv + CRead_a / B_aicunalign + T_headcost</center>

比较T00、T01、T10、T11这四个时间，哪个时间最短，就采用哪种Padding策略。

B_aiv、B_aic512、T_headcost这几个值在硬件相同时，可以简化认为是固定值，可通过实测获得。而B_aicunalign的值，和nd2nz的参数有关（nValue、dValue和srcDValue）需要通过实测数据进行曲线拟合得到计算公式。

以上逻辑为Padding建模的简化过程，具体实现有更细节的考虑，参考[select_kernel_bf16.h](../include/select_kernel_b16.h)，其中的多项式拟合公式仅适用与A2、A3，如果是其他型号，需要自测数据拟合曲线。

### 2.4 特殊场景的读取优化

#### 2.4.1 场景一

![image-20251210102254855](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251210102254855.png)

ND2NZ可以使用普通间隔搬运的DataCopy取代，即for循环每次搬运一行，每一行调用DataCopy Repeat多次进行搬运，代码如下：

```c++
// 使用普通间隔搬运的DataCopy取代ND2NZ，以half为例
for (int i = 0; i < nValue; ++i) {
    AscendC::DataCopyParams dataCopyParams {
        (dValue + 15) / 16, // repeat times
        1, // 一次搬运的数据量，以32B为单位
        0，// Gm上的Stride，以32B为单位
        (256 - 16) / 16  // L1上的Stride，以32B为单位
    };
    AscendC::DataCopy(buffer[i * 16], gmSrc[i * srcDValue], dataCopyParams);
}
```

在M很小的时候，通常是M < 8，ND2NZ的随路转换的搬运效率没有上面的逐行拷贝效率高，所以CommonMatmul在这种场景下采用上面的间隔拷贝的DataCopy替换ND2NZ，以获取更高的搬运效率。

#### 2.4.2 场景二

![image-20251210104718451](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20251210104718451.png)

如图所示当K=16时候，矩阵数据在GM上的数据排布与在L1上的数据排布相同，此时不需要使用随路ND2NZ接口进行搬运，直接进行连续的数据拷贝即可。直接的连续拷贝肯定比ND2NZ的随路转换带宽更高。

#### 2.4.3 场景三

当矩阵A为ColumnMajor，且M=1的时候，此时GM2L1读取的时候，每行只有一个元素，读取效率会非常低，此时将A矩阵当作一个1 x K的RowMajor矩阵进行计算（因为A矩阵相当于是一个向量，既可以是行优先也可以是列优先，两者是等价的），这样读取A矩阵的时候，A矩阵就是一个行向量，读取效率就会高很多。

同理，当矩阵B为RowMajor，且N=1的时候，此时将B矩阵当作一个K x 1的ColumnMajor矩阵进行计算。此处的逻辑参考[select_kernel_bf16.h](../include/select_kernel_b16.h)。

至于K=1的时候，需要交给特殊的模板（采用AIV计算）处理。

