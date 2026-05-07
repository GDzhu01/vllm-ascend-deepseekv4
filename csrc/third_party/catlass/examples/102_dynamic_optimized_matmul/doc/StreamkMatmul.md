# StreamkMamtul

## 1 模板说明

StreamkMatmul模板是为了更极致的负载均衡而设计，相比多核切K模板改进的地方在于，StreamkMatmul模板可以通过K方向的切分将任务量完全均衡的分配到所有核心上。例如M=512，N=2048，K=1280的Matmul，假设L1Tile为m1=128，n1=256，k1=256，这时划分出来的任务数量为32，假设核心数量为20，那么第二轮计算的时候，将有8个核心空闲，如果采用MultiCoreSplitkMamtul模板，将K方向切分一次，那么任务数变为64，那么第四轮计算的时候，有16个核心空闲，还是负载不均衡。但是如果使用StreamkMatmul模板， 可以将第二轮的计算任务均分到20个核心上，从而做到比较严格的负载均衡。

StreamkMatmul的具体详细原理请参考论文[Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/abs/2301.03598)。

### 1.1 模板原理

![image-20260121101922888](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260121101922888.png)

如图所示，对于M=512，N=2048，K=1280的Shape，共有32个任务块，假设有20个核心，需要分两轮完成计算（图中的Swizzle请参考[swizzle_explanation](../../../docs/contents/advanced/swizzle_explanation.md)），第一轮每个核心都有一个任务块，负载是均衡的，所以不关心第一轮，但是第二轮，只有12个任务块计算，有8个核心空闲，如图所示：

![image-20260121102302163](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260121102302163.png)

一共有12个任务块，每个核心计算一个完整任务块，由于K=1280，k1=256，所以每个核心沿K方向切分为5份，在L0C上完成累加，最终得到一个任务块的完整结果。如果第二轮采用Streamk的思想对K进行多核切分，即有`12*5=60`份计算任务，有20个核心，每个核心计算3个任务，如图所示：

![image-20260121103528628](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260121103528628.png)

这样第二轮计算的负载就均匀分到了所有核心上。有的核心将会计算当前任务块尾段的K Tie以及下一个任务块首段的K Tile，例如core 1，这两段计算结果属于不同结果块的部分和，需要分开存储到workspace。

![image-20260121114730452](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260121114730452.png)

每个核心分配两片workspace，总的workspace大小为`2*m1*n1*sizeof(ElementAccumulator)*CoreNum`，workspace大小和shape无关，为固定大小。Mamtul计算完成后，需要AIV进行部分和的累加，得到最终的完整结果，例如20号任务块有两个部分和，分别由core0和core1计算完成，那么由core1对应的两个AIV完成累加。21号块由core1、core2、core3计算完成，那么由core1、core2对应的四个AIV完成累加。

### 1.2 关键优化点：尾轮切分

只针对尾轮进行切分，其他轮次不切分K，直接将计算结果写回到GM_C，尾轮计算时，每个核心先将部分和写入到对应的workspace，然后由对应的AIV完成部分和的累加。因为Streamk的目的是保证负载均衡，非尾轮的计算负载本身就是均衡的，就不需要K切分了，这样做也可以减少累加开销。

### 1.3 关键优化点：提前计算尾轮

将尾轮提前到倒数第二轮进行计算：

![image-20260121121815177](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260121121815177.png)

将尾轮提前好处是，Vecotor可以提前开始累加操作，和后面的Cube计算并行处理，这样Vecotor的累加开销可以被掩盖掉。

### 1.4 其他优化

StreamkMatmul模板中用到了[Preload、ShuffleK、Padding以及特殊场景的读取优化](./CommonMatmul.md)等CommonMamtul中已有的优化点。

## 2 适用场景

1. 尾轮负载不均衡。

