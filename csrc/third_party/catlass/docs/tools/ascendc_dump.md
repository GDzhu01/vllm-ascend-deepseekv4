# 在CATLASS样例工程使用AscendC算子调测API

AscendC算子调测API是AscendC提供的调试能力，可进行kernel内部的打印([`printf`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/ascendcopapi/atlasascendc_api_07_0193.html))、Tensor内容的查看([`DumpTensor`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/ascendcopapi/atlasascendc_api_07_0192.html))。

## 使用示例

下面以`00_basic_matmul`为例，演示基于[AscendC算子调测API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/ascendcopapi/atlasascendc_api_07_0192.html)的测试过程。

### 插入调试代码

在想进行调试的层级，增加调测API调用，如在`include/catlass/gemm/kernel/basic_matmul.hpp`的核函数中添加下述代码。

```diff
// include/catlass/gemm/kernel/basic_matmul.hpp
template <>
CATLASS_DEVICE
void operator()<AscendC::AIC>(Params const &params) {
    BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
    uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
+   AscendC::printf("CoreLoops is %d\n", coreLoops);
    Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad(resource);

    // Represent the full gm
    AscendC::GlobalTensor<ElementA> gmA;
    gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
    AscendC::GlobalTensor<ElementB> gmB;
    gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
+   AscendC::DumpTensor(gmA, coreLoops, 16);
    for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        // Compute block location
        GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
        GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

        // Compute initial location in logical coordinates
        MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
        MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
        MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
        int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
        int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
        int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

        // Compute block-scoped matrix multiply-add
        blockMmad(gmA[gmOffsetA], params.layoutA,
                    gmB[gmOffsetB], params.layoutB,
                    gmC[gmOffsetC], params.layoutC,
                    actualBlockShape);
    }
}
```

### 编译运行

1. 基于[快速上手](../../README.md#快速上手)，打开工具的编译开关`--enable_ascendc_dump`， 使能AscendC算子调测API编译算子样例。

```bash
bash scripts/build.sh --enable_ascendc_dump 00_basic_matmul
```

2. 切换到可执行文件的编译目录`output/bin`下，直接执行算子样例程序。

```bash
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
./00_basic_matmul 256 512 1024 0
```

- ⚠ 注意事项
  - 目前[`AscendC::DumpTensor`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/ascendcopapi/atlasascendc_api_07_0192.html)**不**支持打印`FixPipe`上的数值。

### 输出示例

```bash
./00_basic_matmul 256 512 1024 0
opType=device_gemm, DumpHead: AIC-0, CoreType=AIC, block dim=24, total_block_num=24, block_remain_len=1048408, block_initial_space=1048576, rsv=0, magic=5aa5bccd
CoreLoops is 4
DumpTensor: desc=4, addr=c0013000, data_type=float16, position=GM
[3.402344, -1.056641, 2.830078, 2.984375, 4.117188, -3.025391, -1.647461, 2.681641, -2.222656, 0.539551, -0.226074, 1.289062, -1.352539, 0.134033, 4.523438, 4.160156]
... #每个Cube核都会输出一次信息
Compare success.
```