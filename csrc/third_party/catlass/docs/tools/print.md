# 在CATLASS样例工程进行设备侧打印

基于[CCE Intrinsic](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/cceintrinsicguide/cceprogram_0001.html)提供设备侧打印函数`cce::printf`进行调试，用法与C标准库的`printf`一致。

- 支持`cube/vector/mix`算子
- 支持格式化字符串
- 支持打印常见整型与浮点数、指针、字符

  - ⚠️ **注意** 这个功能在社区版CANN 8.3后（如[8.3.RC1.alpha001](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha001)）开始支持。

## 使用示例

下面以对`09_splitk_matmul`为例，进行设备侧打印的使用说明。

### 插入打印代码

在想进行调试的代码段增加打印代码。

```diff
// include/catlass/gemm/kernel/splitk_matmul.hpp
// ...
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementOut> const &dst,
        AscendC::GlobalTensor<ElementAccumulator> const &src,
        uint64_t elementCount, uint32_t splitkFactor)
    {
        // The vec mte processes 256 bytes of data at a time.
        constexpr uint32_t ELE_PER_VECTOR_BLOCK = 256 / sizeof(ElementAccumulator);
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint64_t taskPerAiv =
            (elementCount / aivNum + ELE_PER_VECTOR_BLOCK - 1) / ELE_PER_VECTOR_BLOCK * ELE_PER_VECTOR_BLOCK;
        if (taskPerAiv == 0) taskPerAiv = ELE_PER_VECTOR_BLOCK;
        uint32_t tileLen;
        if (taskPerAiv > COMPUTE_LENGTH) {
            tileLen = COMPUTE_LENGTH;
        } else {
            tileLen = taskPerAiv;
        }
+       cce::printf("tileLen:%d\n", tileLen);
// ...
    }
```

### 编译运行

1. 基于[快速上手](../../README.md#快速上手)，打开工具的编译开关`--enable_print`， 使能设备侧打印特性编译算子样例。

```bash
bash scripts/build.sh --enable_print 09_splitk_matmul
```

2. 切换到可执行文件的编译目录`output/bin`下，直接执行算子样例程序。

```bash
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
./09_splitk_matmul 256 512 1024 0
```

- ⚠ 注意事项
  - 目前`设备侧打印`仅支持打印`GM`、`UB`和`SB(Scalar Buffer)`上的数值。

### 输出示例

输出结果

```bash
./09_splitk_matmul 256 512 1024 0
-----------------------------------------------------------------------------
---------------------------------HiIPU Print---------------------------------
-----------------------------------------------------------------------------
==> Logical Block 0
=> Physical Block

=> Physical Block
tileLen:2752

=> Physical Block
tileLen:2752

==> Logical Block 1
=> Physical Block

=> Physical Block
tileLen:2752

=> Physical Block
tileLen:2752

... # 此处省略

==> Logical Block 23
=> Physical Block

=> Physical Block
tileLen:2752

=> Physical Block
tileLen:2752

Compare success.
```
