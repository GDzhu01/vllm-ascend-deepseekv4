# QuantMatmulFullLoadA Example Readme
## 代码组织
```
├── 44_quant_matmul_full_loadA_tla
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── quant_matmul_full_loadA_tla.cpp # 主文件
```

## 功能介绍

- 该算子在12_quant_matmul基础上支持A矩阵全载，支持单核将A矩阵全部载入L1Cache并常驻，以减少A矩阵在部分矩阵计算场景中的重复搬运，提高性能。当前A矩阵全载模板暂不支持输入包含bias。
- 如下图所示，模板库矩阵有关键参数 `L1TileShape<M, N, K>`，C矩阵按照`L1TileShape::M`和`L1TileShape::N`参数切分基本块并分核，而后普通Matmul模板会将A矩阵中`L1TileShape::M * L1TileShape::K`大小的矩阵块载入L1，而A矩阵全载模板会直接将`L1TileShape::M * K`大小的矩阵块载入L1，而对于B矩阵，普通模板和A矩阵全载模板都是载入`L1TileShape::K * L1TileShape::N`大小的矩阵块L1。

![A矩阵全载示意图](https://raw.gitcode.com/user-images/assets/7631999/38ccafec-7fd8-4b9f-a4bb-dd57085ffb63/53e318e0feb94c578d888e67365e4c97.png_tplv-a9rns2rl98-image-qvalue.png '53e318e0feb94c578d888e67365e4c97.png~tplv-a9rns2rl98-image-qvalue.png')

- A矩阵全载模板相较普通模板的流水变动如下图所示，使用全载方案，在计算一开始就经MTE2完整搬入A矩阵的数据块，而后pingpong搬入B矩阵执行计算。

![A矩阵全载流水图](https://raw.gitcode.com/user-images/assets/7631999/1de46727-7c46-411e-936c-7a437d951a3a/3e9c799e1de0405d89f07a6bfd7d7c54.png_tplv-a9rns2rl98-image-qvalue.png '3e9c799e1de0405d89f07a6bfd7d7c54.png~tplv-a9rns2rl98-image-qvalue.png')

- 采用A矩阵全载模板时，需要一半的L1空间以放入`L1TileShape::M * problemShape.K`的数据，若L1空间不够A矩阵全载，则返回报错。
- A矩阵全载时，N轴越大，单核越能多次复用L1中的A矩阵、无需再从GM或L2Cache搬运A矩阵，性能收益就越大。
- A矩阵全载时，N轴较小，无法复用A矩阵，性能收益较00_basic_matmul可能会出现劣化。
- 若`problemShape.M > L1TileShape::M`，可使用`GemmIdentityBlockSwizzleL1FullLoad<SwizzleOffset, SwizzleDirection, AicCoreNum>`策略，使得每个核需要处理的基本块尽可能地连在一起，提升A矩阵分核全载时的块间复用率。
- 若`problemShape.M <= L1TileShape::M`，即M方向不切块分核，此时使用`GemmIdentityBlockSwizzle`策略即可适用，常用的策略参数选取可参考[swizzle_explanation](./contents/advanced/swizzle_explanation.md)。
- 以20个cube核为例，常用的`GemmIdentityBlockSwizzle`策略的基本块分核顺序为`0-1-2-...-18-19-0-1-2...-18-19-0-1-2...`，每个核需要处理的基本块跳跃分布，而`GemmIdentityBlockSwizzleL1FullLoad`策略的基本块分核顺序为`0-0..-0-1-1...-1-2-2...-19`，每个核需要处理的基本块连续分布。

## 性能收益

使用相同tileShape和swizzle参数，相比标杆样例12_quant_matmul，具备A矩阵全载特性的样例性能平均提升5%~15%，这是由于全载实现减少了计算过程中的A矩阵搬运，提升了全载矩阵的数据复用率，可参考下表。

| [M,N,K]            | 12_quant_matmul | 44_quant_matmul_full_loadA_tla |
| ------------------ | --------------- | ------------------------------ |
| [512, 4096, 1024]  | 33.26 us        | 31.73 us                       |
| [128,16384, 1024]  | 38.68 us        | 37.05 us                       |
| [1024, 4096, 1024] | 42.66 us        | 41.96 us                       |
| [512,8192,1024]    | 52.36 us        | 44.05 us                       |
| [128,16384,2048]   | 53.77 us        | 52.75 us                       |

说明：

- 标杆为[QuantMatmul](../12_quant_matmul/README.md)算子。
- 统计耗时均为核函数总耗时，使用[msprof](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/devaids/optool/atlasopdev_16_0082.html)工具得到。
- 上述测试例中A、B及输出矩阵均为`layout::RowMajor`排布方式。
- 测试环境说明：NPU型号为910B1，CANN包版本为9.0.0。

## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子

```
# 编译指定用例
bash scripts/build.sh 44_quant_matmul_full_loadA_tla
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./44_quant_matmul_full_loadA_tla 256 512 1024 0
```
执行结果如下，说明精度比对成功。5
```
Compare success.
```