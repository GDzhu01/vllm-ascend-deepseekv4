# MatmulFullLoadA Example Readme
## 代码组织
```
├── 25_matmul_full_loadA
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── matmul_full_loadA.cpp # 主文件
```
## 功能介绍
- 该算子在00_basic_matmul基础上支持A矩阵全载（需要一半的L1空间分配给数据量为`L1TileShape::M * problemShape.K`的矩阵），计算每个基本块时完整搬入A矩阵分块，而后pingpong搬入B矩阵；若L1空间不够A矩阵全载，则返回报错；
- A矩阵全载时，N轴越大，单核越能多次复用L1中的A矩阵、无需再从GM或L2Cache搬运A矩阵，性能收益就越大；
- A矩阵全载时，N轴较小，无法复用A矩阵，性能收益较00_basic_matmul可能会出现劣化；
- 若problemShape.M <= L1TileShape::M，即M方向不切块分核，此时常用GemmIdentityBlockSwizzle策略即可适用；
- 若problemShape.M > L1TileShape::M，新增配套的GemmIdentityBlockSwizzleL1FullLoad<SwizzleOffset, SwizzleDirection, AicCoreNum>，使得每个核需要处理的基本块尽可能地连在一起，提升A矩阵分核全载时的块间复用率；
- 以24个cube核为例，常用的GemmIdentityBlockSwizzle策略，基本块分核顺序为0-1-2-...-22-23-0-1-2...-22-23-0-1-2...，每个核需要处理的基本块跳跃分布。新增的GemmIdentityBlockSwizzleL1FullLoad策略，基本块分核顺序为0-0..-0-1-1...-1-2-2...-23，每个核需要处理的基本块连续分布。
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 25_matmul_full_loadA
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./25_matmul_full_loadA 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```