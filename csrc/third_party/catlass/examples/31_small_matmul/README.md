# SmallMatmul Example Readme
## 代码组织
```
├── 31_small_matmul
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── small_matmul.cpp # 主文件
```
## 功能介绍
- 该算子针对小shape场景，在basic_matmul的基础上减少不必要的scalar计算开销
- 要求切分基本块数不超过cube核数，即ceilDiv(m, L1TileShape::M) * ceilDiv(n, L1TileShape::N) <= aicCoreNum
- 要求k轴不超过L1TileShape::K
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 31_small_matmul
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./31_small_matmul 256 1024 256 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```