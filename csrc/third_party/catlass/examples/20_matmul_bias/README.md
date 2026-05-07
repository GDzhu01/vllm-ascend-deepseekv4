# MatmulBias Example Readme
## 代码组织
```
├── 20_matmul_bias
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── matmul_bias.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- matmul_bias在对float32输入数据计算时，为防止超过L1 Cache上限，推荐tiling如下：
```cpp
using L1TileShape = GemmShape<112, 128, 256>;
using L0TileShape = GemmShape<112, 128, 64>;
```
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 20_matmul_bias
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./20_matmul_bias 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```