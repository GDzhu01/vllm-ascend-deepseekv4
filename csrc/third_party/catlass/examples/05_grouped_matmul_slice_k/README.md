# GroupedMatmulSliceK Example Readme
## 代码组织
```
├── 05_grouped_matmul_slice_k
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── grouped_matmul_slice_k.cpp # 主文件
```
## 功能介绍
该算子支持A矩阵在k轴切分，并支持B矩阵按照group分组进行矩阵乘
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 05_grouped_matmul_slice_k
cd output/bin
# 可执行文件名 group数量|m轴|n轴|k轴|Device ID
./05_grouped_matmul_slice_k 128 512 1024 2048 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```