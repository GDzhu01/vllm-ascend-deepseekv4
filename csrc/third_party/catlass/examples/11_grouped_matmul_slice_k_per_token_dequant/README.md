# GroupedMatmulSliceKPerTokenDequant Example Readme
## 代码组织
```
├── 11_grouped_matmul_slice_k_per_token_dequant
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── 11_grouped_matmul_slice_k_per_token_dequant.cpp # 主文件
```
## 功能介绍
该算子支持A矩阵在k轴切分，和B矩阵按照group分组进行矩阵乘。之后进行per token的量化操作。
A/B矩阵为int8类型，scale为bf16，输出结果为bf16
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 11_grouped_matmul_slice_k_per_token_dequant
cd output/bin
# 可执行文件名|group数量|矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./11_grouped_matmul_slice_k_per_token_dequant 128 512 1024 2048 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```