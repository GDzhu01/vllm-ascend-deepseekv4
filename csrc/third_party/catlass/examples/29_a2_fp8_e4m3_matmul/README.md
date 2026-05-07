# A2Fp8E4M3Matmul Example Readme
## 代码组织
```
├── 29_a2_fp8_e4m3_matmul
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   ├── gen_data.py      # 数据生成脚本
│   └── fp8_matmul.cpp   # 主文件
```
## 功能介绍
该算子支持输入A矩阵和B矩阵的数据类型为FP8 E4M3格式（软实现），然后进行矩阵乘输出C矩阵（FP16）。
## 实现细节
1、输入处理：接收两个FP8 E4M3格式的输入矩阵A和B

2、伪量化：将FP8数据伪量化成FP16格式（per-tensor量化模式）

3、矩阵乘：使用FP16数据进行矩阵乘，中间结果使用FP32精度进行累加

4、输出转换：将最终结果转换成FP16格式输出
## 使用示例
example使用
- 第一步，编译
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
```
# 编译指定用例
bash scripts/build.sh 29_a2_fp8_e4m3_matmul
```
- 第二步，执行`gen_data.py`生成测试数据，测试用例规格从命令行输入
```
cd examples/29_a2_fp8_e4m3_matmul && python gen_data.py 256 512 1024 0 0 && cd ../..
# 输入参数分别对应 m, n, k, trans_a, trans_b
# trans_a表示A矩阵是否转置，0是不转置，1是转置
# trans_b表示B矩阵是否转置，0是不转置，1是转置
```
执行该命令后会在当前路径下生成input和output目录，包含算子的输入数据和用于精度验证的golden数据
```
├── input
│   ├── a_8.bin
│   ├── b_8.bin
└── output
    └── expected_data.bin
```

- 第三步，执行算子，注意提供给算子的输入shape和上面第二步生成数据的shape需一致
```
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./output/bin/29_a2_fp8_e4m3_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```

## 说明

1、 `gen_data.py`的输入支持trans_a和trans_b，但29_a2_fp8_e4m3_matmul可执行文件不支持，仅仅是trans_a和trans_b均为0的example示例。

若要对应转置情况请修改example示例中的layout，因为layout隐式表征转置状态，即layout::RowMajor表示不转置，layout::ColumnMajor表示转置。

其对应关系如下表：

| trans_a | trans_b | LayoutA             | LayoutB             |
|---------|---------|---------------------|---------------------|
| 0       | 0       | layout::RowMajor    | layout::RowMajor    |
| 0       | 1       | layout::RowMajor    | layout::ColumnMajor |
| 1       | 0       | layout::ColumnMajor | layout::RowMajor    |
| 1       | 1       | layout::ColumnMajor | layout::ColumnMajor |

2、对比FP16 Matmul，该样例针对大shape的case有较为明显的显存收益

3、针对小shape场景，可以参考[catlass_optimize_guidance](../../docs/contents/advanced/catlass_optimize_guidance.md#tiling优化)对样例进行tiling调优