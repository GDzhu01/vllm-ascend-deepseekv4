# DynamicOptimizedQuantMatmulPerTokenBasic Example Readme
## 1 背景

基于泛化性要求，本样例在样例102之外，新增了处理量化Matmul的泛化工程实现，本样例目前支持PerToken-PerChannel量化Matmul基础模板。

量化被广泛应用于现代高性能计算的深度学习模型中，特别是在推理过程中。通过量化，模型可以在硬件上更高效地运行，减少计算资源的消耗和加速推理过程，同时降低模型的存储需求。

根据对算子的不同输入Tensor采用不同的量化计算级别，可将目前的量化分为不同的量化模式。目前支持的量化计算模式包括：PerToken量化和PerChannel量化。在以下的介绍中，m、n、k变量分别表示Tensor计算的不同轴大小。左矩阵、右矩阵分别指进行矩阵乘法计算的两个输入Tensor。

- PerToken量化：通常用于量化左矩阵，对每个Token（即左矩阵的一行）使用独立的量化参数进行计算。假设左矩阵shape为(m, k)，k为reduce轴，则生成的PerToken量化参数的shape为(m, )。

- PerChannel量化：通常用于量化右矩阵，对每个Channel（即右矩阵的一列）使用独立的量化参数进行计算。假设右矩阵shape为(k, n)，k为reduce轴，则生成的PerChannel量化参数的shape为(n, )。

一般左矩阵代表激活activation（A）、右矩阵代表权重weight（W），本样例目前支持左右矩阵输入数据类型为int8、对左矩阵进行PerToken量化+对右矩阵进行PerChannel量化的场景，可简记为W8A8 PerToken-PerChannel 全量化Matmul场景。

## 2 文档索引和约束说明
### 2.1 工程说明

泛化量化Matmul工程结构说明可参考：[工程结构说明](../102_dynamic_optimized_matmul_per_token_basic/doc/工程结构介绍.md)。本工程遵循与样例102类似的模板生成、Tiling计算、模板选择等流程，并根据量化Matmul计算特点进行了适配修改。

工程编译前会调用python脚本生成代码，具体包括调用各模板的外围代码，以及launch_map.h(包含tilingKey和具体Kernel的映射关系)。

本工程默认编译为动态库，编译完成执行样例前，请export动态库路径：

```shell
export LD_LIBRARY_PATH=/path/to/catlass/output/shared_lib/lib/:$LD_LIBRARY_PATH
```

### 2.2 工程结构

```shell
├── CMakeLists.txt
├── README.md
├── dynamic_optimized_quant_matmul_per_token_basic.cpp
├── impl
│   ├── kernel
│   │   ├── per_token_matmul_kernel.h
│   ├── scripts
│   │   ├── templates
│   │   │   ├── per_token_matmul_template.py
│   │   ├── utils
│   │   │   └── config.py
│   │   └── wrapper_code_gen.py
│   └── wrapper # 自动生成
│       ├── per_token_matmul_kernel_int8_t_layout00.cpp # 自动生成
│       ├── per_token_matmul_kernel_int8_t_layout01.cpp # 自动生成
│       ├── per_token_matmul_kernel_int8_t_layout10.cpp # 自动生成
│       ├── per_token_matmul_kernel_int8_t_layout11.cpp # 自动生成
└── include
    ├── do_tiling_b8.h
    ├── dynamic_optimized_matmul_w8a8.h
    ├── launch_map.h # 自动生成
    ├── platform_info.h
    ├── select_kernel_b8.h
    ├── tiling_params.h
    └── utils.h
```

### 2.2 模板文档

| 模板名称     | 说明 |
| ------------ | ---- |
| PerTokenBasicMatmul | PerToken 基础模板（文档待补充...） |


### 2.3 约束说明

- A、B矩阵的数据类型支持int8。
- C矩阵的数据类型支持fp16。
- A、B、C矩阵的数据格式支持ND（RowMajor和ColumnMajor）。


## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子

# 编译指定用例

```shell
bash scripts/build.sh 103_dynamic_optimized_quant_matmul_per_token_basic
export LD_LIBRARY_PATH=/path/to/catlass/output/shared_lib/lib/:$LD_LIBRARY_PATH
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|LayoutA|LayoutB|Device ID
# 0 is RowMajor, 1 is ColumnMajor
./103_dynamic_optimized_quant_matmul_per_token_basic 256 512 1024 0 1 0
```

执行结果如下，说明精度比对成功。

```
Compare success.
```

如果需要进行批量性能测试，请注释掉精度比较代码，由于精度比较使用CPU计算golden，耗时较长。

--------------

当前样例输出数据类型为`fp16`，如需修改为`bf16`，请进行以下代码修改后重新编译执行：
- 在`examples/103_dynamic_optimized_quant_matmul_per_token_basic/include/do_tiling_b8.h`中，将所有`DoTilingB8LayoutXX`函数中使用的`fp16_t`替换为`bfloat16`。
- 在`examples/103_dynamic_optimized_quant_matmul_per_token_basic/dynamic_optimized_quant_matmul_per_token_basic.cpp`中，搜索`fp16_t`替换为`bfloat16`。
- 在`examples/103_dynamic_optimized_quant_matmul_per_token_basic/impl/scripts/per_token_matmul_template.py`中，将`element_c`由`half`替换为`bfloat16_t`。
