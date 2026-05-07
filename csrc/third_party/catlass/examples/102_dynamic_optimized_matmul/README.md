# DynamicOptimizedMatmul Example Readme
## 1 背景

在当今高性能计算核深度学习领域，矩阵乘法（Matmul）作为核心算子，其计算效率和泛化性的重要性无需赘述。对计算效率来说，常见的评价指标是算力利用率，也就是实际Flops/理论Flops。而泛化性很难以单一指标评价，在一些业务场景中，Matmul的输入范围变化很大，如推荐场景，MNK变化范围在100000量级，要求在这个范围下，整体有一个不错的性能，也就是泛化性的要求。相比于特定场景（某一类shape）、特定shape的性能要求来说，泛化性提出了更高的要求。

## 2 文档索引和约束说明
### 2.1 工程说明

泛化Matmul工程结构说明请参考：[工程结构说明](./doc/工程结构介绍.md)

工程默认编译成静态库，如果想编译成动态库，请把CMakeLists.txt中的`STATIC`改为`SHARED`，并手动export动态库路径。
工程编译前会调用python脚本生成代码，具体包括调用各模板的外围代码，以及launch_map.h(包含tilingKey和具体Kernel的映射关系)。

DynamicOptimizedMatmul根据shape动态确定Tiling参数，并尝试选择最好的模板进行计算，尽力获取最优性能，但是不保证是最优性能。

### 2.2 模板文档

| 模板名称     | 说明 |
| ------------ | ---- |
| [CommonMatmul](./doc/CommonMatmul.md) | 基础模板 |
| SmallMatmul | 文档待补充... |
| [MultiCoreSplitkMatmul](./doc/MultiCoreSplitkMatmul.md) | 多核切K模板(适用于C矩阵较小场景) |
| [StreamkMatmul](./doc/StreamkMatmul.md) | 负载更均衡的多核切K模板          |
| SingleCoreSplitkMatmul | 文档待补充... |



### 2.3 相关约束

1. A、B、C矩阵的数据类型支持fp16。

2. A、B、C矩阵的数据格式支持ND（RowMajor和ColumnMajor）。

## 3 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 102_dynamic_optimized_matmul
# 动态库编译，需要手动添加动态库路径到LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/output/shared_lib:$LD_LIBRARY_PATH
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|LayoutA|LayoutB|Device ID
# 0 is RowMajor, 1 is ColumnMajor
./102_dynamic_optimized_matmul 256 512 1024 0 1 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```

如果需要进行批量性能测试，请注释掉精度比较代码，由于精度比较使用CPU计算golden，耗时较长。
