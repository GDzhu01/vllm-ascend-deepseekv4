# SingleSplitK_Matmul Example Readme

## 功能说明

 - 算子功能：优化的矩阵乘计算（优化策略详见[单核切K策略说明](../../docs/contents/example_design/34_single_slice_k_matmul.md)）


## 参数说明

本样例直调参数包括m, n, k, deviceId，与[00_basic_matmul参数](../00_basic_matmul/README.md#参数说明)条件一致。
相应地，算子原型有如下设计：
|名称/Name|类型/Class|数据类型/Dtype|维度/Dims|格式/Format|描述/Description|
|---|---|---|---|---|---|
|matA|inTensor|fp16\|bf16\|fp32|[m, k]|ND\|NZ|左矩阵，支持转置|
|matB|inTensor|fp16\|bf16\|fp32|[k, n]|ND\|NZ|右矩阵，支持转置|
|matC|outTensor|fp16\|bf16\|fp32|[m, n]|ND|输出矩阵|

## 约束说明



无

## 代码组织
本样例组织结构如下：
```
├── 34_single_splitk_matmul
│   ├── CMakeLists.txt           # CMake编译文件
│   ├── single_core_splitk.cpp   # 主文件
│   └── README.md                
```

## 使用示例

1. 编译样例代码，并编译生成相应的算子可执行文件。
```bash
# 编译指定用例
bash scripts/build.sh 34_single_core_splitk_matmul
```

2. 切换到可执行文件的编译目录`output/bin`下，并执行算子样例程序。类似于基础样例[00_basic_matmul](../00_basic_matmul/README.md)，测试数据根据命令行输入尺寸随机生成。

```bash
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./34_single_core_splitk_matmul 256 512 1024 0
```
• 256：矩阵m轴

• 512：n轴

• 1024：k轴

• 0：Device ID，可选，默认为0


执行结果如下，说明精度比对成功。
```
Compare success.
```