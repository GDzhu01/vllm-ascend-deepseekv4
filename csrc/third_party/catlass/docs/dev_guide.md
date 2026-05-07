# 基础开发指南


---
> 本文以基础Matmul算子（矩阵乘）为基础（参考样例：[00_basic_matmul](../examples/00_basic_matmul/basic_matmul.cpp)），介绍基于CATLASS的算子开发与执行过程，助力快速上手模板库基础开发实践。

<div align="center">

```mermaid
graph LR
    S[环境准备]--> GroupT[算子开发]

    subgraph GroupT[算子开发]
        direction LR
        A[Kernel层算子] --> B[Device层算子]
        B --> C[Host侧算子调用]
    end

    GroupT --> P[算子编译]
    P --> E[算子执行]

    style S fill:#e1f5fe
    style P fill:#e8f5e8
    style E fill:#fce4ec
```
</div>

[TOC]

---


## 环境准备

模板库依赖于[CANN](https://www.hiascend.com/zh/developer/download/community/result?module=cann)环境，您可查阅[环境准备](./quickstart.md)一节以了解具体配置过程。

## Matmul算子开发

[CATLASS](https://gitcode.com/cann/catlass)模板库为GEMM类算子开发提供了基础组件，要实现基础的Matmul算子只需将各组件按分层结构组装即可。具体而言，CATLASS分层结构由上至下包括`Device`、`Kernel`、`Block`、`Tile`和`Basic`五层（分层示意图详见[Gemm API文档](./contents/advanced/api.md)），每一层对GEMM类算子的计算过程作不同级别的抽象封装，并利用下层基础组件实现该层级的计算逻辑。

为快速上手使用，开发者不必关注各级的具体细节，只需在`Kernel`层、`Device`层以模板参数传递的方式“拼装”出Matmul算子，并调用即可。
 -  下述内容介绍了`BasicMatmul`中的核心组件与设计思路，请参考全量样例[00_basic_matmul](../examples/00_basic_matmul/basic_matmul.cpp)。

### Kernel层算子定义

`Kernel`层模板由`Block`层组件构成，需要利用以下三个`Block`层组件。

1. `BlockMmad`封装了`Block`层的`mmad`计算（矩阵乘计算），对应于昇腾NPU的一个`AI Core`上的计算。
通过模板参数，`BlockMmad_`接收矩阵计算中的`Shape`（特征尺寸）、`Layout`（数据排布，如行优先、列优先排布）与`DType`（数据类型）方面的信息，具体如下：

```c++
using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>; //流水排布使用
using L1TileShape = Catlass::GemmShape<128, 256, 256>; // L1基本块
using L0TileShape = Catlass::GemmShape<128, 256, 64>; // L0基本块
using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;     //封装了A矩阵的数据类型和排布信息
using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;     //封装了B矩阵的数据类型和排布信息
using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;     //封装了C矩阵的数据类型和排布信息

using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy,
    L1TileShape,
    L0TileShape,
    AType,
    BType,
    CType>;
```

2. `BlockEpilogue`封装了`Block`层后处理逻辑，基础Matmul算子不涉及后处理，因此可以为空(`void`)。

```c++
using BlockEpilogue = void;
```

3. `BlockScheduler`描述了矩阵分块的数据走位方式，封装了数据搬运过程中的`Offset`（数据偏移）计算逻辑，这里可使用预定义的`GemmIdentityBlockSwizzle`（详情可参考[Swizzle策略说明](./contents/advanced/swizzle_explanation.md)）。

```c++
using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<>;
```

完成上述三个`Block`层组件定义后，利用预定义的`Kernel`级组件`BasicMatmul`
即可完成算子的`Kernel`层组装。

```c++
using MatmulKernel = Catlass::Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
```



### Device层算子定义

在`Device`层级，只需要对`Kernel`层组装的算子进行一层封装即可完成核函数的编写，包括如下三个步骤：

1. 使用`CATLASS_GLOBAL`修饰符定义Matmul函数，传参包括`A`，`B`，`C`矩阵的地址与排布情况。

```c++
template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
CATLASS_GLOBAL
void BasicMatmul(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC);
```

2. 实例一个`MatmulKernel::Params`参数对象，设置参数值。

```c++
typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};
```

3. 最后，实例化一个`MatmulKernel`对象，并执行该算子(`BasicMatmul`的调用接口为`operator()`运算符)。

```c++
MatmulKernel matmul;
matmul(params);
```

### 算子调用

以上即完成了`BasicMatmul`算子的构建，使用`<<<>>>`的方式调用核函数，传入指定矩阵的输入输出的数据类型和数据排布信息即可。

```c++
BasicMatmul<<<BLOCK_NUM, nullptr, stream>>>(
        options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC);
```

## 算子编译

CATLASS使用CMake进行项目管理，在样例同级目录下创建CMakeLists.txt如下（可参考[CMakeLists.txt](../examples/00_basic_matmul/CMakeLists.txt)）,以支撑算子编译：
```
# CMakeLists.txt
set_source_files_properties(basic_matmul.cpp PROPERTIES LANGUAGE ASCEND)
catlass_example_add_executable(
    00_basic_matmul
    cube
    basic_matmul.cpp
)
```

上述`CMake`文件做了下述几件事：
- 设置源代码对应的编译语言，可以混合使用纯C++代码和算子代码。对于算子文件，需要设定语言为ASCEND。
- 调用`catlass_example_add_executable`函数指定target名称和编译文件。
  - `00_basic_matmul`为target名称
  - `basic_matmul.cpp`为需要编译的文件
  - `cube`为算子类型，可填写`cube`/`vec`/`mix`

首先，在主目录下执行下述指令进行算子编译即可编译出组件：
```bash
bash scripts/build.sh 00_basic_matmul
```
出现`Target '00_basic_matmul' built successfully`即表明样例编译成功。

 - 要编译`examples/`下的全部样例，可使用`bash scripts/build.sh catlass_examples`。


## 算子执行

算子编译产物默认在`output/bin`下，切换至该目录运行算子样例程序如下：

```
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
./00_basic_matmul 256 512 1024 0
```

返回`Compare success.`，说明算子运行成功，精度比较通过。