# 在CATLASS样例工程进行性能调优

CANN对算子开发的两个场景——单算子与整网开发，分别提供了对应的性能调优工具：[**msProf**](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/optool/atlasopdev_16_0082.html)和[**Profiling**](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/Profiling/atlasprofiling_16_0010.html)。

## 性能调优工具简介

### msProf简介

[msProf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/optool/atlasopdev_16_0082.html)是单算子性能分析工具，对应的指令为`msprof op`或`msopprof`。

[msProf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/optool/atlasopdev_16_0082.html)工具用于采集和分析运行在昇腾AI处理器上算子的关键性能指标，用户可根据输出的性能数据，快速定位算子的软、硬件性能瓶颈，提升算子性能的分析效率。

当前支持基于不同运行模式（上板或仿真）和不同文件形式（可执行文件或算子二进制`.o`文件）进行性能数据的采集和自动解析。

### Profiling简介

[Profiling](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/Profiling/atlasprofiling_16_0010.html)是整网性能分析工具，对应的指令为`msprof`。

[Profiling](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/Profiling/atlasprofiling_16_0010.html)工具提供了AI任务运行性能数据、昇腾AI处理器系统数据等性能数据的采集和解析能力。

其中，`msprof`采集通用命令是性能数据采集的基础，用于提供性能数据采集时的基本信息，包括参数说明、AI任务文件、数据存放路径、自定义环境变量等。

## 用`msProf`进行单算子性能分析

以`00_basic_matmul`为例，演示基于[msProf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/optool/atlasopdev_16_0082.html)的性能分析过程。

### 上板性能采集

通过上板性能采集，可以直接测定算子在NPU卡上的运行时间，可判断性能是否初步达到预期标准。

#### `msprof op`使用示例

1. 参考[快速上手](../quickstart.md)，编译算子样例。
2. 使用`msprof op *可选参数* app [arguments]`格式调用msProf工具。

```bash
msprof op --application="./00_basic_matmul 256 512 1024 0"
```

常用参数如下：

| 参数             | 是否必选       | 说明                                         | 取值                          | 配套参数/注意事项            |
| ---------------- | -------------- | -------------------------------------------- | ----------------------------- | ---------------------------- |
| `--application`  | 必选（二选一） | 指定可执行文件/执行指令                      | 有效路径或命令                | 与 `--config` 互斥           |
| `--config`       | 必选（二选一） | 指定二进制文件 `.o`                          | 有效路径                      | 与 `--application` 互斥      |
| `--kernel-name`  | 可选           | 指定采集的算子名称（支持模糊匹配和多个采集） | 例：`"conv*"` 或 `"add\|mul"` | 需配合 `--launch-count` 使用 |
| `--launch-count` | 可选           | 设置最大采集算子数量                         | 1~100 整数（默认 1）          | 需配合 `--kernel-name` 使用  |
| `--warm-up`      | 可选           | 预热次数（解决芯片未提频问题）               | 整数（默认 5）                | 小 shape 场景建议提高到 30   |
| `--output`       | 可选           | 指定数据输出路径                             | 有效路径（默认当前目录）      | 需确保路径可写               |

更多参数可参考[msProf工具概述](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/optool/atlasopdev_16_0082.html)。

- ⚠ 注意事项
  - 工具默认会读取第一个算子的性能，使用example进行调测时可直接获取到结果；若接入其他工程，工程中可能存在其他算子（虽然只跑某一个算子的用例），如果性能分析时未通过`--kernel-name`指定算子名称，则可能读取不到结果。
  - 可设置环境变量`ASCEND_RT_VISIBLE_DEVICES`指定上板调测的Device Id号

```bash
# 指定当前进程仅可使用Device Id为0，1，2，3的Device
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
msprof op ./00_basic_matmul 256 512 1024 0
```

#### 性能数据说明

性能数据文件夹结构示例：

```bash
├──dump                       # 原始的性能数据，用户无需关注
├──ArithmeticUtilization.csv  # cube/vector指令cycle占比，建议优化算子逻辑，减少冗余计算指令
├──L2Cache.csv                # L2 Cache命中率，影响MTE2，建议合理规划数据搬运逻辑，增加命中率
├──Memory.csv                 # UB，L1和主存储器读写带宽速率，单位GB/s
├──MemoryL0.csv               # L0A，L0B，和L0C读写带宽速率，单位GB/s
├──MemoryUB.csv               # Vector和Scalar到UB的读写带宽速率，单位GB/s
├──OpBasicInfo.csv            # 算子基础信息
├──PipeUtilization.csv        # pipe类指令耗时和占比，建议优化数据搬运逻辑，提高带宽利用率
└──ResourceConflictRatio.csv  # UB上的 bank group、bank conflict和资源冲突率在所有指令中的占比， 建议减少/避免对于同一个bank读写冲突或bank group的读读冲突
```

### 性能流水仿真

通过仿真，可以获得**流水图**、**指令与代码行映射**、**代码热点图**、**内存热点图**等可视化数据，以便进一步分析优化算子计算瓶颈。

#### msprof op simulator使用示例

1. 编译脚本增加选项`--simulator`， 以`simulator`模式编译算子。

```bash
bash scripts/build.sh --simulator 00_basic_matmul
```

- 这个选项实际不会改变编译的二进制程序，区别为是否输出第2步的仿真器路径提示。

2. 编译完成后，根据提示，加载仿真器二进制路径。

```bash
# 根据第1步的实际输出执行
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib/libruntime_camodel.so:/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib/libnpu_drv_camodel.so
```

3. 切换到可执行文件的编译目录 `output/bin` 下， 使用`msprof op simulator`执行算子样例程序。

```bash
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
msprof op simulator ./00_basic_matmul 256 512 1024 0
```

- ⚠ 注意事项
  - 若需要查看**代码热点图**，需要在`examples/CMakeLists.txt`中增加`add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-Xaicore-start -g -Xaicore-end")`。
  - 性能结果中有大量明显`Vector`操作（如`Add`、`Div`）映射为`Scalar`操作导致性能结果明显异常(`vector_ratio<10%`，`scalar>90%`)，这是编译优化等级造成的，可在`examples/CMakeLists.txt`中增加`add_compile_options($<$<COMPILE_LANGUAGE:ASCEND>:"-Xaicore-start -O3 -Xaicore-end">)`。
  - 仿真只能在0卡运行，不能指定NPU卡号。

#### 仿真数据说明

```bash
├──dump                    # 原始的性能数据，用户无需关注
└──simulator               # 算子基础信息
   ├──core0.cubecore0
   ├──...
   ├──core23.cubecore0
   ├──trace.json           # Edge/Chrome Trace Viewer/Perfetto呈现文件
   └──visualize_data.bin   # MindStudio Insight呈现文件

```

#### 性能数据可视化查看

- 数据可视化依赖[MindStudio Insight](https://www.hiascend.com/developer/download/community/result?module=sto%2Bcann)工具，需要提前下载安装。

##### 代码热点图

获取仿真输出文件夹`simulator`下的`visualize_data.bin`，通过MindStudio Insight工具加载bin文件查看代码热点图。

![MindStudio Insight Source](https://www.hiascend.com/doc_center/source/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/figure/zh-cn_image_0000002274910673.png)

##### 指令流水图

###### 使用Edge/Chrome Trace Viewer/Perfetto呈现

根据浏览器，选择以下工具：

- [Edge Trace Viewer](edge://tracing)(Edge浏览器)
- [Chrome Trace Viewer](chrome://tracing)(Chrome浏览器/基于Chrome内核的浏览器)
- [Perfetto](https://ui.perfetto.dev/)(通用)

导入`trace.json`即可查看仿真指令流水图。

###### 使用MindStudio Insight呈现

获取仿真输出文件夹`simulator/`下的`visualize_data.bin`。通过MindStudio Insight工具加载bin文件查看仿真流水图。

![MindStudio Insight Timeline](https://www.hiascend.com/doc_center/source/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/figure/zh-cn_image_0000002274910873.png)

### 用MindStudio Insight查看更多可视化数据

msProf工具采集到的数据，可导入可视化工具[MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html)中，以便进一步分析算子计算瓶颈。将`visualize_data.bin`导入工具内，即可可视化地分析算子性能。

![MindStudio Insight Details](https://www.hiascend.com/doc_center/source/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/figure/zh-cn_image_0000002274911037.png)

![MindStudio Insight Cache](https://www.hiascend.com/doc_center/source/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/figure/zh-cn_image_0000002274870637.png)

## 用Profiling进行整网性能分析

虽然CATLASS只提供单算子的调用示例，但单算子调用示例也可使用[Profiling](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/Profiling/atlasprofiling_16_0010.html)工具进行性能分析。

下面以`00_basic_matmul`为例进行演示分析。

### `msProf`使用示例

1. 基于[快速上手](../quickstart.md)，打开工具的编译开关`--enable_profiling`， 使能`Profiling API`编译算子样例。

```bash
bash scripts/build.sh --enable_profiling 00_basic_matmul
```

2. 切换到可执行文件的编译目录`output/bin`下，用`msProf`执行算子样例程序。

```bash
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
msprof ./00_basic_matmul 256 512 1024 0
```

可参考[msProf性能数据文件参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/Profiling/atlasprofiling_16_0057.html)了解性能数据各文件的功能。
