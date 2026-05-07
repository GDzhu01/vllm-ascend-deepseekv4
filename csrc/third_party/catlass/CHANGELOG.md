# CHANGELOG

## CATLASS 1.X

### CATLASS 1.4.0

- 关键特性
  - [Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.4.0/examples/102_dynamic_optimized_matmul/README.md)新增
    - `LocalPaddingCPaddingCommonMatmul`模板，使用局部workSpace对C矩阵做padding
- 更多样例
  - [StreamK Matmul算子](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/37_streamk_matmul/README.md)
  - [W4A4低精度Matmul算子](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/38_w4a4_matmul_per_token_per_channel_dequant/README.md)
  - [Matmul算子L2层级切分+错位分核](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/39_big_matmul_tla/README.md)
  - [Sparse Matmul算子](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/41_sparse_matmul_tla/README.md)
- 工具支持
  - 增加[shared\_lib](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/shared_lib/basic_matmul_shared_lib.cpp)使用示例
- 文档资料
  - 新增[单核切K优化Matmul算子的详设文档](https://gitcode.com/cann/catlass/blob/v1.4.0/docs/contents/example_design/34_single_splitk_matmul.md)，介绍单核切K矩阵乘的设计思路和代码拆解
  - 新增[主页Matmul/GroupedMatmul算子性能展示数据](https://gitcode.com/cann/catlass/blob/v1.4.0/README.md)
  - [msdebug文档](https://gitcode.com/cann/catlass/blob/v1.4.0/docs/tools/msdebug.md)新增驱动支持检查
- Bugfix&优化
  - 优化[grouped\_matmul\_slice\_m](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/02_grouped_matmul_slice_m/README.md)样例支持`groupList`分段式输入
  - BlockMmad增加`TileShape`的32B对齐约束
  - 修复[w4a8 matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/32_w4a8_matmul/w4a8.cpp)样例构造数据长度问题
  - [Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.4.0/examples/102_dynamic_optimized_matmul/README.md)支持Stride大于Shape的场景
  - [msTuner\_CATLASS工具](https://gitcode.com/cann/catlass/tree/v1.4.0/tools/tuner/README.md)支持`quant Matmul`做tiling寻优
  - 修复device侧对`cmath`函数的不规范使用
  - 修复[MatmulSilu](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/28_matmul_silu/README.md)精度问题和命名错误
  - 修复[cast\_int4\_to\_int8](https://gitcode.com/cann/catlass/blob/v1.4.0/include/catlass/gemm/tile/cast_int4_to_int8.hpp)组件的Vector同步问题
  - 修复[w8a16 Matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/30_w8a16_matmul/w8a16_matmul.cpp)算子half类型使用问题
  - 修复[单核切K Matmul算子](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/34_single_core_splitk_matmul/single_core_splitk.cpp)在`RemovePaddingNDAndCastC`为空时的逻辑问题


### CATLASS 1.3.0

 - 关键特性
   - 将`CMake`最低版本要求从3.22降至3.16
   - 支持[`FixPipe`随路量化](https://gitcode.com/cann/catlass/tree/v1.3.0/include/catlass/gemm/tile/tile_copy.hpp#L373)
   - [Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/README.md)新增
     + `PaddingCommonMatmul`
     + `SmallMatmul`
     + `PaddingMultiCoreSplitkMatmul`
     + `PaddingStreamkMatmul`
     + `单核切K系列模板`
     + `动态Swizzle`
 - 更多样例
    - [INT4类型反量化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/32_w4a8_matmul/README.md)
    - [2D卷积算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/33_basic_conv2d/README.md)
    - [单核切K优化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/34_single_core_splitk_matmul/README.md)
 - 工具支持
    - 新增[msOpGen](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0018.html)工具代码示例[basic_matmul](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/advanced/basic_matmul_aclnn/basic_matmul_aclnn.cpp)和[接入文档](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/advanced/basic_matmul_aclnn/README.md)
    - [msTuner_CATLASS工具](https://gitcode.com/cann/catlass/tree/v1.3.0/tools/tuner/README.md)新增
      + `GroupedMatmulSliceM算子`
      + `OptimizedMatmul算子`
- 文档资料
   - 新增[INT8类型反量化GroupedMatmul算子的详设文档](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/example_design/10_grouped_matmul_slice_m_per_token_dequant.md)，介绍`groupMatmul+后处理`类型的算子的设计思路和代码拆解
   - 新增[矩阵乘模板总结文档](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/matmul_template_summary.md)，介绍模板库已有的Matmul模板设计
   - 新增[CommonMatmul说明文档](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/doc/CommonMatmul.md)，介绍泛化Matmul工程中的基础模板
- Bugfix&优化
  - 修复[Flash Attention推理算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/23_flash_attention_infer/README.md)在softmax拷贝mask时引入的内存问题
  - 修复文档错误
    + [catlass_optimize_guidance.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/catlass_optimize_guidance.md)
    + [api.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/api.md)
    + [quickstart.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/quickstart.md)
    + [tutorials.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/tutorials.md)
  - [Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/README.md)更新
    + 修改`TilingParams`读取方式增强可读性
    + 优化原有的`Splitk ReduceAdd`，UB空间利用更充分
    + 新增`CMakeLists.txt`中对python环境的判断
  - 修复[OptimizedMatmul算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/06_optimized_matmul/README.md)在kernel里没有支持PADDING_NZ的问题
  - 优化重构[FP8类型反量化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/29_a2_fp8_e4m3_matmul/README.md)，使其更符合Prologue范式
  - 修复[MatmulBias算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/20_matmul_bias/README.md)精度问题并增加对bf16的校验拦截
  - 优化仿真的编译逻辑以及在A3环境下的编译问题，现在编译`simulator`模式时逻辑与上板模式相同
  - [msTuner_CATLASS工具](https://gitcode.com/cann/catlass/tree/v1.3.0/tools/tuner/README.md)更新
    + 新增接口替换、非法字符、`groupCount`最大值检查等安全校验
    + 修复下发部分算子时默认传入`ffts_addr`被拦截的问题
  - 更改默认的跨核标志位可连续置位次数，避免超过次数后引发的系统卡死问题

### CATLASS 1.2.0

 - 关键特性
   - 算子编译时支持传入计算平台架构
   - 新增[Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/102_dynamic_optimized_matmul/README.md)示例
     + 自动依照特征尺寸确定Tiling参数
     + 可在预设的算子模板中择优选取

 - 更多样例
    - [Flash Attention推理算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/23_flash_attention_infer/README.md)
    - [3D卷积算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/24_conv_bias/README.md)
    - [A矩阵全加载Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/25_matmul_full_loadA/README.md)
    - [小矩阵优化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/31_small_matmul/README.md)
    - [MatmulRelu算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/26_matmul_relu/README.md)
    - [MatmulGelu算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/27_matmul_gelu/README.md)
    - [MatmulSilu算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/28_matmul_silu/README.md)
    - [FP8类型反量化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/29_a2_fp8_e4m3_matmul/README.md)
    - [INT8类型反量化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/30_w8a16_matmul/README.md)

 - 工具支持
    - 更新[Python调用接口](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/python_extension/README.md)内容
        + 调整工程组织结构
        + 支持转置情况
    - 新增[`msTuner_CATLASS`](https://gitcode.com/cann/catlass/tree/v1.2.0/tools/tuner/README.md)工具，用于Tiling自动寻优，在搜索空间内全量运行并获取性能数据
    - 支持使能[`msSanitizer`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/optool/atlasopdev_16_0039.html)地址消毒工具（编译选项加入`--enable_mssanitizer`）
  
 - 文档资料
   - 新增[`catlass_optimize_guidance.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/contents/advanced/catlass_optimize_guidance.md)文档，介绍CATLASS赋能下`Gemm`类算子常用的调优方式

 - Bugfix&优化
   - 优化[`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/06_optimized_matmul/README.md)算子实现，支持任意Padding方式组合
   - 修复`ASCEND_RT_VISIBLE_DEVICES`环境变量使能下，`msTuner_CATLASS`工具无法取得实际运行`DeviceId`的问题
   - 修复[PFA算子样例](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/19_mla/README.md)在单行数据场景下`Set/Wait`错配的异常情形
   - 修复[`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/06_optimized_matmul/README.md)算子样例计算`Workspace`大小出错
   - 隔离使能`AscendC::Dump`及`AscendC::print`功能的代码段
   - 修复[`GroupedMatmulSliceK`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/05_grouped_matmul_slice_k/README.md)算子在Ki=0特例时的输出清零行为，并将真值比较逻辑调整为全尺寸(M,N,K)比较
   - 修改[`performance_tools.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/tools/performance_tools.md)，[`tutorials.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/tutorials.md)等文档中的错误

### CATLASS 1.1.0

 - 关键特性
   - 提供Python调用CATLASS算子的工程组件
     - 可编译[pybind](https://github.com/pybind/pybind11)扩展及[PyTorch](https://pytorch.org/)扩展件
   - 支持算子仿真运行（编译选项启用`--simulator`）
   - 编译过程适配毕昇编译器（[bisheng](https://www.hiascend.com/cann/bisheng)）

 - 更多样例
    - [带偏置的MatmulBias算子](https://gitcode.com/cann/catlass/blob/v1.1.0/examples/20_matmul_bias/README.md)
    - [预加载(Preload)优化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/21_basic_matmul_preload_zN/README.md) （科大讯飞联创贡献）
    - [K轴切分(Split-K)优化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/22_padding_splitk_matmul/README.md) （科大讯飞联创贡献）

 - 工具支持
    - 支持[`AscendC::Dump`](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0075.html)与[`AscendC::printf`](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0075.html)进行打印调试
      - 编译选项中加入`--enable_ascendc_dump`和`--enable_print`以启用上述功能
      - 请参阅文档: [`ascendc_dump`](docs/tools/ascendc_dump.md)和[`print`](docs/tools/print.md)
  
 - 文档资料
   - 新增[tutorials快速上手示例](https://gitee.com/ascend/catlass/tree/v1.1.0/docs/tutorials.md)
   - 新增利用[msProf工具](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0082.html)进行算子性能调测的文档：[msProf](https://gitee.com/ascend/catlass/tree/v1.1.0/docs/tools/performance_tools.md)性能调测

 - Bugfix&优化
   - 优化`Kernel`层AIC程序，添加`PIPE_ALL`避免整网影响
   - 优化[`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/06_optimized_matmul/README.md)算子实现，在非必要Padding场景下不启动AIV核
   - 修复`Block`层预加载`nextBlock`时的错误
   - 隔离Kernel侧`AscendC`的`inline`定义，避免异构编程时无法使用部分标准库
   - 修改`l2offset`设置的重定义问题
   
### CATLASS 1.0.0
  - [CATLASS](https://gitcode.com/cann/catlass/)模板库正式开源发布

 - 关键特性
   - 提供Kernel、Block、Tile、Basic分层算子开发能力

 - 样例参考
   - 提供包括基础Matmul及各种不同的优化策略在内的算子样例
