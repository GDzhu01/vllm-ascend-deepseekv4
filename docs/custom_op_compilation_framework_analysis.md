# vllm-ascend 自定义算子编译框架分析报告

## 一、概述

vllm-ascend 是华为 Ascend NPU 上 vLLM 的大语言模型推理加速框架。该项目的自定义算子编译框架采用分层架构，结合 CMake 构建系统和 Python setuptools，支持多种 Ascend 芯片类型（A2/A3/A5/310P）的算子编译和部署。

## 二、整体架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Python 层 (Pybind11)                         │
│  vllm_ascend_C.so                                                   │
│  ├── torch_binding.cpp (算子注册与 Python 接口)                      │
│  ├── ops.h (算子声明)                                                │
│  └───────────────────────────────────────────────────────────────   │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                      ACLNN Torch Adapter 层                          │
│  aclnn_torch_adapter/                                                │
│  ├── op_api_common.h (ACLNN API 封装，EXEC_NPU_CMD 宏)               │
│  ├── NPUBridge.cpp/h (NPU Tensor 桥接)                               │
│  └───────────────────────────────────────────────────────────────   │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                      算子 Host 层 (op_host)                           │
│  ├── op_api/ (ACLNN 接口实现)                                        │
│  ├── *_def.cpp (算子定义)                                            │
│  ├── *_infershape.cpp (形状推导)                                     │
│  ├── *_tiling.cpp (Tiling 数据生成)                                  │
│  └───────────────────────────────────────────────────────────────   │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                      算子 Kernel 层 (op_kernel)                       │
│  AscendC Kernel 实现                                                 │
│  ├── *.cpp/.h (AscendC 算子核心实现)                                 │
│  ├── tiling_data.h (Tiling 数据结构)                                 │
│  └───────────────────────────────────────────────────────────────   │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                      CANN ACLNN 算子包                               │
│  _cann_ops_custom/                                                   │
│  ├── vendors/custom_transformer/                                    │
│  │   ├── op_api/lib/ (动态库)                                        │
│  │   ├── op_impl/ (Kernel 实现)                                      │
│  │   ├── op_proto/ (算子原型)                                        │
│  └───────────────────────────────────────────────────────────────   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 关键文件结构

| 文件/目录 | 作用 |
|----------|------|
| `setup.py` | Python 包入口，定义编译流程和平台检测 |
| `CMakeLists.txt` (根目录) | 主 CMake 配置，编译 vllm_ascend_C.so |
| `csrc/CMakeLists.txt` | 自定义 ACLNN 算子包编译入口 |
| `csrc/build_aclnn.sh` | ACLNN 自定义算子包构建脚本 |
| `csrc/cmake/*.cmake` | 编译框架辅助脚本 |
| `csrc/torch_binding.cpp` | Pybind11 算子注册 |
| `csrc/aclnn_torch_adapter/` | ACLNN API 与 Torch 桥接 |

## 三、编译流程详解

### 3.1 编译入口 (setup.py)

```python
# 关键编译类
class CMakeExtension(Extension)      # CMake 扩展包装器
class cmake_build_ext(build_ext)     # CMake 构建执行器
class build_and_install_aclnn(Command)  # ACLNN 算子包构建命令
```

**编译流程：**
1. `setup.py` 通过 `npu-smi` 或 `SOC_VERSION` 环境变量检测芯片类型
2. 生成 `_build_info.py` 记录设备类型（A2/A3/A5/310P）
3. 调用 `build_aclnn.sh` 构建 ACLNN 自定义算子包
4. 通过 CMake 编译 Pybind11 扩展模块 `vllm_ascend_C.so`

### 3.2 芯片类型映射

| SOC_VERSION | 设备类型 | 支持的算子集 |
|-------------|---------|------------|
| ascend910b1/b2/b3/b4 | A2 | 完整算子集 + MLAPO |
| ascend910_9391/9381/... | A3 | 完整算子集 + MC2 算子 |
| ascend310p1/p3/p5/... | 310P | 简化算子集 |
| ascend950* | A5 | 禁用 MLAPO 和 batch_matmul_transpose |

### 3.3 ACLNN 算子包构建 (build_aclnn.sh)

```bash
# 构建流程
cd csrc
rm -rf build output build_out
bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}"
./build/cann-ops-transformer*.run --install-path=${install_dir}
```

**关键功能：**
- 根据芯片类型选择不同的算子列表
- 处理第三方依赖（catlass submodule）
- 复制 HCCL 相关头文件用于分布式算子
- 生成 `.run` 安装包并安装到 `vllm_ascend/_cann_ops_custom/`

## 四、CMake 构建框架

### 4.1 CMake 模块结构

```
csrc/cmake/
├── opbuild.cmake       # ACLNN 代码生成
├── func.cmake          # 算子编译函数（add_modules_sources 等）
├── obj_func.cmake      # 对象文件编译
├── intf.cmake          # 接口配置
├── package.cmake       # 打包逻辑
├── custom_build.cmake  # 自定义构建逻辑
├── dependencies.cmake  # 依赖管理
└── ...
```

### 4.2 核心编译函数

**`add_modules_sources()` - 算子源文件添加：**
```cmake
# 功能：添加算子的各个模块源文件
add_modules_sources(OPTYPE moe_grouped_matmul ACLNNTYPE aclnn_inner)
# 会处理：
# - op_api/*.cpp (ACLNN API 实现)
# - *_infershape.cpp (形状推导)
# - *_tiling.cpp (Tiling 逻辑)
# - *_def.cpp (算子定义)
```

**`add_op_to_compiled_list()` - 算子列表管理：**
```cmake
# 记录编译的算子到全局列表
set(COMPILED_OPS ${COMPILED_OPS} ${OP_NAME} CACHE STRING "Compiled Ops" FORCE)
set(COMPILED_OP_DIRS ${COMPILED_OP_DIRS} ${PARENT_DIR} CACHE STRING "Compiled Ops Dirs" FORCE)
```

**`gen_aclnn_with_opdef()` - ACLNN 代码生成：**
```cmake
# 通过 OP_BUILD_TOOL 生成：
# - aclnn_*.cpp/h (ACLNN 接口代码)
# - *_proto.cpp/h (算子原型代码)
# - .ini 算子信息文件
```

### 4.3 算子目录结构标准

```
csrc/<category>/<op_name>/
├── CMakeLists.txt           # 算子编译配置
├── op_host/                 # Host 侧实现
│   ├── CMakeLists.txt
│   ├── <op_name>_def.cpp    # 算子定义
│   ├── <op_name>_infershape.cpp
│   ├── <op_name>_tiling.cpp/h
│   ├── <op_name>_cpu.cpp    # CPU fallback
│   └── op_api/
│       ├── aclnn_<op_name>.cpp/h
│       ├── <op_name>_l0.cpp/h
├── op_kernel/               # Kernel 实现 (AscendC)
│   ├── <op_name>.cpp/h
│   ├── tiling_data.h
│   ├── arch32/              # A2/A3 架构
│   └── arch35/              # A5 架构
```

## 五、算子类型与编译策略

### 5.1 ACLNN 类型分类

| 类型 | 用途 | 编译目标 |
|------|------|----------|
| `aclnn` | 外部调用接口 | op_host_aclnn |
| `aclnn_inner` | 内部调用接口 | op_host_aclnnInner |
| `aclnn_exclude` | 仅原型定义 | op_host_aclnnExc |

### 5.2 算子类别

| 目录 | 算子类型 |
|------|----------|
| `moe/` | MoE 相关：moe_grouped_matmul, moe_gating_top_k, moe_init_routing |
| `attention/` | 注意力：sparse_flash_attention, lightning_indexer, compressor |
| `gmm/` | 分组矩阵乘：grouped_matmul_swiglu_quant |
| `mc2/` | 分布式通信+计算：dispatch_ffn_combine, matmul_allreduce |
| `kernels/` | 基础内核：bgmv, sgmv, get_masked_input_and_mask |

## 六、Pybind11 算子注册

### 6.1 注册机制 (torch_binding.cpp)

```cpp
TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops) {
    ops.def("npu_gemma_rms_norm(Tensor x, Tensor gamma, float epsilon) -> (Tensor y, Tensor rstd)");
    ops.impl("npu_gemma_rms_norm", torch::kPrivateUse1, &vllm_ascend::npu_gemma_rms_norm);

    // 更多算子注册...
}
```

### 6.2 ACLNN 调用封装 (op_api_common.h)

```cpp
#define EXEC_NPU_CMD(aclnn_api, ...) do {
    // 1. 获取 workspace 大小
    // 2. 分配 workspace 内存
    // 3. 调用 ACLNN API
    // 4. 释放资源
} while(false)
```

**关键转换函数：**
- `ConvertType(at::Tensor)` → `aclTensor*`
- `ConvertType(at::Scalar)` → `aclScalar*`
- `ConvertType(at::IntArrayRef)` → `aclIntArray*`

## 七、编译配置选项

### 7.1 环境变量 (vllm_ascend/envs.py)

| 变量 | 说明 |
|------|------|
| `COMPILE_CUSTOM_KERNELS` | 是否编译自定义算子（默认 1） |
| `SOC_VERSION` | 芯片版本 |
| `CMAKE_BUILD_TYPE` | 编译类型（Release/Debug/RelWithDebugInfo） |
| `MAX_JOBS` | 最大并行编译数 |
| `ASCEND_HOME_PATH` | CANN 安装路径 |
| `VERBOSE` | 详细日志 |

### 7.2 CMake 选项

```cmake
option(BUILD_OPEN_PROJECT         "Build open ascend ops project." ON)
option(BUILD_OPS_RTY_KERNEL       "Build return yellow kernel." OFF)
option(ENABLE_CCACHE              "Enable ccache capability" ON)
option(ENABLE_STATIC              "Enable Static" OFF)
```

## 八、安装与部署

### 8.1 安装产物

```
vllm_ascend/
├── vllm_ascend_C.so          # Pybind11 扩展
├── vllm_ascend_kernels.so    # AscendC Kernel 库
├── _cann_ops_custom/         # ACLNN 算子包
│   └── vendors/custom_transformer/
│       ├── op_api/lib/
│       ├── op_impl/ai_core/tbe/kernel/
│       ├── op_proto/
│       └── op_impl/ai_core/tbe/config/
├── _build_info.py            # 设备类型信息
└── _version.py               # 版本信息
```

### 8.2 运行时加载

```python
# vllm_ascend/__init__.py
try:
    import vllm_ascend.vllm_ascend_C as _C
except ImportError:
    pass  # 允许在没有 NPU 的环境运行 UT
```

## 九、关键设计模式

### 9.1 分层解耦
- **Python 层**：用户接口，算子注册
- **Adapter 层**：ACLNN 与 Torch 类型转换
- **Host 层**：算子定义、形状推导、Tiling
- **Kernel 层**：AscendC 实现

### 9.2 芯片适配
- 通过 `SOC_VERSION` 动态选择算子集
- 架构目录分离（arch32/arch35）
- 编译时宏定义区分平台

### 9.3 CMake 函数化
- `add_modules_sources()` 等函数统一处理算子源文件
- 模块化的 cmake 脚本组织
- 自动生成 ACLNN 接口代码

## 十、总结

vllm-ascend 的自定义算子编译框架是一个设计完善的多层次系统：

1. **构建入口**：setup.py 整合芯片检测、ACLNN 构建、CMake 编译
2. **CMake 框架**：函数化设计，模块化管理算子编译
3. **ACLNN 机制**：自动生成接口代码，标准化算子结构
4. **Pybind11 桥接**：标准 TORCH_LIBRARY 注册，EXEC_NPU_CMD 封装
5. **芯片适配**：灵活的芯片类型检测和算子集选择

这套框架支持高效地开发和部署针对华为 Ascend NPU 的自定义算子，为 vLLM 在 Ascend 上的高性能推理提供了底层支撑。