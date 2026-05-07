# basic_matmul_aclnn example

aclnn接口是CANN软件栈一直沿用的接口，msOpGen工具是CANN提供可以生成该接口工程框架的工具，便于用户编写一个具有aclnn接口的算子，并使能CANN软件栈上的各种功能。该样例提供CATLASS算子模板接入msOpGen工程的示例代码与注意事项，并提供CATLASS example风格的调用示例。

下面以basic_matmul接入为例进行示例，利用msOpGen工具接入该算子模板。

## 1. 创建算子工程

参考[创建算子工程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_0060.html)链接编写一个算子原型的json文件，并生成对应工程。

### 编写json

相关示例代码：[catlass_basic_matmul.json](./catlass_basic_matmul.json)

### 生成工程

执行下列脚本，调用msOpGen生成算子工程。

```bash
msopgen gen -i catlass_basic_matmul.json -c ai_core-<soc_version> -lan cpp -out catlass_basic_matmul
```

- 其中`soc_version`可通过`npu-smi info`查看，形如`Ascendxxxyyy`。
- 需保证输入的json配置文件（上例中的`catlass_basic_matmul.json`）具有644的权限
- 需保证输出的文件路径（上例中的`catlass_basic_matmul`）具有755的权限

## 2. 编写Host代码

参考[Host侧Tiling实现-基本流程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_00021.html)实现`TilingFunc`。

若需要使能**算子入图**，请参考[算子入图（GE）图开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_0078.html)实现`InferShape`和`InferDataType`。

相关示例代码：
[op_host/catlass_basic_matmul.cpp](./op_host/catlass_basic_matmul.cpp)
[op_host/catlass_basic_matmul_tiling.h](./op_host/catlass_basic_matmul_tiling.h)

## 3. 编写Device代码

参考[Kernel侧算子实现](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_0063.html)，实现kernel代码。

相关示例代码：
[op_kernel/catlass_basic_matmul.cpp](./op_kernel/catlass_basic_matmul.cpp)
**注意事项**

- 我们需要增加编译选项来引入CATLASS的头文件。在`op_kernel/CMakeLists.txt`中增加算子编译选项。其中`${CATLASS_INCLUDE_PATH}`是CATLASS代码仓下的`include`文件夹的路径，需根据环境实际情况进行配置。

    ```diff
    # set custom compile options
    if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
        add_ops_compile_options(ALL OPTIONS -g -O0)
    endif()
    + add_ops_compile_options(ALL OPTIONS -I${CATLASS_INCLUDE_PATH})
    add_kernels_compile()
    ```

- `msOpGen`工程的分离编译模式不支持直接将结构体（如`Catlass::GemmCoord`）作为kernel的参数传入。当需要使用结构体时，需要通过`tiling`地址传递成员数据，然后在kernel侧重新构造。

    ```cpp
    // 正确
    extern "C" __global__ __aicore__ void
    catlass_basic_matmul(GM_ADDR self, GM_ADDR mat2, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(tiling_data, tiling);
        Catlass::GemmCoord problemShape{tiling_data.m, tiling_data.n, tiling_data.k};
        // ...
    }
    // 暂不支持
    extern "C" __global__ __aicore__ void
    catlass_basic_matmul(GM_ADDR self, GM_ADDR mat2, GM_ADDR out, GM_ADDR workspace,  Catlass::GemmCoord problemShape)
    {
        // ...
    }
    ```

## 4. 编译、部署

参考[算子工程编译](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_0068.html)、[算子包部署](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_0069.html)进行编译、部署，并设定环境变量。

一般来说，调用者需要添加头文件`aclnn_catlass_basic_matmul.h`并链接`libcust_opapi.so`。在不修改工程参数的情况下，这两个文件的位置如下：

```bash
$ASCEND_HOME_PATH/opp/vendors/customize/op_api/include/aclnn_catlass_basic_matmul.h
$ASCEND_HOME_PATH/opp/vendors/customize/op_api/lib/libcust_opapi.so
```

这可作为`Makefile`/`CMakeLists.txt`的编写参考。

## 5. 调用

参考[接口简介](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/aolapi/operatorlist_00001.html)了解aclnn接口的相关概念，并参考[basic_matmul_aclnn.cpp](./basic_matmul_aclnn.cpp)尝试调用。

可参考以下内容编写`CMakeLists.txt`：

```cmake
project(basic_matmul_aclnn)
cmake_minimum_required(VERSION 3.22)
set(CATLASS_REPO_DIR <修改为实际环境上的CATLASS仓库路径>)
add_executable(basic_matmul_aclnn basic_matmul_aclnn.cpp)
target_include_directories(basic_matmul_aclnn PRIVATE
    ${CATLASS_REPO_DIR}/examples/common
    ${CATLASS_REPO_DIR}/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
    $ENV{ASCEND_HOME_PATH}/include/experiment/runtime
    $ENV{ASCEND_HOME_PATH}/include/experiment/msprof
    $ENV{ASCEND_HOME_PATH}/opp/vendors/customize/op_api/include)
target_link_directories(basic_matmul_aclnn PRIVATE
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib/)
target_link_libraries(basic_matmul_aclnn PRIVATE ascendcl cust_opapi nnopbase)
```

## 预置示例

我们对以上操作过程进行了集成，以便快速体验aclnn工程接口。

### 编译指定用例

```bash
bash scripts/build.sh basic_matmul_aclnn
cd output/run
chmod +x ./custom_opp_*.run
./custom_opp_*.run
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./basic_matmul_aclnn 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```
Compare success.
```
