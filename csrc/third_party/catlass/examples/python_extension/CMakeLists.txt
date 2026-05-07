# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CATLASS_SHARED_LIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../shared_lib)
set(CATLASS_SHARED_LIB_INCLUDE_DIR ${CATLASS_SHARED_LIB_SRC_DIR}/include)
set(CATLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../../include)

if(NOT BUILD_PYBIND AND NOT BUILD_TORCH_LIB)
    message("NOT build python extension, skip python related dependencies check.")
    return()
endif()

if(NOT DEFINED Python3_EXECUTABLE)
    message(WARNING "Python3_EXECUTABLE is not defined, using default python3")
else()
    message("Using python: ${Python3_EXECUTABLE}")
endif()

find_package(Python3 COMPONENTS Interpreter Development)

list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}")

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_SITELIB}/torch_npu/include
    ${ASCEND_HOME_PATH}/include
    ${CATLASS_SHARED_LIB_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include)

link_directories(${ASCEND_HOME_PATH}/lib64
    ${Python3_SITELIB}/torch_npu/lib)

link_libraries(${TORCH_LIBRARIES} torch_npu tiling_api)

file(GLOB WRAPPER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper/*.cpp)
add_library(catlass_kernel_wrapper OBJECT ${WRAPPER_SOURCES})
target_compile_options(catlass_kernel_wrapper PRIVATE -fPIC)

if(BUILD_PYBIND)
    set(PYBIND_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/pybind_bindings.cpp)
    pybind11_add_module(_C
        SHARED ${PYBIND_BINDINGS_SRC} $<TARGET_OBJECTS:catlass_kernel_wrapper>)
    target_link_libraries(_C PRIVATE catlass_kernel)
    install(TARGETS _C DESTINATION . COMPONENT _python_extension_lib)
    install(TARGETS catlass_kernel DESTINATION lib COMPONENT _python_extension_lib)
endif()

if(BUILD_TORCH_LIB)
    set(TORCH_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/torch_bindings.cpp)
    add_library(catlass_torch
        SHARED ${TORCH_BINDINGS_SRC} $<TARGET_OBJECTS:catlass_kernel_wrapper>)
    target_link_libraries(catlass_torch PRIVATE catlass_kernel_static)
    install(TARGETS catlass_torch DESTINATION . COMPONENT catlass_torch)
endif()
