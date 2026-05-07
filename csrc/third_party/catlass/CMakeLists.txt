# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ASCEND_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${ASCEND_CMAKE_PATH})
project(Catlass LANGUAGES ASCEND CXX)
include(cmake/funcs.cmake)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(CATLASS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

if(NOT DEFINED CMAKE_BUILD_TYPE OR NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT DEFINED CATLASS_ARCH OR NOT CATLASS_ARCH)
    message(WARNING "CATLASS_ARCH is not defined, use default value \"2201\"")
    set(CATLASS_ARCH 2201)
endif()

# Compile options for secure compile
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_SKIP_RPATH TRUE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fstack-protector-strong>)
    add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-Xhost-start -fstack-protector-strong -Xhost-end>")
    add_link_options(-Wl,-z,relro,-z,now)
    add_link_options(-Wl,-z,noexecstack)
    add_link_options(-s)
endif()

add_compile_definitions(CATLASS_BUILD_LEGACY)

add_subdirectory(examples)
add_subdirectory(tools)

if(DEFINED BUILD_TESTS AND BUILD_TESTS)
    add_subdirectory(tests)
endif()
