# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CATLASS_BUILD_DIR ${PROJECT_SOURCE_DIR}/build)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/examples)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../library/include)
link_directories(${CMAKE_BINARY_DIR}/tools/library)

if(NOT DEFINED ASCEND_HOME_PATH)
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    include_directories(
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/include/aclnn
        ${ASCEND_HOME_PATH}/include/experiment/runtime
        ${ASCEND_HOME_PATH}/include/experiment/msprof
    )
    link_directories(
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/lib64/../devlib/linux/${CMAKE_SYSTEM_PROCESSOR}
    )
endif()
add_subdirectory(dfx_kernel)
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(mstuner_catlass ${SOURCES})

add_dependencies(mstuner_catlass
    catlass_kernels
    dfx_kernel
)

target_link_libraries(mstuner_catlass PRIVATE
    stdc++fs
    pthread
    tiling_api
    platform
    c_sec
    ascendcl
    ascend_hal
    dl
    runtime
    ${CATLASS_BUILD_DIR}/tools/library/libcatlass_kernels.so
    dfx_kernel
    profapi
)

install(TARGETS mstuner_catlass
        DESTINATION bin
        COMPONENT mstuner_catlass)
