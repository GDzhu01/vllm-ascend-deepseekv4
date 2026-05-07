# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# if aicore-arch is different, maybe crash, to find the solution
add_compile_definitions(TILING_KEY_VAR)
add_compile_options($<$<COMPILE_LANGUAGE:ASCEND>:--cce-aicore-arch=${CCE_AICORE_ARCH_BASE}>)
file(GLOB CATLASS_SHARED_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cpp)
set_source_files_properties(${CATLASS_SHARED_LIB_SRC} PROPERTIES LANGUAGE ASCEND)

add_library(catlass_kernel SHARED ${CATLASS_SHARED_LIB_SRC})
add_library(catlass_kernel_static STATIC ${CATLASS_SHARED_LIB_SRC})
set_target_properties(catlass_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(catlass_kernel_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(catlass_kernel PROPERTIES OUTPUT_NAME catlass_kernel)

if(DEFINED ENABLE_MSSANITIZER AND ENABLE_MSSANITIZER)
    target_compile_options(catlass_kernel PRIVATE "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-g --cce-enable-sanitizer>")
    target_link_options(catlass_kernel PRIVATE "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:--cce-enable-sanitizer --cce-aicore-arch=${CCE_AICORE_ARCH_BASE}>")
    target_compile_options(catlass_kernel_static PRIVATE "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-g --cce-enable-sanitizer>")
    target_link_options(catlass_kernel_static PRIVATE "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:--cce-enable-sanitizer --cce-aicore-arch=${CCE_AICORE_ARCH_BASE}>")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(catlass_kernel PRIVATE runtime)
target_link_libraries(catlass_kernel_static PRIVATE runtime)

add_custom_target(shared_lib)
add_dependencies(shared_lib catlass_kernel)
add_dependencies(shared_lib catlass_kernel_static)

if(DEFINED CATLASS_BUILD_USAGE AND CATLASS_BUILD_USAGE)
    add_executable(basic_matmul_shared_lib basic_matmul_shared_lib.cpp)
    target_compile_definitions(basic_matmul_shared_lib PRIVATE CATLASS_EXAMPLE_NAME=basic_matmul_shared_lib)
    target_include_directories(basic_matmul_shared_lib PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/include/aclnn)
    target_link_libraries(basic_matmul_shared_lib PRIVATE catlass_kernel_static)
    add_dependencies(shared_lib basic_matmul_shared_lib)
    install(TARGETS basic_matmul_shared_lib DESTINATION bin COMPONENT shared_lib)
endif()

install(TARGETS catlass_kernel catlass_kernel_static
    DESTINATION shared_lib/lib
    COMPONENT shared_lib)
install(DIRECTORY include DESTINATION shared_lib
    COMPONENT shared_lib)
