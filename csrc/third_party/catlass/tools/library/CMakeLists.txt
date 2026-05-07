# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if ("${CATLASS_LIBRARY_KERNELS}" STREQUAL "")
    message(FATAL_ERROR "CATLASS_LIBRARY_KERNELS is not allowed to be empty")
else()
    message(STATUS "CATLASS_LIBRARY_KERNELS=${CATLASS_LIBRARY_KERNELS}")
endif()

set(ARCH "AtlasA2")

find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/code_generator.py
        --kernels ${CATLASS_LIBRARY_KERNELS}
        --workspace-dir ${CMAKE_CURRENT_BINARY_DIR}
        --arch ${ARCH}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RESULT_VARIABLE CATLASS_LIBRARY_CODE_GENERATION_RESULT
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/catlass_library_code_generation.log
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/catlass_library_code_generation.log
)

if(NOT CATLASS_LIBRARY_CODE_GENERATION_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate catlass library operator code. "
                        "See details in ${CMAKE_CURRENT_BINARY_DIR}/catlass_library_code_generation.log")
endif()

file(GLOB_RECURSE CATLASS_LIBRARY_GENERATED_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated/*.cpp)

message(STATUS "Catlass library operator code was generated successfully.")

list(APPEND CATLASS_LIBRARY_GENERATED_SRCS src/manifest.cpp)

set_source_files_properties(${CATLASS_LIBRARY_GENERATED_SRCS} PROPERTIES LANGUAGE ASCEND)

add_compile_options(
    $<$<COMPILE_LANGUAGE:ASCEND>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:ASCEND>:-Wno-ignored-attributes>

    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-stack-size=0x8000>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-function-stack-size=0x8000>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-record-overflow=true>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-addr-transform>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-dcci-insert-for-scalar=false>"
)

add_compile_definitions(L2_CACHE_HINT)
add_compile_definitions(TILING_KEY_VAR)

add_library(catlass_kernels SHARED
    ${CATLASS_LIBRARY_GENERATED_SRCS}
    src/manifest.cpp
)

set(AICORE_ARCH dav-c220)
target_compile_options(catlass_kernels PRIVATE $<$<COMPILE_LANGUAGE:ASCEND>:--cce-aicore-arch=${AICORE_ARCH}>)

set_target_properties(catlass_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(catlass_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/common
    ${_CMAKE_ASCEND_HOME_PATH}/include
)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../../output CACHE PATH "Installation prefix for CATLASS" FORCE)

install(TARGETS catlass_kernels
        DESTINATION lib64
        COMPONENT catlass_kernels)
