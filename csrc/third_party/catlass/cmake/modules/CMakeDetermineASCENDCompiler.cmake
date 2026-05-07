# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(CMAKE_ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
else()
    message(FATAL_ERROR
        "no, installation path found, should passing -DASCEND_HOME_PATH=<PATH_TO_ASCEND_INSTALLATION> in cmake"
    )
    set(CMAKE_ASCEND_HOME_PATH)
endif()

message(STATUS "ASCEND_HOME_PATH:" "  $ENV{ASCEND_HOME_PATH}")

find_program(CMAKE_ASCEND_COMPILER
    NAMES "bisheng"
    PATHS "$ENV{PATH}"
    DOC "Ascend Compiler")

mark_as_advanced(CMAKE_ASCEND_COMPILER)

message(STATUS "CMAKE_ASCEND_COMPILER: " ${CMAKE_ASCEND_COMPILER})
message(STATUS "Ascend Compiler Information:")
execute_process(
    COMMAND ${CMAKE_ASCEND_COMPILER} --version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_ASCEND_SOURCE_FILE_EXTENSIONS cce)
set(CMAKE_ASCEND_COMPILER_ENV_VAR "Ascend")
message(STATUS "CMAKE_CURRENT_LIST_DIR: " ${CMAKE_CURRENT_LIST_DIR})

set(CMAKE_ASCEND_HOST_IMPLICIT_LINK_DIRECTORIES
    ${CMAKE_ASCEND_HOME_PATH}/lib64
)

set(CMAKE_ASCEND_HOST_IMPLICIT_LINK_LIBRARIES
    stdc++ runtime
)

if(DEFINED ASCEND_ENABLE_MSPROF AND ASCEND_ENABLE_MSPROF)
    list(APPEND CMAKE_ASCEND_HOST_IMPLICIT_LINK_LIBRARIES profapi)
endif()

set(CMAKE_ASCEND_HOST_IMPLICIT_INCLUDE_DIRECTORIES
    ${CMAKE_ASCEND_HOME_PATH}/include
    ${CMAKE_ASCEND_HOME_PATH}/include/experiment/runtime
    ${CMAKE_ASCEND_HOME_PATH}/include/experiment/msprof
)

if(NOT DEFINED ASCEND_ENABLE_ASCENDC OR ASCEND_ENABLE_ASCENDC)
    list(APPEND CMAKE_ASCEND_HOST_IMPLICIT_INCLUDE_DIRECTORIES
        ${CMAKE_ASCEND_HOME_PATH}/compiler/tikcpp
        ${CMAKE_ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
        ${CMAKE_ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
        ${CMAKE_ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    )
endif()

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeASCENDCompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeASCENDCompiler.cmake
    @ONLY
)
