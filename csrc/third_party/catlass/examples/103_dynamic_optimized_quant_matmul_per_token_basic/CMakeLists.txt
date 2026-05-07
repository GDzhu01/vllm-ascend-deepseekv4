# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

option(COMPILE_DYNAMIC_QBMM_OPTIMIZED_MATMUL "Generate wrapper code for generalization matmul" OFF)

if(COMPILE_DYNAMIC_QBMM_OPTIMIZED_MATMUL)

    if(NOT DEFINED Python3_EXECUTABLE)
        message(WARNING "Python3_EXECUTABLE is not defined, using default python3")
        find_program(Python3_EXECUTABLE python3 python)
        if(NOT Python3_EXECUTABLE)
            message(FATAL_ERROR "Python executable not found. Please install Python 3.")
        endif()
    else()
        message("Using python: ${Python3_EXECUTABLE}")
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts/wrapper_code_gen.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts
    )

    add_custom_target(auto_gen_code_103)
    file(GLOB_RECURSE AUTO_GEN_CODE_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts/*.py)
    add_custom_command(
        TARGET auto_gen_code_103
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts/wrapper_code_gen.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts
        DEPENDS ${AUTO_GEN_CODE_SRC}
        COMMENT "Generate wrapper code, launch_map.h ...done"
    )
    
    add_compile_definitions(TILING_KEY_VAR)
    add_compile_options($<$<COMPILE_LANGUAGE:ASCEND>:--cce-aicore-arch=dav-c220>)
    file(GLOB CATLASS_SHARED_LIB_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/impl/wrapper/*.cpp)
    set_source_files_properties(${CATLASS_SHARED_LIB_SRC} PROPERTIES LANGUAGE ASCEND)

    add_library(dynamic_optimized_quant_matmul_per_token_basic SHARED ${CATLASS_SHARED_LIB_SRC})
    set_target_properties(dynamic_optimized_quant_matmul_per_token_basic PROPERTIES POSITION_INDEPENDENT_CODE ON)

    include_directories(
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/include/experiment/runtime
        ${ASCEND_HOME_PATH}/include/experiment/msprof
        ${ASCEND_HOME_PATH}/include/aclnn
        ${CMAKE_CURRENT_SOURCE_DIR}/impl
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

    link_libraries(dynamic_optimized_quant_matmul_per_token_basic runtime)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)

    add_executable(103_dynamic_optimized_quant_matmul_per_token_basic dynamic_optimized_quant_matmul_per_token_basic.cpp)
    add_dependencies(catlass_examples 103_dynamic_optimized_quant_matmul_per_token_basic)
    add_dependencies(103_dynamic_optimized_quant_matmul_per_token_basic dynamic_optimized_quant_matmul_per_token_basic)
    add_dependencies(dynamic_optimized_quant_matmul_per_token_basic auto_gen_code_103)

    install(TARGETS dynamic_optimized_quant_matmul_per_token_basic DESTINATION shared_lib/lib COMPONENT 103_dynamic_optimized_quant_matmul_per_token_basic)
    install(TARGETS dynamic_optimized_quant_matmul_per_token_basic DESTINATION shared_lib/lib COMPONENT catlass_examples)
    install(TARGETS 103_dynamic_optimized_quant_matmul_per_token_basic DESTINATION bin COMPONENT 103_dynamic_optimized_quant_matmul_per_token_basic)
    install(TARGETS 103_dynamic_optimized_quant_matmul_per_token_basic DESTINATION bin COMPONENT catlass_examples)

endif()