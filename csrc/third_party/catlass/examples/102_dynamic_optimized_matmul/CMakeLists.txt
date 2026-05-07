# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

option(COMPILE_DYNAMIC_OPTIMIZED_MATMUL "Generate wrapper code for generalization matmul" OFF)

if(COMPILE_DYNAMIC_OPTIMIZED_MATMUL)

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

    add_custom_target(auto_gen_code)
    file(GLOB_RECURSE AUTO_GEN_CODE_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts/*.py)
    add_custom_command(
        TARGET auto_gen_code
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts/wrapper_code_gen.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts
        DEPENDS ${AUTO_GEN_CODE_SRC}
        COMMENT "Generate wrapper code, launch_map.h ...done"
    )
    
    add_compile_definitions(TILING_KEY_VAR)
    add_compile_options($<$<COMPILE_LANGUAGE:ASCEND>:--cce-aicore-arch=dav-c220>)
    file(GLOB CATLASS_SHARED_LIB_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/impl/wrapper/*.cpp)
    set_source_files_properties(${CATLASS_SHARED_LIB_SRC} PROPERTIES LANGUAGE ASCEND)

    set(BATCH_SIZE 200)
    list(LENGTH CATLASS_SHARED_LIB_SRC TOTAL_FILES_NUM)
    math(EXPR BATCH_COUNT "(${TOTAL_FILES_NUM} + ${BATCH_SIZE} - 1) / ${BATCH_SIZE} - 1")

    set(102_DYNAMIC_OPTIMIZED_MATMUL_LIBS "")
    foreach(BATCH_INDEX RANGE ${BATCH_COUNT})
        math(EXPR START_INDEX "${BATCH_INDEX} * ${BATCH_SIZE}")
        math(EXPR END_INDEX "${START_INDEX} + ${BATCH_SIZE}")
        math(EXPR ACTUAL_SIZE "${BATCH_SIZE}")
        if (${END_INDEX} GREATER TOTAL_FILES_NUM)
            math(EXPR ACTUAL_SIZE "${TOTAL_FILES_NUM} - ${BATCH_INDEX} * ${BATCH_SIZE}")
        endif()
        list(SUBLIST CATLASS_SHARED_LIB_SRC ${START_INDEX} ${ACTUAL_SIZE} CURRENT_FILES)

        set(SHARED_LIB_NAME "102_dynamic_optimized_kernel_batch_${BATCH_INDEX}")
        add_library(${SHARED_LIB_NAME} SHARED ${CURRENT_FILES})
        set_target_properties(${SHARED_LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        list(APPEND 102_DYNAMIC_OPTIMIZED_MATMUL_LIBS ${SHARED_LIB_NAME})
    endforeach()

    include_directories(
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/include/experiment/runtime
        ${ASCEND_HOME_PATH}/include/experiment/msprof
        ${ASCEND_HOME_PATH}/include/aclnn
        ${CMAKE_CURRENT_SOURCE_DIR}/impl
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

    link_libraries(${102_DYNAMIC_OPTIMIZED_MATMUL_LIBS} runtime)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)

    add_executable(102_dynamic_optimized_matmul dynamic_optimized_matmul.cpp)
    add_dependencies(catlass_examples 102_dynamic_optimized_matmul)
    add_dependencies(102_dynamic_optimized_matmul ${102_DYNAMIC_OPTIMIZED_MATMUL_LIBS})
    # target_link_options(102_dynamic_optimized_matmul PRIVATE "-Wl,-rpath,${CMAKE_SOURCE_DIR}/output/shared_lib")

    foreach(LIB ${102_DYNAMIC_OPTIMIZED_MATMUL_LIBS})
        install(TARGETS ${LIB} DESTINATION shared_lib COMPONENT 102_dynamic_optimized_matmul)
        install(TARGETS ${LIB} DESTINATION shared_lib COMPONENT catlass_examples)
    endforeach()
    install(TARGETS 102_dynamic_optimized_matmul DESTINATION bin COMPONENT 102_dynamic_optimized_matmul)
    install(TARGETS 102_dynamic_optimized_matmul DESTINATION bin COMPONENT catlass_examples)

    # show notice
    add_custom_command(TARGET 102_dynamic_optimized_matmul POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "========================================================================="
        COMMAND ${CMAKE_COMMAND} -E echo "                            IMPORTANT NOTICE"
        COMMAND ${CMAKE_COMMAND} -E echo "========================================================================="
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "The shared libraries have been built successfully!"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "To use them, please add the shared library path to your LD_LIBRARY_PATH."
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "You can enter the following command:"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "    export LD_LIBRARY_PATH=${CMAKE_SOURCE_DIR}/output/shared_lib:\\$$LD_LIBRARY_PATH"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "========================================================================="
    )

endif()
