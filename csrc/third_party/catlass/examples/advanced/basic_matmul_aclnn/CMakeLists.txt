# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(MSOPGEN_PROJECT_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/msopgen)
set(MSOPGEN_CMAKE_CURRENT_BINARY_DIR ${MSOPGEN_PROJECT_SOURCE_DIR}/build_out)

if(NOT EXISTS ${MSOPGEN_PROJECT_SOURCE_DIR})
    # create tmp directory
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tmp)
    # msopgen can only generate to a directory with 755 permission
    execute_process(
        COMMAND chmod 755 ${CMAKE_CURRENT_BINARY_DIR}/tmp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    # copy json configuration file (use file(COPY ...) instead of file(COPY_FILE ...) for CMake <3.21)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/catlass_basic_matmul.json DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tmp)
    # msopgen can only generate from a 644 json configuration file
    execute_process(
        COMMAND chmod 644 ${CMAKE_CURRENT_BINARY_DIR}/tmp/catlass_basic_matmul.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    find_program(MSOPGEN NAMES msopgen HINTS ${ASCEND_HOME_PATH})
    add_custom_target(generated_msopgen_project
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MSOPGEN_PROJECT_SOURCE_DIR}
        COMMAND ${MSOPGEN} gen -i ${CMAKE_CURRENT_BINARY_DIR}/tmp/catlass_basic_matmul.json -c ai_core-${NPU_MODEL} -lan cpp -out ${MSOPGEN_PROJECT_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/op_host/catlass_basic_matmul.cpp ${MSOPGEN_PROJECT_SOURCE_DIR}/op_host
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/op_host/catlass_basic_matmul_tiling.h ${MSOPGEN_PROJECT_SOURCE_DIR}/op_host
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/catlass_basic_matmul.cpp ${MSOPGEN_PROJECT_SOURCE_DIR}/op_kernel
        COMMAND ${CMAKE_COMMAND} -E echo "add_ops_compile_options\\(ALL OPTIONS -I${CATLASS_INCLUDE_DIR} -DCATLASS_ARCH=2201\\)" >> ${MSOPGEN_PROJECT_SOURCE_DIR}/op_kernel/CMakeLists.txt
        # uncomment this line to change the vendor name to "catlass" to avoid conflict
        # COMMAND sed -i "s/customize/catlass/g" ${MSOPGEN_PROJECT_SOURCE_DIR}/CMakePresets.json
        BYPRODUCTS ${MSOPGEN_PROJECT_SOURCE_DIR}
    )
else()
    add_custom_target(generated_msopgen_project
        COMMENT "msOpGen project directory already exists"
    )
endif()

execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
set(SYSTEM_INFO "${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}")
set(PACKAGE_NAME "custom_opp_${SYSTEM_INFO}.run")

add_custom_target(basic_matmul_aclnn_run_package
    COMMAND bash -c "${MSOPGEN_PROJECT_SOURCE_DIR}/build.sh"
    WORKING_DIRECTORY ${MSOPGEN_PROJECT_SOURCE_DIR}
    BYPRODUCTS
    "${MSOPGEN_CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}"
)
add_dependencies(basic_matmul_aclnn_run_package generated_msopgen_project)

add_executable(basic_matmul_aclnn basic_matmul_aclnn.cpp)
add_dependencies(basic_matmul_aclnn basic_matmul_aclnn_run_package)
target_include_directories(basic_matmul_aclnn PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/aclnn
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
    ${MSOPGEN_CMAKE_CURRENT_BINARY_DIR}/autogen)
target_link_directories(basic_matmul_aclnn PRIVATE
    ${MSOPGEN_CMAKE_CURRENT_BINARY_DIR}/op_host)
target_link_libraries(basic_matmul_aclnn PRIVATE cust_opapi)

install(
    FILES "${MSOPGEN_CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}"
    DESTINATION run
    COMPONENT basic_matmul_aclnn
)

install(TARGETS basic_matmul_aclnn
    RUNTIME DESTINATION bin/advanced
    COMPONENT basic_matmul_aclnn
)

install(CODE "message(STATUS \"Before executing example for aclnn interfaces, \")" COMPONENT basic_matmul_aclnn)
install(CODE "message(STATUS \"1. Install ${CMAKE_INSTALL_PREFIX}/run/${PACKAGE_NAME}. \")" COMPONENT basic_matmul_aclnn)
install(CODE "message(STATUS \"2. Follow installation logs to set environment variables. \")" COMPONENT basic_matmul_aclnn)


