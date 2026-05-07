# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_source_files_properties(test_self_contained_includes.cpp PROPERTIES LANGUAGE ASCEND)
add_custom_target(test_self_contained_includes)

include_directories(${CATLASS_INCLUDE_DIR})
add_compile_options("--cce-aicore-arch=dav-c220")
add_compile_definitions(CATLASS_ARCH=2201)

file(GLOB_RECURSE CATLASS_HEADERS ${CATLASS_INCLUDE_DIR}/**/*.hpp)
list(FILTER CATLASS_HEADERS EXCLUDE REGEX "block")
list(FILTER CATLASS_HEADERS EXCLUDE REGEX "950")  # 950 compilation not support on CANN 8.5

foreach(CATLASS_HEADER ${CATLASS_HEADERS})
    file(RELATIVE_PATH REL_CATLASS_HEADER ${CATLASS_INCLUDE_DIR} ${CATLASS_HEADER})
    string(REPLACE "/" "_" TEST_TARGET ${REL_CATLASS_HEADER})
    string(REPLACE "." "_" TEST_TARGET ${TEST_TARGET})
    add_library(${TEST_TARGET} OBJECT test_self_contained_includes.cpp)
    target_compile_definitions(${TEST_TARGET} PRIVATE CATLASS_TEST_HEADER=${REL_CATLASS_HEADER})
    add_dependencies(test_self_contained_includes ${TEST_TARGET})

    # install(TARGETS ${TEST_TARGET} DESTINATION obj COMPONENT catlass_test_self_contained_includes)
endforeach()
