# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(MLA_SOURCES
    mla.cpp
    mla_tiling.cpp)
set_source_files_properties(mla.cpp PROPERTIES LANGUAGE ASCEND)
catlass_example_add_executable(19_mla mix ${MLA_SOURCES})
target_include_directories(19_mla PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(19_mla PRIVATE m)
