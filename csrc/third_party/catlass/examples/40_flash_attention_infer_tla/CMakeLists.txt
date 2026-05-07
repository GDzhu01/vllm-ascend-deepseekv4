# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(FAI_SOURCES
    fai.cpp)

set_source_files_properties(fai.cpp PROPERTIES LANGUAGE ASCEND)
catlass_example_add_executable(40_flash_attention_infer_tla mix ${FAI_SOURCES})
target_include_directories(40_flash_attention_infer_tla PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(40_flash_attention_infer_tla PRIVATE m)
