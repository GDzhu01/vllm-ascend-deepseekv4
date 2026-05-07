# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_source_files_properties(grouped_matmul_slice_m.cpp PROPERTIES LANGUAGE ASCEND)
catlass_example_add_executable(02_grouped_matmul_slice_m cube grouped_matmul_slice_m.cpp)
target_compile_definitions(02_grouped_matmul_slice_m PRIVATE L2_CACHE_HINT)

# Option to use segmented group list format instead of prefix-sum format
option(CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED "Use segmented group list format instead of prefix-sum format" OFF)
if(CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED)
    target_compile_definitions(02_grouped_matmul_slice_m PRIVATE CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED)
endif()