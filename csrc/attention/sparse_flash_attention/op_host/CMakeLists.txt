# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# add_ops_compile_options(
#         OP_NAME SparseFlashAttention
#         OPTIONS --cce-auto-sync=off
#                 -Wno-deprecated-declarations
#                 -Werror
#                 -fpermissive
# )

# set(sparse_flash_attention_depends transformer/attention/sparse_flash_attention PARENT_SCOPE)
# target_sources(op_host_aclnn PRIVATE
#         sparse_flash_attention_def.cpp
# )

# target_sources(optiling PRIVATE
#         sparse_flash_attention_tiling.cpp
# )

# if (NOT BUILD_OPEN_PROJECT)
#     target_sources(opmaster_ct PRIVATE
#         sparse_flash_attention_tiling.cpp
#     )
# endif ()

# target_sources(opsproto PRIVATE
#         sparse_flash_attention_proto.cpp
# )

# target_include_directories(optiling PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}
# )

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        sparse_flash_attention_def.cpp
    )
endif()

add_ops_compile_options(
        OP_NAME SparseFlashAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE sparse_flash_attention ACLNNTYPE aclnn)
endif()