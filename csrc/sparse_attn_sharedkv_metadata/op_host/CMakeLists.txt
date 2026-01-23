# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ---------------------------------------------------------------------------------------------------------

set(sparse_attn_sharedkv_metadata_depends
    csrc/sparse_attn_sharedkv
    PARENT_SCOPE
)


# ---- host side: add whatever exists in this directory (def/tiling/infershape) ----
file(GLOB _host_defs   "${CMAKE_CURRENT_LIST_DIR}/*_def.cpp")
file(GLOB _host_tiling "${CMAKE_CURRENT_LIST_DIR}/*_tiling.cpp")
file(GLOB _host_infer  "${CMAKE_CURRENT_LIST_DIR}/*_infershape.cpp")

if (_host_defs)
    target_sources(op_host_aclnn PRIVATE ${_host_defs})
endif()
if (_host_tiling)
    target_sources(optiling PRIVATE ${_host_tiling})
endif()
if (_host_infer)
    target_sources(opsproto PRIVATE ${_host_infer})
endif()

# ---------------- AICPU kernel/json registration ----------------
set(_aicpu_json "${CMAKE_CURRENT_LIST_DIR}/../op_kernel_aicpu/sparse_attn_sharedkv_metadata_aicpu.json")
if (EXISTS "${_aicpu_json}")
    set_property(GLOBAL APPEND PROPERTY AICPU_JSON_FILES "${_aicpu_json}")
endif()

file(GLOB _aicpu_srcs
    "${CMAKE_CURRENT_LIST_DIR}/../op_kernel_aicpu/*.cc"
    "${CMAKE_CURRENT_LIST_DIR}/../op_kernel_aicpu/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../op_kernel_aicpu/*.cxx"
)

if (_aicpu_srcs)
    add_library(sparse_attn_sharedkv_metadata_aicpu_obj OBJECT ${_aicpu_srcs})
    target_link_libraries(sparse_attn_sharedkv_metadata_aicpu_obj PRIVATE vllm_ascend_aicpu_headers)
    target_compile_options(sparse_attn_sharedkv_metadata_aicpu_obj PRIVATE -fPIC)
    target_include_directories(sparse_attn_sharedkv_metadata_aicpu_obj PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    )
    set_property(GLOBAL APPEND PROPERTY AICPU_CUST_OBJ_TARGETS
        sparse_attn_sharedkv_metadata_aicpu_obj
    )
endif()
