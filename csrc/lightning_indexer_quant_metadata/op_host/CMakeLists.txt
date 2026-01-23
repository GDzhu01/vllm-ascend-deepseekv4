# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


set(lightning_indexer_quant_metadata_depends
    csrc/quant_lightning_indexer
    PARENT_SCOPE
)

# ---------------- AICPU kernel/json registration ----------------
# 1) register aicpu json for merge_aicpu_info_json.sh
set(_aicpu_json "${CMAKE_CURRENT_LIST_DIR}/../op_kernel_aicpu/lightning_indexer_quant_metadata_aicpu.json")
if (EXISTS "${_aicpu_json}")
    set_property(GLOBAL APPEND PROPERTY AICPU_JSON_FILES "${_aicpu_json}")
endif()

# 2) compile aicpu kernel to OBJECT, later linked by gen_cust_aicpu_kernel_symbol()
file(GLOB _aicpu_srcs
    "${CMAKE_CURRENT_LIST_DIR}/../op_kernel_aicpu/*.cc"
    "${CMAKE_CURRENT_LIST_DIR}/../op_kernel_aicpu/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../op_kernel_aicpu/*.cxx"
)

if (_aicpu_srcs)
    add_library(lightning_indexer_quant_metadata_aicpu_obj OBJECT ${_aicpu_srcs})
    target_link_libraries(lightning_indexer_quant_metadata_aicpu_obj PRIVATE vllm_ascend_aicpu_headers)
    target_compile_options(lightning_indexer_quant_metadata_aicpu_obj PRIVATE -fPIC)

    # keep it minimal; if compile fails, we add more includes based on error
    target_include_directories(lightning_indexer_quant_metadata_aicpu_obj PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    )

    set_property(GLOBAL APPEND PROPERTY AICPU_CUST_OBJ_TARGETS
        lightning_indexer_quant_metadata_aicpu_obj
    )
endif()
