# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(determine_npu_model var arch)
    # Only continue if arch is a2 or a3
    if(NOT (${arch} STREQUAL "2201" OR ${arch} STREQUAL "3510"))
        message(STATUS "determine_npu_model: arch is '${arch}', not 2201 or 3510, skipping.")
        set(${var} "" PARENT_SCOPE)
        return()
    endif()

    # Use cached NPU_MODEL if defined and non-empty
    if(DEFINED NPU_MODEL AND NOT NPU_MODEL STREQUAL "")
        message(STATUS "NPU_MODEL is defined: ${NPU_MODEL}")
        set(${var} "${NPU_MODEL}" PARENT_SCOPE)
        return()
    endif()

    # check if npu-smi exists.
    execute_process(
        COMMAND which npu-smi
        RESULT_VARIABLE NPU_SMI_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    # if not exists, hint user to check command, or pass NPU_MODEL manually.
    if(NOT NPU_SMI_RESULT EQUAL 0)
        message(FATAL_ERROR "Neither npu-smi command is available nor NPU_MODEL is defined.\nPlease check npu-smi command or set NPU_MODEL manually.")
        return()
    endif()

    # check minimal card id
    execute_process(
        COMMAND bash -c "npu-smi info -l | awk '/NPU ID/ {print \$NF; exit}'"
        RESULT_VARIABLE NPU_SMI_RESULT
        OUTPUT_VARIABLE NPU_SMI_MINIMAL_CARD_ID
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT NPU_SMI_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to get NPU_SMI_MINIMAL_CARD_ID")
        return()
    endif()

    # Execute npu-smi to get NPU_MODEL
    execute_process(
        COMMAND npu-smi info -t board -i ${NPU_SMI_MINIMAL_CARD_ID} -c 0
        RESULT_VARIABLE NPU_SMI_RESULT
        OUTPUT_VARIABLE NPU_SMI_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT NPU_SMI_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to execute 'npu-smi info -t board -i ${NPU_SMI_MINIMAL_CARD_ID} -c 0'")
        return()
    endif()

    # Extract Chip Name from npu-smi output
    execute_process(
        COMMAND bash -c "echo '${NPU_SMI_OUTPUT}' | awk '/Chip Name/ {print \$NF}'"
        OUTPUT_VARIABLE NPU_SMI_CHIP_NAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Compose npu_model according to arch
    if(${NPU_SMI_CHIP_NAME} STREQUAL "Ascend910")
        # Extract NPU Name from npu-smi output
        execute_process(
            COMMAND bash -c "echo '${NPU_SMI_OUTPUT}' | awk '/NPU Name/ {print \$NF}'"
            OUTPUT_VARIABLE NPU_SMI_NPU_NAME
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        set(npu_model "${NPU_SMI_CHIP_NAME}_${NPU_SMI_NPU_NAME}")
    else()
        set(npu_model "Ascend${NPU_SMI_CHIP_NAME}")
    endif()

    set(${var} "${npu_model}" PARENT_SCOPE)
endfunction()