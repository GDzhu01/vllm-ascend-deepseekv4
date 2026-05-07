# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CATLASS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_custom_target(catlass_examples)

# arch specific compile options
if(CATLASS_ARCH STREQUAL "2201")
    set(CCE_AICORE_ARCH_BASE dav-c220)
    add_compile_definitions(CATLASS_ARCH=2201)
    add_compile_options(
        "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-stack-size=0x8000>"
        "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-function-stack-size=0x8000>"
        "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-record-overflow=true>"
        "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-dcci-insert-for-scalar=false>"
    )
endif()

if(CATLASS_ARCH STREQUAL "3510")
    set(CCE_AICORE_ARCH_BASE dav-c310)
    add_compile_definitions(CATLASS_ARCH=3510)
endif()
# common compile options
add_compile_options(
    $<$<COMPILE_LANGUAGE:ASCEND>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:ASCEND>:-Wno-ignored-attributes>
    # these options may be removed after CANN 8.5.0
    $<$<COMPILE_LANGUAGE:ASCEND>:-Wno-unused-value>
)

# these options may be removed after CANN 8.5.0
add_link_options($<$<COMPILE_LANGUAGE:ASCEND>:-Wno-unused-command-line-argument>)

# compile options for debug features
if(DEFINED ENABLE_PRINT AND ENABLE_PRINT)
    add_compile_options($<$<COMPILE_LANGUAGE:ASCEND>:--cce-enable-print>)
endif()

if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
    add_compile_definitions(ENABLE_ASCENDC_DUMP)
    add_compile_definitions(ASCENDC_DUMP=1)
    add_compile_definitions(ASCENDC_DEBUG)
    link_libraries(ascend_dump)
endif()

if(NOT DEFINED NPU_MODEL OR NPU_MODEL STREQUAL "")
    determine_npu_model(NPU_MODEL CATLASS_ARCH)
    if(NPU_MODEL STREQUAL "")
        message(FATAL_ERROR "Failed to determine NPU_MODEL. Please set NPU_MODEL manually by -DNPU_MODEL=Ascendxxxyy.")
        return()
    endif()
endif()
message("NPU MODEL: ${NPU_MODEL}")

if(DEFINED ENABLE_SIMULATOR AND ENABLE_SIMULATOR)
    message("To run examples in simulator mode, execute the following commands in your terminal:")
    message("export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${NPU_MODEL}/lib:\$LD_LIBRARY_PATH")
    message("export LD_PRELOAD=${ASCEND_HOME_PATH}/tools/simulator/${NPU_MODEL}/lib/libruntime_camodel.so:${ASCEND_HOME_PATH}/tools/simulator/${NPU_MODEL}/lib/libnpu_drv_camodel.so")
endif()

macro(catlass_example_add_executable NAME OPTYPE)
    set(EXAMPLE_DESTINATION bin)

    add_executable(${NAME} ${ARGN})
    if(${OPTYPE} STREQUAL "cube" OR ${OPTYPE} STREQUAL "vec")
        set(CCE_AICORE_ARCH "${CCE_AICORE_ARCH_BASE}-${OPTYPE}")
    elseif(${OPTYPE} STREQUAL "mix")
        set(CCE_AICORE_ARCH "${CCE_AICORE_ARCH_BASE}")
    else()
        message("Unsupported op type: ${OPTYPE}")
    endif()
    target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:ASCEND>:--cce-aicore-arch=${CCE_AICORE_ARCH}>)
    target_compile_definitions(${NAME} PRIVATE CATLASS_EXAMPLE_NAME=${NAME})
    if(DEFINED ENABLE_MSSANITIZER AND ENABLE_MSSANITIZER)
        target_compile_options(${NAME} PRIVATE "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-g --cce-enable-sanitizer>")
        target_link_options(${NAME} PRIVATE "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:--cce-enable-sanitizer --cce-aicore-arch=${CCE_AICORE_ARCH}>")
    endif()
    target_include_directories(${NAME} PRIVATE ${ASCEND_HOME_PATH}/include/aclnn)
    add_dependencies(catlass_examples ${NAME})
    install(TARGETS ${NAME} DESTINATION ${EXAMPLE_DESTINATION} COMPONENT ${NAME})
    install(TARGETS ${NAME} DESTINATION ${EXAMPLE_DESTINATION} COMPONENT catlass_examples)
endmacro()

include_directories(
    ${CATLASS_EXAMPLES_COMMON_SOURCE_DIR}
    ${CATLASS_INCLUDE_DIR}
)
link_directories(${ASCEND_HOME_PATH}/lib64)
link_libraries(dl tiling_api platform c_sec nnopbase ascendcl)

link_directories(${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib)
link_libraries(ascend_hal)

# regular examples
set(EXAMPLE_ATLASA2
    00_basic_matmul
    01_batched_matmul
    02_grouped_matmul_slice_m
    03_matmul_add
    04_padding_matmul
    05_grouped_matmul_slice_k
    06_optimized_matmul
    07_grouped_matmul_slice_m_per_token_dequant_moe
    08_grouped_matmul
    09_splitk_matmul
    10_grouped_matmul_slice_m_per_token_dequant
    11_grouped_matmul_slice_k_per_token_dequant
    12_quant_matmul
    13_basic_matmul_tla
    14_optimized_matmul_tla
    15_gemm
    16_group_gemm
    17_gemv_aiv
    18_gemv_aic
    19_mla
    20_matmul_bias
    21_basic_matmul_preload_zN
    22_padding_splitk_matmul
    23_flash_attention_infer
    24_conv_bias
    25_matmul_full_loadA
    26_matmul_relu
    27_matmul_gelu
    28_matmul_silu
    29_a2_fp8_e4m3_matmul
    30_w8a16_matmul
    31_small_matmul
    32_w4a8_matmul
    33_basic_conv2d
    34_single_core_splitk_matmul
    37_streamk_matmul
    38_w4a4_matmul_per_token_per_channel_dequant
    39_big_matmul_tla
    40_flash_attention_infer_tla
    41_sparse_matmul_tla
    42_quant_optimized_matmul_tla
    44_quant_matmul_full_loadA_tla
    102_dynamic_optimized_matmul
    103_dynamic_optimized_quant_matmul_per_token_basic
)

set(EXAMPLE_ASCEND950
    43_ascend950_basic_matmul
)

if(CATLASS_ARCH STREQUAL "2201")
    set(EXAMPLE_LIST ${EXAMPLE_ATLASA2})
elseif(CATLASS_ARCH STREQUAL "3510")
    set(EXAMPLE_LIST ${EXAMPLE_ASCEND950})
endif()

foreach(EXAMPLE ${EXAMPLE_LIST})
    add_subdirectory(${EXAMPLE})
endforeach()

# advanded examples
add_subdirectory(shared_lib)
add_subdirectory(python_extension)
add_subdirectory(advanced/basic_matmul_aclnn)
