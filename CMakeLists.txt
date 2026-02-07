# cmake_minimum_required(VERSION 3.16)
# project(vllm_ascend_C)

# # include(CheckCXXcompilerFlag)
# # check_cxx_compiler_flag("-std=c++17", COMPILER_SUPPORTS_CXX17)
# set(CMAKE_CXX_STANDARD 17)

# include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# # Suppress potential warnings about unused manually-specified variables
# set(ignoreMe "${VLLM_PYTHON_PATH}")

# # TODO: Add 3.12 back when torch-npu support 3.12
# set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11")

# find_package(pybind11 REQUIRED)

# append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
# set(VLLM_ASCEND_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

# find_package(Torch REQUIRED)

# run_python(TORCH_VERSION
#   "import torch; print(torch.__version__)" "Failed to locate torch path")
# # check torch version is 2.8.0
# if(NOT ${TORCH_VERSION} VERSION_EQUAL "2.8.0")
#   message(FATAL_ERROR "Expected PyTorch version 2.8.0, but found ${TORCH_VERSION}")
# endif()

# set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
# set(SOC_VERSION ${SOC_VERSION})
# message(STATUS "Detected SOC version: ${SOC_VERSION}")

# # Normalize for comparisons
# string(TOUPPER "${SOC_VERSION}" SOC_VERSION_UPPER)

# # SOCs that should follow the "minimal" / "exclude some ops" branch
# # - 310P3: existing special handling
# # - ascend910_95: you confirmed A5 passes this value (lowercase), we compare upper
# set(VLLM_SOC_MINIMAL_LIST
#   "ASCEND310P3"
#   "ASCEND910_95"
#   "ASCEND910_9589"
# )

# set(VLLM_MINIMAL_BUILD OFF)
# if(SOC_VERSION_UPPER IN_LIST VLLM_SOC_MINIMAL_LIST)
#   set(VLLM_MINIMAL_BUILD ON)
# endif()

# message(STATUS "VLLM_MINIMAL_BUILD=${VLLM_MINIMAL_BUILD} (SOC_VERSION_UPPER=${SOC_VERSION_UPPER})")

# if (NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
# endif()

# if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
#   set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRINGS "path to install()")
# endif()

# set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
# if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
#     set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
# elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
#     set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
# elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
#     set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
# else()
#     message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
# endif()

# include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# file(GLOB KERNEL_FILES
# ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/*.cpp)

# set(VLLM_ASCEND_CUSTOM_OP
#     ${KERNEL_FILES}
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
# )

# set(VLLM_ASCEND_CUSTOM_OP_EXCLUDE
#     # ${KERNEL_FILES}/bgmv_expand.cpp
#     # ${KERNEL_FILES}/bgmv_shrink.cpp
#     # ${KERNEL_FILES}/sgmv_expand.cpp
#     # ${KERNEL_FILES}/sgmv_shrink.cpp
#     ${KERNEL_FILES}
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
# )

# if(VLLM_MINIMAL_BUILD)
#     list(REMOVE_ITEM VLLM_ASCEND_CUSTOM_OP ${VLLM_ASCEND_CUSTOM_OP_EXCLUDE})
# else()
#   ascendc_library(vllm_ascend_kernels SHARED
#     ${VLLM_ASCEND_CUSTOM_OP}
# )
# endif()


# message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

# if(VLLM_MINIMAL_BUILD)
#     file(GLOB VLLM_ASCEND_SRC
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp)
# else()
#     file(GLOB VLLM_ASCEND_SRC
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_host/tiling/tiling_data.cpp)
# endif()

# include_directories(
#   ${pybind11_INCLUDE_DIRS}
#   ${PYTHON_INCLUDE_PATH}
#   ${TORCH_INCLUDE_DIRS}
#   ${TORCH_NPU_PATH}/include
#   ${ASCEND_HOME_PATH}/include
#   ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
#   ${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/platform
#   ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_host
# )

# set(
#   INCLUDES
#   ${TORCH_INCLUDE_DIRS}
#   ${TORCH_NPU_INCLUDE_DIRS}
#   ${ASCEND_HOME_PATH}/include
#   ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
#   ${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/platform
# )

# pybind11_add_module(vllm_ascend_C ${VLLM_ASCEND_SRC})

# target_link_directories(
#   vllm_ascend_C
#   PRIVATE
#   ${TORCH_LIBRARY_DIRS}
#   ${TORCH_NPU_PATH}/lib/
#   ${ASCEND_HOME_PATH}/lib64
# )


# if(VLLM_MINIMAL_BUILD)
#     target_link_libraries(
#   vllm_ascend_C
#   PUBLIC
#   ${TORCH_LIBRARIES}
#   torch_npu
#   ascendcl
#   tiling_api
#   register
#   platform
#   ascendalog
#   dl
#   opapi
# )
# else()
#   target_link_libraries(
#   vllm_ascend_C
#   PUBLIC
#   ${TORCH_LIBRARIES}
#   torch_npu
#   vllm_ascend_kernels
#   ascendcl
#   tiling_api
#   register
#   platform
#   ascendalog
#   dl
#   opapi
# )
# endif()


# target_link_options(vllm_ascend_C PRIVATE "-Wl,-rpath,$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/vllm-ascend/op_api/lib")


# if(VLLM_MINIMAL_BUILD)
#   install(TARGETS vllm_ascend_C DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
# else()
#   install(TARGETS vllm_ascend_C vllm_ascend_kernels DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
# endif()


cmake_minimum_required(VERSION 3.16)
project(vllm_ascend_C LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(_ignoreMe "${VLLM_PYTHON_PATH}")

# TODO: Add 3.12 back when torch-npu support 3.12
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11")

# ------------------------------------------------------------------------------
# Toolchain / deps
# ------------------------------------------------------------------------------
find_package(pybind11 REQUIRED)

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
set(VLLM_ASCEND_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

find_package(Torch REQUIRED)

# NOTE: if you have patched run_python to set TORCH_DEVICE_BACKEND_AUTOLOAD=0,
# this import becomes safe even when torch_npu cannot be auto-loaded.
run_python(TORCH_VERSION
  "import torch; print(torch.__version__)" "Failed to locate torch path")

if(NOT ${TORCH_VERSION} VERSION_EQUAL "2.8.0")
  message(FATAL_ERROR "Expected PyTorch version 2.8.0, but found ${TORCH_VERSION}")
endif()

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")

# ------------------------------------------------------------------------------
# SOC / build mode policy
# ------------------------------------------------------------------------------
if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
  message(FATAL_ERROR "SOC_VERSION is not set. Please pass -DSOC_VERSION=xxx from setup.py.")
endif()

message(STATUS "Detected SOC version: ${SOC_VERSION}")

string(TOUPPER "${SOC_VERSION}" SOC_VERSION_UPPER)

# Minimal build SOC list: use same behavior bucket (skip ascendc kernels, skip some host tiling, etc.)
set(VLLM_SOC_MINIMAL_LIST
  "ASCEND310P3"
  "ASCEND910_95"
  "ASCEND910_9589"
)

set(VLLM_MINIMAL_BUILD OFF)
if(SOC_VERSION_UPPER IN_LIST VLLM_SOC_MINIMAL_LIST)
  set(VLLM_MINIMAL_BUILD ON)
endif()

message(STATUS "VLLM_MINIMAL_BUILD=${VLLM_MINIMAL_BUILD} (SOC_VERSION_UPPER=${SOC_VERSION_UPPER})")

# CMake build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)" FORCE)
endif()

# Install prefix default override
if(CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE PATH "path to install()" FORCE)
endif()

# ------------------------------------------------------------------------------
# AscendC kernel cmake (only needed for non-minimal build)
# ------------------------------------------------------------------------------
set(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS ON)
if(VLLM_MINIMAL_BUILD)
  set(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS OFF)
endif()

if(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS)
  set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

  if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
  elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
  else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
  endif()

  include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

  file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/*.cpp)

  set(VLLM_ASCEND_CUSTOM_OP
    ${KERNEL_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
  )

  ascendc_library(vllm_ascend_kernels SHARED
    ${VLLM_ASCEND_CUSTOM_OP}
  )
endif()

message(STATUS "TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------
file(GLOB VLLM_ASCEND_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp
)

# Only non-minimal build needs this tiling source
if(NOT VLLM_MINIMAL_BUILD)
  list(APPEND VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_host/tiling/tiling_data.cpp
  )
endif()

# ------------------------------------------------------------------------------
# Python extension module target
# ------------------------------------------------------------------------------
pybind11_add_module(vllm_ascend_C ${VLLM_ASCEND_SRC})

target_include_directories(vllm_ascend_C PRIVATE
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/platform
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_host
)

target_link_directories(vllm_ascend_C PRIVATE
  ${TORCH_LIBRARY_DIRS}
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_HOME_PATH}/lib64
)

# Common libs
target_link_libraries(vllm_ascend_C PUBLIC
  ${TORCH_LIBRARIES}
  torch_npu
  ascendcl
  tiling_api
  register
  platform
  ascendalog
  dl
  opapi
)

# Kernels only on full build
if(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS)
  target_link_libraries(vllm_ascend_C PUBLIC vllm_ascend_kernels)
endif()

# rpath
target_link_options(vllm_ascend_C PRIVATE
  "-Wl,-rpath,$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib"
)

# ------------------------------------------------------------------------------
# Install
# ------------------------------------------------------------------------------
if(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS)
  install(TARGETS vllm_ascend_C vllm_ascend_kernels DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
else()
  install(TARGETS vllm_ascend_C DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
endif()
