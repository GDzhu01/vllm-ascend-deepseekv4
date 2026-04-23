# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/m00663269/vllm-ascend-deepseekv4/csrc/third_party/eigen")
  file(MAKE_DIRECTORY "/home/m00663269/vllm-ascend-deepseekv4/csrc/third_party/eigen")
endif()
file(MAKE_DIRECTORY
  "/home/m00663269/vllm-ascend-deepseekv4/csrc/third_party/third_party/eigen/src/external_eigen_transformer-build"
  "/home/m00663269/vllm-ascend-deepseekv4/csrc/third_party/third_party/eigen"
  "/home/m00663269/vllm-ascend-deepseekv4/csrc/third_party/third_party/eigen/tmp"
  "/home/m00663269/vllm-ascend-deepseekv4/csrc/third_party/third_party/eigen/src/external_eigen_transformer-stamp"
  "/home/m00663269/vllm-ascend-deepseekv4/csrc/third_party/download/eigen"
  "/home/m00663269/vllm-ascend-deepseekv4/csrc/third_party/third_party/eigen/src/external_eigen_transformer-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/m00663269/vllm-ascend-deepseekv4/csrc/third_party/third_party/eigen/src/external_eigen_transformer-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/m00663269/vllm-ascend-deepseekv4/csrc/third_party/third_party/eigen/src/external_eigen_transformer-stamp${cfgdir}") # cfgdir has leading slash
endif()
