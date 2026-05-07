# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include(CMakeCommonLanguageInclude)
include(Compiler/CMakeCommonCompilerMacros)
include(Compiler/Clang)
__compiler_clang(ASCEND)

# Set explicitly, because __compiler_clang() doesn't set this if we're simulating MSVC.
set(CMAKE_DEPFILE_FLAGS_ASCEND "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")

if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
    AND CMAKE_GENERATOR MATCHES "Makefiles|WMake")
    # dependencies are computed by the compiler itself
    set(CMAKE_ASCEND_DEPFILE_FORMAT gcc)
    set(CMAKE_ASCEND_DEPENDS_USE_COMPILER TRUE)
endif()

set(CMAKE_ASCEND_STANDARD ${CMAKE_CXX_STANDARD})

set(CMAKE_ASCEND03_STANDARD_COMPILE_OPTION "-std=c++03")
set(CMAKE_ASCEND03_EXTENSION_COMPILE_OPTION "-std=gnu++03")
# __compiler_clang_cxx_standards(ASCEND)

set(CMAKE_INCLUDE_FLAG_ASCEND "-I")
set(_CMAKE_COMPILE_AS_ASCEND_FLAG "-x cce")

if(UNIX)
    set(CMAKE_ASCEND_OUTPUT_EXTENSION .o)
else()
    set(CMAKE_ASCEND_OUTPUT_EXTENSION .obj)
endif()

# Set implicit links early so compiler-specific modules can use them.
set(__IMPLICIT_LINKS)

foreach(dir ${CMAKE_ASCEND_HOST_IMPLICIT_LINK_DIRECTORIES})
    string(APPEND __IMPLICIT_LINKS " -L\"${dir}\"")
endforeach()

foreach(lib ${CMAKE_ASCEND_HOST_IMPLICIT_LINK_LIBRARIES})
    if(${lib} MATCHES "/")
        string(APPEND __IMPLICIT_LINKS " \"${lib}\"")
    else()
        string(APPEND __IMPLICIT_LINKS " -l${lib}")
    endif()
endforeach()

foreach(dir ${CMAKE_ASCEND_HOST_IMPLICIT_INCLUDE_DIRECTORIES})
    string(APPEND __IMPLICIT_INCLUDES " ${CMAKE_INCLUDE_FLAG_ASCEND}\"${dir}\"")
endforeach()

set(CMAKE_ASCEND_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")

if(DEFINED ASCEND_ENABLE_MSDEBUG AND ASCEND_ENABLE_MSDEBUG)
    set(CMAKE_ASCEND_FLAGS_DEBUG "-O0 -g")
else()
    set(CMAKE_ASCEND_FLAGS_DEBUG "-Xhost-start -O0 -g -Xhost-end -Xaicore-start -O2 -Xaicore-end")
endif()

set(CMAKE_ASCEND_FLAGS_RELEASE "-O2")

set(_INCLUDED_FILE 0)

set(CMAKE_SHARED_LIBRARY_ASCEND_FLAGS -fPIC)
set(CMAKE_SHARED_LIBRARY_CREATE_ASCEND_FLAGS --shared)
set(CMAKE_STATIC_LIBRARY_CREATE_ASCEND_FLAGS --cce-build-static-lib)
set(_CMAKE_LIBRARY_CREATE_ASCEND_FLAGS --cce-fatobj-link)
set(CMAKE_SHARED_LIBRARY_SONAME_ASCEND_FLAG "-Wl,-soname,")

set(CMAKE_ASCEND_CREATE_SHARED_LIBRARY
    "<CMAKE_ASCEND_COMPILER> ${_CMAKE_LIBRARY_CREATE_ASCEND_FLAGS} <CMAKE_SHARED_LIBRARY_ASCEND_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_ASCEND_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>${__IMPLICIT_LINKS}")

set(CMAKE_ASCEND_CREATE_STATIC_LIBRARY
    "<CMAKE_ASCEND_COMPILER> ${_CMAKE_LIBRARY_CREATE_ASCEND_FLAGS} <CMAKE_SHARED_LIBRARY_ASCEND_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> ${CMAKE_STATIC_LIBRARY_CREATE_ASCEND_FLAGS} <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>${__IMPLICIT_LINKS}")

set(CMAKE_ASCEND_CREATE_SHARED_MODULE ${CMAKE_ASCEND_CREATE_SHARED_LIBRARY})

set(CMAKE_ASCEND_COMPILE_OBJECT
    "<CMAKE_ASCEND_COMPILER> <DEFINES> <INCLUDES>${__IMPLICIT_INCLUDES} <FLAGS> ${_CMAKE_COMPILE_AS_ASCEND_FLAG} ${CMAKE_ASCEND_COMPILE_OPTIONS} ${CMAKE_ASCEND_COMMON_COMPILE_OPTIONS} -pthread -o <OBJECT> -c <SOURCE>")

set(CMAKE_ASCEND_LINK_EXECUTABLE
    "<CMAKE_ASCEND_COMPILER> ${_CMAKE_LIBRARY_CREATE_ASCEND_FLAGS} <FLAGS> <CMAKE_ASCEND_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>${__IMPLICIT_LINKS}")

set(CMAKE_ASCEND_INFORMATION_LOADED 1)
