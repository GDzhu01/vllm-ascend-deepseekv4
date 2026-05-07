# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(DEFINED CATLASS_LIBRARY_KERNELS)
    set(CCE_AICORE_ARCH_BASE dav-c220)

    if(CATLASS_ARCH STREQUAL "2201")
        set(CCE_AICORE_ARCH_BASE dav-c220)
        add_compile_definitions(CATLASS_ARCH=2201)
    else()
        message(FATAL_ERROR ${CCE_AICORE_ARCH_BASE} "CCE_AICORE_ARCH_BASE " ${CCE_AICORE_ARCH_BASE} " is unsupported")
    endif()
    add_subdirectory(library)
    add_subdirectory(tuner)
endif()
