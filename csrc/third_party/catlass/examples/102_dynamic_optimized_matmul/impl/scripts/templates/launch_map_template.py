#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os

class LaunchMapTemplate:
    TEMPLATE = """
#ifndef LAUNCH_MAP_H
#define LAUNCH_MAP_H

#include <unordered_map>
#include <string>

#include "acl/acl.h"
#include "tiling_params.h"

#define DECLARE_KERNEL_FUNC(kernelName) \\
    void Launch##kernelName(aclrtStream&, uint64_t, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, TilingParams&); \\
    size_t kernelName##GetWorkspaceSize(TilingParams&);

{declare_list}

std::unordered_map<uint64_t, void(*)(aclrtStream&, uint64_t, 
    uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, TilingParams&)> launchKernelFuncMap = {{ 
{launch_func_list}
}};

using GetWorkspaceFunc = size_t(*)(TilingParams& tilingParams);
std::unordered_map<uint64_t, GetWorkspaceFunc> getWorkspaceFuncMap = {{
{workspace_func_list}
}};

// only for print kernel Info
std::unordered_map<uint64_t, std::string> funcNameMap = {{
{func_name_list}
}};

#endif // LAUNCH_MAP_H
"""

    @staticmethod
    def gen_code(kernel_info):
        declare_list = "\n".join(
            "DECLARE_KERNEL_FUNC({})".format(value) for value in kernel_info.values()
        )
        launch_func_list = ",\n".join(
            "{{ {}, Launch{} }}".format(key, value)
            for key, value in kernel_info.items()
        )
        workspace_func_list = ",\n".join(
            "{{ {}, {}GetWorkspaceSize }}".format(key, value)
            for key, value in kernel_info.items()
        )
        func_name_list = ",\n".join(
            '{{ {}, "{}" }}'.format(key, value) for key, value in kernel_info.items()
        )
        content = LaunchMapTemplate.TEMPLATE.format(
            declare_list=declare_list,
            launch_func_list=launch_func_list,
            workspace_func_list=workspace_func_list,
            func_name_list=func_name_list,
        )

        launch_file = os.path.join("../../include", "launch_map.h")
        try: os.remove(launch_file)
        except FileNotFoundError: pass

        fd = os.open(launch_file, os.O_CREAT | os.O_WRONLY \
                                | os.O_TRUNC, 0o550) # r-xr-x---
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
                fd = None
        finally:
            if fd is not None:
                os.close(fd)
