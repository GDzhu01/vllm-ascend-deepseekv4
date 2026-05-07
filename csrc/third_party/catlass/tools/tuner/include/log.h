/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_LOG_H
#define CATLASS_TUNER_LOG_H

#include <string>
#include <unordered_map>
#include <cstdio>

#define LOG(__level, __msg, ...) printf(__level __msg "\n", ##__VA_ARGS__)
#define LOGI(__msg, ...) LOG("[INFO ] ", __msg, ##__VA_ARGS__)
#define LOGW(__msg, ...) LOG("[WARN ] ", __msg, ##__VA_ARGS__)
#define LOGE(__msg, ...) LOG("[ERROR] ", __msg, ##__VA_ARGS__)
#define LOGM(__msg, ...) LOG("", __msg, ##__VA_ARGS__)

inline const std::unordered_map<char, std::string>& GetInvalidChars()
{
    static const std::unordered_map<char, std::string> INVALID_CHAR = {
        {'\n', "\\n"}, {'\f', "\\f"}, {'\r', "\\r"}, {'\b', "\\b"},
        {'\t', "\\t"}, {'\v', "\\v"}, {'\u007F', "\\u007F"}
    };
    return INVALID_CHAR;
}

inline std::string ReplaceInvalidChars(const std::string &str)
{
    auto &invalidChars = GetInvalidChars();
    std::string replaced;
    for (auto c : str) {
        auto it = invalidChars.find(c);
        replaced += it != invalidChars.cend() ? it->second : std::string{c};
    }
    return replaced;
}

#endif // CATLASS_TUNER_LOG_H
