/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_COMMAND_LINE_PARSER_H
#define CATLASS_TUNER_COMMAND_LINE_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include <algorithm>
#include <cstdint>

namespace Catlass {

#define GET_CHECK(stat, key)                                                        \
    do {                                                                            \
        CommandLineParser::ERROR_CODE err = stat;                                   \
        if (err != CommandLineParser::ERROR_CODE::NONE) {                           \
            LOGE("%s:%d get key --" key " failed, err: %s", __FILE__, __LINE__,     \
                CommandLineParser::GetErrorStr(err).data());                        \
        }                                                                           \
    } while (false)

class CommandLineParser {
public:
    enum class ERROR_CODE : uint32_t {
        NONE = 0,
        CAST_FAILED,
        KEY_NOT_EXIST,
        INTEGER_OVERFLOW,
        EXPECT_UNSIGNED_INTEGER,
        NOT_DIGITAL_FORMAT,
        END,
    };

    static std::string_view GetErrorStr(ERROR_CODE er)
    {
        static const std::string_view strs[] = {
            "none",
            "cast value from string failed",
            "key not exist",
            "integer overflow",
            "expect unsigned integer",
            "input is not digital format",
            "unknown error"
        };
        auto idx = std::min(static_cast<int>(er), static_cast<int>(ERROR_CODE::END));
        return strs[idx];
    }

    [[nodiscard]] inline bool HasKey(const std::string& key) const
    {
        return dataMap_.find(key) != dataMap_.end();
    }

    [[nodiscard]] inline bool Help() const { return help_; }

    void Parse(int argc, const char* argv[]);
    void PrintHelp() const;
    void PrintUnusedKeys() const;
    [[nodiscard]] std::vector<std::string> Keys() const;
    template <typename T>
    ERROR_CODE Get(const std::string& key, T &target);

private:
    // 去除字符串两端空格
    static std::string Trim(const std::string& str)
    {
        auto start = std::find_if_not(str.begin(), str.end(), [](unsigned char c) {
            return std::isspace(c);
        });
        auto end = std::find_if_not(str.rbegin(), str.rend(), [](unsigned char c) {
            return std::isspace(c);
        }).base();
        return (start < end) ? std::string(start, end) : "";
    }

    static bool IsDigitFormat(const std::string& str);

    auto FindKey(const std::string& key)
    {
        auto it = dataMap_.find(key);
        if (it != dataMap_.end()) {
            usedSet_.insert(key);
        }
        return it;
    }

    std::map<std::string, std::string> dataMap_;
    std::unordered_set<std::string> usedSet_;
    bool help_{false};
};

template<>
CommandLineParser::ERROR_CODE CommandLineParser::Get<std::string>(const std::string& key, std::string &target);
template<>
CommandLineParser::ERROR_CODE CommandLineParser::Get<std::string_view>(const std::string& key,
                                                                       std::string_view &target);
template<>
CommandLineParser::ERROR_CODE CommandLineParser::Get<int64_t>(const std::string& key, int64_t &target);
template<>
CommandLineParser::ERROR_CODE CommandLineParser::Get<uint64_t>(const std::string& key, uint64_t &target);
template<>
CommandLineParser::ERROR_CODE CommandLineParser::Get<int32_t>(const std::string& key, int32_t &target);
template<>
CommandLineParser::ERROR_CODE CommandLineParser::Get<uint32_t>(const std::string& key, uint32_t &target);
template<>
CommandLineParser::ERROR_CODE CommandLineParser::Get<double>(const std::string& key, double &target);
template<>
CommandLineParser::ERROR_CODE CommandLineParser::Get<float>(const std::string& key, float &target);
template<>
CommandLineParser::ERROR_CODE CommandLineParser::Get<bool>(const std::string& key, bool &target);

} // namespace Catlass
#endif // CATLASS_TUNER_COMMAND_LINE_PARSER_H
