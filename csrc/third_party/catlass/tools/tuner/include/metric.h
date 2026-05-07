/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_METRIC_H
#define CATLASS_TUNER_METRIC_H

#include <map>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <iomanip>
#include "catlass/library/operation.h"

namespace Catlass {

enum class ClassicMetric : uint32_t {
    CASE_ID = 0,
    TASK_DURATION,
    DEVICE_ID,
    OPERATION,
    DESCRIPTION,
    L0,             // L0_TILE_SHAPE
    L1,             // L1_TILE_SHAPE
    SWIZZLE,
    M,
    N,
    K,
    A,
    B,
    C,
    END
};

// When modify strings below, sync Metrics::HEAD
struct ClassicMetricStr {
    static constexpr std::string_view CASE_ID = "case_id";
    static constexpr std::string_view TASK_DURATION = "task_duration(us)";
    static constexpr std::string_view DEVICE_ID = "device_id";
    static constexpr std::string_view OPERATION = "operation";
    static constexpr std::string_view DESCRIPTION = "description";
    static constexpr std::string_view L0 = "l0_tile_shape";
    static constexpr std::string_view L1 = "l1_tile_shape";
    static constexpr std::string_view SWIZZLE = "swizzle";
    static constexpr std::string_view M = "m";
    static constexpr std::string_view N = "n";
    static constexpr std::string_view K = "k";
    static constexpr std::string_view A = "A";
    static constexpr std::string_view B = "B";
    static constexpr std::string_view C = "C";
};

class Metric {
public:
    explicit Metric();
    [[nodiscard]] std::string ToString() const;
    [[nodiscard]] std::string ToTerminalString() const;
    void SaveOperator(Library::Operation *op);
    void SetField(const std::string &key, const std::string &value);

    template<ClassicMetric key, typename T>
    void SetField(const T &value)
    {
        if constexpr (IsString<T>::value) {
            classic_[static_cast<uint32_t>(key)] = value;
            if constexpr (key == ClassicMetric::DESCRIPTION) {
                std::string_view sv = classic_[static_cast<uint32_t>(key)];
                auto r = sv.rfind('_');
                if (r == std::string::npos || r == 0) {
                    return;
                }
                classic_[static_cast<uint32_t>(ClassicMetric::SWIZZLE)] = sv.substr(r + 1);
                auto l = sv.rfind('_', r - 1);
                if (l == std::string::npos || l == 0) {
                    return;
                }
                classic_[static_cast<uint32_t>(ClassicMetric::L0)] = sv.substr(l + 1, r - l - 1);
                r = l - 1;
                l = sv.rfind('_', r);
                if (l == std::string::npos) {
                    return;
                }
                classic_[static_cast<uint32_t>(ClassicMetric::L1)] = sv.substr(l + 1, r - l);
            }
        } else {
            if constexpr (key == ClassicMetric::TASK_DURATION) {
                taskDuration_ = value;
                std::stringstream ss;
                constexpr size_t PREC = 3;
                ss << std::fixed << std::setprecision(PREC) << value;
                classic_[static_cast<uint32_t>(key)] = ss.str();
            } else {
                classic_[static_cast<uint32_t>(key)] = std::to_string(value);
            }
        }
    }

    inline const std::string& Field(ClassicMetric key) const
    {
        return classic_[static_cast<uint32_t>(key)];
    }

    inline const std::string& Field(const std::string &key) const
    {
        // Don't change the default return value unless reviewed all references to the function
        static std::string empty{};
        if (auto it = CLASSIC_STR_TO_E.find(key); it != CLASSIC_STR_TO_E.end()) {
            return Field(it->second);
        } else if (auto itf = fields_.find(key); itf != fields_.end()) {
            return itf->second;
        }
        return empty;
    }

    inline const auto& Fields() const { return fields_; }

    inline double GetTaskDuration() const { return taskDuration_; }

    const static std::unordered_map<std::string, ClassicMetric> CLASSIC_STR_TO_E;

private:
    template <typename T>
    struct IsString : std::false_type {};

    double taskDuration_{0};
    std::map<std::string, std::string> fields_;
    std::vector<std::string> classic_ = std::vector<std::string>(static_cast<uint32_t>(ClassicMetric::END));
};

template <>
struct Metric::IsString<std::string> : std::true_type {};
template <>
struct Metric::IsString<std::string_view> : std::true_type {};
template <typename T>
struct Metric::IsString<T*> : std::is_same<std::remove_const_t<T>, char> {};
template <typename T>
struct Metric::IsString<T* const> : std::is_same<std::remove_const_t<T>, char> {};
template <size_t N>
struct Metric::IsString<char[N]> : std::true_type {};
template <size_t N>
struct Metric::IsString<const char[N]> : std::true_type {};

} // namespace Catlass
#endif // CATLASS_TUNER_METRIC_H
