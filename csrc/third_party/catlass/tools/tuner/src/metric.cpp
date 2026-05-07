/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "metric.h"
#include <sstream>
#include <iomanip>
#include "log.h"
#include "library_helper.h"

namespace Catlass {

using namespace Library;

namespace {

std::string GetTensorDescription(const TensorDescription &td)
{
    std::string dtype{LibraryHelper::GetDataTypeStr(td.element)};
    dtype.append(":");
    dtype.append(LibraryHelper::GetLayoutStr(td.layout));
    return dtype;
}
}

const std::unordered_map<std::string, ClassicMetric> Metric::CLASSIC_STR_TO_E = {
    {"case_id", ClassicMetric::CASE_ID},
    {"task_duration(us)", ClassicMetric::TASK_DURATION},
    {"device_id", ClassicMetric::DEVICE_ID},
    {"operation", ClassicMetric::OPERATION},
    {"description", ClassicMetric::DESCRIPTION},
    {"l0 tile shape", ClassicMetric::L0},
    {"l1 tile shape", ClassicMetric::L1},
    {"swizzle", ClassicMetric::SWIZZLE},
    {"m", ClassicMetric::M},
    {"n", ClassicMetric::N},
    {"k", ClassicMetric::K},
    {"A", ClassicMetric::A},
    {"B", ClassicMetric::B},
    {"C", ClassicMetric::C},
};

void Metric::SaveOperator(Library::Operation *op)
{
    auto &desp = op->GetDescription();
    SetField<ClassicMetric::DESCRIPTION>(desp.name);
    if (desp.kind == Library::OperationKind::Gemm) {
        SetField<ClassicMetric::OPERATION>("Gemm");
        auto &mdesp = static_cast<const Library::GemmOperationDescription &>(desp);
        SetField<ClassicMetric::A>(GetTensorDescription(mdesp.A));
        SetField<ClassicMetric::B>(GetTensorDescription(mdesp.B));
        SetField<ClassicMetric::C>(GetTensorDescription(mdesp.C));
    }
}

std::string Metric::ToString() const
{
    std::stringstream ss;
    ss << Field(ClassicMetric::CASE_ID) << "," << Field(ClassicMetric::TASK_DURATION) << ","
       << Field(ClassicMetric::DEVICE_ID) << "," << Field(ClassicMetric::OPERATION) << ","
       << Field(ClassicMetric::DESCRIPTION) << "," << Field(ClassicMetric::M) << ","
       << Field(ClassicMetric::N) << "," << Field(ClassicMetric::K) << "," << Field(ClassicMetric::A) << ","
       << Field(ClassicMetric::B) << "," << Field(ClassicMetric::C);
    for (const auto &p : fields_) {
        ss << "," << p.second;
    }
    return ss.str();
}

std::string Metric::ToTerminalString() const
{
    constexpr size_t LEFT_ALIGN = 20;
    std::stringstream ss;
    auto format = [&](const std::string_view &sv, ClassicMetric metric) {
        ss << std::setw(LEFT_ALIGN) << sv << " : " << Field(metric) << std::endl;
    };
    ss << std::right;
    format(ClassicMetricStr::CASE_ID, ClassicMetric::CASE_ID);
    format(ClassicMetricStr::TASK_DURATION, ClassicMetric::TASK_DURATION);
    format(ClassicMetricStr::DEVICE_ID, ClassicMetric::DEVICE_ID);
    format(ClassicMetricStr::OPERATION, ClassicMetric::OPERATION);
    format(ClassicMetricStr::DESCRIPTION, ClassicMetric::DESCRIPTION);
    format(ClassicMetricStr::L0, ClassicMetric::L0);
    format(ClassicMetricStr::L1, ClassicMetric::L1);
    format(ClassicMetricStr::SWIZZLE, ClassicMetric::SWIZZLE);
    format(ClassicMetricStr::M, ClassicMetric::M);
    format(ClassicMetricStr::N, ClassicMetric::N);
    format(ClassicMetricStr::K, ClassicMetric::K);
    format(ClassicMetricStr::A, ClassicMetric::A);
    format(ClassicMetricStr::B, ClassicMetric::B);
    format(ClassicMetricStr::C, ClassicMetric::C);
    for (const auto &p : fields_) {
        ss << std::setw(LEFT_ALIGN) << p.first << " : " << p.second << std::endl;
    }
    return ss.str();
}

Metric::Metric()
{
    SetField<ClassicMetric::CASE_ID>(0);
    SetField<ClassicMetric::TASK_DURATION>(0);
    SetField<ClassicMetric::DEVICE_ID>(0);
    SetField<ClassicMetric::OPERATION>("");
    SetField<ClassicMetric::DESCRIPTION>("");
    SetField<ClassicMetric::L0>("");
    SetField<ClassicMetric::L1>("");
    SetField<ClassicMetric::SWIZZLE>("");
    SetField<ClassicMetric::M>(0);
    SetField<ClassicMetric::N>(0);
    SetField<ClassicMetric::K>(0);
    SetField<ClassicMetric::A>("");
    SetField<ClassicMetric::B>("");
    SetField<ClassicMetric::C>("");
}

void Metric::SetField(const std::string &key, const std::string &value)
{
    if (auto it = CLASSIC_STR_TO_E.find(key); it != CLASSIC_STR_TO_E.end()) {
        // This branch should not be entered during runtime
        LOGE("Step into Unexpected branch, please call SetField<ClassicMetrics>");
        return;
    }
    fields_[key] = value;
}

} // namespace Catlass
