/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_METRICS_H
#define CATLASS_TUNER_METRICS_H

#include <set>
#include "metric.h"
#include "op_config.h"

namespace Catlass {

class Metrics {
public:
    inline void SetDeviceId(int32_t device) { deviceId_ = device; }

    bool SetOutputPath(std::string_view output);
    void Dump();
    void Add(const std::shared_ptr<OpConfig>& opConfig, Library::Operation *op);
    void SetDurationAndPrint(double duration);

private:
    static constexpr std::string_view HEAD = "case_id,task_duration(us),device_id,operation,description,"
                                             "m,n,k,A,B,C";
    static constexpr std::string_view DIVIDE = "================================\n";

    void PrintTop10(const std::string &head);
    std::string GetHead();

    std::string outputPath_;
    std::vector<Metric> metrics_;
    std::set<std::string> extraHeads_;
    size_t durationIdx_{0};
    int32_t deviceId_{0};
};

} // namespace Catlass
#endif // CATLASS_TUNER_METRICS_H
