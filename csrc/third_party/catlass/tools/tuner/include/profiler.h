/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_PROFILER_H
#define CATLASS_TUNER_PROFILER_H

#include <thread>
#include <utility>
#include <vector>
#include <cstdint>
#include <functional>
#include <condition_variable>
#include <queue>
#include "m_t_var.h"

namespace Catlass {

class Profiler {
public:
    using CallBackFunc = std::function<void(std::vector<char>&&)>;
    explicit Profiler(int32_t deviceId = 0) : deviceId_(deviceId) {}
    ~Profiler();

    inline void SetDeviceId(int32_t deviceId) { deviceId_ = deviceId; }
    inline void RegisterCallback(CallBackFunc callBackFunc) { callBack_ = std::move(callBackFunc); }

    bool Start();
    void Stop();
    void GetDurations(const std::vector<char> &data, std::vector<uint64_t> &starts, std::vector<uint64_t> &ends);

private:
    void CreateReadThread();

    std::thread readThread_{};
    std::vector<char> data_{};
    CallBackFunc callBack_{};
    int32_t deviceId_;
    MTVar<bool> running_{false};
};

class ProfileDataHandler {
public:
    ~ProfileDataHandler()
    {
        Synchronize();
    }

    bool Init();
    void Synchronize();
    std::pair<int64_t, int32_t> GetAicoreFreq();
    bool SetDeviceId(int32_t deviceId);

    inline std::vector<double> GetDurations()
    {
        std::vector<double> durations{};
        durations_.DoTransaction<void>([&](auto &val) {
            durations.swap(val);
        });
        return durations;
    }

private:
    void ProfileDataThread();
    int64_t GetAicpuFreq();

    Profiler profiler_{};
    std::condition_variable profileDataCV_;
    std::mutex mtx_;
    std::queue<std::vector<char>> profileDataQueue_;
    std::thread profileDataThread_;
    MTVar<std::vector<double>> durations_{};
    MTVar<bool> finish_{false};
    int64_t freq_;
    int32_t deviceId_;
};

} // namespace Catlass
#endif // CATLASS_TUNER_PROFILER_H
