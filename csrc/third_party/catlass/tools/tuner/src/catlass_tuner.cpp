/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "catlass_tuner.h"
#include "m_t_var.h"

#include "tiling/platform/platform_ascendc.h"

namespace Catlass {

static constexpr int RUN_TIMES = 5;

CatlassTuner::CatlassTuner(CommandLineParser parser) : parser_(std::move(parser))
{
    if (parser_.HasKey("device")) {
        deviceId_ = -1;
        GET_CHECK(parser_.Get<decltype(deviceId_)>("device", deviceId_), "device");
        if (deviceId_ == -1) {
            return;
        }
        metrics_.SetDeviceId(deviceId_);
    }
    // 调优通道需要做device id映射
    if (!profileHandler_.SetDeviceId(deviceId_)) {
        return;
    }

    if (parser_.HasKey("output")) {
        std::string_view output;
        GET_CHECK(parser_.Get<std::string_view>("output", output), "output");
        if (output.empty() || !metrics_.SetOutputPath(output)) {
            return;
        }
    }
    if (!profileHandler_.Init()) {
        LOGE("Start profile channel failed, will not run operators");
        return;
    } else if (stream_ = DeviceMemoryManager::Instance().Initialize(deviceId_); !stream_) {
        LOGE("Initialize device failed, will not run kernels");
    } else if (!DeviceMemoryManager::Instance().InitCacheClear()) {
        LOGW("Init resource for clear l2cache failed, won't clear l2cache before each kernel");
    }
}

CatlassTuner::~CatlassTuner()
{
    DeviceMemoryManager::Instance().Finalize();
}

bool CatlassTuner::InitOperators(OpConfigPool &pool)
{
    std::string_view kernel;
    if (parser_.HasKey("kernels")) {
        GET_CHECK(parser_.Get<decltype(kernel)>("kernels", kernel), "kernels");
        for (auto c : kernel) {
            auto uc = static_cast<unsigned char>(c);
            if (!std::isdigit(uc) && !std::isalpha(uc) && c != '_') {
                LOGE("--kernels can only contain [0-9, a-z, A-Z, _]");
                return false;
            }
        }
    }

    for (auto op : manifest_.GetOperations()) {
        if (!pool.Register(op, parser_, kernel)) {
            return false;
        }
    }
    size_t sum = 0;
    for (auto &p : pool.GetPool()) {
        if (!p.first->Invalid()) {
            sum += p.second.size();
        }
    }
    LOGI("Initializing %lu operations", sum);
    return true;
}

void CatlassTuner::Run()
{
    if (!stream_) {
        return;
    }

    OpConfigPool pool;
    if (manifest_.Initialize() != Status::kSuccess) {
        LOGE("Initialize operator manifest failed");
        return;
    } else if (!InitOperators(pool)) {
        return;
    }

    parser_.PrintUnusedKeys();
    // Get the number of cube cores of the current hardware
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    for (auto &p : pool.GetPool()) {
        auto &opConfig = p.first;
        if (!opConfig || opConfig->Invalid()) {
            continue;
        }
        for (auto op : p.second) {
            metrics_.Add(opConfig, op);
            auto stat = RunOp(opConfig, op, aicCoreNum);
            UpdateMetrics();
            if (stat != OpRunStatus::FATAL) {
                continue;
            }
            LOGE("Running kernel %s failed, try restart profiling", op->GetDescription().name);
            Synchronize();
            if (!profileHandler_.Init()) {
                LOGE("Restart profiling failed, end subsequent operator execution.");
                break;
            }
        }
    }
    Synchronize();
    metrics_.Dump();
}

OpRunStatus CatlassTuner::RunOp(const std::shared_ptr<OpConfig>& opConfig, Library::Operation *op, uint32_t aicCoreNum)
{
    std::vector<KernelType> kernels;
    std::shared_ptr<void> defer(nullptr, [&](void*) {
        // all kernel type ran by current operator
        kernelsQueue_.emplace(kernels);
    });
    OpLauncher launcher(opConfig, op, aicCoreNum);
    if (launcher.Init() != OpRunStatus::SUCCESS) {
        LOGE("Initialize operator %s failed", op->GetDescription().name);
        return OpRunStatus::FAILED;
    }
    auto freq = profileHandler_.GetAicoreFreq();
    if (freq.first > freq.second) {
        LOGW("Current freq %d is lower than rated freq %ld, run warm up", freq.second, freq.first);
        constexpr int TIMEOUT = 10;
        for (int i = 0; i < TIMEOUT && freq.first > freq.second; ++i) {
            constexpr size_t WARM_UP_TIMES = 10;
            auto stat = launcher(stream_, WARM_UP_TIMES, false);
            std::vector<KernelType> tmp(WARM_UP_TIMES, KernelType::OPERATOR);
            kernels.insert(kernels.end(), tmp.begin(), tmp.end());
            auto err = aclrtSynchronizeStream(stream_);
            if (stat != OpRunStatus::SUCCESS || err != ACL_SUCCESS) {
                LOGE("Warm up failed, aclrtSynchronizeStream ret: %d", err);
                return OpRunStatus::FATAL;
            }
            freq = profileHandler_.GetAicoreFreq();
        }
        LOGI("Warm up finished, rated freq %ld, current freq %d", freq.first, freq.second);
    }
    OpRunStatus stat = OpRunStatus::SUCCESS;
    for (int i = 0; i < RUN_TIMES; ++i) {
        if (DeviceMemoryManager::Instance().ClearL2Cache(aicCoreNum)) {
            kernels.emplace_back(KernelType::CACHE_CLEAR);
        }
        stat = launcher(stream_);
        kernels.emplace_back(KernelType::OPERATOR);
        if (stat != OpRunStatus::SUCCESS) {
            break;
        }
    }
    return stat;
}

void CatlassTuner::UpdateMetrics(bool readAll)
{
    auto tmp = profileHandler_.GetDurations();
    durations_.insert(durations_.end(), tmp.begin(), tmp.end());
    if (durations_.empty()) {
        return;
    }
    size_t i = 0;
    auto setDuration = [&](const std::vector<KernelType> &kernel) {
        double time = 0;
        size_t end = std::min(kernel.size(), std::max(durations_.size(), i) - i);
        for (size_t j = 0; j < end; ++j) {
            if (kernel[j] == KernelType::OPERATOR) {
                // use task duration from the last execution,
                // but the collected durations also includes warm up operators and ClearL2Cache
                time = durations_[i + j];
            }
        }
        metrics_.SetDurationAndPrint(time);
        i += end;
    };

    while (!kernelsQueue_.empty() && durations_.size() >= kernelsQueue_.front().size() + i) {
        auto kernel = std::move(kernelsQueue_.front());
        kernelsQueue_.pop();
        setDuration(kernel);
    }
    Erase(durations_, i);
    i = 0;
    if (!readAll) {
        return;
    }

    while (!kernelsQueue_.empty()) {
        auto kernel = std::move(kernelsQueue_.front());
        kernelsQueue_.pop();
        if (durations_.size() < kernel.size() + i + 1) {
            LOGW("This operator's kernel run times are more than profile data collected");
        }
        setDuration(kernel);
    }
    durations_.clear();
}

void CatlassTuner::Synchronize()
{
    profileHandler_.Synchronize();
    UpdateMetrics(true);
}

} // namespace Catlass
