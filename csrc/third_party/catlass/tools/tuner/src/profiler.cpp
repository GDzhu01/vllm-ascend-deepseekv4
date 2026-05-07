/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "profiler.h"
#include <algorithm>
#include "log.h"
#include "device_memory_manager.h"

namespace {
constexpr uint32_t STARS_ENABLE_FLAG = 1;
constexpr uint32_t PROF_CFG_PERIOD = 0;
constexpr uint32_t PROF_REAL = 1;

enum class PROF_CHANNEL_TYPE {
    PROF_TS_TYPE,
    PROF_PERIPHERAL_TYPE,
    PROF_CHANNEL_TYPE_MAX,
};

using ProfStartParaT = struct prof_start_para {
    PROF_CHANNEL_TYPE channelType;
    uint32_t samplePeriod;
    uint32_t realTime;
    void *userData;
    uint32_t userDataSize;
};

// ts data code
using StarsSocLogConfigT = struct TagStarsSocLogConfig {
    uint32_t acsqTask;         // 1-enable,2-disable
    uint32_t accPmu;           // 1-enable,2-disable
    uint32_t cdqmReg;          // 1-enable,2-disable
    uint32_t dvppVpcBlock;     // 1-enable,2-disable
    uint32_t dvppJpegdBlock;   // 1-enable,2-disable
    uint32_t dvppJpedeBlock;   // 1-enable,2-disable
    uint32_t fftsThreadTask;   // 1-enable,2-disable
    uint32_t fftsBlock;        // 1-enable,2-disable
    uint32_t sdmaDmu;          // 1-enable,2-disable
};

using ProfPollInfoT = struct prof_poll_info {
    uint32_t deviceId;
    uint32_t channelId;
};

constexpr uint32_t CHANNEL_STARS_SOC_LOG_BUFFER = 50;   /* add for ascend910B */

enum class TimeType : uint16_t {
    START = 0,
    END,
    OTHERS
};

struct AcsqBean {
public:
    explicit AcsqBean(const std::vector<char> &bin)
    {
        constexpr uint16_t typeIndex = 0;
        constexpr uint16_t streamIdIndex = 2;
        constexpr uint16_t streamIdOffset = 11;
        constexpr uint16_t taskIdIndex = 3;
        constexpr uint16_t taskTypeOffset = 10;
        constexpr uint16_t funcTypeAndOperation = 63;
        constexpr uint16_t systemTimeIndex = 0;
        auto ptr = reinterpret_cast<const AcsqConstruct*>(&bin[0]);
        acsqData_.taskType = ptr->shortNums1[typeIndex] >> taskTypeOffset;
        acsqData_.funcType = ptr->shortNums1[typeIndex] & funcTypeAndOperation;
        acsqData_.systemTime = ptr->longlongNums[systemTimeIndex];
    
        uint16_t hardwareStreamId = ptr->shortNums1[streamIdIndex];
        uint16_t hardwareTaskId = ptr->shortNums1[taskIdIndex];
        if ((hardwareStreamId & 0x1000) != 0) {
            acsqData_.taskId = (hardwareTaskId & 0x1FFF) | (hardwareStreamId & 0xE000);
            acsqData_.streamId = hardwareStreamId % (1 << streamIdOffset);
        } else if ((hardwareStreamId & 0x2000) != 0) {
            acsqData_.taskId = (hardwareTaskId & 0xF000) | (hardwareStreamId & 0x0FFF);
            acsqData_.streamId = hardwareTaskId & 0x0FFF;
        } else {
            acsqData_.taskId = hardwareTaskId;
            acsqData_.streamId = hardwareStreamId % (1 << streamIdOffset);
        }
    }

    uint16_t GetTaskType() const
    {
        return acsqData_.taskType;
    }

    TimeType GetTimeType() const
    {
        if (acsqData_.funcType == 0) {
            return TimeType::START;
        } else if (acsqData_.funcType == 1) {
            return TimeType::END;
        }
        return TimeType::OTHERS;
    }

    uint16_t GetStreamId() const
    {
        return acsqData_.streamId;
    }

    uint16_t GetTaskId() const
    {
        return acsqData_.taskId;
    }

    uint64_t GetSystemTime() const
    {
        return acsqData_.systemTime;
    }

private:
    // format: 4short 1longlong 2short 11int, total 64B
    struct AcsqConstruct {
        uint16_t shortNums1[4];
        uint64_t longlongNums[1];
        uint16_t shortNums2[2];
        uint32_t intNums[11];
    };

    struct AcsqData {
        uint16_t taskType;
        uint16_t funcType;
        uint16_t taskId;
        uint16_t streamId;
        uint64_t systemTime;
    } acsqData_{};
};

bool GetStarsTask(ProfStartParaT &starsProfStartPara)
{
    uint32_t starsConfigSize = sizeof(StarsSocLogConfigT);
    auto *starsConfigPtr = static_cast<StarsSocLogConfigT *>(malloc(starsConfigSize));
    starsProfStartPara.userData = nullptr;
    if (starsConfigPtr == nullptr) {
        LOGE("Can not get user data pointer while getting stars task");
        return false;
    }
    std::fill_n(reinterpret_cast<uint8_t *>(starsConfigPtr), starsConfigSize, 0);

    starsConfigPtr->acsqTask = STARS_ENABLE_FLAG;
    starsConfigPtr->fftsThreadTask = STARS_ENABLE_FLAG;
    starsConfigPtr->accPmu = STARS_ENABLE_FLAG;

    starsProfStartPara.channelType = PROF_CHANNEL_TYPE::PROF_TS_TYPE;
    starsProfStartPara.samplePeriod = PROF_CFG_PERIOD;
    starsProfStartPara.realTime = PROF_REAL;
    starsProfStartPara.userData = starsConfigPtr;
    starsProfStartPara.userDataSize = starsConfigSize;
    return true;
}
}

extern "C" {
int prof_drv_start(unsigned int deviceId, unsigned int channelId, struct prof_start_para *startPara);
int prof_channel_read(unsigned int deviceId, unsigned int channelId, char *outBuf, unsigned int bufSize);
int prof_stop(unsigned int deviceId, unsigned int channelId);
int prof_channel_poll(struct prof_poll_info *outBuf, int num, int timeout);
int halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t *value);
int halGetDeviceInfoByBuff(uint32_t deviceId, int32_t aicoreType, int32_t frequeType, void* freq, int32_t* size);
}

namespace Catlass {

bool Profiler::Start()
{
    if (running_) {
        Stop();
    }
    data_.clear();
    ProfStartParaT starsProfStartPara;
    if (!GetStarsTask(starsProfStartPara)) {
        LOGE("Set stars task data failed.");
        return false;
    }
    int drvRes = prof_drv_start(deviceId_, CHANNEL_STARS_SOC_LOG_BUFFER, &starsProfStartPara);
    if (drvRes != 0) {
        free(starsProfStartPara.userData);
        LOGE("Start channel %u failed", CHANNEL_STARS_SOC_LOG_BUFFER);
        return false;
    }
    free(starsProfStartPara.userData);
    running_ = true;
    CreateReadThread();
    return true;
}

void Profiler::Stop()
{
    if (!running_) {
        return;
    }

    int drvRes = prof_stop(deviceId_, CHANNEL_STARS_SOC_LOG_BUFFER);
    if (drvRes != 0) {
        LOGE("Channel %u prof_stop failed, %d", CHANNEL_STARS_SOC_LOG_BUFFER, drvRes);
    }
    running_ = false;
    if (readThread_.joinable()) {
        readThread_.join();
    }
}

void Profiler::GetDurations(const std::vector<char> &data, std::vector<uint64_t> &starts, std::vector<uint64_t> &ends)
{
    constexpr uint16_t ACSQ_LENGTH = 64;
    for (size_t i = 0; i + ACSQ_LENGTH <= data.size(); i += ACSQ_LENGTH) {
        std::vector<char> splitBinData{&data[i], &data[i] + ACSQ_LENGTH};
        AcsqBean acsqBean(splitBinData);
        uint16_t taskType = acsqBean.GetTaskType();
        TimeType timeType = acsqBean.GetTimeType();
        uint64_t systemTime = acsqBean.GetSystemTime();
        // AI_CORE type = 0, save last group time
        if (taskType == 0) {
            if (timeType == TimeType::START) {
                starts.emplace_back(systemTime);
            } else if (timeType == TimeType::END) {
                ends.emplace_back(systemTime);
            }
        }
    }
}

void Profiler::CreateReadThread()
{
    readThread_ = std::thread([&]() {
        constexpr int PROF_CHANNEL_NUM = 2;
        static constexpr int PROF_CHANNEL_BUFFER_SIZE = 1024 * 1024 * 2;
        std::vector<ProfPollInfoT> channels(PROF_CHANNEL_NUM);
        std::vector<char> outBuf_ = std::vector<char>(PROF_CHANNEL_BUFFER_SIZE);
        for (bool read = true; running_ || read;) {
            std::vector<char> data;
            if (!running_) {
                // try poll once more when stopped
                read = false;
            }
            int ret = prof_channel_poll(channels.data(), PROF_CHANNEL_NUM, 1);
            for (int i = 0; i < ret; ++i) {
                int curLen = prof_channel_read(channels[i].deviceId, channels[i].channelId,
                    &outBuf_[0], outBuf_.size());
                data.insert(data.end(), outBuf_.begin(), outBuf_.begin() + curLen);
            }
            if (!data.empty() && callBack_) {
                callBack_(std::move(data));
            }
        }
    });
}

Profiler::~Profiler()
{
    Stop();
}

void ProfileDataHandler::ProfileDataThread()
{
    std::vector<uint64_t> starts;
    std::vector<uint64_t> ends;
    std::vector<char> data;
    auto flush = [&]() {
        while (!profileDataQueue_.empty()) {
            auto front = profileDataQueue_.front();
            data.insert(data.end(), front.begin(), front.end());
            profileDataQueue_.pop();
        }
    };
    auto getDuration = [&]() {
        if (data.empty()) {
            return;
        }
        profiler_.GetDurations(data, starts, ends);
        data.clear();
        size_t i = 0;
        auto freq = static_cast<double>(GetAicpuFreq());
        durations_.DoTransaction<void>([&](auto &val) {
            while (i < std::min(starts.size(), ends.size())) {
                uint64_t time = ends[i] >= starts[i] ? ends[i] - starts[i] : 0UL;
                val.emplace_back(static_cast<double>(time) * 1000 / freq);
                ++i;
            }
        });
        Erase(starts, i);
        Erase(ends, i);
    };

    while (!finish_) {
        {
            std::unique_lock<decltype(mtx_)> lock(mtx_);
            (void)profileDataCV_.wait_for(lock, std::chrono::seconds(1), [this]() {
                return !profileDataQueue_.empty() || finish_;
            });
            flush();
        }
        getDuration();
    }
    // finish时profileDataQueue不会有其他线程访问
    flush();
    getDuration();
}

bool ProfileDataHandler::Init()
{
    finish_ = false;
    Profiler::CallBackFunc callback = [&](std::vector<char> &&data) {
        {
            std::lock_guard<decltype(mtx_)> lock(mtx_);
            profileDataQueue_.emplace(std::move(data));
        }
        profileDataCV_.notify_all();
    };
    profiler_.RegisterCallback(callback);
    if (!profiler_.Start()) {
        LOGE("Start profiling failed");
        return false;
    }
    profileDataThread_ = std::thread([this]() { ProfileDataThread(); });
    return true;
}

void ProfileDataHandler::Synchronize()
{
    if (finish_) {
        return;
    }

    // 等待所有数据获取更新完成
    constexpr int SLEEP_TIME = 100;
    std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
    profiler_.Stop();
    finish_ = true;
    profileDataCV_.notify_all();
    if (profileDataThread_.joinable()) {
        profileDataThread_.join();
    }
}

std::pair<int64_t, int32_t> ProfileDataHandler::GetAicoreFreq()
{
    constexpr int32_t MODULE_TYPE_AICORE = 4;
    constexpr int32_t INFO_TYPE_FREQUE = 4;
    constexpr int32_t INFO_TYPE_CURRENT_FREQ = 32;
    static int64_t aicoreFreq = 0;
    if (aicoreFreq == 0) {
        // get rated freq
        auto ret = halGetDeviceInfo(deviceId_, MODULE_TYPE_AICORE, INFO_TYPE_FREQUE, &aicoreFreq);
        if (ret != 0) {
            LOGW("Get device rated freq failed, ret %d", ret);
            aicoreFreq = 0;
        }
    }
    // get current freq
    int32_t curFreq;
    int32_t size = sizeof(int32_t);
    auto ret = halGetDeviceInfoByBuff(deviceId_, MODULE_TYPE_AICORE, INFO_TYPE_CURRENT_FREQ,
                                      static_cast<void*>(&curFreq), &size);
    if (ret != 0) {
        LOGW("Get device current freq failed, ret %d", ret);
        curFreq = 0;
    }
    return {aicoreFreq, curFreq};
}

int64_t ProfileDataHandler::GetAicpuFreq()
{
    if (freq_ != 0) {
        return freq_;
    }
    constexpr int64_t DEFAULT_TSCPU_FREQ = 50000;
    constexpr int32_t MODULE_TYPE_SYSTEM = 0;
    constexpr int32_t INFO_TYPE_DEV_OSC_FREQUE = 25;
    int ret = halGetDeviceInfo(0, MODULE_TYPE_SYSTEM, INFO_TYPE_DEV_OSC_FREQUE, &freq_);
    if (ret != 0 || freq_ == 0) {
        LOGW("Get device freq failed, use default freq");
        freq_ = DEFAULT_TSCPU_FREQ;
        return freq_;
    }
    return freq_;
}

bool ProfileDataHandler::SetDeviceId(int32_t deviceId)
{
    constexpr char const *VIS = "ASCEND_RT_VISIBLE_DEVICES";
    int32_t convertedId = deviceId;
    auto env = getenv(VIS);
    if (env) {
        auto error = rtGetVisibleDeviceIdByLogicDeviceId(deviceId, &convertedId);
        if (error != 0) {
            LOGE("Call rtGetVisibleDeviceIdByLogicDeviceId failed, error: %d. Please disable %s or try again.",
                 error, VIS);
            return false;
        }
    }
    deviceId_ = convertedId;
    profiler_.SetDeviceId(convertedId);
    return true;
}

} // namespace Catlass
