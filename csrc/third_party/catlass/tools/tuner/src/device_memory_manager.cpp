/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "device_memory_manager.h"

namespace Catlass {

void DoClearL2Cache(uint32_t blockDim, uint8_t* l2ctrl, uint8_t* stream, uint8_t* buffer, uint8_t* tilingSize);

namespace {
struct L2CacheClearTiling {
    uint64_t clearSizePerCore; // B
    uint32_t aicCoreNum;
};

bool GetTiling(L2CacheClearTiling &tiling)
{
    const static std::unordered_map<std::string_view, L2CacheClearTiling> TILING_MAP = {
        {"Ascend910B1", {8388608, 24}},
        {"Ascend910B2", {8388608, 24}},
        {"Ascend910B2C", {8388608, 24}},
        {"Ascend910B3", {10485760, 20}},
        {"Ascend910B4", {5242880, 20}},
        {"Ascend910B4-1", {10485760, 20}},
    };
    auto soc = aclrtGetSocName();
    if (!soc) {
        LOGW("Call aclrtGetSocName failed");
        return false;
    }
    auto it = TILING_MAP.find(soc);
    if (it == TILING_MAP.end()) {
        LOGW("Cannot get l2cache clear params of current soc: %s", soc);
        return false;
    }
    tiling = it->second;
    return true;
}
}

bool DeviceMemoryManager::MallocWorkspace(DeviceMemoryParam &param)
{
    if (!Expand(&workspace_, workspaceSize_, param.size)) {
        LOGE("Expand device memory for workspace failed");
        return false;
    }
    *param.addr = workspace_;
    return true;
}

bool DeviceMemoryManager::MallocArguments(std::vector<DeviceMemoryParam> &params)
{
    std::vector<uint64_t> sizes(params.size());
    uint64_t sum = 0;
    bool overflow = false;
    constexpr uint64_t MAX_MEMORY_SIZE = 100UL * 1024 * 1024 * 1024;
    std::transform(params.begin(), params.end(), sizes.begin(),
        [&](const DeviceMemoryParam& p) {
            uint64_t size = Align(p.size);
            if (MAX_MEMORY_SIZE - sum < size) {
                overflow = true;
                sum = 0;
            }
            sum += size;
            return size;
        });
    if (overflow) {
        LOGE("Kernel arguments size larger than 100GB, cannot malloc device memory");
        return false;
    }
    if (!Expand(&arg_, argSize_, sum)) {
        LOGW("Expand device memory for kernel arguments failed");
        return false;
    }
    auto addr = reinterpret_cast<uintptr_t>(arg_);
    for (std::size_t i = 0; i < params.size(); ++i) {
        *params[i].addr = sizes[i] > 0 ? reinterpret_cast<void*>(addr) : nullptr;
        addr += sizes[i];
    }
    return true;
}

aclrtStream DeviceMemoryManager::Initialize(int32_t deviceId)
{
    if (stream_) {
        return stream_;
    }
    LOGI("Start to initialize device %d", deviceId);
    aclError err = aclInit(nullptr);
    if (err != ACL_SUCCESS) {
        LOGE("Call aclInit failed: %d", err);
        return nullptr;
    }
    deviceId_ = deviceId;
    err = aclrtSetDevice(deviceId_);
    if (err != ACL_SUCCESS) {
        LOGE("Call aclrtSetDevice failed: %d, device id: %d", err, deviceId_);
        return nullptr;
    }
    err = aclrtCreateStream(&stream_);
    if (err != ACL_SUCCESS) {
        LOGE("Call aclrtCreateStream failed: %d", err);
        return nullptr;
    }
    LOGI("Initializing device %d success", deviceId_);
    return stream_;
}

void DeviceMemoryManager::Finalize()
{
    if (!stream_) {
        return;
    }
    Free(arg_);
    Free(workspace_);
    Free(cacheClear_.buffer);
    Free(cacheClear_.tilingSize);
    Free(cacheClear_.flushBuffer);
    ACL_CHECK(aclrtDestroyStream(stream_), "aclrtDestroyStream");
    ACL_CHECK(aclrtResetDevice(deviceId_), "aclrtResetDevice");
    ACL_CHECK(aclFinalize(), "aclFinalize");
    stream_ = nullptr;
}

bool DeviceMemoryManager::Expand(void **addr, uint64_t &size, uint64_t target)
{
    target = Align(target);
    if (*addr && size >= target) {
        return true;
    }
    if (!Free(*addr)) {
        return false;
    }
    *addr = nullptr;
    auto err = aclrtMalloc(addr, target, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS) {
        LOGE("Call aclrtMalloc failed, %d, size %lu", err, target);
        size = 0;
        return false;
    }
    size = target;
    return true;
}

bool DeviceMemoryManager::Free(void *addr)
{
    if (addr) {
        auto err = aclrtFree(addr);
        if (err != ACL_SUCCESS) {
            LOGE("Call aclrtFree failed, %d, release memory for 0x%lx failed", err, (uint64_t)addr);
            return false;
        }
    }
    return true;
}

bool DeviceMemoryManager::InitCacheClear()
{
    L2CacheClearTiling tiling{};
    if (!GetTiling(tiling)) {
        return false;
    }
    cacheClear_.cacheSize = tiling.clearSizePerCore * tiling.aicCoreNum;
    int err = aclrtMalloc(&cacheClear_.buffer, cacheClear_.cacheSize, ACL_MEM_MALLOC_HUGE_FIRST);
    std::shared_ptr<void> defer(nullptr, [&](void*) {
        if (err != ACL_SUCCESS) {
            Free(cacheClear_.buffer);
            Free(cacheClear_.tilingSize);
            Free(cacheClear_.flushBuffer);
            for (auto &j : cacheClear_.cmoBuffers) {
                Free(j);
            }
            cacheClear_.buffer = nullptr;
            cacheClear_.tilingSize = nullptr;
            cacheClear_.flushBuffer = nullptr;
            cacheClear_.cmoBuffers.clear();
        }
    });
    if (err != ACL_SUCCESS) {
        LOGE("Call aclrtMalloc failed, err %d, size %lu", err, cacheClear_.cacheSize);
        return false;
    }
    err = aclrtMalloc(&cacheClear_.flushBuffer, cacheClear_.cacheSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS) {
        LOGE("Call aclrtMalloc failed, err %d, size %lu", err, cacheClear_.cacheSize);
        return false;
    }
    err = SetCacheClearTiling(tiling.clearSizePerCore);
    if (err != ACL_SUCCESS) {
        return false;
    }

    constexpr int CACHE_CLEAR_BUFF = 1;
    cacheClear_.cmoBuffers.resize(CACHE_CLEAR_BUFF);
    for (int i = 0; i < CACHE_CLEAR_BUFF; ++i) {
        err = aclrtMalloc(&cacheClear_.cmoBuffers[i], cacheClear_.cacheSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            LOGE("Call aclrtMalloc failed, err: %d, size 32", err);
            return false;
        }
    }
    return true;
}

aclError DeviceMemoryManager::SetCacheClearTiling(uint64_t clearSizePerCore)
{
    constexpr size_t TILING_SIZE = 32;
    auto err = aclrtMalloc(&cacheClear_.tilingSize, TILING_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS) {
        LOGE("Call aclrtMalloc failed, err: %d, size %lu", err, TILING_SIZE);
        return err;
    }

    void *host_tilingSize;
    err = aclrtMallocHost(&host_tilingSize, TILING_SIZE);
    if (err != ACL_SUCCESS) {
        LOGE("Call aclrtMallocHost failed, err: %d, size %lu", err, TILING_SIZE);
        return err;
    }
    std::shared_ptr<void> deferC(nullptr, [&host_tilingSize](void*) {
        aclrtFreeHost(host_tilingSize);
    });

    *reinterpret_cast<uint64_t*>(host_tilingSize) = clearSizePerCore;
    err = aclrtMemcpyAsync(cacheClear_.tilingSize, sizeof(uint64_t), host_tilingSize, sizeof(uint64_t),
                           ACL_MEMCPY_HOST_TO_DEVICE, stream_);
    if (err != ACL_SUCCESS || (err = aclrtSynchronizeStream(stream_)) != ACL_SUCCESS) {
        LOGE("Set cache clear data failed, err: %d", err);
        return err;
    }
    return ACL_SUCCESS;
}

// return true if calls ClearL2Cache
bool DeviceMemoryManager::ClearL2Cache(uint32_t blockDim)
{
    bool res = false;
    if (cacheClear_.buffer && cacheClear_.tilingSize && cacheClear_.flushBuffer) {
        DoClearL2Cache(blockDim, nullptr, reinterpret_cast<uint8_t*>(stream_),
                       reinterpret_cast<uint8_t*>(cacheClear_.buffer),
                       reinterpret_cast<uint8_t*>(cacheClear_.tilingSize));
        ACL_CHECK(aclrtMemcpyAsync(cacheClear_.flushBuffer, cacheClear_.cacheSize,
                                   cacheClear_.buffer, cacheClear_.cacheSize, ACL_MEMCPY_DEVICE_TO_DEVICE, stream_),
                  "aclrtMemcpyAsync");
        ACL_CHECK(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
        res = true;
    }
    for (auto &cmoBuffer : cacheClear_.cmoBuffers) {
        int err = aclrtCmoAsync(cmoBuffer, cacheClear_.cacheSize, ACL_RT_CMO_TYPE_PREFETCH, stream_);
        if (err != ACL_SUCCESS) {
            LOGE("Call aclrtCmoAsync failed, err %d", err);
            break;
        }
    }
    ACL_CHECK(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
    return res;
}

bool DeviceMemoryManager::FillDeviceData(void *dst, size_t size, void *host) const
{
    auto d = reinterpret_cast<uint64_t>(dst);
    auto addr = reinterpret_cast<uint64_t>(arg_);
    auto addr2 = reinterpret_cast<uint64_t>(workspace_);
    if (!((d >= addr && d + size <= addr + argSize_) || (d >= addr2 && d + size <= addr2 + workspaceSize_))) {
        LOGE("Try to copy host data to invalid addr 0x%lx, size %lu", d, size);
        return false;
    }
    auto err = aclrtMemcpyAsync(dst, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_);
    if (err != ACL_SUCCESS) {
        LOGE("Fill device data failed when call aclrtMemcpyAsync, err: %d", err);
        return false;
    }
    err = aclrtSynchronizeStream(stream_);
    if (err != ACL_SUCCESS) {
        LOGE("Fill device data failed when call aclrtSynchronizeStream, err: %d", err);
        return false;
    }
    return true;
}

} // namespace Catlass
