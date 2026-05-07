/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_OP_CONFIG_H
#define CATLASS_TUNER_OP_CONFIG_H

#include <memory>
#include <unordered_map>

#include <opdev/bfloat16.h>

#include "command_line_parser.h"
#include "m_t_var.h"
#include "device_memory_manager.h"
#include "metric.h"

#include "catlass/library/operation.h"

namespace Catlass {

struct TensorConfig {
    TensorConfig(Library::DataType data = Library::DataType::Invalid,
                 Library::LayoutType layout = Library::LayoutType::Invalid)
        : dataType(data), layoutType(layout) {}

    void SetInputData(size_t len, uint8_t *dst)
    {
        switch (dataType) {
            case Library::DataType::U8:
                FillData<uint8_t>(len, dst);
                break;
            case Library::DataType::Int8:
                FillData<int8_t>(len, dst);
                break;
            case Library::DataType::Int32:
                FillData<int32_t>(len, dst);
                break;
            case Library::DataType::Fp16:
                FillData<op::fp16_t>(len, dst);
                break;
            case Library::DataType::Bf16:
                FillData<op::bfloat16>(len, dst);
                break;
            case Library::DataType::Fp32:
                FillData<float>(len, dst);
                break;
            default:
                LOGE("Invalid tensor data type, generate input data failed");
        }
    }

    Library::DataType dataType{Library::DataType::Invalid};
    Library::LayoutType layoutType{Library::LayoutType::Invalid};

private:
    template<typename T>
    void FillData(size_t len, uint8_t *dst)
    {
        T lower = 1;
        T upper = 5;
        size_t size;
        if (!SafeMul<size_t>({len, sizeof(T)}, size)) {
            return;
        }
        std::vector<T> host(len);
        FillRandomData<T>(host, lower, upper);
        DeviceMemoryManager::Instance().FillDeviceData(dst, size, host.data());
    }
};

class OpConfig {
public:
    explicit OpConfig(const Library::OperationDescription &desp) : kind_(static_cast<uint32_t>(desp.kind)) {}
    static std::shared_ptr<OpConfig> GetOpConfig(const Library::OperationDescription &desp);
    static bool GetTensorConfig(const std::string &key, CommandLineParser &parser, TensorConfig &config);

    virtual ~OpConfig() = default;
    virtual void* GetConfig() = 0;
    virtual void* GetArg() = 0;
    virtual bool Filter(Library::Operation *op) = 0;
    virtual bool InitConfig(CommandLineParser &parser) = 0; // call once each OpConfig
    virtual bool InitArgument(Library::Operation *op) = 0; // call each Operator
    virtual void SaveMetric(Metric &metric) = 0;

    bool operator<(const OpConfig& other) const
    {
        if (kind_ != other.kind_) {
            return kind_ < other.kind_;
        }
        return subKind_ < other.subKind_;
    }

    inline bool MallocDeviceMemory(std::vector<DeviceMemoryParam> &params)
    {
        if (!DeviceMemoryManager::Instance().MallocArguments(params)) {
            if (!DeviceMemoryManager::Instance().FreeWorkspace() ||
                !DeviceMemoryManager::Instance().MallocArguments(params)) {
                LOGE("Malloc device memory for operator failed");
                return false;
            }
        }
        return true;
    }

    inline uint32_t GetKind() const { return kind_; }
    inline uint32_t GetSubKind() const { return subKind_; }
    inline bool Invalid() const { return invalid_; }

protected:
    uint32_t kind_; // Library::OperationKind
    uint32_t subKind_{UINT32_MAX}; // sth like Library::GemmKind
    bool invalid_{false};
};

class OpConfigPool {
public:
    bool Register(Library::Operation *op, CommandLineParser &parser, const std::string_view kernel);
    auto &GetPool() { return pool_; }

private:
    struct OpConfigCompare {
        bool operator()(const std::shared_ptr<OpConfig>& lhs,
                        const std::shared_ptr<OpConfig>& rhs) const
        {
            if (lhs && rhs) {
                return *lhs < *rhs;
            }
            return lhs < rhs;
        }
    };

    std::map<std::shared_ptr<OpConfig>, std::vector<Library::Operation*>, OpConfigCompare> pool_;
};

} // namespace Catlass
#endif // CATLASS_TUNER_OP_CONFIG_H
