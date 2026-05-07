/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "gemm_op_config.h"
#include "library_helper.h"

namespace Catlass {

using namespace Library;

bool OpConfig::GetTensorConfig(const std::string &key, CommandLineParser &parser, TensorConfig &config)
{
    if (!parser.HasKey(key)) {
        return true;
    }

    std::string_view val;
    auto err = parser.Get<std::string_view>(key, val);
    if (err != CommandLineParser::ERROR_CODE::NONE) {
        LOGE("Get command line input failed, key: %s, err: %s", key.c_str(),
             CommandLineParser::GetErrorStr(err).data());
        return false;
    }
    auto i = val.find(':');
    if (!val.empty() && i >= val.size() - 1) {
        LOGE("Command line input --%s value format is invalid, it should be string like fp16:row, "
             "please check README.md. ", key.c_str());
        return false;
    }
    config.dataType = LibraryHelper::GetDataTypeEnum(val.substr(0, i));
    config.layoutType = LibraryHelper::GetLayoutEnum(val.substr(i + 1));
    if (config.dataType == DataType::Invalid || config.layoutType == LayoutType::Invalid) {
        LOGE("Command line input --%s value format is invalid, it should be string like fp16:row, "
             "please check README.md. ", key.c_str());
        return false;
    }
    return true;
}

std::shared_ptr<OpConfig> GetGemmOpConfig(const OperationDescription &desp)
{
    if (desp.kind != OperationKind::Gemm) {
        LOGE("Operate is not matmul kind");
        return nullptr;
    }
    auto mDesp = static_cast<const GemmOperationDescription&>(desp);
    switch (mDesp.gemmKind) {
        case GemmKind::BasicMatmul:
            return std::make_shared<BasicGemmOpConfig>(desp);
        case GemmKind::GroupedMatmul:
            return std::make_shared<GroupedGemmOpConfig>(desp);
        case GemmKind::GroupedMatmulSliceM:
            return std::make_shared<GroupedSliceMGemmOpConfig>(desp);
        case GemmKind::OptimizedMatmul:
            return std::make_shared<OptimizedGemmOpConfig>(desp);
        case GemmKind::QuantMatmul:
            return std::make_shared<QuantMatmulGemmOpConfig>(desp);
        case GemmKind::MatmulGelu:
            return std::make_shared<MatmulGeluGemmOpConfig>(desp);
        default:
            LOGE("Matmul op type is invalid %u, config create failed", static_cast<uint32_t>(mDesp.gemmKind));
            break;
    }
    return nullptr;
}

std::shared_ptr<OpConfig> OpConfig::GetOpConfig(const OperationDescription &desp)
{
    using FuncType = std::shared_ptr<OpConfig>(*)(const OperationDescription &desp);
    std::vector<FuncType> func{
        GetGemmOpConfig
    };
    size_t i = static_cast<size_t>(desp.kind);
    if (i >= func.size()) {
        LOGE("description kind invalid %ld", i);
        return nullptr;
    }
    return func[i](desp);
}

// return true means exist normally, false means meet sth wrong
bool OpConfigPool::Register(Operation *op, CommandLineParser &parser, const std::string_view kernel)
{
    auto &desp = op->GetDescription();
    std::string_view name = desp.name;
    if (!kernel.empty() && name.find(kernel) == std::string_view::npos) {
        return true;
    }
    std::shared_ptr<OpConfig> config = OpConfig::GetOpConfig(desp);
    if (!config) {
        LOGE("Get op config failed, op name %s", desp.name);
        return false;
    }
    auto p = pool_.insert({config, {}});
    if (p.second && !config->InitConfig(parser)) {
        LOGE("Initialize config for %s failed", desp.name);
        return false;
    }
    config = p.first->first;
    if (!config->Invalid() && config->Filter(op)) {
        p.first->second.emplace_back(op);
    }
    return true;
}

} // namespace Catlass
