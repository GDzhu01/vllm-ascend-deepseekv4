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
#include "device_memory_manager.h"
#include "metrics.h"
#include "library_helper.h"
#include "catlass/gemm_coord.hpp"
#include "catlass/catlass.hpp"
#include "tiling/platform/platform_ascendc.h"

namespace Catlass {

namespace {

template <typename T>
std::vector<T> GenGroupList(uint32_t groupCount, uint32_t m)
{
    std::vector<T> groupList;
    groupList.resize(groupCount);
    FillRandomData<T, uint32_t>(groupList, 0, m);
    std::sort(groupList.begin(), groupList.end());
    groupList[0] = 0;
    groupList.back() = static_cast<T>(m);
    return groupList;
}
}

void GemmOpConfig::SaveMetric(Metric &metric)
{
    metric.SetField<ClassicMetric::M>(m_);
    metric.SetField<ClassicMetric::N>(n_);
    metric.SetField<ClassicMetric::K>(k_);
}

bool GemmOpConfig::InitConfig(CommandLineParser &parser)
{
    if (parser.HasKey("m")) {
        m_ = 0;
        GET_CHECK(parser.Get<decltype(m_)>("m", m_), "m");
    }
    if (parser.HasKey("n")) {
        n_ = 0;
        GET_CHECK(parser.Get<decltype(n_)>("n", n_), "n");
    }
    if (parser.HasKey("k")) {
        k_ = 0;
        GET_CHECK(parser.Get<decltype(k_)>("k", k_), "k");
    }
    if (m_ == 0 || n_ == 0 || k_ == 0 || !GetTensorConfig("A", parser, tcA_) ||
        !GetTensorConfig("B", parser, tcB_) || !GetTensorConfig("C", parser, tcC_)) {
        invalid_ = true;
        return false;
    }
    return true;
}

bool GemmOpConfig::Filter(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::GemmOperationDescription&>(op->GetDescription());
    if (UnMatch(tcA_.dataType, mdesp.A.element) || UnMatch(tcA_.layoutType, mdesp.A.layout) ||
        UnMatch(tcB_.dataType, mdesp.B.element) || UnMatch(tcB_.layoutType, mdesp.B.layout) ||
        UnMatch(tcC_.dataType, mdesp.C.element) || UnMatch(tcC_.layoutType, mdesp.C.layout)) {
        return false;
    }
    return true;
}

bool BasicGemmOpConfig::InitConfig(CommandLineParser &parser)
{
    bool res = GemmOpConfig::InitConfig(parser);
    if (!res) {
        return false;
    }
    config_.m = m_;
    config_.n = n_;
    config_.k = k_;
    return true;
}

bool BasicGemmOpConfig::InitArgument(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::GemmOperationDescription &>(op->GetDescription());
    size_t lenA;
    size_t lenB;
    size_t lenC;
    constexpr std::string_view log = "Arguments size overflows, please check command line input"
                                     " --m --n --k";
    if (!SafeMul<uint32_t>({config_.m, config_.k}, lenA) ||
        !SafeMul<uint32_t>({config_.k, config_.n}, lenB) ||
        !SafeMul<uint32_t>({config_.m, config_.n}, lenC)) {
        LOGE("%s", log.data());
        return false;
    }

    size_t sizeA;
    size_t sizeB;
    size_t sizeC;
    if (!SafeMul<size_t>({lenA, LibraryHelper::GetDataTypeSize(mdesp.A.element)}, sizeA) ||
        !SafeMul<size_t>({lenB, LibraryHelper::GetDataTypeSize(mdesp.B.element)}, sizeB) ||
        !SafeMul<size_t>({lenC, LibraryHelper::GetDataTypeSize(mdesp.C.element)}, sizeC)) {
        LOGE("%s", log.data());
        return false;
    }
    std::vector<DeviceMemoryParam> params{
        {reinterpret_cast<void**>(&arg_.A), sizeA},
        {reinterpret_cast<void**>(&arg_.B), sizeB},
        {reinterpret_cast<void**>(&arg_.C), sizeC},
    };
    if (!MallocDeviceMemory(params)) {
        return false;
    }
    return true;
}

void GroupedGemmOpConfig::SaveMetric(Metric &metric)
{
    GemmOpConfig::SaveMetric(metric);
    metric.SetField("group_count", std::to_string(config_.groupCount));
}

bool GroupedGemmOpConfig::InitConfig(CommandLineParser &parser)
{
    bool res = GemmOpConfig::InitConfig(parser);
    if (!res) {
        return false;
    }
    config_.m = m_;
    config_.n = n_;
    config_.k = k_;
    if (!parser.HasKey("group_count")) {
        config_.groupCount = 128; // 未指定group_count时默认使用128
    } else {
        GET_CHECK(parser.Get<decltype(config_.groupCount)>("group_count", config_.groupCount), "group_count");
        if (config_.groupCount == 0) {
            LOGE("The --group_count should be a positive integer");
            invalid_ = true;
            return false;
        }
        constexpr uint32_t GROUP_COUNT_MAX_LIMIT = 65535U;
        if (config_.groupCount > GROUP_COUNT_MAX_LIMIT) {
            LOGE("The --group_count should be not larger than %u", GROUP_COUNT_MAX_LIMIT);
            invalid_ = true;
            return false;
        }
    }
    groupList_ = GenGroupList<int32_t>(config_.groupCount, config_.m);
    return true;
}

bool GroupedGemmOpConfig::CheckArgument(const Library::GemmOperationDescription &mdesp, ArgumentSize &argSize)
{
    argSize.layoutASize = LibraryHelper::GetLayoutSize(mdesp.A.layout);
    argSize.layoutBSize = LibraryHelper::GetLayoutSize(mdesp.B.layout);
    argSize.layoutCSize = LibraryHelper::GetLayoutSize(mdesp.C.layout);
    if (!SafeMul<uint32_t>({config_.m, config_.k}, argSize.lenA) ||
        !SafeMul<uint32_t>({config_.k, config_.n}, argSize.lenB) ||
        !SafeMul<uint32_t>({config_.m, config_.n, config_.groupCount}, argSize.lenC) ||
        !SafeMul<size_t>({argSize.lenA, LibraryHelper::GetDataTypeSize(mdesp.A.element)}, argSize.sizeA) ||
        !SafeMul<size_t>({argSize.lenB, LibraryHelper::GetDataTypeSize(mdesp.B.element)}, argSize.sizeB) ||
        !SafeMul<size_t>({argSize.lenC, LibraryHelper::GetDataTypeSize(mdesp.C.element)}, argSize.sizeC) ||
        !SafeMul<size_t>({config_.groupCount, argSize.layoutASize}, argSize.sizeLayoutAList) ||
        !SafeMul<size_t>({config_.groupCount, argSize.layoutBSize}, argSize.sizeLayoutBList) ||
        !SafeMul<size_t>({config_.groupCount, argSize.layoutCSize}, argSize.sizeLayoutCList) ||
        !SafeMul<size_t>({config_.groupCount, sizeof(GemmCoord)}, argSize.sizeProblemShapeList)) {
        LOGE("Arguments size overflows, please check command line input --m --n --k --group_count");
        return false;
    }
    return true;
}

void GroupedGemmOpConfig::GenerateInput(const Library::GemmOperationDescription &mdesp,
                                        const ArgumentSize &argSize)
{
    std::vector<GemmCoord> problemShapeList(config_.groupCount);
    std::vector<uint8_t> layoutAList(argSize.layoutASize * config_.groupCount);
    std::vector<uint8_t> layoutBList(argSize.layoutBSize * config_.groupCount);
    std::vector<uint8_t> layoutCList(argSize.layoutCSize * config_.groupCount);
    for (uint32_t i = 0, a = 0, b = 0, c = 0;
         i < config_.groupCount;
         ++i, a += argSize.layoutASize, b += argSize.layoutBSize, c += argSize.layoutCSize) {
        uint32_t currentK = (i == 0) ? groupList_[0] : (groupList_[i] - groupList_[i - 1]);
        problemShapeList[i] = GemmCoord{config_.m, config_.n, currentK};
        LibraryHelper::ConstructLayout(mdesp.A.layout, mdesp.A.element, config_.m, currentK, &layoutAList[a]);
        LibraryHelper::ConstructLayout(mdesp.B.layout, mdesp.B.element, currentK, config_.n, &layoutBList[b]);
        LibraryHelper::ConstructLayout(mdesp.C.layout, mdesp.C.element, config_.m, config_.n, &layoutCList[c]);
    }
    DeviceMemoryManager::Instance().FillDeviceData(arg_.problemShapeList, argSize.sizeProblemShapeList,
                                                   problemShapeList.data());
    DeviceMemoryManager::Instance().FillDeviceData(arg_.layoutAList, argSize.sizeLayoutAList, layoutAList.data());
    DeviceMemoryManager::Instance().FillDeviceData(arg_.layoutBList, argSize.sizeLayoutBList, layoutBList.data());
    DeviceMemoryManager::Instance().FillDeviceData(arg_.layoutCList, argSize.sizeLayoutCList, layoutCList.data());
}

bool GroupedGemmOpConfig::InitArgument(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::GemmOperationDescription &>(op->GetDescription());
    ArgumentSize safeArg{};
    if (!CheckArgument(mdesp, safeArg)) {
        return false;
    }

    std::vector<DeviceMemoryParam> params{
        {reinterpret_cast<void**>(&arg_.problemShapeList), safeArg.sizeProblemShapeList},
        {reinterpret_cast<void**>(&arg_.A), safeArg.sizeA},
        {reinterpret_cast<void**>(&arg_.layoutAList), safeArg.sizeLayoutAList},
        {reinterpret_cast<void**>(&arg_.B), safeArg.sizeB},
        {reinterpret_cast<void**>(&arg_.layoutBList), safeArg.sizeLayoutBList},
        {reinterpret_cast<void**>(&arg_.C), safeArg.sizeC},
        {reinterpret_cast<void**>(&arg_.layoutCList), safeArg.sizeLayoutCList},
    };
    if (!MallocDeviceMemory(params)) {
        return false;
    }

    GenerateInput(mdesp, safeArg);
    return true;
}

bool GroupedSliceMGemmOpConfig::InitConfig(CommandLineParser &parser)
{
    // m/n/k 为通用参数，在父类中解析
    bool res = GemmOpConfig::InitConfig(parser);
    if (!res) {
        return false;
    }
    config_.m = m_;
    config_.n = n_;
    config_.k = k_;

    // 特殊参数获取
    if (!parser.HasKey("group_count")) {
        config_.groupCount = 128;
    } else {
        GET_CHECK(parser.Get<decltype(config_.groupCount)>("group_count", config_.groupCount), "group_count");
        if (config_.groupCount == 0) {
            LOGE("The --group_count should be a positive integer");
            // 获取失败时需要标注 invalid_ 位，跳过本类型算子的运行
            invalid_ = true;
            return false;
        }
        constexpr uint32_t GROUP_COUNT_MAX_LIMIT = 65535U;
        if (config_.groupCount > GROUP_COUNT_MAX_LIMIT) {
            LOGE("The --group_count should be not larger than %u", GROUP_COUNT_MAX_LIMIT);
            invalid_ = true;
            return false;
        }
    }
    return true;
}

bool GroupedSliceMGemmOpConfig::InitArgument(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::GemmOperationDescription &>(op->GetDescription());
    ArgumentSize argSize{};
    // 安全校验，和计算 device 侧内存空间占用，公式和 examples 内是一致的，可以对照看一下
    if (!SafeMul<uint32_t>({config_.m, config_.k}, argSize.lenA) ||
        !SafeMul<uint32_t>({config_.k, config_.n, config_.groupCount}, argSize.lenB) ||
        !SafeMul<uint32_t>({config_.m, config_.n}, argSize.lenC) ||
        !SafeMul<size_t>({argSize.lenA, LibraryHelper::GetDataTypeSize(mdesp.A.element)}, argSize.sizeA) ||
        !SafeMul<size_t>({argSize.lenB, LibraryHelper::GetDataTypeSize(mdesp.B.element)}, argSize.sizeB) ||
        !SafeMul<size_t>({argSize.lenC, LibraryHelper::GetDataTypeSize(mdesp.C.element)}, argSize.sizeC) ||
        !SafeMul<size_t>({config_.groupCount, sizeof(int64_t)}, argSize.sizeGroupList)) {
        LOGE("Arguments size overflows, please check command line input --m --n --k --group_count");
        return false;
    }

    // 申请 device 侧内存，除了 workspace 外，所有内存统一申请
    std::vector<DeviceMemoryParam> params{
        {reinterpret_cast<void**>(&arg_.deviceGroupList), argSize.sizeGroupList},
        {reinterpret_cast<void**>(&arg_.A), argSize.sizeA},
        {reinterpret_cast<void**>(&arg_.B), argSize.sizeB},
        {reinterpret_cast<void**>(&arg_.C), argSize.sizeC},
    };
    if (!MallocDeviceMemory(params)) {
        return false;
    }

    // 填充数据，layout 的案例在 GroupedGemmOpConfig 类
    std::vector<int64_t> groupList = GenGroupList<int64_t>(config_.groupCount, config_.m);
    DeviceMemoryManager::Instance().FillDeviceData(arg_.deviceGroupList, argSize.sizeGroupList,
                                                   groupList.data());
    return true;
}

void GroupedSliceMGemmOpConfig::SaveMetric(Metric &metric)
{
    GemmOpConfig::SaveMetric(metric);
    metric.SetField("group_count", std::to_string(config_.groupCount));
}

bool OptimizedGemmOpConfig::InitConfig(CommandLineParser &parser)
{
    bool res = GemmOpConfig::InitConfig(parser);
    if (!res) {
        return false;
    }
    config_.m = m_;
    config_.n = n_;
    config_.k = k_;
    return true;
}

bool OptimizedGemmOpConfig::InitArgument(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::GemmOperationDescription &>(op->GetDescription());
    size_t lenA;
    size_t lenB;
    size_t lenC;
    constexpr std::string_view log = "Arguments size overflows, please check command line input"
                                     " --m --n --k";
    if (!SafeMul<uint32_t>({config_.m, config_.k}, lenA) ||
        !SafeMul<uint32_t>({config_.k, config_.n}, lenB) ||
        !SafeMul<uint32_t>({config_.m, config_.n}, lenC)) {
        LOGE("%s", log.data());
        return false;
    }

    size_t sizeA;
    size_t sizeB;
    size_t sizeC;
    if (!SafeMul<size_t>({lenA, LibraryHelper::GetDataTypeSize(mdesp.A.element)}, sizeA) ||
        !SafeMul<size_t>({lenB, LibraryHelper::GetDataTypeSize(mdesp.B.element)}, sizeB) ||
        !SafeMul<size_t>({lenC, LibraryHelper::GetDataTypeSize(mdesp.C.element)}, sizeC)) {
        LOGE("%s", log.data());
        return false;
    }
    std::vector<DeviceMemoryParam> params{
        {reinterpret_cast<void**>(&arg_.A), sizeA},
        {reinterpret_cast<void**>(&arg_.B), sizeB},
        {reinterpret_cast<void**>(&arg_.C), sizeC},
    };
    if (!MallocDeviceMemory(params)) {
        return false;
    }
    return true;
}

void QuantMatmulGemmOpConfig::SaveMetric(Metric &metric)
{
    GemmOpConfig::SaveMetric(metric);
}

bool QuantMatmulGemmOpConfig::InitConfig(CommandLineParser &parser)
{
    bool res = GemmOpConfig::InitConfig(parser);
    if (!res) {
        return false;
    }
    config_.m = m_;
    config_.n = n_;
    config_.k = k_;
    return true;
}

bool QuantMatmulGemmOpConfig::CheckArgument(const Library::QuantMatmulGemmOperationDescription &mdesp, ArgumentSize &argSize)
{
    argSize.layoutASize = LibraryHelper::GetLayoutSize(mdesp.A.layout);
    argSize.layoutBSize = LibraryHelper::GetLayoutSize(mdesp.B.layout);
    argSize.layoutDSize = LibraryHelper::GetLayoutSize(mdesp.D.layout);
    argSize.layoutScaleSize = LibraryHelper::GetLayoutSize(mdesp.Scale.layout);
    argSize.layoutPerTokenScaleSize = LibraryHelper::GetLayoutSize(mdesp.PerTokenScale.layout);

    if (!SafeMul<uint32_t>({config_.m, config_.k}, argSize.lenA) ||
        !SafeMul<uint32_t>({config_.k, config_.n}, argSize.lenB) ||
        !SafeMul<uint32_t>({config_.m, config_.n}, argSize.lenC) ||
        !SafeMul<uint32_t>({config_.n}, argSize.lenScale) ||
        !SafeMul<uint32_t>({config_.m}, argSize.lenPerTokenScale) ||
        !SafeMul<size_t>({argSize.lenA, LibraryHelper::GetDataTypeSize(mdesp.A.element)}, argSize.sizeA) ||
        !SafeMul<size_t>({argSize.lenB, LibraryHelper::GetDataTypeSize(mdesp.B.element)}, argSize.sizeB) ||
        !SafeMul<size_t>({argSize.lenC, LibraryHelper::GetDataTypeSize(mdesp.D.element)}, argSize.sizeD) ||
        !SafeMul<size_t>({argSize.lenScale, LibraryHelper::GetDataTypeSize(mdesp.Scale.element)}, argSize.sizeScale) ||
        !SafeMul<size_t>({argSize.lenPerTokenScale, LibraryHelper::GetDataTypeSize(mdesp.PerTokenScale.element)}, argSize.sizePerTokenScale))
    {
        LOGE("Arguments size overflows, please check command line input --m --n --k");
        return false;
    }
    return true;
}

bool QuantMatmulGemmOpConfig::InitArgument(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::QuantMatmulGemmOperationDescription &>(op->GetDescription());
    ArgumentSize safeArg{};
    if (!CheckArgument(mdesp, safeArg)) {
        return false;
    }

    arg_.problemShape = Catlass::GemmCoord{config_.m, config_.n, config_.k};
    arg_.aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    std::vector<DeviceMemoryParam> params{
        {reinterpret_cast<void**>(&arg_.ptrA), safeArg.sizeA},
        {reinterpret_cast<void**>(&arg_.ptrB), safeArg.sizeB},
        {reinterpret_cast<void**>(&arg_.ptrD), safeArg.sizeD},
        {reinterpret_cast<void**>(&arg_.ptrScale), safeArg.sizeScale},
        {reinterpret_cast<void**>(&arg_.ptrPerTokenScale), safeArg.sizePerTokenScale}
    };

    if (!MallocDeviceMemory(params)) {
        return false;
    }

    return true;
}

bool MatmulGeluGemmOpConfig::InitConfig(CommandLineParser &parser)
{
    bool res = GemmOpConfig::InitConfig(parser);
    if (!res) {
        return false;
    }
    config_.m = m_;
    config_.n = n_;
    config_.k = k_;
    
    if (!parser.HasKey("accu_dtype")) {
        config_.elementSize = 4; // Default: fp32 (4 bytes)
    } else {
        std::string_view accuDtype;
        GET_CHECK(parser.Get<std::string_view>("accu_dtype", accuDtype), "accu_dtype");
        config_.elementSize = LibraryHelper::GetDataTypeSize(
            LibraryHelper::GetDataTypeEnum(accuDtype)
        );
    }
    return true;
}

bool MatmulGeluGemmOpConfig::InitArgument(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::MatmulGeluGemmOperationDescription &>(op->GetDescription());
    ArgumentSize argSize{};
    if (!SafeMul<uint32_t>({config_.m, config_.k}, argSize.lenA) ||
        !SafeMul<uint32_t>({config_.k, config_.n}, argSize.lenB) ||
        !SafeMul<uint32_t>({config_.m, config_.n}, argSize.lenD) || 
        !SafeMul<size_t>({argSize.lenA, LibraryHelper::GetDataTypeSize(mdesp.A.element)}, argSize.sizeA) ||
        !SafeMul<size_t>({argSize.lenB, LibraryHelper::GetDataTypeSize(mdesp.B.element)}, argSize.sizeB) ||
        !SafeMul<size_t>({argSize.lenD, LibraryHelper::GetDataTypeSize(mdesp.D.element)}, argSize.sizeD)) {
        LOGE("Arguments size overflows, please check command line input --m --n --k");
        return false;
    }

    std::vector<DeviceMemoryParam> params{
        {reinterpret_cast<void**>(&arg_.ptrA), argSize.sizeA},
        {reinterpret_cast<void**>(&arg_.ptrB), argSize.sizeB},
        {reinterpret_cast<void**>(&arg_.ptrD), argSize.sizeD},
    };
    if (!MallocDeviceMemory(params)) {
        return false;
    }
    
    return true;
}

void MatmulGeluGemmOpConfig::SaveMetric(Metric &metric)
{
    GemmOpConfig::SaveMetric(metric);
    metric.SetField("element_size", std::to_string(config_.elementSize));
}
} // namespace Catlass
