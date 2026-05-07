/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_GEMM_OP_CONFIG_H
#define CATLASS_TUNER_GEMM_OP_CONFIG_H

#include "op_config.h"

namespace Catlass {

class GemmOpConfig : public OpConfig {
public:
    explicit GemmOpConfig(const Library::OperationDescription &desp) : OpConfig(desp) {}
    ~GemmOpConfig() override = default;
    void SaveMetric(Metric &metric) override;
    bool InitConfig(CommandLineParser &parser) override;
    bool Filter(Library::Operation *op) override;

protected:
    TensorConfig tcA_{};
    TensorConfig tcB_{};
    TensorConfig tcC_{};
    uint32_t m_{256};  // 256 为example/00_basic_matmul算子默认配置
    uint32_t n_{512};  // 512 为example/00_basic_matmul算子默认配置
    uint32_t k_{1024}; // 1024 为example/00_basic_matmul算子默认配置

private:
    template<class T>
    [[nodiscard]] inline bool UnMatch(T exp, T val) const
    {
        return exp != T::Invalid && exp != val;
    }
};

class BasicGemmOpConfig : public GemmOpConfig {
public:
    explicit BasicGemmOpConfig(const Library::OperationDescription &desp)
        : GemmOpConfig(desp)
    {
        subKind_ = static_cast<uint32_t>(Library::GemmKind::BasicMatmul);
    }

    bool InitConfig(CommandLineParser &parser) override;
    bool InitArgument(Library::Operation *op) override;

    void* GetConfig() override { return &config_; };
    void* GetArg() override { return &arg_; };

private:
    Library::BasicMatmulGemmArguments arg_{};
    Library::BasicMatmulGemmConfiguration config_{};
};

class GroupedGemmOpConfig : public GemmOpConfig {
public:
    explicit GroupedGemmOpConfig(const Library::OperationDescription &desp)
        : GemmOpConfig(desp)
    {
        subKind_ = static_cast<uint32_t>(Library::GemmKind::GroupedMatmul);
    }

    bool InitConfig(CommandLineParser &parser) override;
    bool InitArgument(Library::Operation *op) override;

    void SaveMetric(Metric &metric) override;
    void* GetConfig() override { return &config_; };
    void* GetArg() override { return &arg_; };

private:
    struct ArgumentSize {
        size_t lenA;
        size_t lenB;
        size_t lenC;
        size_t sizeA;
        size_t sizeB;
        size_t sizeC;
        size_t sizeLayoutAList;
        size_t sizeLayoutBList;
        size_t sizeLayoutCList;
        size_t sizeProblemShapeList;
        size_t layoutASize;
        size_t layoutBSize;
        size_t layoutCSize;
    };

    bool CheckArgument(const Library::GemmOperationDescription &mdesp, ArgumentSize &argSize);
    void GenerateInput(const Library::GemmOperationDescription &mdesp, const ArgumentSize &argSize);

    Library::GroupedMatmulGemmArguments arg_{};
    Library::GroupedMatmulGemmConfiguration config_{};
    std::vector<int32_t> groupList_;
};

class GroupedSliceMGemmOpConfig: public GemmOpConfig {
public:
    explicit GroupedSliceMGemmOpConfig(const Library::OperationDescription &desp)
        : GemmOpConfig(desp)
    {
        // 构造函数必须初始化 subKind_ 为本类对应的枚举
        subKind_ = static_cast<uint32_t>(Library::GemmKind::GroupedMatmulSliceM);
    }

    // 用于从命令行参数初始化 m/n/k/group_count 等配置
    bool InitConfig(CommandLineParser &parser) override;
    bool InitArgument(Library::Operation *op) override;
    void SaveMetric(Metric &metric) override;

    void* GetConfig() override { return &config_; };
    void* GetArg() override { return &arg_; };

private:
    // 临时存储各个缓冲区长度
    struct ArgumentSize {
        size_t lenA;
        size_t lenB;
        size_t lenC;
        size_t sizeA;
        size_t sizeB;
        size_t sizeC;
        size_t sizeGroupList;
    };

    // config_ 和 arg_ 分别需要定义成 catlass/tools/library/src/library.h 对应本类的对象
    Library::GroupedMatmulSliceMGemmArguments arg_{};
    Library::GroupedMatmulSliceMGemmConfiguration config_{};
};

class OptimizedGemmOpConfig : public GemmOpConfig {
public:
    explicit OptimizedGemmOpConfig(const Library::OperationDescription &desp)
        : GemmOpConfig(desp)
    {
        subKind_ = static_cast<uint32_t>(Library::GemmKind::OptimizedMatmul);
    }

    bool InitConfig(CommandLineParser &parser) override;
    bool InitArgument(Library::Operation *op) override;

    void* GetConfig() override { return &config_; };
    void* GetArg() override { return &arg_; };

private:
    Library::BasicMatmulGemmArguments arg_{};
    Library::BasicMatmulGemmConfiguration config_{};
};


class QuantMatmulGemmOpConfig : public GemmOpConfig {
public:
    explicit QuantMatmulGemmOpConfig(const Library::OperationDescription &desp)
        : GemmOpConfig(desp)
    {
        subKind_ = static_cast<uint32_t>(Library::GemmKind::QuantMatmul);
    }

    bool InitConfig(CommandLineParser &parser) override;
    bool InitArgument(Library::Operation *op) override;

    void SaveMetric(Metric &metric) override;
    void* GetConfig() override { return &config_; };
    void* GetArg() override { return &arg_; };

private:
    struct ArgumentSize {
        size_t lenA;
        size_t lenB;
        size_t lenC;
        size_t lenScale;
        size_t lenPerTokenScale;
        size_t sizeA;
        size_t sizeB;
        size_t sizeD;  // ptrD的内存大小
        size_t sizeScale;
        size_t sizePerTokenScale;
        size_t layoutASize;
        size_t layoutBSize;
        size_t layoutDSize;
        size_t layoutScaleSize;
        size_t layoutPerTokenScaleSize;
    };

    bool CheckArgument(const Library::QuantMatmulGemmOperationDescription &mdesp, ArgumentSize &argSize);
    void GenerateInput(const Library::QuantMatmulGemmOperationDescription &mdesp, const ArgumentSize &argSize);

    Library::QuantMatmulGemmArguments arg_{};
    Library::QuantMatmulGemmConfiguration config_{};
};


struct MatmulGeluGemmOpConfig : public GemmOpConfig {
    explicit MatmulGeluGemmOpConfig(const Library::OperationDescription &desp)
        : GemmOpConfig(desp)
    {
        subKind_ = static_cast<uint32_t>(Library::GemmKind::MatmulGelu);
    }

    bool InitConfig(CommandLineParser &parser) override;
    bool InitArgument(Library::Operation *op) override;
    void SaveMetric(Metric &metric) override;
    
    void* GetConfig() override { return &config_; };
    void* GetArg() override { return &arg_; }; 


private:
    struct ArgumentSize {
        size_t lenA;
        size_t lenB;
        size_t lenD;
        size_t sizeA;
        size_t sizeB;
        size_t sizeD;
    };

    bool CheckArgument(const Library::MatmulGeluGemmOperationDescription &mdesp, ArgumentSize &argSize);
    void GenerateInput(const Library::MatmulGeluGemmOperationDescription &mdesp, const ArgumentSize &argSize);

    Library::MatmulGeluGemmArguments arg_{};
    Library::MatmulGeluGemmConfiguration config_{};
};
} // namespace Catlass
#endif // CATLASS_TUNER_GEMM_OP_CONFIG_H
