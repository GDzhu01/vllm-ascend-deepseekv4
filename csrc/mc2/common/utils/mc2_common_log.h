/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_common_log.h
 * \brief
 */

#ifndef MC2_COMMON_LOG_H
#define MC2_COMMON_LOG_H
#include <stdarg.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "base/err_msg.h"
#include "log/log.h"
#include "securec.h"
#if __has_include("error_manager/error_manager.h")
#include "error_manager/error_manager.h"
#else
#include "err_mgr.h"
#endif

struct ErrorResult {
    operator bool() const
    {
        return false;
    }
    operator ge::graphStatus() const
    {
        return ge::GRAPH_PARAM_INVALID;
    }
    template <typename T>
    operator std::unique_ptr<T>() const
    {
        return nullptr;
    }
    template <typename T>
    operator std::shared_ptr<T>() const
    {
        return nullptr;
    }
    template <typename T>
    operator T *() const
    {
        return nullptr;
    }
    template <typename T>
    operator std::vector<std::shared_ptr<T>>() const
    {
        return {};
    }
    template <typename T>
    operator std::vector<T>() const
    {
        return {};
    }
    operator std::string() const
    {
        return "";
    }
    template <typename T>
    operator T() const
    {
        return T();
    }
};

inline std::vector<char> CreateErrorMsg(const char *format, ...) __attribute__((format(printf, 1, 2)));

inline std::vector<char> CreateErrorMsg();

inline std::vector<char> CreateErrorMsg(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    va_list args_copy;
    va_copy(args_copy, args);
    const size_t len = 4095;
    va_end(args_copy);
    std::vector<char> msg(len + 1U, '\0');
    const auto ret = vsnprintf_s(msg.data(), len + 1U, len, format, args);
    va_end(args);
    return (ret > 0) ? msg : std::vector<char>{};
}

inline std::vector<char> CreateErrorMsg()
{
    return {};
}

#if !defined(__ANDROID__) && !defined(ANDROID)
#define OPS_ERR_IF(COND, LOG_FUNC, EXPR)                                                                               \
    if (__builtin_expect(COND, 0)) {                                                                                   \
        LOG_FUNC;                                                                                                      \
        EXPR;                                                                                                          \
    }

#define OPS_CHECK(COND, LOG_FUNC, EXPR)                                                                                \
    if (COND) {                                                                                                        \
        LOG_FUNC;                                                                                                      \
        EXPR;                                                                                                          \
    }

#define OPS_LOG_FULL(LEVEL, OPS_DESC, ...)

#define OPS_LOG_E(opName, ...) D_OP_LOGE(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OPS_LOG_W(opName, ...) D_OP_LOGW(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OPS_LOG_I(opName, ...) D_OP_LOGI(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OPS_LOG_D(opName, ...) D_OP_LOGD(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGE_IF(condition, returnValue, opName, fmt, ...)                                                           \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool");       \
    do {                                                                                                               \
        if (unlikely(condition)) {                                                                                     \
            OP_LOGE(Ops::Base::GetOpInfo(opName), fmt, ##__VA_ARGS__);                                                 \
            return returnValue;                                                                                        \
        }                                                                                                              \
    } while (0)
#define OP_LOGI_IF_RETURN(condition, returnValue, opName, fmt, ...)                                                    \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool");       \
    do {                                                                                                               \
        if (unlikely(condition)) {                                                                                     \
            OP_LOGI(Ops::Base::GetOpInfo(opName), fmt, ##__VA_ARGS__);                                                 \
            return returnValue;                                                                                        \
        }                                                                                                              \
    } while (0)
#define GE_ASSERT(exp, ...)                                                                                            \
    do {                                                                                                               \
        if (!(exp)) {                                                                                                  \
            auto msg = CreateErrorMsg(__VA_ARGS__);                                                                    \
            if (msg.empty()) {                                                                                         \
                REPORT_INNER_ERR_MSG("E19999", "Assert %s failed", #exp);                                              \
                return ge::FAILED;                                                                                     \
            } else {                                                                                                   \
                REPORT_INNER_ERR_MSG("E19999", "%s", msg.data());                                                      \
                return ge::FAILED;                                                                                     \
            }                                                                                                          \
        }                                                                                                              \
    } while (false)

namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                                      \
    if ((ptr) == nullptr) {                                                                                            \
        const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();                   \
        OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr);                                                          \
        REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                                          \
        return ge::GRAPH_FAILED;                                                                                       \
    }
#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                                             \
    if ((ptr) == nullptr) {                                                                                            \
        const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();                   \
        OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr);                                                          \
        REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                                          \
        return ret;                                                                                                    \
    }
} // namespace ops

#define GE_ASSERT_SUCCESS(v, ...) GE_ASSERT(((v) == ge::SUCCESS), __VA_ARGS__)
#define GE_ASSERT_NOTNULL(v, ...) GE_ASSERT(((v) != nullptr), __VA_ARGS__)
#define GE_ASSERT_GRAPH_SUCCESS(v, ...) GE_ASSERT(((v) == 0), __VA_ARGS__)
#define GE_ASSERT_TRUE(v, ...) GE_ASSERT((v), __VA_ARGS__)
#define GE_ASSERT_EQ(x, y)                                                                                             \
    do {                                                                                                               \
        const auto &xv = (x);                                                                                          \
        const auto &yv = (y);                                                                                          \
        if (xv != yv) {                                                                                                \
            std::stringstream ss;                                                                                      \
            ss << "Assert (" << #x << " == " << #y << ")failed, expect " << yv << " actual " << xv;                    \
            REPORT_INNER_ERR_MSG("E19999", "%s", ss.str().c_str());                                                    \
            return ::ErrorResult();                                                                                    \
        }                                                                                                              \
    } while (0)

#else
#define OPS_ERR_IF(COND, LOG_FUNC, EXPR)
#define OPS_CHECK(COND, LOG_FUNC, EXPR)
#define OPS_LOG_FULL(LEVEL, OPS_DESC, ...)
#define OPS_LOG_E(opName, ...)
#define OPS_LOG_W(opName, ...)
#define OPS_LOG_I(opName, ...)
#define OPS_LOG_D(opName, ...)
#define GE_ASSERT_SUCCESS(v, ...)
#define GE_ASSERT_NOTNULL(v, ...)
#endif
#endif