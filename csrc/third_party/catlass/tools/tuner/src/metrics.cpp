/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "metrics.h"
#include <filesystem>
#include <securec.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iterator>
#include "library_helper.h"

namespace Catlass {

using namespace Library;

static constexpr std::string_view PATH_SEP = "/";
static constexpr mode_t SAVE_DATA_FILE_AUTHORITY = 0640;
static constexpr mode_t SAVE_DIR_AUTHORITY = 0750;

namespace {

std::string StandardizePath(const std::string_view path_view)
{
    std::error_code ec;
    std::filesystem::path path(path_view);

    if (path.is_absolute()) {
        return path.lexically_normal();
    }

    if (std::filesystem::path cwd = std::filesystem::current_path(ec); !ec) {
        return (cwd / path).lexically_normal();
    }
    LOGE("Get current working dir failed when parse absolute path: %s", ec.message().c_str());
    return {};
}

inline bool IsSoftLink(std::string_view path)
{
    std::error_code ec;
    std::filesystem::path absPath = path;
    bool res = is_symlink(absPath, ec);
    if (ec) {
        LOGE("Check soft link failed: %s", ec.message().c_str());
        return false;
    }
    return res;
}

inline bool IsExist(std::string_view path)
{
    std::error_code ec;
    std::filesystem::path p(path);
    bool res = std::filesystem::exists(p, ec);
    if (ec) {
        LOGE("Check path %s exists failed %s", p.c_str(), ec.message().c_str());
        return false;
    }
    return res;
}

std::string_view GetLastExistPath(std::string_view path)
{
    if (IsExist(path)) {
        return path;
    }
    std::string_view last = PATH_SEP;
    auto i = path.find(PATH_SEP, 1);
    while (i != std::string_view::npos) {
        auto cur = path.substr(0, i);
        if (!IsExist(cur)) {
            break;
        }
        last = cur;
        i = path.find(PATH_SEP, i + 1);
    }
    return last;
}

inline bool IsRootUser()
{
    constexpr __uid_t root = 0;
    return getuid() == root;
}

bool CheckPermission(std::string_view path)
{
    if (IsRootUser()) {
        return true;
    }
    struct stat fileStat{};
    std::string pathStr{path};
    if (stat(pathStr.data(), &fileStat) != 0) {
        LOGE("Get Path permission error: %s", pathStr.c_str());
        return false;
    }
    if ((fileStat.st_mode & S_IWOTH) || (fileStat.st_mode & S_IWGRP)) {
        LOGE("Path %s cannot be writable by group or other users", pathStr.c_str());
        return false;
    }
    if (!(fileStat.st_mode & S_IRUSR) && !(fileStat.st_mode & S_IXUSR)) {
        LOGE("Path %s is not readable or executable", pathStr.c_str());
        return false;
    }
    if (fileStat.st_uid == 0 || fileStat.st_uid == getuid()) {
        return true;
    }
    LOGE("Path %s is not belong to current user or root", pathStr.c_str());
    return false;
}

bool CheckInvalidChar(std::string_view path)
{
    auto &invalidChars = GetInvalidChars();
    for (auto c : path) {
        if (auto it = invalidChars.find(c); it != invalidChars.cend()) {
            LOGE("Path contains invalid character %s", it->second.c_str());
            return false;
        }
    }
    return true;
}

bool IsSafePath(std::string_view path)
{
    std::string_view existPath = GetLastExistPath(path);
    if (!CheckPermission(existPath)) {
        return false;
    }
    return true;
}

bool MkdirRecursively(std::string_view path)
{
    std::error_code ec;
    std::filesystem::path absPath = path;
    if (IsExist(path)) {
        if (std::filesystem::is_directory(absPath, ec) && !ec) {
            return ec.value() == 0;
        }
        if (ec) {
            LOGE("Check is dir failed: %s", ec.message().c_str());
        }
        return false;
    }
    std::string cur = "/";
    cur.reserve(path.size());
    size_t slow = 1;
    size_t fast = path.find(PATH_SEP, slow);
    for (; slow < path.size(); slow = fast + 1, fast = path.find(PATH_SEP, slow)) {
        cur.append(path.substr(slow, fast - slow));
        cur.append(PATH_SEP);
        if (!IsExist(cur)) {
            if ((!std::filesystem::create_directory(cur, ec) && ec) ||
                chmod(cur.c_str(), SAVE_DIR_AUTHORITY) != 0) {
                std::string filteredCur = ReplaceInvalidChars(cur);
                LOGE("Create dir %s failed: %s", filteredCur.c_str(), ec.message().c_str());
                return false;
            }
        }
        if (fast == std::string_view::npos) {
            break;
        }
    }
    return true;
}
} // namespace

void Metrics::Add(const std::shared_ptr<OpConfig>& opConfig, Library::Operation *op)
{
    Metric metric{};
    metric.SetField<ClassicMetric::DEVICE_ID>(deviceId_);
    metric.SetField<ClassicMetric::CASE_ID>(metrics_.size() + 1);
    metric.SaveOperator(op);
    opConfig->SaveMetric(metric);
    metrics_.emplace_back(metric);
    for (auto &field : metric.Fields()) {
        extraHeads_.insert(field.first);
    }
}

bool Metrics::SetOutputPath(std::string_view output)
{
    std::string absPath = StandardizePath(output);
    if (absPath.empty() || absPath.back() == '/') {
        LOGE("--output is not a valid file path");
        return false;
    }
    constexpr size_t TAIL_LEN = 4;
    if (absPath.size() < TAIL_LEN || absPath.find(".csv", absPath.size() - TAIL_LEN) == std::string::npos) {
        absPath.append(".csv");
    }
    // check file security
    if (IsExist(absPath)) {
        if (IsSoftLink(absPath)) {
            LOGE("--output cannot be a soft link");
            return false;
        } else if (!IsSafePath(absPath)) {
            return false;
        } else if (std::error_code ec; std::filesystem::is_directory(absPath, ec) && !ec) {
            LOGE("--output cannot be an existing directory: %s", absPath.c_str());
            return false;
        }
    }
    std::string_view absView = absPath;
    auto sep = absView.rfind(PATH_SEP);
    std::string_view dir = absView.substr(0, sep);
    if (!CheckInvalidChar(absView) || !IsSafePath(dir) || !MkdirRecursively(dir)) {
        return false;
    }
    outputPath_ = std::move(absPath);
    LOGI("Set profile output file %s", outputPath_.c_str());
    return true;
}

void Metrics::PrintTop10(const std::string &head)
{
    std::vector<Metric> tmp = metrics_;
    auto normalEnd = std::remove_if(tmp.begin(), tmp.end(), [](const Metric &m) {
        return m.GetTaskDuration() == 0;
    });
    std::sort(tmp.begin(), normalEnd, [](const Metric &l, const Metric &r) {
        return l.GetTaskDuration() < r.GetTaskDuration();
    });
    constexpr size_t NUM = 10;
    LOGM("%sTop %lu:\n%s", DIVIDE.data(), NUM, head.c_str());
    for (size_t i = 0; i < std::min(NUM, tmp.size()); ++i) {
        LOGM("%s", tmp[i].ToString().c_str());
    }
}

void Metrics::Dump()
{
    std::string head = GetHead();
    std::vector<std::string> extraHeads{extraHeads_.begin(), extraHeads_.end()};
    for (auto &metric : metrics_) {
        for (auto &s : extraHeads) {
            // Field returns "" when key not exist, using this to set default value and align with column names
            metric.SetField(s, metric.Field(s));
        }
    }
    PrintTop10(head);
    if (outputPath_.empty()) {
        return;
    }
    std::ofstream file(outputPath_);
    if (!file.is_open() || chmod(outputPath_.c_str(), SAVE_DATA_FILE_AUTHORITY) != 0) {
        LOGE("Create file %s failed", outputPath_.c_str());
        return;
    }
    std::ostream_iterator<std::string> output_iterator(file, "\n");
    output_iterator++ = head;
    std::transform(metrics_.begin(), metrics_.end(), output_iterator, [&](Metric& metric) {
        return metric.ToString();
    });
    file.close();
    LOGI("Save profile data to %s success", outputPath_.c_str());
}

void Metrics::SetDurationAndPrint(double duration)
{
    if (durationIdx_ >= metrics_.size()) {
        LOGE("SetDuration idx %lu > metrics size", durationIdx_);
        return;
    }
    metrics_[durationIdx_].SetField<ClassicMetric::TASK_DURATION>(duration);
    LOGM("%s\n%s", DIVIDE.data(), metrics_[durationIdx_].ToTerminalString().c_str());
    ++durationIdx_;
}

std::string Metrics::GetHead()
{
    std::string head{HEAD};
    for (auto &s : extraHeads_) {
        head.append(",");
        head.append(s);
    }
    return head;
}

} // namespace Catlass
