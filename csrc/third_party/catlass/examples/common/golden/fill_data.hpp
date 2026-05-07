/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef EXAMPLES_COMMON_GOLDEN_FILL_DATA_HPP
#define EXAMPLES_COMMON_GOLDEN_FILL_DATA_HPP

#include <stack>
#include <vector>
#include <cstdlib>
#include <ctime>

namespace Catlass::golden {

template <class Element, class ElementRandom>
void FillRandomData(std::vector<Element>& data, ElementRandom low, ElementRandom high)
{
    for (uint64_t i = 0; i < data.size(); ++i) {
        ElementRandom randomValue = low +
            (static_cast<ElementRandom>(rand()) / static_cast<ElementRandom>(RAND_MAX)) * (high - low);
        data[i] = static_cast<Element>(randomValue);
    }
}

template <>
void FillRandomData<int8_t, int>(std::vector<int8_t>& data, int low, int high)
{
    for (uint64_t i = 0; i < data.size(); ++i) {
        int randomValue = low + rand() % (high - low + 1);
        data[i] = static_cast<int8_t>(randomValue);
    }
}


template <typename T>
void QuickSort(std::vector<T>& arr, int left, int right)
{
    std::stack<std::pair<int, int>> stk;
    stk.push({left, right});
    
    while (!stk.empty()) {
        auto [l, r] = stk.top();
        stk.pop();
        
        if (l >= r) {
            continue;
        }
        
        T pivot = arr[(l + r) / 2];
        int i = l;
        int j = r;  
        while (i <= j) {
            while (arr[i] < pivot) {
                i++;
            }
            while (arr[j] > pivot) {
                j--;
            }
            if (i <= j) {
                std::swap(arr[i], arr[j]);
                i++;
                j--;
            }
        }

        if (l < j) {
            stk.push({l, j});
        }
        if (i < r) {
            stk.push({i, r});
        }
    }
}

// Generate an ascending random sequence as grouplist
// If CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED is defined, convert to segmented form
// Otherwise, keep prefix-sum form
template <typename T = int32_t>
std::vector<T> GenerateGroupList(uint32_t m, uint32_t problemCount)
{
    std::vector<T> groupList(problemCount);
    std::srand(0);
    for (int i = 0; i < problemCount; ++i) {
        groupList[i] = rand() % (m + 1);
    }
    QuickSort(groupList, 0, groupList.size() - 1);

#ifdef CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED
    for (int i = problemCount - 1; i > 0; --i) {
        groupList[i] -= groupList[i - 1];
    }
#endif

    return groupList;
}

} // namespace Catlass::golden

#endif // EXAMPLES_COMMON_GOLDEN_FILL_DATA_HPP
