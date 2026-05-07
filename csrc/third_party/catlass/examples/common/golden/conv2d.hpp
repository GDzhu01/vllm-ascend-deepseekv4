/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_COMMON_GOLDEN_CONV2D_HPP
#define EXAMPLES_COMMON_GOLDEN_CONV2D_HPP

#include <vector>

#include "catlass/conv_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::golden {

// simple conv2d
template <class ElementFmap, class LayoutFmap, class ElementFilter, class LayoutFilter, class ElementGolden, class LayoutGolden>
void ComputeConv2d(
    const Conv2dParams &params,
    const std::vector<ElementFmap> &dataFmap,
    const LayoutFmap &layoutFmap,
    const std::vector<ElementFilter> &dataFilter,
    const LayoutFilter &layoutFilter,
    std::vector<ElementGolden> &dataGolden,
    const LayoutGolden &layoutGolden
)
{
    for (uint32_t batch = 0; batch < params.batch(); batch++) {
        for (uint32_t ho = 0; ho < params.ho(); ho++) {
            for (uint32_t wo = 0; wo < params.wo(); wo++) {
                for (uint32_t cout = 0; cout < params.cout(); cout++) {
                    ElementGolden accumulator = 0;
                    for (uint32_t kh = 0; kh < params.kh(); kh++) {
                        for (uint32_t kw = 0; kw < params.kw(); kw++) {
                            uint32_t hi = ho * params.strideH() + kh * params.dilationH() - params.padTop();
                            uint32_t wi = wo * params.strideW() + kw * params.dilationW() - params.padLeft();
                            if (hi < 0 || hi > params.hi() - 1 || wi < 0 || wi > params.wi() - 1)
                                continue;
                            for (uint32_t cin1 = 0; cin1 < params.cin1(); cin1++) {
                                for (uint32_t c0 = 0; c0 < params.C0; c0++) {
                                    accumulator += static_cast<ElementGolden>(
                                                       dataFmap
                                                           [batch * params.cin1() * params.hi() * params.wi() * params.C0
                                                            + cin1 * params.hi() * params.wi() * params.C0
                                                            + hi * params.wi() * params.C0 + wi * params.C0 + c0]
                                                   )
                                                   * static_cast<ElementGolden>(
                                                       dataFilter
                                                           [cin1 * params.kh() * params.kw() * params.cout() * params.C0
                                                            + kh * params.kw() * params.cout() * params.C0
                                                            + kw * params.cout() * params.C0 + cout * params.C0 + c0]
                                                   );
                                }
                            }
                        }
                    }
                    uint32_t cout1 = cout / params.C0;
                    uint32_t c0 = cout - cout1 * params.C0;
                    dataGolden
                        [batch * params.cout1() * params.ho() * params.wo() * params.C0
                         + cout1 * params.ho() * params.wo() * params.C0 + ho * params.wo() * params.C0 + wo * params.C0
                         + c0] = static_cast<ElementGolden>(accumulator);
                }
            }
        }
    }
}

} // namespace Catlass::golden

#endif // EXAMPLES_COMMON_GOLDEN_CONV2D_HPP
