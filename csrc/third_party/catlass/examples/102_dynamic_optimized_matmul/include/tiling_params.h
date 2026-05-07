/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILING_PARAMS_H
#define TILING_PARAMS_H

#include <cstdint>

enum class LayoutTag : uint8_t { TagRowMajor = 0, TagColumnMajor = 1};

/*
 * Bit field layout description (little-endian):
 * -------------------------------------------------------------------------
 * | Bit Range | Size | Field Name            | Description                |
 * |-----------|------|-----------------------|----------------------------|
 * | 0-3       | 4    | layoutTagC            | Layout tag for C matrix    |
 * | 4-7       | 4    | layoutTagB            | Layout tag for B matrix    |
 * | 8-11      | 4    | layoutTagA            | Layout tag for A matrix    |
 * | 12-15     | 4    | paddingTagC           | Padding tag for C matrix   |
 * | 16-19     | 4    | paddingTagB           | Padding tag for B matrix   |
 * | 20-23     | 4    | paddingTagA           | Padding tag for A matrix   |
 * | 24-51     | 28   | reserveBit            | Reserved for future use    |
 * | 52-55     | 4    | dtype                 | Data type specification    |
 * | 56-63     | 8    | templateKernelSerial  | Template kernel serial ID  |
 * -------------------------------------------------------------------------
 */

union TilingKey {
    uint64_t value;
    struct {
        uint64_t layoutTagC : 4;  // 0-3
        uint64_t layoutTagB : 4;  // 4-7
        uint64_t layoutTagA : 4;  // 8-11
        uint64_t paddingTagC : 4; // 12-15
        uint64_t paddingTagB : 4; // 16-19
        uint64_t paddingTagA : 4; // 20-23
        uint64_t reserveBit : 28; // 24-51 May be used in the future
        uint64_t dtype : 4;       // 52-55
        uint64_t templateKernelSerial : 8; // 56-63
    } bits;

    TilingKey() : value(0) {}
    TilingKey(uint64_t v) : value(v) {}

    uint8_t GetLayoutTagA() const {return bits.layoutTagA;}
    uint8_t GetLayoutTagB() const {return bits.layoutTagB;}
    uint8_t GetLayoutTagC() const {return bits.layoutTagC;}
    uint8_t GetPaddingTagTagA() const {return bits.paddingTagA;}
    uint8_t GetPaddingTagTagB() const {return bits.paddingTagB;}
    uint8_t GetPaddingTagTagC() const {return bits.paddingTagC;}
    uint8_t GetDtype() const {return bits.dtype;}
    uint8_t GetKernelSerial() const {return bits.templateKernelSerial;}

    void SetKernelSerial(uint8_t kernelSerial) { bits.templateKernelSerial = kernelSerial;}
    void SetLayoutTagA(uint8_t layoutTagA) { bits.layoutTagA = layoutTagA & 0xF; }
    void SetLayoutTagB(uint8_t layoutTagB) { bits.layoutTagB = layoutTagB & 0xF; }
    void SetLayoutTagC(uint8_t layoutTagC) { bits.layoutTagC = layoutTagC & 0xF; }
    void SetPaddingTagA(uint8_t paddingTagA) { bits.paddingTagA = paddingTagA & 0xF; }
    void SetPaddingTagB(uint8_t paddingTagB) { bits.paddingTagB = paddingTagB & 0xF; }
    void SetPaddingTagC(uint8_t paddingTagC) { bits.paddingTagC = paddingTagC & 0xF; }
    void SetDtype(uint8_t dtype) { bits.dtype = dtype & 0xF; }

    void SetTilingKey(uint8_t kernelSerial, uint8_t layoutTagA, uint8_t layoutTagB, uint8_t layoutTagC,
        uint8_t paddingTagA, uint8_t paddingTagB, uint8_t paddingTagC, uint8_t dtype = 0)
    {
        SetKernelSerial(kernelSerial);
        SetLayoutTagA(layoutTagA);
        SetLayoutTagB(layoutTagB);
        SetLayoutTagC(layoutTagC);
        SetPaddingTagA(paddingTagA);
        SetPaddingTagB(paddingTagB);
        SetPaddingTagC(paddingTagC);
        SetDtype(dtype);
    }
};

struct TilingParams {
    uint64_t strideA{0};
    uint64_t strideB{0};
    uint64_t strideC{0};
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    uint16_t m1{0};
    uint16_t n1{0};
    uint16_t k1{0};
    uint8_t swizzleOffset{1};
    uint8_t swizzleDirection{0};
    uint16_t splitkFactor{1};
    uint16_t m0{0};
    uint16_t n0{0};
    uint16_t k0{0};
    uint16_t m1Factor{1};
    uint16_t n1Factor{1};
    // The following parameters are only used in tiling and are not read by the kernel.
    uint8_t layoutTagA{0};
    uint8_t layoutTagB{0};
    uint8_t layoutTagC{0};
    uint8_t paddingTagA{0};
    uint8_t paddingTagB{0};
    uint8_t paddingTagC{0};
    uint8_t blockDim{0};
    TilingKey tilingKey{0};

    TilingParams() {}

    TilingParams(uint32_t m_, uint32_t n_, uint32_t k_, 
        LayoutTag layoutTagA_, LayoutTag layoutTagB_, LayoutTag layoutTagC_)
        : m(m_), n(n_), k(k_), layoutTagA(static_cast<uint8_t>(layoutTagA_)),
          layoutTagB(static_cast<uint8_t>(layoutTagB_)), layoutTagC(static_cast<uint8_t>(layoutTagC_))
    {
        strideA = k;
        strideB = n;
        strideC = n;
        if (static_cast<LayoutTag>(layoutTagA) == LayoutTag::TagColumnMajor) {
            strideA = m;
        }
        if (static_cast<LayoutTag>(layoutTagB) == LayoutTag::TagColumnMajor) {
            strideB = k;
        }
        if (static_cast<LayoutTag>(layoutTagC) == LayoutTag::TagColumnMajor) {
            strideC = m;
        }
    }

    TilingParams(uint32_t m_, uint32_t n_, uint32_t k_, size_t strideA_, size_t strideB_, size_t strideC_,
        LayoutTag layoutTagA_, LayoutTag layoutTagB_, LayoutTag layoutTagC_)
        : m(m_), n(n_), k(k_), strideA(strideA_), strideB(strideB_), strideC(strideC_),
          layoutTagA(static_cast<uint8_t>(layoutTagA_)), layoutTagB(static_cast<uint8_t>(layoutTagB_)),
          layoutTagC(static_cast<uint8_t>(layoutTagC_))
    {}

    void SetParams(uint32_t m_, uint32_t n_, uint32_t k_, size_t strideA_, size_t strideB_, size_t strideC_,
        LayoutTag layoutTagA_, LayoutTag layoutTagB_, LayoutTag layoutTagC_)
    {
        m = m_;
        n = n_;
        k = k_;
        strideA = strideA_;
        strideB = strideB_;
        strideC = strideC_;
        layoutTagA = static_cast<uint8_t>(layoutTagA_);
        layoutTagB = static_cast<uint8_t>(layoutTagB_);
        layoutTagC = static_cast<uint8_t>(layoutTagC_);
    }

    void SetParams(uint32_t m_, uint32_t n_, uint32_t k_, LayoutTag layoutTagA_, LayoutTag layoutTagB_, LayoutTag layoutTagC_)
    {
        m = m_;
        n = n_;
        k = k_;
        strideA = k;
        strideB = n;
        strideC = n;
        if (static_cast<LayoutTag>(layoutTagA) == LayoutTag::TagColumnMajor) {
            strideA = m;
        }
        if (static_cast<LayoutTag>(layoutTagB) == LayoutTag::TagColumnMajor) {
            strideB = k;
        }
        if (static_cast<LayoutTag>(layoutTagC) == LayoutTag::TagColumnMajor) {
            strideC = m;
        }
        layoutTagA = static_cast<uint8_t>(layoutTagA_);
        layoutTagB = static_cast<uint8_t>(layoutTagB_);
        layoutTagC = static_cast<uint8_t>(layoutTagC_);
    }
};

#endif // TILING_PARAMS_H
