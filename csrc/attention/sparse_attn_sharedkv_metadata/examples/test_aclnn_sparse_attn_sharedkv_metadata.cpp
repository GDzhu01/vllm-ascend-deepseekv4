/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <stdio.h>
#include <iostream>
#include <tuple>
#include <vector>
#include "acl/acl.h"
//#include "aclnnop/aclnn_sparse_attn_sharedkv_metadata.h"
#include "../../sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_metadata.h"
#include "../op_api/aclnn_sparse_attn_sharedkv_metadata.h"

static const uint32_t batchSize = 4;
static const uint32_t numHeadsQ = 128;
static const uint32_t numHeadsKV = 1;
static const uint32_t headDim = 512;
static const uint32_t topK = 128;
static const uint32_t cmpRatio = 4;
static const uint32_t oriMaskMode = 4;
static const uint32_t cmpMaskMode = 3;
static const uint32_t oriWinLeft = 128;
static const uint32_t oriWinRight = 0;
static const bool hasOriKv = true;
static const bool hasCmpKv = true;
static std::string layoutQuery = "BSND";
static std::string layoutKV = "PA_ND";
static std::string socVersion = "ascend910B";
static const uint32_t aicCoreNum = 24;
static const uint32_t aivCoreNum = 48;

static const std::vector<int32_t> cuSeqlensQ = {3, 6, 9, 12};
static const std::vector<int32_t> sequsedKv = {10240, 10240, 10240, 10240};
static const std::vector<int64_t> cuSeqlensQShape = {batchSize};
static const std::vector<int64_t> sequsedKvShape = {batchSize};
static const std::vector<int64_t> cuSeqlensQStride = {1};
static const std::vector<int64_t> sequsedKvStride = {1};
static const std::vector<int64_t> metadataShape = {optiling::SCFA_META_SIZE};
static const std::vector<int64_t> metadataStride = {1};

static const std::string dumpFile = "./dump";

static const bool enableCuSeqlensQ = true;
static const bool enableSequsedKv = true;

std::tuple<aclTensor*, void*> CreateTensor(size_t size,  // in bytes
                                           std::vector<int64_t> shape,
                                           std::vector<int64_t> stride,
                                           aclDataType dType,
                                           const void* hostData = nullptr) {
  void* devicePtr = nullptr;
  auto ret = aclrtMalloc(&devicePtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    printf("aclrtMalloc %d\n", ret);
    return {nullptr, nullptr};
  }

  aclTensor* tensor = aclCreateTensor(&shape[0], shape.size(), dType,
                                      &stride[0], 0, aclFormat::ACL_FORMAT_ND,
                                      &shape[0], shape.size(), devicePtr);
  if (tensor == nullptr) {
    aclrtFree(devicePtr);
    return {nullptr, nullptr};
  }

  if (hostData != nullptr) {
    aclrtMemcpy(devicePtr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
  }
  return {tensor, devicePtr};
}

static void DumpMeta(void *data) {
  optiling::detail::ScfaMetaData *metaDataPtr =
      (optiling::detail::ScfaMetaData *)data;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  aclError ret = 0;
  aclTensor* qSeqLenTensor = nullptr;
  void* qSeqLenDevPtr = nullptr;
  aclTensor* kvSeqLenTensor = nullptr;
  void* kvSeqLenDevPtr = nullptr;
  aclTensor* metadataTensor = nullptr;
  void* metadataDevPtr = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  void* workspace = nullptr;

  ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    printf("aclInit %d\n", ret);
    return -1;
  }

  ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) {
    printf("aclrtSetDevice %d\n", ret);
    return -1;
  }

  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) {
    printf("aclrtCreateStream %d\n", ret);
    return -1;
  }

  if (enableCuSeqlensQ) {
    std::tie(qSeqLenTensor, qSeqLenDevPtr) = CreateTensor(
        cuSeqlensQ.size() * sizeof(cuSeqlensQ[0]), cuSeqlensQShape,
        cuSeqlensQStride, aclDataType::ACL_INT32, &cuSeqlensQ[0]);
    if (qSeqLenTensor == nullptr) {
      return -1;
    }
  }

  if (enableSequsedKv) {
    std::tie(kvSeqLenTensor, kvSeqLenDevPtr) = CreateTensor(
        sequsedKv.size() * sizeof(sequsedKv[0]), sequsedKvShape,
        sequsedKvStride, aclDataType::ACL_INT32, &sequsedKv[0]);
    if (kvSeqLenTensor == nullptr) {
      return -1;
    }
  }


  std::tie(metadataTensor, metadataDevPtr) =
      CreateTensor(sizeof(int32_t) * optiling::SCFA_META_SIZE, metadataShape,
                   metadataStride, aclDataType::ACL_INT32);
  if (metadataTensor == nullptr) {
    return -1;
  }

  ret = aclnnSparseAttnSharedkvMetadataGetWorkspaceSize(
      qSeqLenTensor, kvSeqLenTensor, batchSize, numHeadsQ, numHeadsKV, headDim,
      topK, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
      &layoutQuery[0], &layoutKV[0], hasOriKv, hasCmpKv, &socVersion[0],
      aicCoreNum, aivCoreNum, metadataTensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    printf("aclnnSparseAttnSharedkvMetadataGetWorkspaceSize %d\n",
           ret);
    return -1;
  }

  ret = aclnnSparseAttnSharedkvMetadata(workspace, workspaceSize,
                                                 executor, stream);
  if (ret != ACL_SUCCESS) {
    printf("aclnnSparseAttnSharedkvMetadata %d\n", ret);
    return -1;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    printf("aclrtSynchronizeStream %d\n", ret);
    return -1;
  }

  std::vector<int32_t> metdataHost(optiling::SCFA_META_SIZE);
  ret = aclrtMemcpy(metdataHost.data(),
                    metdataHost.size() * sizeof(metdataHost[0]), metadataDevPtr,
                    optiling::SCFA_META_SIZE * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    printf("aclrtMemcpy %d\n", ret);
    return -1;
  }

  DumpMeta(&metdataHost[0]);

  aclDestroyTensor(qSeqLenTensor);
  aclDestroyTensor(kvSeqLenTensor);
  aclDestroyTensor(metadataTensor);

  aclrtFree(qSeqLenDevPtr);
  aclrtFree(kvSeqLenDevPtr);
  aclrtFree(metadataDevPtr);
  aclrtFree(workspace);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
