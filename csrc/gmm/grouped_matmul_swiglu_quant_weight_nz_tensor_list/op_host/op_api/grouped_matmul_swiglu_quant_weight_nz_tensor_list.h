#ifndef OP_API_INC_LEVEL0_OP_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_OP_H
#define OP_API_INC_LEVEL0_OP_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_OP_H

#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<aclTensor*, aclTensor*> GroupedMatmulSwigluQuantWeightNzTensorList(const aclTensor *x,
                                                                  const aclTensorList *weight,
                                                                  const aclTensorList *perChannelScale,
                                                                  const aclTensor *perTokenScale,
                                                                  const aclTensor *groupList,
                                                                  aclOpExecutor *executor);
}

#endif