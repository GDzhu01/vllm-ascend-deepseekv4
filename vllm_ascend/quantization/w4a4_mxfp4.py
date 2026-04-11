#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from typing import Any, Callable, Dict, Optional

import torch
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context

from vllm_ascend.flash_common3_context import get_flash_common3_context
from vllm_ascend.mxfp_compat import (FLOAT4_E2M1FN_X2_DTYPE,
                                     FLOAT8_E8M0FNU_DTYPE)
from vllm_ascend.ops.fused_moe.experts_selector import (select_experts,
                                                        zero_experts_compute)
from vllm_ascend.utils import (AscendDeviceType, get_ascend_device_type)

from .w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod


class AscendW4A4MXFP4DynamicFusedMoEMethod(AscendW8A8DynamicFusedMoEMethod):
    """Fused MoE method for FP4 experts with MX per-group scales on A5."""

    def __init__(self, tid2eid=None):
        super().__init__(tid2eid=tid2eid)
        if get_ascend_device_type() != AscendDeviceType.A5:
            raise RuntimeError("MoE MXFP4 quantization is only supported on Ascend A5.")
        # Read group_size from quant config instead of inheriting W8A8 defaults.
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get(
            "group_size", 32)

    @staticmethod
    def get_weight(num_experts: int, intermediate_size_per_partition: int,
                   hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "w13_weight":
            torch.empty(num_experts,
                        2 * intermediate_size_per_partition,
                        hidden_sizes // 2,
                        dtype=torch.uint8),
            "w2_weight":
            torch.empty(num_experts,
                        hidden_sizes,
                        intermediate_size_per_partition // 2,
                        dtype=torch.uint8),
        }

    def get_dynamic_quant_param(self, num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "w13_weight_scale":
            torch.empty(num_experts,
                        2 * intermediate_size_per_partition,
                        hidden_sizes // self.group_size,
                        dtype=torch.uint8),
            "w2_weight_scale":
            torch.empty(num_experts,
                        hidden_sizes,
                        intermediate_size_per_partition // self.group_size,
                        dtype=torch.uint8),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
        n_shared_experts = layer.n_shared_experts
        valid_global_expert_num = (global_num_experts -
                                   global_redundant_expert_num -
                                   n_shared_experts)
        if zero_expert_num == 0 or zero_expert_type is None:
            assert router_logits.shape[1] == valid_global_expert_num, \
                "Number of global experts mismatch (excluding redundancy)"

        if self.multistream_overlap_gate:
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            topk_weights = fc3_context.topk_weights
            topk_ids = fc3_context.topk_ids
        else:
            topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                routed_scaling_factor=routed_scaling_factor,
                mix_placement=layer.mix_placement,
                num_logical_experts=router_logits.shape[1],
                num_shared_experts=n_shared_experts,
                global_num_experts=global_num_experts,
                tid2eid=self.tid2eid)

        assert topk_ids is not None
        assert topk_weights is not None
        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=global_num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0),
                                       global_num_experts -
                                       global_redundant_expert_num,
                                       device=topk_ids.device)
            topk_ids = torch.argsort(
                random_matrix, dim=1)[:, :topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(self.in_dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        final_hidden_states = moe_comm_method.fused_experts(
            hidden_states=x,
            pertoken_scale=pertoken_scale,
            w1=layer.w13_weight,
            w1_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_mxfp4_moe=True,
            expert_map=expert_map,
            log2phy=log2phy,
            dynamic_eplb=self.dynamic_eplb,
            mc2_mask=kwargs.get("mc2_mask", None),
            act_quant_type=FLOAT4_E2M1FN_X2_DTYPE,
            weight_quant_type=FLOAT4_E2M1FN_X2_DTYPE,
            scale_type=FLOAT8_E8M0FNU_DTYPE,
            per_token_scale_type=FLOAT8_E8M0FNU_DTYPE,
            use_bf16=x.dtype == torch.bfloat16)
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states

    def process_weights_after_loading(self, layer):
        g_num, n_size, k_size = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(
            g_num, n_size, k_size // 2, 2)
        g_num, n_size, k_size = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(
            g_num, n_size, k_size // 2, 2)
        layer.w13_weight.data = layer.w13_weight.data.transpose(
            1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(
            1, 2).contiguous()
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(
            1, 2).contiguous()
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(
            1, 2).contiguous()
