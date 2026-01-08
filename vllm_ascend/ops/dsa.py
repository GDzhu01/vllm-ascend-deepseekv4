# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from typing import Optional

import torch
from torch import nn
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.layer import MLAAttention
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_config import get_ascend_config


from dataclasses import dataclass

import torch

from vllm.attention.layer import MLAAttention
from vllm.config import CacheConfig
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization import QuantizationConfig


@dataclass
class DSAModules:
    """Modules used in SFA V2."""

    wq_a: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    wkv: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    indexer: torch.nn.Module | None
    compressor: torch.nn.Module | None
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None

@CustomOp.register("deepseek_sparse_attention")
class DeepseekSparseAttentionWrapper(CustomOp):
    """MLA layer registered as CustomOp to allow OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently MLA ignores the enable/disable mechanism of CustomOp
    because there is only one in-tree implementation in forward_native.
    TODO: implement this with a new PluggableLayer mechanism.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        dsa_modules: DSAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

    def forward_native(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hidden_states
    
    def forward_cuda(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)
    

class AscendDeepseekSparseAttention(DeepseekSparseAttentionWrapper):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        dsa_modules: DSAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.dim=dim
        self.n_heads=n_heads
        self.scale=scale
        self.n_local_heads=n_local_heads
        self.o_lora_rank=o_lora_rank
        self.head_dim=head_dim 
        self.rope_head_dim=rope_head_dim
        self.nope_head_dim=nope_head_dim
        self.n_group=n_group
        self.n_local_groups=n_local_groups
        self.window_size = window_size
        self.compress_ratio=compress_ratio
        
        self.wq_a = dsa_module.wq_a
        self.q_norm = dsa_module.q_norm
        self.wq_b = dsa_module.wq_b
        self.wkv = dsa_module.wkv
        self.kv_norm = dsa_module.kv_norm
        self.wo_a = dsa_module.wo_a
        self.wo_b = dsa_module.wo_b
        self.indexer = dsa_module.indexer
        self.compressor = dsa_module.compressor
        self.topk_indices_buffer = dsa_module.topk_indices_buffer
        self.indexer_rotary_emb = dsa_module.indexer_rotary_emb


        self.dsa_attn = DSAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        need_gather_q_kv = get_forward_context().sp_enabled
        output_shape = hidden_states.shape
        # FIXME: This does not seem right, should make sure the buffer is fixed
        output = torch.empty(output_shape,
                             dtype=hidden_states.dtype,
                             device=hidden_states.device)
        torch.ops.vllm.dsa_forward(hidden_states, need_gather_q_kv, output,
                                   self.prefix)
        output = output.view(-1, output_shape[-1])
        return output


def dsa_forward(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if forward_context.attn_metadata:
        attn_metadata = forward_context.attn_metadata[self.dsa_attn.layer_name]
    else:
        attn_metadata = forward_context.attn_metadata
    kv_cache = self.dsa_attn.kv_cache[forward_context.virtual_engine]
    self.dsa_attn.impl.forward(self.dsa_attn.layer_name, hidden_states,
                               kv_cache, attn_metadata, need_gather_q_kv,
                               output)
    return


def dsa_forward_fake(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="dsa_forward",
    op_func=dsa_forward,
    mutates_args=["output"],
    fake_impl=dsa_forward_fake,
    dispatch_key="PrivateUse1",
)
