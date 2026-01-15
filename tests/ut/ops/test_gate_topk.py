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

import torch
import torch_npu
import custom_ops
import torch.nn.functional as F

from tests.ut.base import TestBase


def gating_topk_ref(scores, topk, bias, input_ids, tid2eid, route_scale, norm_type="softplus"):
    if norm_type == "softmax":
        scores = scores.softmax(dim=-1)
    elif norm_type == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    original_scores = scores
    if bias is not None:
        scores = scores + bias
    if tid2eid is not None: # Note: if hash
        indices = tid2eid[input_ids]
    else:
        indices = scores.topk(topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    if norm_type != "softmax":
        weights /= weights.sum(dim=-1, keepdim=True)
    weights *= route_scale
    return weights, indices


class TestMoeGatingTopk(TestBase):
    def setUp(self):
        torch.manual_seed(42)
    
        self.use_hash = True
        self.input_size = 512

        self.n_activated_experts = 6
        self.vocab_size = 129280
        self.n_routed_experts = 256
        self.route_scale = 2.0
        self.norm_type = "softplus"
        self.norm_type_int = 2 # 0-Softmax，1-Sigmoid，2-Softplus

    def test_cumsum_group_list_with_type_0(self):

        torch.npu.set_device(0)
        scores = torch.randn((self.input_size,self.n_routed_experts), dtype=torch.float32).npu()
        scores_ref = scores.clone()
        input_ids = torch.randint(0, self.vocab_size, (self.input_size,),dtype=torch.int64).npu()
        if self.use_hash:
            tid2eid = torch.empty(self.vocab_size, self.n_activated_experts, dtype=torch.int32).npu()
            bias = None
        else:
            bias = torch.empty(self.n_routed_experts, dtype=torch.float32).npu()

        ns = torch.ops.custom
        ops = [name for name in dir(ns) if not name.startswith("_")]
        print(f'ops: {ops}')
        for op_name in sorted(ops):
            op = getattr(ns, op_name)
            print(f"\n== custom::{op_name} ==")
            # 有些对象能 dir 出 overload 名字
            overloads = [n for n in dir(op) if not n.startswith("_")]
            # 过滤掉一些明显不是 overload 的属性（经验规则，不保证 100%）
            overloads = [n for n in overloads if n not in ("default",)]
            if hasattr(op, "default"):
                print("  - overload: default")
            for ov in overloads:
                print(f"  - overload: {ov}")
        
        weights, indices, _ = torch.ops.custom.npu_moe_gating_top_k(
            x=scores, 
            k=self.n_activated_experts,
            bias=bias,
            input_ids=input_ids,
            tid2eid=tid2eid,
            routed_scaling_factor=self.route_scale,
            norm_type=self.norm_type_int
            )

        weights_ref, indices_ref = gating_topk_ref(scores_ref, self.n_activated_experts, bias,
            input_ids, tid2eid, self.route_scale, self.norm_type)

        self.assertTrue(torch.allclose(weights, weights_ref))
        self.assertTrue(torch.equal(indices, indices_ref))