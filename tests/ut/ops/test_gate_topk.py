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
import torch.nn.functional as F

from tests.ut.base import TestBase
from vllm_ascend.utils import enable_custom_op

import torch
import torch_npu
import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F
enable_custom_op()

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))
        

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

        print(f'torch.ops._C_ascend.moe_gating_top_k:{torch.ops._C_ascend.moe_gating_top_k}')
        weights, indices, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
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




class TestAddRmsNromBias(TestBase):
    def setUp(self):
        torch.manual_seed(42)

    def test_add_rms_norm_bias(self):

        torch.npu.set_device(0)

        print(f'torch.ops._C_ascend.npu_add_rms_norm_bias:{torch.ops._C_ascend.npu_add_rms_norm_bias}')


def _hc_post_cpu(x, residual, post, comb):
    data_type = x.dtype
    x = x.float()
    residual = residual.float()
    post = post.float()
    comb = comb.float()
    hc = residual.shape[2]
    out_shape = list(residual.shape)
    out = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    out = out.to(data_type)
    out = out.reshape(out_shape)
    return out


class TestHcPost(TestBase):
    def setUp(self):
        torch.manual_seed(42)

    def test_hc_post_b4_s4(self):
        b = 4
        s = 4
        hc = 4
        d = 4096

        np.random.seed(0)
        x = torch.tensor(np.random.uniform(-10, 10, (b, s, d))).to(torch.bfloat16)
        residual = torch.tensor(np.random.uniform(-10, 10, (b, s, hc, d))).to(torch.bfloat16)
        post = torch.tensor(np.random.uniform(-10, 10, (b, s, hc))).to(torch.bfloat16)
        comb = torch.tensor(np.random.uniform(-10, 10, (b, s, hc, hc))).to(torch.bfloat16)
        cpuout = _hc_post_cpu(x, residual, post, comb)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)
        residual = residual.to("npu:%s" % DEVICE_ID)
        post = post.to("npu:%s" % DEVICE_ID)
        comb = comb.to("npu:%s" % DEVICE_ID)
        # start run custom ops
        print(f'======================== PTA eager BEGIN ========================')
        npu_out = torch.ops._C_ascend.npu_hc_post(x, residual, post, comb)

        # compare result
        npu_out = npu_out.reshape(b, s, hc, d).cpu().float()
        cpuout = cpuout.reshape(b, s, hc, d).float()
        for i in range(b):
            for j in range(s):
                for k in range(hc):
                    for l in range(d):
                        if torch.abs(npu_out[i][j][k][l] - cpuout[i][j][k][l]):
                            print("i j k l npu cpu = ", i, j, k, l, npu_out[i][j][k][l], cpuout[i][j][k][l])
        print(f'======================== PTA eager FINISH ========================')

    def test_hc_post_b4_s4_float(self):
        b = 4
        s = 4
        hc = 4
        d = 4096

        np.random.seed(0)
        x = torch.tensor(np.random.uniform(-10, 10, (b, s, d))).to(torch.float)
        residual = torch.tensor(np.random.uniform(-10, 10, (b, s, hc, d))).to(torch.float)
        post = torch.tensor(np.random.uniform(-10, 10, (b, s, hc))).to(torch.float)
        comb = torch.tensor(np.random.uniform(-10, 10, (b, s, hc, hc))).to(torch.float)
        cpuout = _hc_post_cpu(x, residual, post, comb)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)
        residual = residual.to("npu:%s" % DEVICE_ID)
        post = post.to("npu:%s" % DEVICE_ID)
        comb = comb.to("npu:%s" % DEVICE_ID)
        # start run custom ops
        print(f'======================== PTA eager BEGIN ========================')
        npu_out = torch.ops._C_ascend.npu_hc_post(x, residual, post, comb)

        # compare result
        npu_out = npu_out.reshape(b, s, hc, d).cpu().float()
        cpuout = cpuout.reshape(b, s, hc, d).float()
        for i in range(b):
            for j in range(s):
                for k in range(hc):
                    for l in range(d):
                        if torch.abs(npu_out[i][j][k][l] - cpuout[i][j][k][l]) > 0.001:
                            print("i j k l npu cpu = ", i, j, k, l, npu_out[i][j][k][l], cpuout[i][j][k][l])
        print(f'======================== PTA eager FINISH ========================')
        
        
def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def data_compare(npu_out, cpu_out, diff_thd=0.001, pct_thd=0.005, max_diff_hd=0.001):
    real_data = npu_out.flatten()
    data_compe = cpu_out.flatten()
    start = 0
    end = real_data.size - 1
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        return result, 0.0, max_error
    
    split_count = int(end - start + 1) if end != start else 1
    diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    diff_index = np.where(diff_abs > 0)
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32),
                                 data_compe[diff_index].astype(np.float32), diff_thd)
    
    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    error_cnt = err_diff.size

    fulfill_num = split_count - error_cnt
    fulfill_percent = float(fulfill_num) / float(split_count) * 100.0

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    # if len(err_diff) > 0:
    #     max_error = max(err_diff)
    #     if max(err_diff) >= max_diff_hd:
    #         result = "Failed"

    return result, fulfill_percent, max_error

def hc_split_sinkhorn_torch(
        mixes: torch.Tensor, 
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6):
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + eps
    post = 2 * F.sigmoid(post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult].unsqueeze(0).unsqueeze(0))
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult).unsqueeze(0).unsqueeze(0)

    comb = comb.softmax(-1) + eps
    col_sum = comb.sum(-2, keepdim=True)
    comb = comb / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb

def _hc_pre(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int, hc_sinkhorn_iters: int, norm_eps: float, hc_eps: float):
    # x: [b, s, hc, d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b, s, d]
    shape, dtype = x.size(), x.dtype
    if x.dim() == 4:
        x = x.flatten(2).float()
    elif x.dim() == 3:
        x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt

    pre, post, comb = hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype), post, comb

class TestCustomHcPre(TestBase):
    def setUp(self):
        torch.manual_seed(42)

    def test_hc_pre_eager(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        b = 4
        s = 4
        hc_mix = 24
        hc_mult = 4
        d = 4096
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        np.random.seed(42)

        # create input tensor
        hc_scale = torch.tensor(np.random.uniform(-2, 2, (3))).to(torch.float32)
        hc_base = torch.tensor(np.random.uniform(-2, 2, (hc_mix))).to(torch.float32)
        hc_fn = torch.tensor(np.random.uniform(-2, 2, (hc_mix, hc_mult * d))).to(torch.float32)
        x = torch.tensor(np.random.uniform(-2, 2, (b, s, hc_mult, d))).to(torch.bfloat16)

        # to NPU
        hc_scale_npu = hc_scale.to("npu:%s" % DEVICE_ID)
        hc_base_npu = hc_base.to("npu:%s" % DEVICE_ID)
        hc_fn_npu = hc_fn.to("npu:%s" % DEVICE_ID)
        x_npu = x.to("npu:%s" % DEVICE_ID)

        print(f'======================== PTA eager test ========================')

        # Golden结果
        golden_yOut, golden_postOut, golden_comb_fragOut = _hc_pre(
            x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

        # NPU call
        npu_yOut, npu_postOut, npu_comb_fragOut = torch.ops._C_ascend.npu_hc_pre(
            x_npu, hc_fn_npu, hc_scale_npu, hc_base_npu, hc_mult=hc_mult, hc_sinkhorn_iters=hc_sinkhorn_iters, norm_eps=norm_eps, hc_eps=hc_eps)

        # to CPU
        npu_yOut_cpu = npu_yOut.cpu().float().numpy()
        npu_postOut_cpu = npu_postOut.cpu().float().numpy()
        npu_comb_fragOut_cpu = npu_comb_fragOut.cpu().float().numpy()

        golden_yOut_cpu = golden_yOut.cpu().float().numpy()
        golden_postOut_cpu = golden_postOut.cpu().numpy()
        golden_comb_fragOut_cpu = golden_comb_fragOut.cpu().numpy()

        # compare result
        compare_y = data_compare(golden_yOut_cpu, npu_yOut_cpu)
        compare_post = data_compare(golden_postOut_cpu, npu_postOut_cpu)
        compare_comb_frag = data_compare(golden_comb_fragOut_cpu, npu_comb_fragOut_cpu)

        assert(compare_y[0] == "Pass")
        assert(compare_post[0] == "Pass")
        assert(compare_comb_frag[0] == "Pass")
   

MAX_INT8_VALUE = 127
MIN_INT8_VALUE = -128

def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def data_compare(npu_out, cpu_out, diff_thd=0.0001, pct_thd=0.0005, max_diff_hd=0.0001):
    real_data = npu_out.flatten()
    data_compe = cpu_out.flatten()
    start = 0
    end = real_data.size - 1
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        return result, 0.0, max_error
    
    split_count = int(end - start + 1) if end != start else 1
    diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    diff_index = np.where(diff_abs > 0)
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32),
                                 data_compe[diff_index].astype(np.float32), diff_thd)
    
    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    error_cnt = err_diff.size

    fulfill_num = split_count - error_cnt
    fulfill_percent = float(fulfill_num) / float(split_count) * 100.0

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff)
        if max(err_diff) >= max_diff_hd:
            result = "Failed"

    return result, fulfill_percent, max_error


def _hc_pre_inv_rms(x, epsilon=1e-20):
    if x.dim() == 4: 
        x = x.flatten(2)
    elif x.dim() == 3:
        x = x.flatten(1)
    x = x.float()
    y = torch.rsqrt(x.square().mean(-1, keepdim = True) + epsilon)
    return y

class TestCustomHcPreInvRms(TestBase):
    def setUp(self):
        torch.manual_seed(42)

    def test_hc_pre_inv_rms_eager(self):
        b = 4
        s = 4
        hc = 4
        d = 4096
        eps = 1e-6

        np.random.seed(0)

        # start run custom ops
        print(f'======================== PTA eager BEGIN ========================')
        # float32 input test (flattened input)
        x = torch.tensor(np.random.uniform(-1, 1, (b, s, hc, d))).to(torch.float32)
        cpu_y = _hc_pre_inv_rms(x, epsilon=eps)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)

        npu_y = torch.ops._C_ascend.npu_hc_pre_inv_rms(x, epsilon=eps)
        
        compare_y = data_compare(cpu_y.float().numpy(), npu_y.cpu().float().numpy())
        assert(compare_y[0] == "Pass")


        # bfloat16 input test (flattened input)
        x = torch.tensor(np.random.uniform(-1, 1, (b, s, hc, d))).to(torch.bfloat16)
        cpu_y = _hc_pre_inv_rms(x, epsilon=eps)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)

        npu_y = torch.ops._C_ascend.npu_hc_pre_inv_rms(x, epsilon=eps)
        
        compare_y = data_compare(cpu_y.float().numpy(), npu_y.cpu().float().numpy())
        assert(compare_y[0] == "Pass")


        # float16 input test (orgin 4 dims input)
        x = torch.tensor(np.random.uniform(-1, 1, (b * s, hc ,d))).to(torch.float16)
        cpu_y = _hc_pre_inv_rms(x, epsilon=eps)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)

        npu_y = torch.ops._C_ascend.npu_hc_pre_inv_rms(x, epsilon=eps)
        
        compare_y = data_compare(cpu_y.float().numpy(), npu_y.cpu().float().numpy())
        assert(compare_y[0] == "Pass")
        print(f'======================== PTA eager FINISH ========================')


