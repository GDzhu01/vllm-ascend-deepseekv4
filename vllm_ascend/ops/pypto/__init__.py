import torch
from .attention_post_impl import npu_attention_post_v4
from .hc_post_impl import npu_hc_post
from .hc_pre_impl import npu_hc_pre

pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define("hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base) -> (Tensor, Tensor, Tensor)")

@torch.library.impl(pyptolib, "hc_pre", "Meta")
def hc_pre(x, hc_fn, hc_scale, hc_base):
    y = torch.empty([x.size(0), x.size(2)], dtype=x.dtype, device=f'{x.device}')
    post = torch.empty([x.size(0), x.size(1)], dtype=hc_scale.dtype, device=f'{hc_scale.device}')
    comb = torch.empty([x.size(0), x.size(1), x.size(1)], dtype=hc_scale.dtype, device=f'{hc_scale.device}')
    return y, post, comb


@torch.library.impl(pyptolib, "hc_pre", "NPU")
def hc_pre(x, hc_fn, hc_scale, hc_base):
    return npu_hc_pre(x, hc_fn, hc_scale, hc_base)

class HC_PRE(torch.nn.Module):
    def forward(self, x, hc_fn, hc_scale, hc_base):
        return torch.ops.pypto.hc_pre(x, hc_fn, hc_scale, hc_base)
    
pyptolib.define("attn_post(Tensor atten_res, Tensor cos, Tensor sin, Tensor wo_a, Tensor wo_b) -> (Tensor)")

@torch.library.impl(pyptolib, "attn_post", "Meta")
def attn_post(atten_res, cos, sin, wo_a, wo_b):
    y = torch.empty([atten_res.size(0), wo_b.size(1)], dtype=atten_res.dtype, device=atten_res.device)
    return y

@torch.library.impl(pyptolib, "attn_post", "NPU")
def attn_post(atten_res, cos, sin, wo_a, wo_b):
    return npu_attention_post_v4(atten_res, cos, sin, wo_a, wo_b)

class AttentionPostV4(torch.nn.Module):
    def forward(self, attn_res, cos, sin, wo_a, wo_b):
        return torch.ops.pypto.attn_post(attn_res, cos, sin, wo_a, wo_b)
    
pyptolib.define("hc_post(Tensor x, Tensor residual, Tensor post, Tensor comb) -> (Tensor)")

@torch.library.impl(pyptolib, "hc_post", "Meta")
def hc_post(x, residual, post, comb):
    y = torch.empty([x.size(0), residual.size(1), residual.size(2)], dtype=x.dtype, device=x.device)
    return y

@torch.library.impl(pyptolib, "hc_post", "NPU")
def hc_post(x, residual, post, comb):
    return npu_hc_post(x, residual, post, comb)

class HC_POST(torch.nn.Module):
    def forward(self, x, residual, post, comb):
        return torch.ops.pypto.hc_post(x, residual, post, comb)