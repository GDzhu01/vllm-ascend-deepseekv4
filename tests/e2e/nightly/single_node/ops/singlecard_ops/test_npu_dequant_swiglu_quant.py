import gc

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

# enable internal format
torch_npu.npu.config.allow_internal_format = True
# enable vllm-ascend custom ops
enable_custom_op()


def dequant_swiglu_quant_golden(
    x: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    group_index: torch.Tensor,
    clamp_limit: float,
    glu_alpha: float,
    glu_bias: float,
):
    m, n = x.shape
    output = torch.empty((m, n // 2), dtype=torch.int8)
    output_scale = torch.empty((m,), dtype=torch.float32)

    start_idx = 0
    pre_v = 0
    for group_idx, curr_v in enumerate(group_index.tolist()):
        curr_v = int(curr_v)
        group_tokens = curr_v - pre_v
        pre_v = curr_v
        if group_tokens <= 0:
            continue

        dequant_out = x[start_idx:curr_v].to(
            torch.float32) * weight_scale[group_idx].view(1, -1)
        dequant_out = dequant_out * activation_scale[start_idx:curr_v].view(
            -1, 1)

        gate, up = dequant_out.chunk(2, dim=-1)
        gate = torch.clamp(gate, max=clamp_limit)
        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
        swiglu_out = gate * torch.sigmoid(glu_alpha * gate) * (up + glu_bias)

        abs_max = torch.max(torch.abs(swiglu_out), dim=-1).values
        quant_scale = 127 / abs_max
        output[start_idx:curr_v] = torch.round(
            swiglu_out * quant_scale.view(-1, 1)).to(torch.int8)
        output_scale[start_idx:curr_v] = 1 / quant_scale
        start_idx = curr_v

    return output, output_scale


@torch.inference_mode()
def test_npu_dequant_swiglu_quant_with_limit():
    torch.manual_seed(0)

    m = 512
    hidden_size = 1024
    group_num = 4
    x = torch.randint(-500, 500, (m, hidden_size * 2), dtype=torch.int32)
    x[x == 0] = 1
    weight_scale = torch.rand(group_num, hidden_size * 2,
                              dtype=torch.float32) * 0.10 + 0.05
    activation_scale = torch.rand(m, dtype=torch.float32) * 0.10 + 0.05
    group_index = torch.tensor([128, 256, 384, 512], dtype=torch.int64)
    clamp_limit = 1.0
    glu_alpha = 1.0
    glu_bias = 0.0

    output_golden, output_scale_golden = dequant_swiglu_quant_golden(
        x, weight_scale, activation_scale, group_index, clamp_limit, glu_alpha,
        glu_bias)

    output, output_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
        x=x.npu(),
        weight_scale=weight_scale.npu(),
        activation_scale=activation_scale.npu(),
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=group_index.npu(),
        activate_left=True,
        quant_mode=1,
        swiglu_mode=1,
        clamp_limit=clamp_limit,
        glu_alpha=glu_alpha,
        glu_bias=glu_bias,
    )

    torch.testing.assert_close(output.cpu(), output_golden, atol=1, rtol=0)
    torch.testing.assert_close(output_scale.cpu(),
                               output_scale_golden,
                               atol=1e-4,
                               rtol=5e-3)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
