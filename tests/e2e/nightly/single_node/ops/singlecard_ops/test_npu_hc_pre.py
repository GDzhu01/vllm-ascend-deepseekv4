import gc

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

HC_MULT = 4
HC_MIX = (2 + HC_MULT) * HC_MULT
HIDDEN_SIZE = 4096
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
INPUT_SHAPES = [(16, HC_MULT, HIDDEN_SIZE), (1, 16, HC_MULT, HIDDEN_SIZE)]


def _create_inputs(shape: tuple[int, ...]):
    generator = torch.Generator()
    generator.manual_seed(0)

    x = torch.randn(shape, generator=generator, dtype=torch.float32)
    x = (x * 0.1).to(torch.bfloat16).npu()
    hc_fn = (torch.randn(
        (HC_MIX, HC_MULT * HIDDEN_SIZE),
        generator=generator,
        dtype=torch.float32,
    ) * 0.01).npu()
    hc_scale = (torch.randn((3, ), generator=generator, dtype=torch.float32) *
                0.1).npu()
    hc_base = (torch.randn((HC_MIX, ),
                           generator=generator,
                           dtype=torch.float32) * 0.1).npu()

    return x, hc_fn, hc_scale, hc_base


def _run_hc_pre(op, x, hc_fn, hc_scale, hc_base):
    return op(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        HC_MULT,
        HC_SINKHORN_ITERS,
        NORM_EPS,
        HC_EPS,
    )


@torch.inference_mode()
def test_npu_hc_pre_v1_v2_bf16_input_consistency():
    for shape in INPUT_SHAPES:
        _compare_hc_pre_outputs(shape)


def _compare_hc_pre_outputs(shape: tuple[int, ...]):
    x, hc_fn, hc_scale, hc_base = _create_inputs(shape)

    v1_outputs = _run_hc_pre(torch.ops._C_ascend.npu_hc_pre, x, hc_fn,
                             hc_scale, hc_base)
    v2_outputs = _run_hc_pre(torch.ops._C_ascend.npu_hc_pre_v2, x, hc_fn,
                             hc_scale, hc_base)

    for v1_output, v2_output in zip(v1_outputs, v2_outputs):
        assert v1_output.shape == v2_output.shape
        assert v1_output.dtype == v2_output.dtype
        torch.testing.assert_close(v1_output.cpu(),
                                   v2_output.cpu(),
                                   rtol=5e-2,
                                   atol=5e-2)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
