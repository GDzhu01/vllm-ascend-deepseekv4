from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_ascend.ops.layernorm import AscendRMSNorm
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.utils import is_310p as is_310p_hw

enable_custom_op()


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


def mock_rms_norm(x, weight, eps):
    return x + 1, None


def mock_add_rms_norm(x, residual, weight, eps):
    return 2 * x, None, 2 * residual


def mock_add_rms_norm_bias(x, residual, weight, bias, eps):
    if bias is None:
        return 2 * x, None, 2 * residual
    else:
        return 2 * x + bias, None, 2 * residual


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]
    mock_config.quant_config = None

    with set_current_vllm_config(mock_config):
        yield mock_config


@pytest.mark.skip("Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
@pytest.mark.skipif(is_310p_hw(), reason="non_310P device unittest case.")
@pytest.mark.parametrize("residual", [None, torch.randn(4, 8, dtype=torch.float32)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
@patch("torch.ops._C_ascend.npu_add_rms_norm_bias", side_effect=mock_add_rms_norm_bias)
def test_RMSNorm_forward(
    mock_add_rms_norm_bias, mock_add_rmsnorm, mock_rmsnorm, residual, dummy_tensor, default_vllm_config
):
    layer = RMSNorm(hidden_size=8, eps=1e-05)
    if residual is not None:
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = 2 * dummy_tensor
        expected_out_residual = 2 * residual
        mock_add_rms_norm_bias.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
        assert torch.allclose(out_residual, expected_out_residual)
    else:
        out_x = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = dummy_tensor + 1

        mock_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)


@pytest.mark.skipif(not is_310p_hw(), reason="310P device unittest case.")
@pytest.mark.parametrize("residual", [None, torch.randn(4, 8, dtype=torch.float16)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
def test_RMSNorm_forward_310p(mock_add_rmsnorm, mock_rmsnorm, residual, dummy_tensor, default_vllm_config):
    layer = RMSNorm(hidden_size=8, eps=1e-05)
    if residual is not None:
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = 2 * dummy_tensor
        expected_out_residual = 2 * residual
        mock_add_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
        assert torch.allclose(out_residual, expected_out_residual)
    else:
        out_x = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = dummy_tensor + 1
        mock_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)


@patch("vllm_ascend.ops.layernorm.vllm_is_batch_invariant", return_value=True)
@patch("vllm_ascend.ops.layernorm.enable_custom_op", return_value=True)
@patch("torch.ops.vllm.maybe_chunk_residual", side_effect=lambda x, residual: residual)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
@patch("torch.ops._C_ascend.npu_add_rms_norm_bias", side_effect=AssertionError("custom op should be skipped"))
def test_ascend_rmsnorm_batch_invariant_skips_custom_op(
    mock_add_rms_norm_bias,
    mock_add_rms_norm,
    _mock_maybe_chunk_residual,
    _mock_enable_custom_op,
    _mock_batch_invariant,
    dummy_tensor,
    default_vllm_config,
):
    residual = torch.randn(4, 8, dtype=torch.float16)
    layer = AscendRMSNorm(hidden_size=8, eps=1e-5)

    out_x, out_residual = layer.forward_oot(dummy_tensor, residual)

    expected_out_x = 2 * dummy_tensor
    expected_out_residual = 2 * residual
    mock_add_rms_norm.assert_called_once()
    mock_add_rms_norm_bias.assert_not_called()
    assert torch.allclose(out_x, expected_out_x)
    assert torch.allclose(out_residual, expected_out_residual)


@patch("vllm_ascend.ops.layernorm.supports_add_rms_norm_bias", return_value=False)
@patch("vllm_ascend.ops.layernorm.vllm_is_batch_invariant", return_value=False)
@patch("vllm_ascend.ops.layernorm.enable_custom_op", return_value=True)
@patch("torch.ops.vllm.maybe_chunk_residual", side_effect=lambda x, residual: residual)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
@patch("torch.ops._C_ascend.npu_add_rms_norm_bias", side_effect=AssertionError("custom op should be skipped"))
def test_ascend_rmsnorm_skips_unsupported_add_rms_norm_bias(
    mock_add_rms_norm_bias,
    mock_add_rms_norm,
    _mock_maybe_chunk_residual,
    _mock_enable_custom_op,
    _mock_batch_invariant,
    _mock_supports_add_rms_norm_bias,
    dummy_tensor,
    default_vllm_config,
):
    residual = torch.randn(4, 8, dtype=torch.float16)
    layer = AscendRMSNorm(hidden_size=8, eps=1e-5)

    out_x, out_residual = layer.forward_oot(dummy_tensor, residual)

    expected_out_x = 2 * dummy_tensor
    expected_out_residual = 2 * residual
    mock_add_rms_norm.assert_called_once()
    mock_add_rms_norm_bias.assert_not_called()
    assert torch.allclose(out_x, expected_out_x)
    assert torch.allclose(out_residual, expected_out_residual)


@patch("vllm_ascend.ops.layernorm.get_weight_prefetch_method")
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
def test_ascend_rmsnorm_without_residual_uses_rms_norm(
    mock_npu_rms_norm,
    mock_get_weight_prefetch_method,
    dummy_tensor,
    default_vllm_config,
):
    layer = AscendRMSNorm(hidden_size=8, eps=1e-5)

    out_x = layer.forward_oot(dummy_tensor, None)

    mock_npu_rms_norm.assert_called_once()
    mock_get_weight_prefetch_method.return_value.maybe_prefetch_mlp_weight_postprocess.assert_called_once()
    assert torch.allclose(out_x, dummy_tensor + 1)
