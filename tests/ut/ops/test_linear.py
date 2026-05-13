import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from vllm import config

from tests.ut.base import TestBase
from vllm_ascend import ascend_config
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.linear import (AscendColumnParallelLinear,
                                    AscendMergedColumnParallelLinear,
                                    AscendReplicatedLinear,
                                    AscendRowParallelLinear,
                                    AscendUnquantizedLinearMethod)
from vllm_ascend.ops.linear_op import (DeepseekV4OProjColumnParallelOp,
                                       OProjRowParallelOp)
from vllm_ascend.quantization.method_adapters import AscendLinearMethod


class BaseLinearTest(unittest.TestCase):

    def setUp(self):
        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0

        parallel_state._MLP_TP = self.mock_group
        parallel_state._OTP = self.mock_group

        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.finegrained_tp_config.oproj_tensor_parallel_size = 2
        self.mock_ascend_config.finegrained_tp_config.mlp_tensor_parallel_size = 2

        self.patches = [
            patch("vllm_ascend.ascend_config.get_ascend_config",
                  return_value=self.mock_ascend_config),
            patch("vllm_ascend.distributed.parallel_state.get_otp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.distributed.parallel_state.get_mlp_tp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.ops.linear_op.get_tp_group",
                  return_value=self.mock_group),
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=self.mock_group,
            ),
            patch("vllm_ascend.utils.mlp_tp_enable", return_value=True),
            patch("vllm_ascend.utils.oproj_tp_enable", return_value=True)
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()


class TestAscendUnquantizedLinearMethod(TestBase):

    def setUp(self):
        self.method = AscendUnquantizedLinearMethod()
        self.layer = mock.MagicMock()
        mock_dtype = mock.PropertyMock(return_value=torch.float16)
        type(self.layer.weight.data).dtype = mock_dtype

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz0(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "1"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz1(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "2"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz2(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_called_once()


class TestAscendRowParallelLinear(BaseLinearTest):

    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method",
           return_value=MagicMock())
    def test_mlp_optimize(self, mock_get_weight_prefetch_method):

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="down_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)

    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method",
           return_value=MagicMock())
    def test_oproj_tp(self, mock_get_weight_prefetch_method):

        config._current_vllm_config = MagicMock()

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="o_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._OTP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)

    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method",
           return_value=MagicMock())
    def test_deepseek_v4_wo_b_oproj_tp(self, mock_get_weight_prefetch_method):

        config._current_vllm_config = MagicMock()

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="model.layers.0.self_attn.wo_b",
        )
        self.assertIsInstance(linear.custom_op, OProjRowParallelOp)
        self.assertEqual(linear.custom_op.comm_group, parallel_state._OTP)

    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method",
           return_value=MagicMock())
    def test_deepseek_v4_mtp_wo_b_skips_oproj_tp(
            self, mock_get_weight_prefetch_method):

        config._current_vllm_config = MagicMock()

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="mtp.0.self_attn.wo_b",
        )
        self.assertNotIsInstance(linear.custom_op, OProjRowParallelOp)

    @patch("vllm_ascend.quantization.method_adapters.get_otp_group")
    @patch("vllm_ascend.quantization.method_adapters.oproj_tp_enable",
           return_value=True)
    def test_deepseek_v4_wo_b_quant_uses_otp_rank(
        self,
        mock_oproj_tp_enable,
        mock_get_otp_group,
    ):
        mock_group = MagicMock()
        mock_group.rank_in_group = 1
        mock_get_otp_group.return_value = mock_group

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="model.layers.0.self_attn.wo_b",
        )

        scheme = MagicMock()
        scheme.apply = MagicMock(return_value=torch.empty(1, 1))
        method = AscendLinearMethod(scheme)
        method.apply(linear, torch.empty(1, 8))

        scheme.apply.assert_called_once()
        self.assertEqual(scheme.apply.call_args.args[3], 1)

    @patch(
        "vllm_ascend.quantization.method_adapters.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch("vllm_ascend.quantization.method_adapters.get_otp_group")
    @patch("vllm_ascend.quantization.method_adapters.oproj_tp_enable",
           return_value=True)
    def test_deepseek_v4_mtp_wo_b_quant_skips_otp_rank(
        self,
        mock_oproj_tp_enable,
        mock_get_otp_group,
        mock_get_tensor_model_parallel_rank,
    ):
        mock_group = MagicMock()
        mock_group.rank_in_group = 1
        mock_get_otp_group.return_value = mock_group

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="mtp.0.self_attn.wo_b",
        )

        scheme = MagicMock()
        scheme.apply = MagicMock(return_value=torch.empty(1, 1))
        method = AscendLinearMethod(scheme)
        method.apply(linear, torch.empty(1, 8))

        scheme.apply.assert_called_once()
        self.assertEqual(scheme.apply.call_args.args[3], 0)


class TestAscendMergedColumnParallelLinear(BaseLinearTest):

    def test_merged_mlp_tp_init(self):

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="gate_up_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)


class TestAscendColumnParallelLinear(BaseLinearTest):

    def _set_hf_config(self, model_type: str):
        config._current_vllm_config = MagicMock()
        hf_config = config._current_vllm_config.model_config.hf_text_config
        hf_config.model_type = model_type
        hf_config.o_groups = 4
        hf_config.o_lora_rank = 4
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

    def test_deepseek_v4_wo_a_oproj_tp(self):
        self._set_hf_config("deepseek_v4")

        linear = AscendColumnParallelLinear(
            input_size=8,
            output_size=16,
            prefix="model.layers.0.self_attn.wo_a",
        )

        self.assertIsInstance(linear.custom_op,
                              DeepseekV4OProjColumnParallelOp)
        self.assertEqual(linear.custom_op.comm_group, parallel_state._OTP)
        self.assertEqual(linear.tp_size, 2)
        self.assertEqual(linear.n_local_groups, 2)

    def test_deepseek_v4_mtp_wo_a_skips_oproj_tp(self):
        self._set_hf_config("deepseek_v4")

        linear = AscendColumnParallelLinear(
            input_size=8,
            output_size=16,
            prefix="mtp.0.self_attn.wo_a",
        )

        self.assertNotIsInstance(linear.custom_op,
                                 DeepseekV4OProjColumnParallelOp)
        self.assertEqual(linear.tp_size, 2)
        self.assertEqual(linear.n_local_groups, 2)

    def test_wo_a_oproj_tp_rejects_non_deepseek_v4(self):
        self._set_hf_config("qwen3")

        with self.assertRaises(AssertionError):
            AscendColumnParallelLinear(
                input_size=8,
                output_size=16,
                prefix="model.layers.0.self_attn.wo_a",
            )


class TestAscendReplicatedLinear(BaseLinearTest):

    def test_init_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(
            isinstance(linear.quant_method, AscendUnquantizedLinearMethod))

    def test_init_without_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(
            isinstance(linear.quant_method, AscendUnquantizedLinearMethod))


if __name__ == '__main__':
    unittest.main()
