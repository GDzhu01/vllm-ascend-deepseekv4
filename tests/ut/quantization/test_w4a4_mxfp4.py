from unittest.mock import MagicMock, Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.mxfp_compat import (FLOAT4_E2M1FN_X2_DTYPE,
                                     FLOAT8_E8M0FNU_DTYPE)
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.quantization.w4a4_mxfp4 import \
    AscendW4A4MXFP4DynamicFusedMoEMethod
from vllm_ascend.utils import AscendDeviceType, QuantType


class TestAscendW4A4MXFP4DynamicFusedMoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128
    group_size = 32

    @patch("torch.distributed.get_rank")
    @patch("vllm_ascend.quantization.w4a4_mxfp4.get_ascend_device_type",
           return_value=AscendDeviceType.A5)
    @patch("vllm_ascend.quantization.w8a8_dynamic.get_ascend_config")
    @patch("vllm_ascend.quantization.w8a8_dynamic.get_ep_group")
    @patch("vllm_ascend.quantization.w8a8_dynamic.get_mc2_group")
    @patch(
        "vllm_ascend.quantization.w8a8_dynamic.get_current_vllm_config")
    def setUp(self, mock_get_current_vllm_config, mock_get_mc2_group,
              mock_get_ep_group, mock_get_ascend_config,
              mock_get_ascend_device_type, mock_get_rank):
        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock(
            quant_description={"group_size": self.group_size})
        mock_vllm_config.parallel_config = Mock(enable_expert_parallel=True)
        mock_vllm_config.compilation_config = Mock(mode=None)
        mock_vllm_config.model_config = Mock(enforce_eager=False,
                                             dtype=torch.bfloat16)
        mock_get_current_vllm_config.return_value = mock_vllm_config

        mock_ascend_config = Mock()
        mock_ascend_config.dynamic_eplb = False
        mock_ascend_config.expert_map_record_path = None
        mock_ascend_config.multistream_overlap_gate = False
        mock_get_ascend_config.return_value = mock_ascend_config

        mock_ep_group.return_value = Mock(world_size=1)
        mock_device_group = MagicMock()
        mock_device_group._get_backend.return_value.get_hccl_comm_name.return_value = "hccl_0"
        mock_get_mc2_group.return_value = Mock(device_group=mock_device_group)
        mock_get_rank.return_value = 0

        self.quant_method = AscendW4A4MXFP4DynamicFusedMoEMethod()

    def test_get_quant_type_reports_mxfp4(self):
        layer = object.__new__(AscendFusedMoE)
        layer.quant_method = MagicMock(quant_method=self.quant_method)

        self.assertEqual(layer._get_quant_type(), QuantType.MXFP4)

    def test_get_weight(self):
        params = self.quant_method.get_weight(self.num_experts,
                                              self.intermediate_size,
                                              self.hidden_size,
                                              torch.bfloat16)
        self.assertEqual(params["w13_weight"].dtype, torch.uint8)
        self.assertEqual(params["w13_weight"].shape,
                         (self.num_experts, 2 * self.intermediate_size,
                          self.hidden_size // 2))
        self.assertEqual(params["w2_weight"].dtype, torch.uint8)
        self.assertEqual(params["w2_weight"].shape,
                         (self.num_experts, self.hidden_size,
                          self.intermediate_size // 2))

    def test_get_dynamic_quant_param(self):
        params = self.quant_method.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size,
            torch.bfloat16)
        self.assertEqual(params["w13_weight_scale"].dtype, torch.uint8)
        self.assertEqual(params["w13_weight_scale"].shape,
                         (self.num_experts, 2 * self.intermediate_size,
                          self.hidden_size // self.group_size))
        self.assertEqual(params["w2_weight_scale"].dtype, torch.uint8)
        self.assertEqual(params["w2_weight_scale"].shape,
                         (self.num_experts, self.hidden_size,
                          self.intermediate_size // self.group_size))

    def test_process_weights_after_loading(self):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros((self.num_experts, 2 * self.intermediate_size,
                         self.hidden_size // 2),
                        dtype=torch.uint8),
            requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros((self.num_experts, self.hidden_size,
                         self.intermediate_size // 2),
                        dtype=torch.uint8),
            requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.zeros((self.num_experts, 2 * self.intermediate_size,
                         self.hidden_size // self.group_size),
                        dtype=torch.uint8),
            requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.zeros((self.num_experts, self.hidden_size,
                         self.intermediate_size // self.group_size),
                        dtype=torch.uint8),
            requires_grad=False)

        self.quant_method.process_weights_after_loading(layer)

        self.assertEqual(layer.w13_weight.data.shape,
                         (self.num_experts, self.hidden_size // 2,
                          2 * self.intermediate_size))
        self.assertEqual(layer.w2_weight.data.shape,
                         (self.num_experts, self.intermediate_size // 2,
                          self.hidden_size))
        self.assertEqual(layer.w13_weight_scale.data.shape,
                         (self.num_experts,
                          (self.hidden_size // self.group_size) // 2,
                          2 * self.intermediate_size, 2))
        self.assertEqual(layer.w2_weight_scale.data.shape,
                         (self.num_experts,
                          (self.intermediate_size // self.group_size) // 2,
                          self.hidden_size, 2))

    def test_weight_loader_loads_packed_w13_weight(self):
        layer = object.__new__(AscendFusedMoE)
        layer.quant_method = MagicMock(quant_method=self.quant_method)
        layer._expert_map = None
        layer.tp_rank = 0
        layer.moe_config = MagicMock(is_act_and_mul=True)

        param = torch.nn.Parameter(
            torch.zeros((self.num_experts, 2 * self.intermediate_size,
                         self.hidden_size // 2),
                        dtype=torch.uint8),
            requires_grad=False)
        loaded_weight = torch.randint(0,
                                      256,
                                      (self.intermediate_size,
                                       self.hidden_size // 2),
                                      dtype=torch.uint8)

        success = layer.weight_loader(
            param,
            loaded_weight,
            "model.layers.0.mlp.experts.3.gate_proj.weight",
            shard_id="w1",
            expert_id=3,
            return_success=True)

        self.assertTrue(success)
        self.assertTrue(
            torch.equal(param.data[3, :self.intermediate_size], loaded_weight))
        self.assertEqual(
            torch.count_nonzero(param.data[3, self.intermediate_size:]).item(),
            0)

    @patch("vllm_ascend.quantization.w4a4_mxfp4.select_experts")
    @patch("vllm_ascend.quantization.w4a4_mxfp4.get_forward_context")
    def test_apply_passes_mxfp4_runtime_args(self, mock_get_forward_context,
                                             mock_select_experts):
        x = torch.randn(4, self.hidden_size, dtype=torch.bfloat16)
        router_logits = torch.randn(4, self.num_experts)
        topk_weights = torch.rand(4, 2)
        topk_ids = torch.randint(0, self.num_experts, (4, 2))
        mock_select_experts.return_value = (topk_weights, topk_ids)

        moe_comm_method = MagicMock()
        moe_comm_method.fused_experts.return_value = torch.randn_like(x)
        mock_get_forward_context.return_value = MagicMock(
            moe_comm_method=moe_comm_method)

        layer = MagicMock()
        layer.zero_expert_num = 0
        layer.zero_expert_type = None
        layer.n_shared_experts = 0
        layer.mix_placement = False
        layer.w13_weight = torch.empty(self.num_experts,
                                       2 * self.intermediate_size,
                                       self.hidden_size // 2,
                                       dtype=torch.uint8)
        layer.w2_weight = torch.empty(self.num_experts,
                                      self.hidden_size,
                                      self.intermediate_size // 2,
                                      dtype=torch.uint8)
        layer.w13_weight_scale = torch.empty(self.num_experts,
                                             2 * self.intermediate_size,
                                             self.hidden_size //
                                             self.group_size,
                                             dtype=torch.uint8)
        layer.w2_weight_scale = torch.empty(self.num_experts,
                                            self.hidden_size,
                                            self.intermediate_size //
                                            self.group_size,
                                            dtype=torch.uint8)

        output = self.quant_method.apply(layer=layer,
                                         x=x,
                                         router_logits=router_logits,
                                         top_k=2,
                                         renormalize=False,
                                         global_num_experts=self.num_experts)

        self.assertEqual(output.shape, x.shape)
        moe_comm_method.fused_experts.assert_called_once()
        kwargs = moe_comm_method.fused_experts.call_args.kwargs
        self.assertTrue(kwargs["use_mxfp4_moe"])
        self.assertEqual(kwargs["act_quant_type"], FLOAT4_E2M1FN_X2_DTYPE)
        self.assertEqual(kwargs["weight_quant_type"],
                         FLOAT4_E2M1FN_X2_DTYPE)
        self.assertEqual(kwargs["scale_type"], FLOAT8_E8M0FNU_DTYPE)
        self.assertEqual(kwargs["per_token_scale_type"],
                         FLOAT8_E8M0FNU_DTYPE)
        self.assertTrue(kwargs["use_bf16"])
