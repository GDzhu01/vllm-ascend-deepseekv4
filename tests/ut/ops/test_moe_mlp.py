import unittest
from typing import ClassVar
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.mxfp_compat import (FLOAT4_E2M1FN_X2_DTYPE,
                                     FLOAT8_E8M0FNU_DTYPE)
from vllm_ascend.ops.fused_moe.moe_mlp import (cumsum_group_list,
                                               unified_apply_mlp)
from vllm_ascend.utils import AscendDeviceType


class TestCumsumGroupList(unittest.TestCase):
    glist_dict: ClassVar[dict[int, torch.Tensor]]

    @classmethod
    def setUpClass(cls):
        cls.glist_dict = {
            0: torch.tensor([0, 2, 3, 3]),
            1: torch.tensor([0, 2, 1, 0]),
            2: torch.tensor([[1, 2], [2, 1], [0, 0], [0, 0]])
        }

    support_combine = [(0, 0), (1, 0), (0, 1)]
    unsupport_combine = [(0, 2), (2, 1), (1, 2)]

    def test_cumsum_group_list_supported_conversion(self):
        for src_list_type, dst_list_type in self.support_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                result = cumsum_group_list(self.glist_dict[src_list_type],
                                           src_list_type,
                                           dst_list_type,
                                           expert_num=4)
                self.assertTrue(
                    torch.equal(result, self.glist_dict[dst_list_type]))

    def test_cumsum_group_list_invalid_type_valueerror(self):
        with self.assertRaises(ValueError) as excinfo:
            cumsum_group_list(self.glist_dict[0], 4, 0)
        self.assertIn("group_list_type should be in [0, 1, 2], but received",
                      str(excinfo.exception))

    def test_cumsum_group_list_unsupported_conversion_notimplementederror(
            self):
        for src_list_type, dst_list_type in self.unsupport_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                with self.assertRaises(NotImplementedError) as excinfo:
                    cumsum_group_list(self.glist_dict[0], src_list_type,
                                      dst_list_type)
                self.assertIn("This feature is under development.",
                              str(excinfo.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestUnifiedApplyMLPA5MXFP4(TestBase):

    @patch("vllm_ascend.ops.fused_moe.moe_mlp.get_weight_prefetch_method",
           return_value=None)
    @patch("vllm_ascend.ops.fused_moe.moe_mlp.get_ascend_device_type",
           return_value=AscendDeviceType.A5)
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_grouped_matmul_swiglu_quant_v2")
    def test_unified_apply_mlp_passes_mxfp4_dtypes(
            self, mock_gmm1, mock_gmm2, mock_get_device_type,
            mock_get_weight_prefetch_method):
        hidden_states = torch.randn(4, 16, dtype=torch.bfloat16)
        group_list = torch.tensor([2, 2], dtype=torch.int64)
        w1 = [torch.randint(0, 16, (1, 16, 16), dtype=torch.uint8)]
        w1_scale = [torch.randint(0, 16, (1, 1, 16, 2), dtype=torch.uint8)]
        w2 = [torch.randint(0, 16, (1, 8, 16), dtype=torch.uint8)]
        w2_scale = [torch.randint(0, 16, (1, 1, 16, 2), dtype=torch.uint8)]
        dynamic_scale = torch.randint(0,
                                      16, (4, 2),
                                      dtype=torch.uint8).to(torch.float32)

        mock_gmm1.return_value = (
            torch.randn(4, 8, dtype=torch.int8),
            torch.randint(0, 16, (4, 2), dtype=torch.uint8).to(torch.float32),
        )
        mock_gmm2.return_value = [torch.randn(4, 16, dtype=torch.bfloat16)]

        output = unified_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            group_list=group_list,
            dynamic_scale=dynamic_scale,
            with_quant=True,
            act_quant_type=FLOAT4_E2M1FN_X2_DTYPE,
            weight_quant_type=FLOAT4_E2M1FN_X2_DTYPE,
            scale_type=FLOAT8_E8M0FNU_DTYPE,
            per_token_scale_type=FLOAT8_E8M0FNU_DTYPE,
            use_bf16=True)

        self.assertEqual(output.shape, (4, 16))
        mock_gmm1.assert_called_once()
        gmm1_kwargs = mock_gmm1.call_args.kwargs
        self.assertEqual(gmm1_kwargs["quant_dtype"], FLOAT4_E2M1FN_X2_DTYPE)
        self.assertEqual(gmm1_kwargs["x_dtype"], FLOAT4_E2M1FN_X2_DTYPE)
        self.assertEqual(gmm1_kwargs["weight_dtype"],
                         FLOAT4_E2M1FN_X2_DTYPE)
        self.assertEqual(gmm1_kwargs["weight_scale_dtype"],
                         FLOAT8_E8M0FNU_DTYPE)
        self.assertEqual(gmm1_kwargs["x_scale_dtype"],
                         FLOAT8_E8M0FNU_DTYPE)
        self.assertEqual(gmm1_kwargs["x_scale"].shape, (4, 1, 2))

        mock_gmm2.assert_called_once()
        gmm2_kwargs = mock_gmm2.call_args.kwargs
        self.assertEqual(gmm2_kwargs["x_dtype"], FLOAT4_E2M1FN_X2_DTYPE)
        self.assertEqual(gmm2_kwargs["weight_dtype"],
                         FLOAT4_E2M1FN_X2_DTYPE)
        self.assertEqual(gmm2_kwargs["scale_dtype"], FLOAT8_E8M0FNU_DTYPE)
        self.assertEqual(gmm2_kwargs["per_token_scale_dtype"],
                         FLOAT8_E8M0FNU_DTYPE)
