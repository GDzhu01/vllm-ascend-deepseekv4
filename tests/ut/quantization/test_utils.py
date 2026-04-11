import types
from unittest.mock import MagicMock, patch

from tests.ut.base import TestBase
from vllm_ascend.quantization.utils import (ASCEND_QUANTIZATION_METHOD_MAP,
                                            get_quant_method)


class TestGetQuantMethod(TestBase):

    def setUp(self):
        self.original_quantization_method_map = ASCEND_QUANTIZATION_METHOD_MAP.copy(
        )
        for quant_type, layer_map in ASCEND_QUANTIZATION_METHOD_MAP.items():
            for layer_type in layer_map.keys():
                def exec_body(ns):
                    ns["__init__"] = lambda self, *args, **kwargs: None

                ASCEND_QUANTIZATION_METHOD_MAP[quant_type][layer_type] = (
                    types.new_class(f"{quant_type}_{layer_type}", (),
                                    exec_body=exec_body))

    def tearDown(self):
        # Restore original map
        ASCEND_QUANTIZATION_METHOD_MAP.clear()
        ASCEND_QUANTIZATION_METHOD_MAP.update(
            self.original_quantization_method_map)

    def test_linear_quant_methods(self):
        for quant_type, layer_map in ASCEND_QUANTIZATION_METHOD_MAP.items():
            if "linear" in layer_map.keys():
                prefix = "linear_layer"
                cls = layer_map["linear"]
                method = get_quant_method({"linear_layer.weight": quant_type},
                                          prefix, "linear")
                self.assertIsInstance(method, cls)

    def test_moe_quant_methods(self):
        for quant_type, layer_map in ASCEND_QUANTIZATION_METHOD_MAP.items():
            if "moe" in layer_map.keys():
                prefix = "layer"
                cls = layer_map["moe"]
                method = get_quant_method({"layer.weight": quant_type}, prefix,
                                          "moe")
                self.assertIsInstance(method, cls)

    @patch("vllm_ascend.quantization.utils.get_current_vllm_config")
    def test_fp4_expert_dtype_overrides_moe_quant_type(self,
                                                       mock_get_current_vllm_config):
        mock_config = MagicMock()
        mock_config.model_config.hf_text_config.expert_dtype = "fp4"
        mock_get_current_vllm_config.return_value = mock_config

        method = get_quant_method({"layer.weight": "W8A8_MXFP8"}, "layer",
                                  "moe")
        self.assertIsInstance(method,
                              ASCEND_QUANTIZATION_METHOD_MAP["W4A4_MXFP4"]
                              ["moe"])

    def test_invalid_layer_type(self):
        quant_description = {"linear_layer.weight": "W8A8"}
        with self.assertRaises(NotImplementedError):
            get_quant_method(quant_description, "linear_layer", "unsupported")

    def test_invalid_quant_type(self):
        quant_description = {"linear_layer.weight": "UNKNOWN"}
        with self.assertRaises(NotImplementedError):
            get_quant_method(quant_description, "linear_layer", "linear")
