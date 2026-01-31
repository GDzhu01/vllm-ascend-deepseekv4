from typing import TYPE_CHECKING, Any, Optional, cast

import torch
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS, register_quantization_config)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsScheme
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target, is_activation_quantization_format,
    should_ignore_layer)

from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.quant_config import (AscendFusedMoEMethod,
                                                   AscendLinearMethod,
                                                   AscendQuantConfig)
from vllm_ascend.quantization.w4a16 import AscendW4A16FusedMoEMethod
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.quantization.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import FP8_METHOD

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

QUANTIZATION_SCHEME_MAP_TYPE = dict[str, Optional[dict[str, QuantizationArgs]]]


def remove_quantization_method():
    if FP8_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(FP8_METHOD)


remove_quantization_method()


@register_quantization_config(FP8_METHOD)
class AscendFp8Config(QuantizationConfig):

    def __init__(
        self,
        ignore: list[str],
        quant_format: str,
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        self.quant_description = config

    def get_name(self) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float8_e4m3fn, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware dose not support \"get_min_capability\" feature.")

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str,
                                      Any]) -> "AscendFp8Config":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))

        return cls(
            ignore=ignore,
            quant_format=quant_format,
            config=config,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            layer.ascend_quant_method = FP8_METHOD

            ascend_quant_config = AscendQuantConfig(self.quant_description
                                                    or {})
            quant_method = AscendLinearMethod(ascend_quant_config, prefix,
                                                None, layer)
            return quant_method
        if isinstance(layer, FusedMoE):
            layer.ascend_quant_method = FP8_METHOD
            ascend_quant_config = AscendQuantConfig(self.quant_description
                                                    or {})
            quant_method = AscendFusedMoEMethod(
                ascend_quant_config, prefix,
                ascend_quant_config.packed_modules_mapping, layer)
            return quant_method
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        self.target_scheme_map = hf_to_vllm_mapper.apply_dict(
            self.target_scheme_map)
        self.ignore = hf_to_vllm_mapper.apply_list(self.ignore)
