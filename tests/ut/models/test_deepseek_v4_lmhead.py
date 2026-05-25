import unittest
from unittest.mock import MagicMock, patch

from torch import nn
from vllm.config.vllm import set_current_vllm_config

from vllm_ascend.models import deepseek_v4
from vllm_ascend.ops.vocab_parallel_embedding import (
    AscendLogitsProcessor, AscendParallelLMHead)


class DummyDeepseekV4Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layers = []
        self.make_empty_intermediate_tensors = MagicMock()


class TestDeepseekV4LmHead(unittest.TestCase):

    def test_main_model_uses_ascend_lmhead_and_logits_processor(self):
        vllm_config = MagicMock()
        vllm_config.model_config.hf_config.vocab_size = 64
        vllm_config.model_config.hf_config.hidden_size = 16
        vllm_config.model_config.hf_config.num_hidden_layers = 0
        vllm_config.compilation_config.custom_ops = ["all"]
        vllm_config.quant_config = None

        pp_group = MagicMock()
        pp_group.is_last_rank = True

        lmhead_group = MagicMock()
        lmhead_group.world_size = 8
        lmhead_group.rank_in_group = 0

        tp_group = MagicMock()
        tp_group.world_size = 2
        tp_group.rank_in_group = 0

        with patch.object(deepseek_v4.AscendDeepseekV4ForCausalLM, "model_cls", DummyDeepseekV4Model), \
            patch.object(deepseek_v4.AscendDeepseekV4ForCausalLM, "set_moe_parameters", lambda self: None), \
            patch.object(deepseek_v4, "get_pp_group", return_value=pp_group), \
            patch("vllm_ascend.ops.vocab_parallel_embedding.lmhead_tp_enable", return_value=True), \
            patch("vllm_ascend.ops.vocab_parallel_embedding.embedding_tp_enable", return_value=False), \
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group", return_value=lmhead_group), \
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_tp_group", return_value=tp_group), \
            patch("vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank", return_value=0), \
            patch("vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size", return_value=2):
            with set_current_vllm_config(vllm_config):
                model = deepseek_v4.AscendDeepseekV4ForCausalLM(
                    vllm_config=vllm_config)

        self.assertIsInstance(model.lm_head, AscendParallelLMHead)
        self.assertTrue(model.lm_head.use_lmhead_tp)
        self.assertIs(model.lm_head.comm_group, lmhead_group)
        self.assertIsInstance(model.logits_processor, AscendLogitsProcessor)
