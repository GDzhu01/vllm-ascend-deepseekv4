import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import (
    AscendSampler,
    AscendTopKTopPSampler,
    _apply_top_k_one_tie_break,
    _apply_top_k_top_p_pytorch,
)


class TestAscendSampler(TestBase):

    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, 'topk_topp_sampler'))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)

    def test_top_k_one_uses_single_argmax_on_tied_logits(self):
        logits = torch.tensor([[0.0, 5.0, 5.0, -1.0],
                               [3.0, 1.0, 3.0, 0.0]],
                              dtype=torch.float32)
        k = torch.tensor([1, 1], dtype=torch.int32)

        logits = _apply_top_k_top_p_pytorch(logits, k, None)
        logits = _apply_top_k_one_tie_break(logits, k, None)

        self.assertEqual(torch.isfinite(logits[0]).nonzero().view(-1).tolist(),
                         [1])
        self.assertEqual(torch.isfinite(logits[1]).nonzero().view(-1).tolist(),
                         [0])
