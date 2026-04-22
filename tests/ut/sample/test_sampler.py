from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler


class TestAscendSampler(TestBase):

    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, 'topk_topp_sampler'))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)

    @patch("vllm_ascend.sample.sampler.compute_logprobs_batch_invariant")
    @patch("vllm_ascend.sample.sampler.vllm_is_batch_invariant", return_value=True)
    def test_compute_logprobs_uses_batch_invariant_helper(self, _mock_batch_invariant, mock_compute_logprobs):
        logits = torch.randn(2, 4)
        expected = torch.randn(2, 4)
        mock_compute_logprobs.return_value = expected

        output = AscendSampler.compute_logprobs(logits)

        mock_compute_logprobs.assert_called_once_with(logits)
        self.assertIs(output, expected)

    @patch("vllm_ascend.sample.sampler.TopKTopPSampler.forward_native")
    @patch("vllm_ascend.sample.sampler.compute_logprobs_batch_invariant")
    @patch("vllm_ascend.sample.sampler.vllm_is_batch_invariant", return_value=True)
    def test_topk_topp_sampler_recomputes_processed_logprobs_in_batch_invariant(
        self,
        _mock_batch_invariant,
        mock_compute_logprobs,
        mock_super_forward,
    ):
        logits = torch.randn(2, 4)
        sampled = torch.tensor([1, 0])
        expected = torch.randn(2, 4)
        mock_super_forward.return_value = (sampled, torch.randn(2, 4))
        mock_compute_logprobs.return_value = expected

        sampler = AscendTopKTopPSampler(logprobs_mode="processed_logprobs")
        output_sampled, output_logprobs = sampler.forward_native(logits, {}, None, None)

        self.assertIs(output_sampled, sampled)
        self.assertIs(output_logprobs, expected)
        self.assertEqual(mock_super_forward.call_count, 1)
        self.assertIs(mock_super_forward.call_args.args[0], logits)
        self.assertEqual(mock_super_forward.call_args.args[1], {})
        self.assertIsNone(mock_super_forward.call_args.args[2])
        self.assertIsNone(mock_super_forward.call_args.args[3])
        mock_compute_logprobs.assert_called_once_with(logits)
