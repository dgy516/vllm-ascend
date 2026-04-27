from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler


class TestAscendSampler(TestBase):

    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)

    def test_init_propagates_logprobs_mode_to_topk_topp_sampler(self):
        sampler = AscendSampler(logprobs_mode="processed_logits")
        self.assertEqual(
            sampler.topk_topp_sampler.logprobs_mode,
            "processed_logits",
        )

    def test_topk_topp_sampler_uses_npu_top_k_top_p_sample_with_async_q(self):
        sampler = AscendTopKTopPSampler()
        logits = torch.randn(2, 4, dtype=torch.float32)
        q = torch.ones_like(logits)
        event = Mock()
        sampler.set_q_event(q, event)

        with (
            patch(
                "vllm_ascend.sample.sampler.vllm_is_batch_invariant",
                return_value=False,
            ),
            patch(
                "vllm_ascend.sample.sampler.get_ascend_config",
                return_value=SimpleNamespace(enable_async_exponential=True),
            ),
            patch.object(
                sampler,
                "_can_use_npu_top_k_top_p_sample",
                return_value=True,
            ),
            patch(
                "vllm_ascend.sample.sampler.torch_npu.npu_top_k_top_p_sample",
                return_value=(torch.tensor([1, 2]), torch.empty_like(logits)),
            ) as mock_sample,
        ):
            sampled, logits_to_return = sampler.forward_native(logits, {}, None, None)

        self.assertIsNone(logits_to_return)
        self.assertEqual(sampled.tolist(), [1, 2])
        event.synchronize.assert_called_once()
        sample_logits, sample_k, sample_p, sample_q = mock_sample.call_args.args
        self.assertEqual(sample_logits.dtype, torch.bfloat16)
        self.assertEqual(sample_k.dtype, torch.int32)
        self.assertEqual(sample_k.tolist(), [4, 4])
        self.assertEqual(sample_p.dtype, torch.bfloat16)
        self.assertEqual(sample_p.tolist(), [1.0, 1.0])
        self.assertIs(sample_q, q)

    def test_topk_topp_sampler_preserves_processed_logits_fallback(self):
        sampler = AscendTopKTopPSampler(logprobs_mode="processed_logits")
        logits = torch.randn(2, 4, dtype=torch.float32)

        with (
            patch(
                "vllm_ascend.sample.sampler.vllm_is_batch_invariant",
                return_value=False,
            ),
            patch(
                "vllm_ascend.sample.sampler.get_ascend_config",
                return_value=SimpleNamespace(enable_async_exponential=False),
            ),
            patch.object(
                sampler,
                "_can_use_npu_top_k_top_p_sample",
                return_value=True,
            ),
            patch.object(
                sampler,
                "apply_top_k_top_p",
                return_value=logits,
            ) as mock_apply,
            patch(
                "vllm_ascend.sample.sampler.random_sample",
                return_value=torch.tensor([1, 2]),
            ) as mock_random,
            patch(
                "vllm_ascend.sample.sampler.torch_npu.npu_top_k_top_p_sample"
            ) as mock_sample,
        ):
            sampled, logits_to_return = sampler.forward_native(logits, {}, None, None)

        self.assertEqual(sampled.tolist(), [1, 2])
        self.assertIs(logits_to_return, logits)
        mock_apply.assert_called_once()
        mock_random.assert_called_once()
        mock_sample.assert_not_called()
