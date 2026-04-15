import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestNPUModelRunnerGreedyFastPath(unittest.TestCase):

    @staticmethod
    def _make_vllm_config(max_num_seqs: int = 4):
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs)
        )

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.broadcast_pp_output = False
        runner.input_batch = SimpleNamespace(
            sampling_metadata=SimpleNamespace(
                all_greedy=True,
                max_num_logprobs=None,
                no_penalties=True,
                allowed_token_ids_mask=None,
                bad_words_token_ids={},
            ),
            logitsprocs=LogitsProcessors(),
            logitsprocs_need_output_token_ids=False,
            num_reqs=2,
        )
        return runner

    def test_is_greedy_fastpath_candidate_requires_clean_sampling_state(self):
        runner = self._build_runner()

        self.assertTrue(runner._is_greedy_fastpath_candidate(None))

        runner.input_batch.sampling_metadata.bad_words_token_ids = {0: [[1, 2]]}
        self.assertFalse(runner._is_greedy_fastpath_candidate(None))

    def test_is_greedy_fastpath_candidate_allows_inactive_builtin_processors(self):
        runner = self._build_runner()
        min_tokens = MinTokensLogitsProcessor(
            self._make_vllm_config(),
            torch.device("cpu"),
            False,
        )
        min_p = MinPLogitsProcessor(
            self._make_vllm_config(),
            torch.device("cpu"),
            False,
        )
        min_p.min_p_count = 1
        runner.input_batch.logitsprocs = LogitsProcessors([min_tokens, min_p])

        self.assertTrue(runner._is_greedy_fastpath_candidate(None))

    def test_is_spec_greedy_fastpath_candidate_requires_spec_metadata(self):
        runner = self._build_runner()
        metadata = SpecDecodeMetadata.make_dummy([[1], [2]], torch.device("cpu"))

        self.assertTrue(runner._is_spec_greedy_fastpath_candidate(metadata))
        self.assertFalse(runner._is_spec_greedy_fastpath_candidate(None))

    def test_is_spec_greedy_fastpath_candidate_allows_inactive_min_tokens(self):
        runner = self._build_runner()
        runner.input_batch.logitsprocs = LogitsProcessors([
            MinTokensLogitsProcessor(
                self._make_vllm_config(),
                torch.device("cpu"),
                False,
            )
        ])
        metadata = SpecDecodeMetadata.make_dummy([[1], [2]], torch.device("cpu"))

        self.assertTrue(runner._is_spec_greedy_fastpath_candidate(metadata))

    def test_is_greedy_fastpath_candidate_rejects_active_min_tokens(self):
        runner = self._build_runner()
        min_tokens = MinTokensLogitsProcessor(
            self._make_vllm_config(),
            torch.device("cpu"),
            False,
        )
        min_tokens.min_toks[0] = (2, [1], {2})
        runner.input_batch.logitsprocs = LogitsProcessors([min_tokens])

        self.assertFalse(runner._is_greedy_fastpath_candidate(None))

    def test_is_greedy_fastpath_candidate_rejects_active_logit_bias(self):
        runner = self._build_runner()
        logit_bias = LogitBiasLogitsProcessor(None, torch.device("cpu"), False)
        logit_bias.biases = {0: {1: 1.5}}
        runner.input_batch.logitsprocs = LogitsProcessors([logit_bias])

        self.assertFalse(runner._is_greedy_fastpath_candidate(None))

    def test_is_greedy_fastpath_candidate_rejects_unknown_non_argmax_processor(self):
        runner = self._build_runner()

        class CustomNonArgmaxProcessor(LogitsProcessor):

            def __init__(self):
                pass

            def apply(self, logits: torch.Tensor) -> torch.Tensor:
                return logits

            def is_argmax_invariant(self) -> bool:
                return False

            def update_state(self, batch_update) -> None:
                return None

        runner.input_batch.logitsprocs = LogitsProcessors(
            [CustomNonArgmaxProcessor()]
        )

        self.assertFalse(runner._is_greedy_fastpath_candidate(None))

    def test_get_top_tokens_for_model_trims_lmhead_tp_padding(self):
        runner = self._build_runner()
        logits_processor = MagicMock()
        logits_processor.get_top_tokens.return_value = torch.tensor(
            [7, 8, 9],
            dtype=torch.int64,
        )
        runner.model = SimpleNamespace(
            logits_processor=logits_processor,
            lm_head=SimpleNamespace(bias=None),
        )

        top_tokens = runner._get_top_tokens_for_model(torch.randn(2, 4))

        torch.testing.assert_close(
            top_tokens,
            torch.tensor([7, 8], dtype=torch.int64),
        )
        logits_processor.get_top_tokens.assert_called_once()

    def test_get_top_tokens_for_model_uses_language_model_from_wrapper(self):
        runner = self._build_runner()
        logits_processor = MagicMock()
        logits_processor.get_top_tokens.return_value = torch.tensor(
            [4, 5, 6],
            dtype=torch.int64,
        )
        language_model = SimpleNamespace(
            logits_processor=logits_processor,
            lm_head=SimpleNamespace(bias=None),
        )
        runner.model = SimpleNamespace(
            get_language_model=MagicMock(return_value=language_model),
            compute_logits=MagicMock(),
        )

        top_tokens = runner._get_top_tokens_for_model(torch.randn(2, 4))

        torch.testing.assert_close(
            top_tokens,
            torch.tensor([4, 5], dtype=torch.int64),
        )
        runner.model.get_language_model.assert_called_once()
        logits_processor.get_top_tokens.assert_called_once()
        runner.model.compute_logits.assert_not_called()

    def test_get_top_tokens_for_model_preserves_remapped_compute_logits(self):
        runner = self._build_runner()
        logits_processor = MagicMock()
        runner.model = SimpleNamespace(
            draft_id_to_target_id=torch.tensor([1, 2], dtype=torch.int32),
            logits_processor=logits_processor,
            lm_head=SimpleNamespace(bias=None),
            compute_logits=MagicMock(
                return_value=torch.tensor(
                    [[1.0, 9.0, 2.0], [5.0, 4.0, 6.0]],
                    dtype=torch.float32,
                )
            ),
        )

        top_tokens = runner._get_top_tokens_for_model(torch.randn(2, 4))

        torch.testing.assert_close(
            top_tokens,
            torch.tensor([1, 2], dtype=torch.int64),
        )
        runner.model.compute_logits.assert_called_once()
        logits_processor.get_top_tokens.assert_not_called()

    def test_build_spec_greedy_sampler_output_uses_target_and_bonus_indices(self):
        runner = self._build_runner()
        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.tensor([5, 6], dtype=torch.int32),
            num_draft_tokens=[1, 1],
            cu_num_draft_tokens=torch.tensor([1, 2], dtype=torch.int32),
            cu_num_sampled_tokens=torch.tensor([2, 4], dtype=torch.int32),
            target_logits_indices=torch.tensor([0, 2], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([1, 3], dtype=torch.int32),
            logits_indices=torch.tensor([8, 9, 10, 11], dtype=torch.int32),
        )
        runner._get_top_tokens_for_model = MagicMock(
            return_value=torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        )
        fake_sampled = torch.tensor([[10, 20], [30, 40]], dtype=torch.int32)

        with patch(
            "vllm_ascend.worker.model_runner_v1.greedy_rejection_sample",
            return_value=fake_sampled,
        ) as mock_helper:
            sampler_output = runner._build_spec_greedy_sampler_output(
                torch.randn(4, 3),
                metadata,
            )

        runner._get_top_tokens_for_model.assert_called_once()
        _, kwargs = mock_helper.call_args
        torch.testing.assert_close(
            kwargs["target_argmax"],
            torch.tensor([10, 30], dtype=torch.int64),
        )
        torch.testing.assert_close(
            kwargs["bonus_token_ids"],
            torch.tensor([[20], [40]], dtype=torch.int32),
        )
        torch.testing.assert_close(sampler_output.sampled_token_ids, fake_sampled)
        self.assertIsNone(sampler_output.logprobs_tensors)


if __name__ == "__main__":
    unittest.main()
