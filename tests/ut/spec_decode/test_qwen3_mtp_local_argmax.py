from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP, Qwen3_5MoeMTP
from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP

from vllm_ascend.patch.worker.patch_qwen3_mtp_local_argmax import (
    _qwen_mtp_get_top_tokens,
)
from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer


class _DraftArgmaxHarness:
    _uses_draft_vocab_remapping = SpecDecodeBaseProposer._uses_draft_vocab_remapping
    _can_use_local_argmax_reduction = SpecDecodeBaseProposer._can_use_local_argmax_reduction
    _draft_argmax = SpecDecodeBaseProposer._draft_argmax


def test_qwen3_mtp_models_expose_get_top_tokens():
    assert hasattr(Qwen3_5MTP, "get_top_tokens")
    assert hasattr(Qwen3_5MoeMTP, "get_top_tokens")
    assert hasattr(Qwen3NextMTP, "get_top_tokens")


def test_qwen3_mtp_get_top_tokens_uses_logits_processor():
    expected = torch.tensor([1, 2], dtype=torch.int64)
    hidden_states = torch.randn(2, 4)
    lm_head = object()
    logits_processor = Mock()
    logits_processor.get_top_tokens.return_value = expected
    model = SimpleNamespace(
        lm_head=lm_head,
        logits_processor=logits_processor,
    )

    actual = _qwen_mtp_get_top_tokens(model, hidden_states)

    assert actual is expected
    logits_processor.get_top_tokens.assert_called_once_with(lm_head, hidden_states)


def test_draft_argmax_uses_local_argmax_reduction():
    expected = torch.tensor([1, 2], dtype=torch.int64)
    hidden_states = torch.randn(2, 4)
    model = SimpleNamespace(
        get_top_tokens=Mock(return_value=expected),
        compute_logits=Mock(),
    )
    proposer = _DraftArgmaxHarness()
    proposer.model = model
    proposer.use_local_argmax_reduction = True

    with patch("vllm_ascend.spec_decode.eagle_proposer.lmhead_tp_enable", return_value=False):
        actual = proposer._draft_argmax(hidden_states, num_indices=2)

    assert actual is expected
    model.get_top_tokens.assert_called_once_with(hidden_states)
    model.compute_logits.assert_not_called()


def test_draft_argmax_falls_back_to_compute_logits_when_disabled():
    logits = torch.tensor([[1.0, 3.0], [4.0, 2.0]])
    hidden_states = torch.randn(2, 4)
    model = SimpleNamespace(
        get_top_tokens=Mock(),
        compute_logits=Mock(return_value=logits),
    )
    proposer = _DraftArgmaxHarness()
    proposer.model = model
    proposer.use_local_argmax_reduction = False

    with patch("vllm_ascend.spec_decode.eagle_proposer.lmhead_tp_enable", return_value=False):
        actual = proposer._draft_argmax(hidden_states, num_indices=2)

    assert torch.equal(actual, torch.tensor([1, 0]))
    model.compute_logits.assert_called_once_with(hidden_states)
    model.get_top_tokens.assert_not_called()


def test_draft_argmax_falls_back_for_lmhead_tp_padding():
    logits = torch.tensor([[1.0, 3.0], [4.0, 2.0], [5.0, 7.0]])
    hidden_states = torch.randn(3, 4)
    model = SimpleNamespace(
        get_top_tokens=Mock(),
        compute_logits=Mock(return_value=logits),
    )
    proposer = _DraftArgmaxHarness()
    proposer.model = model
    proposer.use_local_argmax_reduction = True

    with patch("vllm_ascend.spec_decode.eagle_proposer.lmhead_tp_enable", return_value=True):
        actual = proposer._draft_argmax(hidden_states, num_indices=2)

    assert torch.equal(actual, torch.tensor([1, 0]))
    model.compute_logits.assert_called_once_with(hidden_states)
    model.get_top_tokens.assert_not_called()


def test_draft_argmax_requires_get_top_tokens_when_enabled():
    hidden_states = torch.randn(2, 4)
    model = SimpleNamespace(compute_logits=Mock())
    proposer = _DraftArgmaxHarness()
    proposer.model = model
    proposer.use_local_argmax_reduction = True

    with patch("vllm_ascend.spec_decode.eagle_proposer.lmhead_tp_enable", return_value=False):
        with pytest.raises(ValueError, match="get_top_tokens"):
            proposer._draft_argmax(hidden_states, num_indices=2)
