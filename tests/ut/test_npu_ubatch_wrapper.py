from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.worker.npu_ubatch_wrapper import AscendUBatchWrapper


class _RunnerWithModel:
    def __init__(self, model):
        self.model = model


def test_get_model_runnable_uses_runner_model_when_present():
    model = object()
    wrapper = object.__new__(AscendUBatchWrapper)
    wrapper.runnable = _RunnerWithModel(model)

    assert wrapper._get_model_runnable() is model


def test_get_model_runnable_falls_back_to_runnable():
    model = object()
    wrapper = object.__new__(AscendUBatchWrapper)
    wrapper.runnable = model

    assert wrapper._get_model_runnable() is model


def test_get_safe_ubatch_cudagraph_mode_keeps_full_for_dbo():
    wrapper = object.__new__(AscendUBatchWrapper)
    wrapper.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(enable_dbo=True))

    assert wrapper._get_safe_ubatch_cudagraph_mode(CUDAGraphMode.FULL) is CUDAGraphMode.FULL


def test_get_safe_ubatch_cudagraph_mode_keeps_non_dbo_full():
    wrapper = object.__new__(AscendUBatchWrapper)
    wrapper.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(enable_dbo=False))

    assert wrapper._get_safe_ubatch_cudagraph_mode(CUDAGraphMode.FULL) is CUDAGraphMode.FULL


def test_slice_model_inputs_supports_inputs_embeds_only():
    tokens_slice = slice(1, 4)
    positions = torch.arange(6)
    inputs_embeds = torch.randn(6, 3)

    sliced_input_ids, sliced_positions, sliced_inputs_embeds, sliced_intermediate_tensors = (
        AscendUBatchWrapper._slice_model_inputs(
            object(),
            tokens_slice,
            None,
            positions,
            inputs_embeds,
            None,
        )
    )

    assert sliced_input_ids is None
    assert torch.equal(sliced_positions, positions[tokens_slice])
    assert torch.equal(sliced_inputs_embeds, inputs_embeds[tokens_slice])
    assert sliced_intermediate_tensors is None
