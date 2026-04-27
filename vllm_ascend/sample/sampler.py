import torch
import torch_npu
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.penalties import apply_all_penalties
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_device_type,
    global_stream,
    npu_stream_switch,
)

DEFAULT_LOGPROBS_MODE = "raw_logprobs"
NPU_TOP_K_TOP_P_SAMPLE_SUPPORTED_DEVICES = [AscendDeviceType.A2, AscendDeviceType.A3]
PROCESSED_LOGPROBS_MODES = ("processed_logits", "processed_logprobs")


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-NPU synchronization.
    """
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())
    return probs.div_(q).argmax(dim=-1).view(-1)


def generate_random_q(
    shape: torch.Size,
    device: torch.device,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Generate exponential randoms for device-side sampling."""
    with npu_stream_switch(global_stream()):
        q = torch.empty(shape, device=device, dtype=torch.float32)
        if len(generators) != q.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())
    return q


class AscendSampler(Sampler):
    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """Use Triton-Ascend penalties on NPU when Triton is available; else vLLM default."""
        if not HAS_TRITON:
            return Sampler.apply_penalties(logits, sampling_metadata, output_token_ids)

        if sampling_metadata.no_penalties:
            return logits
        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )

    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        # TODO: support logprobs_mode in vllm-ascend
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler(
            logprobs_mode=logprobs_mode,
        )
        self.async_exponential_event = torch.npu.Event()

    def set_q_event(self, q, event):
        self.topk_topp_sampler.set_q_event(q, event)

    def do_async_exponential(self, b_s, head_dim, generators):
        # Calculating exponential randoms in a different stream
        # and overlapping with model executing.
        with torch.npu.stream(global_stream()):
            global_stream().wait_stream(torch.npu.current_stream())
            q = torch.empty((b_s, head_dim), device="npu", dtype=torch.float32)
            # Goes to async exponential with AI-CPU exponential or default exponential.
            if len(generators) != q.shape[0]:
                q.exponential_()
            if generators:
                for i, generator in generators.items():
                    q[i].exponential_(generator=generator)
            self.async_exponential_event.record()
        self.set_q_event(q, self.async_exponential_event)


class AscendTopKTopPSampler(TopKTopPSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_top_k_top_p = apply_top_k_top_p

    def set_q_event(self, q, event):
        # Pass in async exponential results.
        # Also pass in event to prevent synchronize errors.
        self.q = q
        self.async_event = event

    @staticmethod
    def _can_use_npu_top_k_top_p_sample() -> bool:
        return (
            get_ascend_device_type() in NPU_TOP_K_TOP_P_SAMPLE_SUPPORTED_DEVICES
            and hasattr(torch_npu, "npu_top_k_top_p_sample")
        )

    @staticmethod
    def _prepare_npu_top_k_top_p_inputs(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # npu_top_k_top_p_sample supports fp16/bf16 logits only. Sampler logits
        # are usually promoted to fp32 by vLLM before reaching this layer.
        if logits.dtype in (torch.float16, torch.bfloat16):
            sample_logits = logits
        else:
            sample_logits = logits.to(torch.bfloat16)

        batch_size, vocab_size = sample_logits.shape
        if k is None:
            sample_k = torch.full(
                (batch_size,),
                vocab_size,
                dtype=torch.int32,
                device=sample_logits.device,
            )
        else:
            sample_k = k.to(device=sample_logits.device, dtype=torch.int32)

        if p is None:
            sample_p = torch.ones(
                (batch_size,),
                dtype=sample_logits.dtype,
                device=sample_logits.device,
            )
        else:
            sample_p = p.to(
                device=sample_logits.device,
                dtype=sample_logits.dtype,
            )
        return sample_logits, sample_k, sample_p

    def _sample_with_npu_top_k_top_p(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample_logits, sample_k, sample_p = (
            self._prepare_npu_top_k_top_p_inputs(logits, k, p)
        )
        if q is None:
            q = generate_random_q(
                sample_logits.shape,
                sample_logits.device,
                generators,
            )
        sampled_token_ids, _ = torch_npu.npu_top_k_top_p_sample(
            sample_logits,
            sample_k,
            sample_p,
            q,
        )
        return sampled_token_ids.view(-1)

    def forward_native(self, logits, generators, k, p):
        """Override pytorch native implementation to torch_npu"""
        # when batch_invariant mode is enabled, we should use vllm's implementation.
        # or it will make batch_invariant mode not working.
        if vllm_is_batch_invariant():
            return super().forward_native(logits, generators, k, p)

        if (self.logprobs_mode not in PROCESSED_LOGPROBS_MODES
                and self._can_use_npu_top_k_top_p_sample()):
            if get_ascend_config().enable_async_exponential:
                # Add synchronize to prevent synchronize error.
                self.async_event.synchronize()
                return self._sample_with_npu_top_k_top_p(
                    logits,
                    generators,
                    k,
                    p,
                    self.q,
                ), None
            return self._sample_with_npu_top_k_top_p(
                logits,
                generators,
                k,
                p,
            ), None

        logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        if get_ascend_config().enable_async_exponential:
            # Add synchronize to prevent synchronize error.
            self.async_event.synchronize()
            return probs.div_(self.q).argmax(dim=-1).view(-1), logits_to_return
        return random_sample(probs, generators), logits_to_return


def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    return logits


def _apply_top_k_top_p_ascendc(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    return torch.ops._C_ascend.npu_apply_top_k_top_p(logits, k=k, p=p)


apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    else _apply_top_k_top_p_pytorch
)
