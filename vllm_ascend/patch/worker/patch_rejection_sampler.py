import vllm.v1.sample.rejection_sampler as rs
import torch

from vllm_ascend.sample.rejection_sampler import apply_sampling_constraints, expand_batch_to_tokens, rejection_sample


def _get_logprobs_tensors_without_nonzero(
    self,
    max_num_logprobs: int,
    metadata,
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    bonus_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
):
    cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
    cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

    bonus_logits_indices = metadata.bonus_logits_indices
    target_logits_indices = metadata.target_logits_indices
    final_logits = torch.zeros_like(logits, dtype=torch.float32)
    final_logits[target_logits_indices] = target_logits.to(torch.float32)
    final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)

    logit_start_indices = cu_num_sampled_tokens
    offsets = torch.arange(
        sampled_token_ids.shape[-1],
        device=logit_start_indices.device,
        dtype=logit_start_indices.dtype,
    )
    accepted_logit_indices = (logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)).flatten()
    accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)

    accepted_tokens = sampled_token_ids.clone().flatten()
    # Avoid boolean advanced indexing here. On NPU it lowers through NonZero,
    # which is unstable in the 16-token MTP full-graph path.
    accepted_tokens.clamp_(min=0)

    accepted_logits = final_logits[accepted_logit_indices]
    accepted_logprobs = (
        accepted_logits if self.is_logits_logprobs_mode else self.sampler.compute_logprobs(accepted_logits)
    )
    return self.sampler.gather_logprobs(
        accepted_logprobs,
        max_num_logprobs,
        accepted_tokens.to(torch.int64),
    )


# TODO: delete this patch after apply_sampling_constraints and rejection_sample
#   are extracted to as class func of RejectionSampler
rs.apply_sampling_constraints = apply_sampling_constraints
rs.rejection_sample = rejection_sample
rs.expand_batch_to_tokens = expand_batch_to_tokens
rs.RejectionSampler._get_logprobs_tensors = _get_logprobs_tensors_without_nonzero
