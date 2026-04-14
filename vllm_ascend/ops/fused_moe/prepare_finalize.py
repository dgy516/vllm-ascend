# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.distributed.utils import fc3_all_gather_and_maybe_unpad_impl
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEPrepareOutput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_sp, enable_sp_by_pass, npu_stream_switch, prefill_context_parallel_enable


class PrepareAndFinalize(ABC):
    """
    Abstract base class for MoE (Mixture-of-Experts) tensor preparation and finalization
    in distributed environments. Subclasses implement specific communication strategies
    (e.g., AllGather, All2All, MC2) to handle tensor padding, slicing,
    broadcasting, and reduction across TP/DP/EP groups.

    Attributes:
        moe_config (FusedMoEConfig): Configuration object containing TP/DP/EP group info,
                                     sizes, ranks, and communication settings.
    """

    quant_stream: torch.npu.Stream | None = None

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config
        ascend_config = get_ascend_config()
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        if self.multistream_overlap_gate and PrepareAndFinalize.quant_stream is None:
            PrepareAndFinalize.quant_stream = torch.npu.Stream()

    @abstractmethod
    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Prepare tensors before MoE computation. May involve:
          - Padding to align communication boundaries
          - Slicing across tensor-parallel ranks
          - Broadcasting across data-parallel ranks

        Args:
            hidden_states (torch.Tensor): Input features, shape [num_tokens, hidden_size]
            router_logits (torch.Tensor): Router outputs, shape [num_tokens, num_experts]
            enable_shared_expert_dp (bool): Skip DP communication for shared experts
            replace_allreduce (bool): Bypass default all-reduce behavior
            quant_type: none, w8a8, w4a8 or mxfp8

        Returns:
            MoEPrepareOutput:
                - processed hidden_states (may be padded/sliced/broadcasted)
                - processed router_logits (may be recomputed or broadcasted)
                - optional communication mask (e.g., mc2_mask for sparse ops)
                - optional padded hidden state shape for finalization
                - optional per-token scale for quantized path
        """
        raise NotImplementedError("Prepare not implemented.")

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Finalize MoE output. May involve:
          - Gathering sliced tensors across TP ranks
          - Reducing or scattering across DP ranks
          - Unpadding to original token count
          - Applying all-reduce across TP/EP if requested

        Args:
            hidden_states (torch.Tensor): MoE layer output, possibly padded or sliced
            reduce_results (bool): Whether to apply all-reduce across TP/EP groups

        Returns:
            torch.Tensor: Final output with shape [original_num_tokens, hidden_size]
        """
        raise NotImplementedError("Finalize function not implemented.")

    @staticmethod
    def _pad_along_first_dim(tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        if pad_size <= 0:
            return tensor

        pad_args = [0, 0] * tensor.dim()
        pad_args[-1] = pad_size
        return nn.functional.pad(tensor, tuple(pad_args))


class PrepareAndFinalizeWithAll2All(PrepareAndFinalize):
    """
    MoE communication strategy using All-to-All style slicing.
    Similar to MC2 but does not use mc2_mask; instead pads to TP size for uniform slicing.
    Will be used when num_tokens exceed mc2's limitation (512 tokens/rank).
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        """Restore original TP configuration (same as MC2)."""
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Pad hidden_states and router_logits to next multiple of TP size.
          2. If TP > 1, split along token dim and select current TP rank's slice.
          3. Save splits for later all-gather in finalize.

        Skips if `enable_shared_expert_dp` or `replace_allreduce` is True.

        Returns:
            MoEPrepareOutput where `mc2_mask` is None for All2All path.
        """
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp

        padded_hidden_states_shape = hidden_states.shape
        if not (self.replace_allreduce or self.enable_shared_expert_dp):
            self.num_tokens, _ = hidden_states.shape
            pad_size = self.tp_size - self.num_tokens  # Pad to TP size (cyclic)

            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                padded_hidden_states_shape = hidden_states.shape

            if self.tp_size > 1:
                split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
                split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)

                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=padded_hidden_states_shape,
            pertoken_scale=None,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Finalization steps:
          1. If TP > 1, all-gather slices to reconstruct full tensor.
          2. Unpad to original token count.
          3. Return [original_num_tokens, hidden_size] tensor.

        Skips if `enable_shared_expert_dp` or `replace_allreduce` is True.
        """

        if not (self.enable_shared_expert_dp or self.replace_allreduce):
            if self.tp_size > 1:
                assert padded_hidden_states_shape is not None
                # Cannot reuse `split_hidden_states` from prepare phase as it
                # may share memory with original hidden_states. Since shared
                # experts may use the original tensor, reusing it would cause
                # in-place modification during all_gather, corrupting the data.
                gathered_hidden_states = torch.empty(
                    padded_hidden_states_shape, device=hidden_states.device, dtype=hidden_states.dtype
                )
                split_hidden_states = torch.tensor_split(gathered_hidden_states, self.tp_size, dim=0)
                dist.all_gather(list(split_hidden_states), hidden_states, self.moe_config.tp_group.device_group)
                hidden_states = gathered_hidden_states

            if self.num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[: self.num_tokens]

        return hidden_states


class PrepareAndFinalizeWithMC2(PrepareAndFinalizeWithAll2All):
    """
    MoE communication strategy using MC2, which is based on All2All. Hence, it inherits
    All2All and share the same finalize method.
    Designed for Ascend or environments requiring explicit padding and slicing control.
    Relies on `mc2_mask` and `padded_num_tokens` from forward_context for alignment.
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        """
        Restore original TP configuration.
        vLLM flattens TP and DP into a single dimension; this method recovers
        the true TP world size and rank for correct tensor slicing.
        """
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Fetch `mc2_mask` and target padding length from forward context.
          2. Pad `hidden_states` and `router_logits` to target length if needed.
          3. If TP > 1, split tensors along token dimension and select current TP rank's slice.
          4. Split and return corresponding `mc2_mask`.

        Skips padding/slicing if `enable_shared_expert_dp` or `replace_allreduce` is True.

        Returns:
            MoEPrepareOutput, possibly sliced/padded.
        """
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp
        mc2_mask = _EXTRA_CTX.mc2_mask
        if self.tp_size > 1:
            # Also slice mc2_mask
            split_mc2_mask = torch.tensor_split(mc2_mask, self.tp_size, dim=0)
            mc2_mask = split_mc2_mask[self.tp_rank]

        padded_hidden_states_shape = hidden_states.shape
        if not self.replace_allreduce:
            self.num_tokens, _ = hidden_states.shape
            target_pad_length = _EXTRA_CTX.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens

            # Pad if necessary (unless shared expert DP is enabled)
            if pad_size > 0 and not self.enable_shared_expert_dp:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                padded_hidden_states_shape = hidden_states.shape

            # Slice across TP ranks
            if self.tp_size > 1 and not self.enable_shared_expert_dp:
                split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
                split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)
                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=mc2_mask,
            padded_hidden_states_shape=padded_hidden_states_shape,
            pertoken_scale=None,
        )


class PrepareAndFinalizeWithAllGather(PrepareAndFinalize):
    """
    MoE communication strategy using All-Gather + Reduce-Scatter on EP group.
    There are two sets of prepare and finalize:
    1. _prepare_with_dp_group/_finalize_with_dp_group: When sequence parallelism is not enabled,
    we gather inputs across DP ranks before MoE, scatter outputs after.
    The communication and calculation process is as follows (AG, AR and RS
    are abbreviations for All-Gather, All-Reduce and Reduce-Scatter, respectively):

    Attn → TP AR → DP AG → MoE → DP RS → TP AR

    2. _prepare_with_ep_group/_finalize_with_ep_group: When sequence parallelism is enabled,
    the above process becomes:

    TP AG → Attn → TP RS → TP AG → DP AG → MoE → DP RS → TP RS

    This strategy further combines TP AG + DP AG into EP All-Gather and TP RS + DP RS
    into EP Reduce-Scatter to improve communication performance. The optimized process is as follows:

    TP AG → Attn → TP RS → EP AG → MoE → EP RS
    """

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          AllGather hidden_states and router_logits to form global tensors.

        Returns:
            MoEPrepareOutput with global tensors.
        """
        if enable_sp() or enable_sp_by_pass():
            return self._prepare_with_ep_group(hidden_states, router_logits, quant_type)

        return self._prepare_with_dp_group(hidden_states, router_logits, enable_shared_expert_dp, replace_allreduce)

    def _prepare_with_ep_group(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, quant_type=QuantType.NONE
    ) -> MoEPrepareOutput:
        pertoken_scale = None
        if quant_type == QuantType.W8A8:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        elif quant_type == QuantType.MXFP8:
            # TODO(linfeng): MXFP8 with AllGather+EP currently does not pre-quantize
            # per-token activations in prepare. Keep quantization in the MoE MLP path.
            pass

        if self.multistream_overlap_gate:
            assert PrepareAndFinalize.quant_stream is not None
            PrepareAndFinalize.quant_stream.wait_stream(torch.npu.current_stream())
            with npu_stream_switch(PrepareAndFinalize.quant_stream, enabled=self.multistream_overlap_gate):
                hidden_states = fc3_all_gather_and_maybe_unpad_impl(hidden_states)
        else:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, True, True)
            router_logits = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(router_logits, True, True)

        if pertoken_scale is not None:
            pertoken_scale = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(pertoken_scale, True, True)

        if self.multistream_overlap_gate:
            torch.npu.current_stream().wait_stream(PrepareAndFinalize.quant_stream)

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=pertoken_scale,
        )

    def _prepare_with_dp_group(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Fetch max token count across DP group from forward context.
          2. Pad local tensors to that size.
          3. All-gather across DP group to form global input tensor.

        Returns:
            MoEPrepareOutput with global tensors.
        """
        self.enable_shared_expert_dp = enable_shared_expert_dp
        if self.moe_config.dp_size > 1:
            max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp

            self.num_tokens = hidden_states.shape[0]
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))

            # All-gather across DP group
            hidden_states = self.moe_config.dp_group.all_gather(hidden_states, 0)
            router_logits = self.moe_config.dp_group.all_gather(router_logits, 0)

        if prefill_context_parallel_enable() and self.moe_config.pcp_size > 1:
            max_tokens_across_pcp = _EXTRA_CTX.max_tokens_across_pcp

            self.num_tokens_pcp = hidden_states.shape[0]
            pad_size = max_tokens_across_pcp - self.num_tokens_pcp
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))

            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=None,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Finalization steps:
          Reduce Scatter hidden states.

        Returns:
            Tensor with shape [local_num_tokens, hidden_size]
        """
        if enable_sp() or enable_sp_by_pass():
            return self._finalize_with_ep_group(hidden_states)

        return self._finalize_with_dp_group(hidden_states, reduce_results)

    def _finalize_with_ep_group(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Argument `reduce_results` is not needed in this func. Given sequence parallelism is enabled:
        1. Reduce_results is False usually happens when models have shared experts and need to
        allreduce hidden states after results of shared experts and routed experts are added in FusedMoe.
        We do reduce scatter for hidden states here, then skip allreudce in FusedMoe and add it to the
        result of shared experts.
        2 Reduce_results is True usually happens when model has no shared experts. We still do reduce scatter
        here, then skip allreudce in FusedMoe.
        """
        hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states, True)

        return hidden_states

    def _finalize_with_dp_group(self, hidden_states: torch.Tensor, reduce_results: bool) -> torch.Tensor:
        """
        Finalization steps:
          1. If DP > 1 and not shared expert, reduce-scatter output across DP group.
          2. Slice to original local token count.
          3. If `reduce_results=True` and TP/EP > 1, apply tensor_model_parallel_all_reduce.

        Returns:
            Tensor with shape [original_local_num_tokens, hidden_size]
        """
        if self.moe_config.dp_size > 1 and not self.enable_shared_expert_dp:
            hidden_states = get_dp_group().reduce_scatter(hidden_states, 0)
            hidden_states = hidden_states[: self.num_tokens]

        if prefill_context_parallel_enable() and self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)
        return hidden_states


class PrepareAndFinalizeWithMoETPAllGather(PrepareAndFinalize):
    """MoE-TP prepare/finalize path for DPxTP flattened routed MoE execution.

    Phase 1 canonicalizes MoE inputs by selecting a single source TP rank inside
    each DP replica, gathering unique inputs across the source ranks, then
    broadcasting the global batch to the paired TP peer. Final outputs are
    reduced on the dedicated MoE-TP group before each source rank slices its
    local DP batch and broadcasts it back to the TP peer.
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self.source_tp_rank = getattr(moe_config, "source_tp_rank", 0)
        self.moe_tp_group = getattr(moe_config, "moe_tp_group", None)
        self.moe_source_group = getattr(moe_config, "moe_source_group", None)
        self.moe_source_group_world_size = getattr(moe_config, "moe_source_group_world_size", None)
        self.moe_source_group_index = getattr(moe_config, "moe_source_group_index", None)
        self.moe_peer_group = getattr(moe_config, "moe_peer_group", None)

        assert self.moe_tp_group is not None, "moe_tp_group is required for MoE-TP mode."
        assert self.moe_peer_group is not None, "moe_peer_group is required for MoE-TP mode."
        assert self.moe_source_group_world_size is not None, "moe_source_group_world_size is required for MoE-TP mode."
        assert self.moe_source_group_index is not None, "moe_source_group_index is required for MoE-TP mode."

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        if replace_allreduce:
            raise ValueError("MoE-TP tensor_parallel mode does not support FLASHCOMM1 replace_allreduce path.")
        if enable_shared_expert_dp:
            raise ValueError("MoE-TP tensor_parallel mode does not support shared expert DP.")

        self.num_tokens = hidden_states.shape[0]
        self.max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp or self.num_tokens
        self.is_source_rank = self.moe_peer_group.rank_in_group == self.source_tp_rank
        should_precompute_routing = (
            quant_type == QuantType.W8A8
            and getattr(self.moe_config, "custom_routing_function", None) is None
        )
        _EXTRA_CTX.moe_tp_topk_weights = None
        _EXTRA_CTX.moe_tp_topk_ids = None
        if self.is_source_rank:
            assert self.moe_source_group is not None, "source ranks must initialize moe_source_group."
        global_num_tokens = self.max_tokens_across_dp * self.moe_source_group_world_size
        pertoken_scale = None
        hidden_states_dtype = hidden_states.dtype
        hidden_size = hidden_states.shape[-1]

        if self.is_source_rank:
            if should_precompute_routing:
                local_topk_weights, local_topk_ids = select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    top_k=self.moe_config.experts_per_token,
                    use_grouped_topk=getattr(self.moe_config, "use_grouped_topk", False),
                    renormalize=getattr(self.moe_config, "renormalize", False),
                    topk_group=getattr(self.moe_config, "topk_group", None),
                    num_expert_group=getattr(self.moe_config, "num_expert_group", None),
                    custom_routing_function=None,
                    scoring_func=getattr(self.moe_config, "scoring_func", "softmax"),
                    routed_scaling_factor=getattr(self.moe_config, "routed_scaling_factor", 1.0),
                    e_score_correction_bias=getattr(self.moe_config, "e_score_correction_bias", None),
                    global_num_experts=self.moe_config.num_experts,
                )
            if quant_type == QuantType.W8A8:
                # Only the source TP rank owns the canonical MoE inputs. Quantize
                # there before communication so peers do not pay duplicate work.
                hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            pad_size = self.max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                hidden_states = self._pad_along_first_dim(hidden_states, pad_size)
                router_logits = self._pad_along_first_dim(router_logits, pad_size)
                if pertoken_scale is not None:
                    pertoken_scale = self._pad_along_first_dim(pertoken_scale, pad_size)
                if should_precompute_routing:
                    local_topk_weights = self._pad_along_first_dim(local_topk_weights, pad_size)
                    local_topk_ids = self._pad_along_first_dim(local_topk_ids, pad_size)
            if should_precompute_routing and local_topk_ids.dtype != torch.int32:
                local_topk_ids = local_topk_ids.to(torch.int32)

            hidden_states = self.moe_source_group.all_gather(hidden_states, 0)
            if should_precompute_routing:
                _EXTRA_CTX.moe_tp_topk_weights = self.moe_source_group.all_gather(local_topk_weights, 0)
                _EXTRA_CTX.moe_tp_topk_ids = self.moe_source_group.all_gather(local_topk_ids, 0)
                router_logits = router_logits.new_empty((0, router_logits.shape[-1]))
            else:
                router_logits = self.moe_source_group.all_gather(router_logits, 0)
            if pertoken_scale is not None:
                pertoken_scale = self.moe_source_group.all_gather(pertoken_scale, 0)
        else:
            hidden_states = torch.empty(
                (global_num_tokens, hidden_size),
                device=hidden_states.device,
                dtype=torch.int8 if quant_type == QuantType.W8A8 else hidden_states_dtype,
            )
            router_logits = router_logits.new_empty((global_num_tokens, router_logits.shape[-1]))
            if quant_type == QuantType.W8A8:
                pertoken_scale = torch.empty((global_num_tokens,), device=hidden_states.device, dtype=torch.float32)
                if should_precompute_routing:
                    _EXTRA_CTX.moe_tp_topk_weights = torch.empty(
                        (global_num_tokens, self.moe_config.experts_per_token),
                        device=hidden_states.device,
                        dtype=torch.float32,
                    )
                    _EXTRA_CTX.moe_tp_topk_ids = torch.empty(
                        (global_num_tokens, self.moe_config.experts_per_token),
                        device=hidden_states.device,
                        dtype=torch.int32,
                    )
                    router_logits = router_logits.new_empty((0, router_logits.shape[-1]))

        hidden_states = self.moe_peer_group.broadcast(hidden_states, src=self.source_tp_rank)
        if router_logits.numel() > 0:
            router_logits = self.moe_peer_group.broadcast(router_logits, src=self.source_tp_rank)
        if pertoken_scale is not None:
            pertoken_scale = self.moe_peer_group.broadcast(pertoken_scale, src=self.source_tp_rank)
        if _EXTRA_CTX.moe_tp_topk_weights is not None:
            _EXTRA_CTX.moe_tp_topk_weights = self.moe_peer_group.broadcast(
                _EXTRA_CTX.moe_tp_topk_weights,
                src=self.source_tp_rank,
            )
            _EXTRA_CTX.moe_tp_topk_ids = self.moe_peer_group.broadcast(
                _EXTRA_CTX.moe_tp_topk_ids,
                src=self.source_tp_rank,
            )

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=pertoken_scale,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del reduce_results, padded_hidden_states_shape

        hidden_states = self.moe_tp_group.all_reduce(hidden_states)
        local_batch_offset = self.moe_source_group_index * self.max_tokens_across_dp
        hidden_states = hidden_states[local_batch_offset : local_batch_offset + self.num_tokens]
        return hidden_states
