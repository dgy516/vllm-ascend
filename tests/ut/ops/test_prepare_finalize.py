import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalizeWithAll2All, PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2, PrepareAndFinalizeWithMoETPAllGather)
from vllm_ascend.quantization.quant_type import QuantType


class TestPrepareAndFinalize(unittest.TestCase):

    def setUp(self):
        # Mock FusedMoEConfig
        fake_stream = MagicMock()
        patcher = patch("torch.npu.Stream", return_value=fake_stream)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.moe_config = MagicMock(spec=FusedMoEConfig)
        self.moe_config.tp_group = MagicMock()
        self.moe_config.tp_group.device_group = MagicMock()
        self.moe_config.dp_size = 1
        self.moe_config.tp_size = 1
        self.moe_config.ep_size = 1
        self.moe_config.dp_group = MagicMock()
        self.moe_config.moe_tp_group = MagicMock()
        self.moe_config.moe_source_group = MagicMock()
        self.moe_config.moe_source_group_world_size = 1
        self.moe_config.moe_source_group_index = 0
        self.moe_config.moe_peer_group = MagicMock()
        self.moe_config.source_tp_rank = 0
        self.moe_config.original_num_experts = 8

    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_world_size",
        return_value=1)
    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    def test_mc2_prepare_finalize(self, mock_get_forward_context, mock_tp_rank,
                                  mock_tp_size):
        mock_context = MagicMock()
        mock_context.mc2_mask = torch.tensor([1, 0, 1])
        mock_context.padded_num_tokens = 4
        mock_get_forward_context.return_value = mock_context

        layer = PrepareAndFinalizeWithMC2(self.moe_config)

        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)

        prepare_output = layer.prepare(hidden_states, router_logits)
        h_out = prepare_output.hidden_states
        r_out = prepare_output.router_logits
        mask = prepare_output.mc2_mask
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # Check padding and split
        self.assertEqual(h_out.shape[0], 4)
        self.assertEqual(r_out.shape[0], 4)
        self.assertEqual(mask.tolist(), [1, 0, 1])
        self.assertEqual(padded_hidden_states_shape, torch.Size([4, 8]))

        # Finalize
        result = layer.finalize(h_out,
                                reduce_results=False,
                                padded_hidden_states_shape=padded_hidden_states_shape)
        self.assertEqual(result.shape[0], 3)

    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_world_size",
        return_value=2)
    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("torch.distributed.all_gather")
    def test_mc2_tp_split_allgather(self, mock_all_gather,
                                    mock_get_forward_context, mock_tp_rank,
                                    mock_tp_size):
        mock_context = MagicMock()
        mock_context.mc2_mask = torch.tensor([1, 0, 1, 0])
        mock_context.padded_num_tokens = 4
        mock_get_forward_context.return_value = mock_context

        layer = PrepareAndFinalizeWithMC2(self.moe_config)
        hidden_states = torch.randn(4, 8)
        router_logits = torch.randn(4, 2)

        prepare_output = layer.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp=False,
            replace_allreduce=False)
        h_out = prepare_output.hidden_states
        r_out = prepare_output.router_logits
        mask = prepare_output.mc2_mask
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # With TP=2, should split into 2 parts
        self.assertEqual(h_out.shape[0], 2)
        self.assertEqual(padded_hidden_states_shape, torch.Size([4, 8]))

        # Mock all_gather behavior
        def mock_all_gather_func(tensor_list, tensor, group=None):
            tensor_list[0] = tensor
            tensor_list[1] = tensor.clone()

        mock_all_gather.side_effect = mock_all_gather_func

        layer.split_hidden_states = [
            torch.zeros_like(h_out),
            torch.zeros_like(h_out)
        ]
        final_result = layer.finalize(h_out,
                                      reduce_results=False,
                                      padded_hidden_states_shape=padded_hidden_states_shape)

        # Should concat back to original size
        self.assertEqual(final_result.shape[0], 4)

    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_world_size",
        return_value=1)
    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    def test_all2all_prepare_finalize(self, mock_tp_rank, mock_tp_size):
        layer = PrepareAndFinalizeWithAll2All(self.moe_config)
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)

        prepare_output = layer.prepare(hidden_states, router_logits)
        h_out = prepare_output.hidden_states
        r_out = prepare_output.router_logits
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # Pad to tp_size=1, so no change
        self.assertEqual(h_out.shape[0], 3)
        self.assertEqual(padded_hidden_states_shape, torch.Size([3, 8]))

        result = layer.finalize(h_out,
                                reduce_results=False,
                                padded_hidden_states_shape=padded_hidden_states_shape)
        self.assertEqual(result.shape[0], 3)

    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_world_size",
        return_value=2)
    @patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    @patch("torch.distributed.all_gather")
    def test_all2all_tp_split_allgather(self, mock_all_gather, mock_tp_rank,
                                        mock_tp_size):
        layer = PrepareAndFinalizeWithAll2All(self.moe_config)
        hidden_states = torch.randn(2, 8)
        router_logits = torch.randn(2, 2)

        prepare_output = layer.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp=False,
            replace_allreduce=False)
        h_out = prepare_output.hidden_states
        r_out = prepare_output.router_logits
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # Split due to TP=2
        self.assertEqual(h_out.shape[0], 1)
        self.assertEqual(padded_hidden_states_shape, torch.Size([2, 8]))

        # Mock all_gather
        def mock_all_gather_func(tensor_list, tensor, group=None):
            tensor_list[0] = tensor
            tensor_list[1] = tensor.clone()

        mock_all_gather.side_effect = mock_all_gather_func

        layer.split_hidden_states = [
            torch.zeros_like(h_out),
            torch.zeros_like(h_out)
        ]
        final_result = layer.finalize(h_out,
                                      reduce_results=False,
                                      padded_hidden_states_shape=padded_hidden_states_shape)

        # Should concat back
        self.assertEqual(final_result.shape[0], 2)

    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_dp_group")
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.enable_sp",
           return_value=False)
    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.enable_sp_by_pass",
        return_value=False)
    def test_allgather_prepare_finalize(self, mock_enable_sp_by_pass,
                                        mock_enable_sp,
                                        mock_get_forward_context,
                                        mock_get_dp_group):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.max_tokens_across_dp = 6
        mock_get_forward_context.return_value = mock_context

        # Create a proper mock for DP group with working all_gather
        mock_dp_group = MagicMock()

        def mock_all_gather_func(tensor, dim):
            # Simulate DP=2: repeat the tensor along the specified dimension
            return torch.cat([tensor, tensor], dim=dim)

        mock_dp_group.all_gather = mock_all_gather_func
        mock_get_dp_group.return_value = mock_dp_group

        self.moe_config.dp_size = 2
        self.moe_config.tp_size = 1
        self.moe_config.ep_size = 1
        self.moe_config.dp_group = mock_dp_group

        layer = PrepareAndFinalizeWithAllGather(self.moe_config)

        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)

        prepare_output = layer.prepare(hidden_states, router_logits)
        h_out = prepare_output.hidden_states
        r_out = prepare_output.router_logits
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # After all-gather with DP=2, should double the batch size
        self.assertEqual(h_out.shape[0], 12)
        self.assertEqual(r_out.shape[0], 12)
        self.assertIsNone(padded_hidden_states_shape)

        # Finalize with reduce_scatter
        def mock_reduce_scatter_func(tensor, dim):
            # Simulate reduce_scatter: take first half
            return tensor[:3]

        mock_dp_group.reduce_scatter = mock_reduce_scatter_func
        result = layer.finalize(h_out,
                                reduce_results=False,
                                padded_hidden_states_shape=padded_hidden_states_shape)

        self.assertEqual(result.shape[0], 3)

        result_with_tp = layer.finalize(h_out, reduce_results=True)
        self.assertEqual(result_with_tp.shape[0], 3)

    @patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX")
    def test_moe_tp_prepare_finalize_source_rank(self, mock_extra_ctx):
        mock_extra_ctx.max_tokens_across_dp = 4

        self.moe_config.moe_source_group.world_size = 2
        self.moe_config.moe_source_group_world_size = 2
        self.moe_config.moe_source_group.rank_in_group = 1
        self.moe_config.moe_source_group_index = 1
        self.moe_config.moe_peer_group.rank_in_group = 0

        def mock_source_all_gather(tensor, dim):
            return torch.cat([tensor, tensor + 10], dim=dim)

        self.moe_config.moe_source_group.all_gather = mock_source_all_gather
        self.moe_config.moe_peer_group.broadcast = lambda tensor, src: tensor
        self.moe_config.moe_tp_group.all_reduce = lambda tensor: tensor + 100

        layer = PrepareAndFinalizeWithMoETPAllGather(self.moe_config)
        hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
        router_logits = torch.arange(12, dtype=torch.float32).view(3, 4)

        prepare_output = layer.prepare(hidden_states, router_logits)
        self.assertEqual(prepare_output.hidden_states.shape, torch.Size([8, 2]))
        self.assertEqual(prepare_output.router_logits.shape, torch.Size([8, 4]))

        result = layer.finalize(prepare_output.hidden_states, reduce_results=True)
        self.assertTrue(torch.equal(result, hidden_states + 110))

    @patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX")
    def test_moe_tp_prepare_finalize_peer_rank(self, mock_extra_ctx):
        mock_extra_ctx.max_tokens_across_dp = 3

        expected_global_hidden = torch.full((6, 2), 7.0)
        expected_global_router = torch.full((6, 4), 11.0)

        self.moe_config.moe_source_group = None
        self.moe_config.moe_source_group_world_size = 2
        self.moe_config.moe_source_group_index = 1
        self.moe_config.moe_peer_group.rank_in_group = 1

        def mock_peer_broadcast(tensor, src):
            if tuple(tensor.shape) == (6, 2):
                tensor.copy_(expected_global_hidden)
            elif tuple(tensor.shape) == (6, 4):
                tensor.copy_(expected_global_router)
            return tensor

        self.moe_config.moe_peer_group.broadcast = mock_peer_broadcast
        self.moe_config.moe_tp_group.all_reduce = lambda tensor: tensor + 5

        layer = PrepareAndFinalizeWithMoETPAllGather(self.moe_config)
        hidden_states = torch.arange(4, dtype=torch.float32).view(2, 2)
        router_logits = torch.arange(8, dtype=torch.float32).view(2, 4)

        prepare_output = layer.prepare(hidden_states, router_logits)
        self.assertTrue(torch.equal(prepare_output.hidden_states, expected_global_hidden))
        self.assertTrue(torch.equal(prepare_output.router_logits, expected_global_router))

        result = layer.finalize(prepare_output.hidden_states, reduce_results=True)
        self.assertTrue(torch.equal(result, expected_global_hidden[3:5] + 5))

    @patch("torch_npu.npu_dynamic_quant")
    @patch("vllm_ascend.ops.fused_moe.prepare_finalize.select_experts")
    @patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX")
    def test_moe_tp_prepare_finalize_w8a8_quantizes_before_communication(
        self,
        mock_extra_ctx,
        mock_select_experts,
        mock_dynamic_quant,
    ):
        mock_extra_ctx.max_tokens_across_dp = 4

        quant_hidden_states = torch.full((3, 2), 5, dtype=torch.int8)
        quant_scales = torch.full((3,), 0.25, dtype=torch.float32)
        mock_dynamic_quant.return_value = (quant_hidden_states, quant_scales)
        local_topk_weights = torch.full((3, 2), 0.5, dtype=torch.float32)
        local_topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
        mock_select_experts.return_value = (local_topk_weights, local_topk_ids)

        self.moe_config.moe_source_group.world_size = 2
        self.moe_config.moe_source_group_world_size = 2
        self.moe_config.moe_source_group.rank_in_group = 0
        self.moe_config.moe_source_group_index = 0
        self.moe_config.moe_peer_group.rank_in_group = 0

        def mock_source_all_gather(tensor, dim):
            return torch.cat([tensor, tensor], dim=dim)

        self.moe_config.moe_source_group.all_gather = mock_source_all_gather
        self.moe_config.moe_peer_group.broadcast = lambda tensor, src: tensor
        self.moe_config.moe_tp_group.all_reduce = lambda tensor: tensor

        layer = PrepareAndFinalizeWithMoETPAllGather(self.moe_config)
        hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
        router_logits = torch.arange(12, dtype=torch.float32).view(3, 4)

        prepare_output = layer.prepare(hidden_states, router_logits, quant_type=QuantType.W8A8)

        mock_dynamic_quant.assert_called_once()
        self.assertEqual(prepare_output.hidden_states.dtype, torch.int8)
        self.assertEqual(prepare_output.hidden_states.shape, torch.Size([8, 2]))
        self.assertEqual(prepare_output.pertoken_scale.shape, torch.Size([8]))
        self.assertEqual(prepare_output.router_logits.shape, torch.Size([0, 4]))
        self.assertEqual(mock_extra_ctx.moe_tp_topk_weights.shape, torch.Size([8, 2]))
        self.assertEqual(mock_extra_ctx.moe_tp_topk_ids.shape, torch.Size([8, 2]))
        self.assertEqual(mock_extra_ctx.moe_tp_topk_ids.dtype, torch.int32)

    @patch("torch_npu.npu_dynamic_quant")
    @patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX")
    def test_moe_tp_prepare_finalize_w8a8_peer_rank_skips_duplicate_quant(self, mock_extra_ctx, mock_dynamic_quant):
        mock_extra_ctx.max_tokens_across_dp = 3

        expected_global_hidden = torch.full((6, 2), 5, dtype=torch.int8)
        expected_global_router = torch.full((6, 4), 11.0)
        expected_global_scale = torch.full((6,), 0.125, dtype=torch.float32)

        self.moe_config.moe_source_group = None
        self.moe_config.moe_source_group_world_size = 2
        self.moe_config.moe_source_group_index = 1
        self.moe_config.custom_routing_function = object()
        self.moe_config.moe_peer_group.rank_in_group = 1

        def mock_peer_broadcast(tensor, src):
            if tuple(tensor.shape) == (6, 2):
                tensor.copy_(expected_global_hidden)
            elif tuple(tensor.shape) == (6, 4):
                tensor.copy_(expected_global_router)
            elif tuple(tensor.shape) == (6,):
                tensor.copy_(expected_global_scale)
            return tensor

        self.moe_config.moe_peer_group.broadcast = mock_peer_broadcast
        self.moe_config.moe_tp_group.all_reduce = lambda tensor: tensor

        layer = PrepareAndFinalizeWithMoETPAllGather(self.moe_config)
        hidden_states = torch.arange(4, dtype=torch.float32).view(2, 2)
        router_logits = torch.arange(8, dtype=torch.float32).view(2, 4)

        prepare_output = layer.prepare(hidden_states, router_logits, quant_type=QuantType.W8A8)

        mock_dynamic_quant.assert_not_called()
        self.assertTrue(torch.equal(prepare_output.hidden_states, expected_global_hidden))
        self.assertTrue(torch.equal(prepare_output.router_logits, expected_global_router))
        self.assertTrue(torch.equal(prepare_output.pertoken_scale, expected_global_scale))
