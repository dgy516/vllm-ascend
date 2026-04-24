import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from vllm import config

from tests.ut.base import TestBase
from vllm_ascend import ascend_config
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.linear import (AscendMergedColumnParallelLinear,
                                    AscendReplicatedLinear,
                                    AscendRowParallelLinear,
                                    AscendUnquantizedLinearMethod)
from vllm_ascend.ops.linear_op import (SequenceColumnParallelOp,
                                       _maybe_all_gather_and_maybe_unpad_with_cache)


class BaseLinearTest(unittest.TestCase):

    def setUp(self):
        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0

        parallel_state._MLP_TP = self.mock_group
        parallel_state._OTP = self.mock_group

        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.finegrained_tp_config.oproj_tensor_parallel_size = 2
        self.mock_ascend_config.finegrained_tp_config.mlp_tensor_parallel_size = 2

        self.patches = [
            patch("vllm_ascend.ascend_config.get_ascend_config",
                  return_value=self.mock_ascend_config),
            patch("vllm_ascend.distributed.parallel_state.get_otp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.distributed.parallel_state.get_mlp_tp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.ops.linear_op.get_tp_group",
                  return_value=self.mock_group),
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=self.mock_group,
            ),
            patch("vllm_ascend.utils.mlp_tp_enable", return_value=True),
            patch("vllm_ascend.utils.oproj_tp_enable", return_value=True)
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()


class TestAscendUnquantizedLinearMethod(TestBase):

    def setUp(self):
        self.method = AscendUnquantizedLinearMethod()
        self.layer = mock.MagicMock()
        mock_dtype = mock.PropertyMock(return_value=torch.float16)
        type(self.layer.weight.data).dtype = mock_dtype

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz0(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "1"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz1(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "2"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz2(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_called_once()


class TestAscendRowParallelLinear(BaseLinearTest):

    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method",
           return_value=MagicMock())
    def test_mlp_optimize(self, mock_get_weight_prefetch_method):

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="down_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)

    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method",
           return_value=MagicMock())
    def test_oproj_tp(self, mock_get_weight_prefetch_method):

        config._current_vllm_config = MagicMock()

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="o_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._OTP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)


class TestAscendMergedColumnParallelLinear(BaseLinearTest):

    def test_merged_mlp_tp_init(self):

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="gate_up_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)


class TestAscendReplicatedLinear(BaseLinearTest):

    def test_init_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(
            isinstance(linear.quant_method, AscendUnquantizedLinearMethod))

    def test_init_without_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(
            isinstance(linear.quant_method, AscendUnquantizedLinearMethod))


class TestLinearOpGatherReuseCache(unittest.TestCase):

    @patch("vllm_ascend.ops.linear_op.get_forward_context")
    @patch("vllm_ascend.ops.linear_op._EXTRA_CTX")
    @patch("vllm_ascend.ops.linear_op.torch.ops.vllm.maybe_all_gather_and_maybe_unpad")
    def test_reuses_all_gather_for_same_input(self, mock_all_gather, mock_extra_ctx, mock_get_forward_context):
        x = torch.randn(2, 4)
        gathered = torch.randn(4, 4)
        mock_get_forward_context.return_value = MagicMock()
        mock_extra_ctx.all_gather_reuse_cache = {}
        mock_all_gather.return_value = gathered

        out1 = _maybe_all_gather_and_maybe_unpad_with_cache(x, label=True)
        out2 = _maybe_all_gather_and_maybe_unpad_with_cache(x, label=True)

        self.assertIs(out1, gathered)
        self.assertIs(out2, gathered)
        mock_all_gather.assert_called_once_with(x, label=True)

    @patch("vllm_ascend.ops.linear_op.get_forward_context")
    @patch("vllm_ascend.ops.linear_op._EXTRA_CTX")
    @patch("vllm_ascend.ops.linear_op.torch.ops.vllm.maybe_all_gather_and_maybe_unpad")
    def test_different_views_do_not_share_cache(self, mock_all_gather, mock_extra_ctx, mock_get_forward_context):
        x = torch.randn(4, 4)
        x_view = x[:2]
        mock_get_forward_context.return_value = MagicMock()
        mock_extra_ctx.all_gather_reuse_cache = {}
        mock_all_gather.side_effect = [torch.randn(8, 4), torch.randn(4, 4)]

        _maybe_all_gather_and_maybe_unpad_with_cache(x, label=True)
        _maybe_all_gather_and_maybe_unpad_with_cache(x_view, label=True)

        self.assertEqual(mock_all_gather.call_count, 2)

    @patch("vllm_ascend.ops.linear_op._maybe_all_gather_and_maybe_unpad_with_cache")
    def test_sequence_column_uses_cached_all_gather(self, mock_all_gather):
        x = torch.randn(2, 4)
        gathered_x = torch.randn(4, 4)
        mock_all_gather.return_value = gathered_x

        op = object.__new__(SequenceColumnParallelOp)
        op.layer = MagicMock(prefix="model.layers.3.self_attn.qkv_proj")
        op.bias = None
        op.skip_bias_add = False
        op.quant_method = MagicMock()
        op.quant_method.apply.return_value = torch.randn(2, 4)
        op.gather_output = False

        SequenceColumnParallelOp.apply_impl(op, x)

        mock_all_gather.assert_called_once_with(x, label=True)
        op.quant_method.apply.assert_called_once_with(op.layer, gathered_x, None)


if __name__ == '__main__':
    unittest.main()
