import unittest
from unittest.mock import MagicMock, patch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType, select_moe_comm_method


class TestAscendForwardContext(unittest.TestCase):

    @patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
    @patch("vllm_ascend.ascend_config.get_ascend_config")
    def test_select_moe_comm_method_prefers_moe_tp_mode(self, mock_get_ascend_config, mock_is_moe_model):
        del mock_is_moe_model

        mock_ascend_config = MagicMock()
        mock_ascend_config.moe_parallel_config.enabled = True
        mock_get_ascend_config.return_value = mock_ascend_config

        vllm_config = MagicMock()

        moe_comm_type = select_moe_comm_method(num_tokens=32, vllm_config=vllm_config)

        self.assertEqual(moe_comm_type, MoECommType.MOE_TP_ALLGATHER)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_extra_forward_context_proxy_supports_moe_tp_topk_attrs(self, mock_get_forward_context):
        mock_context = MagicMock()
        mock_get_forward_context.return_value = mock_context

        topk_weights = object()
        topk_ids = object()

        _EXTRA_CTX.moe_tp_topk_weights = topk_weights
        _EXTRA_CTX.moe_tp_topk_ids = topk_ids

        self.assertIs(_EXTRA_CTX.moe_tp_topk_weights, topk_weights)
        self.assertIs(_EXTRA_CTX.moe_tp_topk_ids, topk_ids)
