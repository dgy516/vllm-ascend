import unittest
from unittest.mock import MagicMock, patch

from vllm_ascend.ascend_forward_context import MoECommType, select_moe_comm_method


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
