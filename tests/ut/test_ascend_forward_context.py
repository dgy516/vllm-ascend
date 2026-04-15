from unittest.mock import MagicMock, patch

from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import MoECommType, select_moe_comm_method
from vllm_ascend.utils import AscendDeviceType


class TestSelectMoeCommMethod(TestBase):
    @patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
    @patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=128)
    @patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.ascend_forward_context.get_ep_group")
    def test_select_moe_comm_method_allows_fused_mc2_for_draft_decode(
        self,
        mock_get_ep_group,
        _mock_get_ascend_device_type,
        _mock_get_mc2_tokens_capacity,
        _mock_is_moe_model,
    ):
        mock_get_ep_group.return_value.world_size = 8
        vllm_config = MagicMock()
        vllm_config.parallel_config.enable_expert_parallel = True

        with patch("vllm_ascend.ascend_forward_context.envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2", 1, create=True):
            moe_comm_type = select_moe_comm_method(
                num_tokens=64,
                vllm_config=vllm_config,
                is_draft_model=True,
            )

        self.assertEqual(moe_comm_type, MoECommType.FUSED_MC2)

    @patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
    @patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=128)
    @patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.ascend_forward_context.get_ep_group")
    def test_select_moe_comm_method_allows_fused_mc2_for_draft_prefill(
        self,
        mock_get_ep_group,
        _mock_get_ascend_device_type,
        _mock_get_mc2_tokens_capacity,
        _mock_is_moe_model,
    ):
        mock_get_ep_group.return_value.world_size = 8
        vllm_config = MagicMock()
        vllm_config.parallel_config.enable_expert_parallel = True

        with patch("vllm_ascend.ascend_forward_context.envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2", 1, create=True):
            moe_comm_type = select_moe_comm_method(
                num_tokens=256,
                vllm_config=vllm_config,
                is_draft_model=True,
            )

        self.assertEqual(moe_comm_type, MoECommType.FUSED_MC2)
