from unittest.mock import MagicMock, patch

from vllm.config import CUDAGraphMode
from tests.ut.base import TestBase
import vllm_ascend.ascend_forward_context as ascend_forward_context
from vllm_ascend.ascend_forward_context import (
    AscendDeviceType,
    MoECommType,
    get_mc2_max_num_tokens,
    select_moe_comm_method,
    set_mc2_tokens_capacity,
)


class TestAscendForwardContext(TestBase):

    def setUp(self):
        ascend_forward_context._mc2_tokens_capacity = None

    def tearDown(self):
        ascend_forward_context._mc2_tokens_capacity = None

    @staticmethod
    def _build_vllm_config():
        mock_config = MagicMock()
        mock_config.compilation_config.cudagraph_capture_sizes = [1024, 2048, 4096, 8192]
        mock_config.compilation_config.max_cudagraph_capture_size = 8192
        mock_config.parallel_config.tensor_parallel_size = 8
        mock_config.parallel_config.enable_expert_parallel = True
        mock_config.parallel_config.world_size_across_dp = 8
        mock_config.parallel_config.pipeline_parallel_size = 1
        mock_config.model_config.hf_text_config.moe_quantize = None
        mock_config.model_config.hf_text_config.quantize = None
        mock_config.scheduler_config.max_num_seqs = 64
        mock_config.speculative_config = None
        return mock_config

    @patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3)
    def test_get_mc2_max_num_tokens_caps_a3_graph_capacity(self, _mock_soc):
        vllm_config = self._build_vllm_config()

        max_num_tokens = get_mc2_max_num_tokens(vllm_config, max_num_reqs=64, uniform_decode_query_len=1)

        self.assertEqual(max_num_tokens, 4096)

    @patch("vllm_ascend.ascend_forward_context.get_ep_group")
    @patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
    @patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3)
    def test_select_moe_comm_method_falls_back_for_8k_prefill(
        self,
        _mock_soc,
        _mock_is_moe,
        mock_ep_group,
    ):
        vllm_config = self._build_vllm_config()
        mock_ep_group.return_value.world_size = 8
        set_mc2_tokens_capacity(vllm_config, max_num_reqs=64, uniform_decode_query_len=1)

        moe_comm_type = select_moe_comm_method(8192, vllm_config)

        self.assertEqual(moe_comm_type, MoECommType.ALLTOALL)

    @patch("vllm_ascend.ascend_forward_context.get_ep_group")
    @patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
    @patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3)
    def test_select_moe_comm_method_uses_allgather_for_8k_graph_prefill(
        self,
        _mock_soc,
        _mock_is_moe,
        mock_ep_group,
    ):
        vllm_config = self._build_vllm_config()
        mock_ep_group.return_value.world_size = 8
        set_mc2_tokens_capacity(vllm_config, max_num_reqs=64, uniform_decode_query_len=1)

        moe_comm_type = select_moe_comm_method(
            8192,
            vllm_config,
            aclgraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        )

        self.assertEqual(moe_comm_type, MoECommType.ALLGATHER)
