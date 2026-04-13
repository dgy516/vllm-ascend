from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE


def _fake_fused_moe_init(self, *args, **kwargs):
    torch.nn.Module.__init__(self)
    self.quant_config = MagicMock()
    self.quant_method_from_config = MagicMock()
    self.quant_method_from_config.supports_eplb = False
    self.quant_method_from_config.create_weights = MagicMock()
    self.quant_method_from_config.quant_method = None
    self.quant_config.get_quant_method.return_value = self.quant_method_from_config

    self.moe_parallel_config = MagicMock(
        tp_size=8,
        tp_rank=3,
        dp_size=4,
        dp_rank=1,
        pcp_size=1,
        pcp_rank=0,
        ep_size=1,
        ep_rank=0,
        use_ep=False,
        use_pplx_kernels=False,
        use_deepep_ht_kernels=False,
        use_deepep_ll_kernels=False,
        use_mori_kernels=False,
    )
    self.moe_config = MagicMock()
    self.moe_config.moe_parallel_config = self.moe_parallel_config
    self.moe_config.num_experts = kwargs["num_experts"]
    self.moe_config.num_local_experts = kwargs["num_experts"]
    self.moe_config.experts_per_token = kwargs["top_k"]
    self.num_experts = kwargs["num_experts"]
    self.top_k = kwargs["top_k"]
    self.layer_name = "model.layers.0.moe"
    self.custom_routing_function = None
    self.e_score_correction_bias = None
    self.hidden_size = kwargs["hidden_size"]
    self.intermediate_size_per_partition = kwargs["intermediate_size"] // self.moe_parallel_config.tp_size
    self.weight_loader = MagicMock()
    self.params_dtype = torch.bfloat16
    self.reduce_results = True
    self.tp_size = self.moe_parallel_config.tp_size
    self.tp_rank = self.moe_parallel_config.tp_rank
    self.dp_size = self.moe_parallel_config.dp_size
    self.dp_rank = self.moe_parallel_config.dp_rank
    self.ep_size = self.moe_parallel_config.ep_size
    self.ep_rank = self.moe_parallel_config.ep_rank
    self.activation = "silu"
    self.vllm_config = MagicMock(parallel_config=MagicMock(enable_dbo=False))
    self.gate = MagicMock()
    self.router = MagicMock()
    self.shared_experts = None
    self._routed_input_transform = MagicMock()


class TestAscendFusedMoEMoeTP(TestBase):
    @patch("vllm_ascend.ops.fused_moe.fused_moe.AscendFusedMoE._init_runner", return_value=MagicMock())
    @patch("vllm_ascend.ops.fused_moe.fused_moe.setup_moe_comm_method")
    @patch("vllm_ascend.ops.fused_moe.fused_moe.init_eplb_config", return_value=(None, None, None, 0))
    @patch("vllm_ascend.ops.fused_moe.fused_moe.get_moe_peer_group")
    @patch("vllm_ascend.ops.fused_moe.fused_moe.get_moe_source_group")
    @patch("vllm_ascend.ops.fused_moe.fused_moe.get_moe_tp_group")
    @patch("vllm_ascend.ops.fused_moe.fused_moe.get_mc2_group")
    @patch("vllm_ascend.ops.fused_moe.fused_moe.get_ep_group")
    @patch("vllm_ascend.ops.fused_moe.fused_moe.get_dp_group")
    @patch("vllm_ascend.ops.fused_moe.fused_moe.get_tp_group")
    @patch("vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config")
    @patch("vllm_ascend.ops.fused_moe.fused_moe.FusedMoE.__init__", new=_fake_fused_moe_init)
    def test_init_wires_moe_tp_groups(
        self,
        mock_get_ascend_config,
        mock_get_tp_group,
        mock_get_dp_group,
        mock_get_ep_group,
        mock_get_mc2_group,
        mock_get_moe_tp_group,
        mock_get_moe_source_group,
        mock_get_moe_peer_group,
        mock_init_eplb_config,
        mock_setup_moe_comm_method,
        mock_init_runner,
    ):
        del mock_init_eplb_config, mock_init_runner

        tp_group = MagicMock()
        dp_group = MagicMock()
        ep_group = MagicMock()
        mc2_group = MagicMock()
        moe_tp_group = MagicMock()
        moe_source_group = MagicMock()
        moe_peer_group = MagicMock()

        mock_get_tp_group.return_value = tp_group
        mock_get_dp_group.return_value = dp_group
        mock_get_ep_group.return_value = ep_group
        mock_get_mc2_group.return_value = mc2_group
        mock_get_moe_tp_group.return_value = moe_tp_group
        mock_get_moe_source_group.return_value = moe_source_group
        mock_get_moe_peer_group.return_value = moe_peer_group

        mock_ascend_config = MagicMock()
        mock_ascend_config.moe_parallel_config.enabled = True
        mock_ascend_config.moe_parallel_config.source_tp_rank = 0
        mock_ascend_config.multistream_overlap_gate = False
        mock_ascend_config.enable_shared_expert_dp = False
        mock_ascend_config.ascend_compilation_config.enable_static_kernel = False
        mock_ascend_config.eplb_config.dynamic_eplb = False
        mock_get_ascend_config.return_value = mock_ascend_config

        moe = AscendFusedMoE(
            num_experts=8,
            top_k=2,
            hidden_size=16,
            intermediate_size=32,
            quant_config=MagicMock(),
        )

        self.assertTrue(moe.moe_tp_mode)
        self.assertIs(moe.moe_config.tp_group, tp_group)
        self.assertIs(moe.moe_config.dp_group, dp_group)
        self.assertIs(moe.moe_config.ep_group, ep_group)
        self.assertIs(moe.moe_config.mc2_group, mc2_group)
        self.assertIs(moe.moe_config.moe_tp_group, moe_tp_group)
        self.assertIs(moe.moe_config.moe_source_group, moe_source_group)
        self.assertIs(moe.moe_config.moe_peer_group, moe_peer_group)
        self.assertEqual(moe.moe_config.source_tp_rank, 0)
        mock_setup_moe_comm_method.assert_called_once_with(moe.moe_config)
