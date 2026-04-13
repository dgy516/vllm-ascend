from unittest.mock import MagicMock, patch

import pytest
from vllm.config import ParallelConfig

import vllm_ascend.distributed.parallel_state as parallel_state
from vllm_ascend.distributed.parallel_state import (
    _FLASHCOMM2_ODP,
    _FLASHCOMM2_OTP,
    _LMTP,
    _MC2,
    _OTP,
    _P_TP,
    destroy_ascend_model_parallel,
    get_flashcomm2_odp_group,
    get_flashcomm2_otp_group,
    get_lmhead_tp_group,
    get_mc2_group,
    get_moe_peer_group,
    get_moe_source_group,
    get_moe_tp_group,
    get_otp_group,
    get_p_tp_group,
    init_ascend_model_parallel,
)


@pytest.fixture
def parallel_config():
    return ParallelConfig(
        data_parallel_size=2,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
    )


@pytest.fixture
def mock_distributed():
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_world_size', return_value=16), \
         patch('torch.distributed.get_backend', return_value='nccl'), \
         patch('vllm_ascend.distributed.parallel_state.get_world_group') as mock_group, \
         patch('vllm_ascend.distributed.parallel_state.get_tp_group') as mock_tp_group:
        mock_group.return_value.local_rank = 0
        mock_group.return_value.device_group = MagicMock()
        mock_tp_group.return_value.world_size = 4
        yield


def test_init_ascend_model_parallel(mock_distributed, parallel_config):
    mock_ascend_config = MagicMock()
    mock_ascend_config.finegrained_tp_config.lmhead_tensor_parallel_size = 2
    mock_ascend_config.finegrained_tp_config.oproj_tensor_parallel_size = 2
    mock_ascend_config.finegrained_tp_config.embedding_tensor_parallel_size = 2
    mock_ascend_config.finegrained_tp_config.mlp_tensor_parallel_size = 2
    mock_ascend_config.moe_parallel_config.enabled = False
    mock_ascend_config.flashcomm2_oproj_tensor_parallel_size = 2
    mock_ascend_config.pd_tp_ratio = 2
    mock_ascend_config.num_head_replica = 0
    mock_ascend_config.pd_head_ratio = 2
    mock_ascend_config.eplb_config.dynamic_eplb = False
    mock_ascend_config.multistream_overlap_gate = False
    mock_ascend_config.layer_sharding = None
    mock_vllm_config = MagicMock()
    mock_vllm_config.kv_transfer_config.is_kv_producer = True
    mock_envs_ascend = MagicMock()
    mock_envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE = 2
    mock_envs_ascend.VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL = 0
    with patch('vllm_ascend.distributed.parallel_state.model_parallel_initialized', return_value=False), \
         patch('vllm_ascend.distributed.parallel_state.init_model_parallel_group'), \
         patch('vllm_ascend.distributed.parallel_state.get_current_vllm_config', return_value=mock_vllm_config), \
         patch('vllm_ascend.distributed.parallel_state.get_ascend_config', return_value=mock_ascend_config), \
         patch('vllm_ascend.utils.envs_ascend', new=mock_envs_ascend), \
         patch('vllm_ascend.utils.get_ascend_config', return_value=mock_ascend_config):
        init_ascend_model_parallel(parallel_config)

        mc2_group = get_mc2_group()
        lmheadtp_group = get_lmhead_tp_group()
        otp_group = get_otp_group()
        flashcomm2_otp_group = get_flashcomm2_otp_group()
        flashcomm2_odp_group = get_flashcomm2_odp_group()
        p_tp_group = get_p_tp_group()
        assert mc2_group is not None
        assert otp_group is not None
        assert flashcomm2_otp_group is not None
        assert flashcomm2_odp_group is not None
        assert lmheadtp_group is not None
        assert p_tp_group is not None

        destroy_ascend_model_parallel()
        assert _MC2 is None
        assert _LMTP is None
        assert _OTP is None
        assert _FLASHCOMM2_OTP is None
        assert _FLASHCOMM2_ODP is None
        assert _P_TP is None


def test_init_ascend_model_parallel_with_moe_tp(mock_distributed):
    parallel_config = ParallelConfig(
        data_parallel_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )

    mock_ascend_config = MagicMock()
    mock_ascend_config.finegrained_tp_config.lmhead_tensor_parallel_size = 0
    mock_ascend_config.finegrained_tp_config.oproj_tensor_parallel_size = 0
    mock_ascend_config.finegrained_tp_config.embedding_tensor_parallel_size = 0
    mock_ascend_config.finegrained_tp_config.mlp_tensor_parallel_size = 0
    mock_ascend_config.moe_parallel_config.enabled = True
    mock_ascend_config.moe_parallel_config.source_tp_rank = 0
    mock_ascend_config.flashcomm2_oproj_tensor_parallel_size = 1
    mock_ascend_config.pd_tp_ratio = 1
    mock_ascend_config.num_head_replica = 0
    mock_ascend_config.pd_head_ratio = 1
    mock_ascend_config.eplb_config.dynamic_eplb = False
    mock_ascend_config.multistream_overlap_gate = False
    mock_ascend_config.layer_sharding = None

    mock_vllm_config = MagicMock()
    mock_vllm_config.kv_transfer_config = None
    mock_envs_ascend = MagicMock()
    mock_envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE = 1
    mock_envs_ascend.VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL = 0

    created_groups = {}

    def _init_group_side_effect(group_ranks, local_rank, backend, group_name):
        group = MagicMock(name=group_name)
        group.group_name = group_name
        created_groups[group_name] = (group, group_ranks)
        return group

    with patch("vllm_ascend.distributed.parallel_state.model_parallel_initialized", return_value=False), patch(
        "vllm_ascend.distributed.parallel_state.init_model_parallel_group", side_effect=_init_group_side_effect
    ), patch(
        "vllm_ascend.distributed.parallel_state.init_optional_model_parallel_group", side_effect=_init_group_side_effect
    ), patch(
        "vllm_ascend.distributed.parallel_state.get_current_vllm_config", return_value=mock_vllm_config
    ), patch(
        "vllm_ascend.distributed.parallel_state.get_ascend_config", return_value=mock_ascend_config
    ), patch("vllm_ascend.utils.envs_ascend", new=mock_envs_ascend), patch(
        "vllm_ascend.utils.get_ascend_config", return_value=mock_ascend_config
    ):
        init_ascend_model_parallel(parallel_config)

        moe_tp_group = get_moe_tp_group()
        moe_source_group = get_moe_source_group()
        moe_peer_group = get_moe_peer_group()

        assert moe_tp_group is created_groups["moe_tp"][0]
        assert moe_source_group is created_groups["moe_source"][0]
        assert moe_peer_group is not None
        assert created_groups["moe_tp"][1] == [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
        assert created_groups["moe_source"][1] == [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]

        destroy_ascend_model_parallel()
        assert parallel_state._MOE_TP is None
        assert parallel_state._MOE_SOURCE is None


def test_init_ascend_model_parallel_with_moe_tp_peer_rank_builds_tp_rank_source_group():
    parallel_config = ParallelConfig(
        data_parallel_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )

    mock_world_group = MagicMock()
    mock_world_group.local_rank = 1
    mock_world_group.rank = 1
    mock_world_group.device_group = MagicMock()
    mock_tp_group = MagicMock()
    mock_tp_group.world_size = 2

    mock_ascend_config = MagicMock()
    mock_ascend_config.finegrained_tp_config.lmhead_tensor_parallel_size = 0
    mock_ascend_config.finegrained_tp_config.oproj_tensor_parallel_size = 0
    mock_ascend_config.finegrained_tp_config.embedding_tensor_parallel_size = 0
    mock_ascend_config.finegrained_tp_config.mlp_tensor_parallel_size = 0
    mock_ascend_config.moe_parallel_config.enabled = True
    mock_ascend_config.moe_parallel_config.source_tp_rank = 0
    mock_ascend_config.flashcomm2_oproj_tensor_parallel_size = 1
    mock_ascend_config.pd_tp_ratio = 1
    mock_ascend_config.num_head_replica = 0
    mock_ascend_config.pd_head_ratio = 1
    mock_ascend_config.eplb_config.dynamic_eplb = False
    mock_ascend_config.multistream_overlap_gate = False
    mock_ascend_config.layer_sharding = None

    mock_vllm_config = MagicMock()
    mock_vllm_config.kv_transfer_config = None
    mock_envs_ascend = MagicMock()
    mock_envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE = 1
    mock_envs_ascend.VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL = 0

    created_groups = {}

    def _init_group_side_effect(group_ranks, local_rank, backend, group_name):
        group = MagicMock(name=group_name)
        group.group_name = group_name
        created_groups[group_name] = (group, group_ranks)
        return group

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_world_size", return_value=16
    ), patch("torch.distributed.get_backend", return_value="nccl"), patch(
        "vllm_ascend.distributed.parallel_state.get_world_group", return_value=mock_world_group
    ), patch(
        "vllm_ascend.distributed.parallel_state.get_tp_group", return_value=mock_tp_group
    ), patch(
        "vllm_ascend.distributed.parallel_state.model_parallel_initialized", return_value=False
    ), patch(
        "vllm_ascend.distributed.parallel_state.init_model_parallel_group", side_effect=_init_group_side_effect
    ), patch(
        "vllm_ascend.distributed.parallel_state.init_optional_model_parallel_group", side_effect=_init_group_side_effect
    ), patch(
        "vllm_ascend.distributed.parallel_state.get_current_vllm_config", return_value=mock_vllm_config
    ), patch(
        "vllm_ascend.distributed.parallel_state.get_ascend_config", return_value=mock_ascend_config
    ), patch("vllm_ascend.utils.envs_ascend", new=mock_envs_ascend), patch(
        "vllm_ascend.utils.get_ascend_config", return_value=mock_ascend_config
    ):
        init_ascend_model_parallel(parallel_config)

        assert get_moe_tp_group() is created_groups["moe_tp"][0]
        assert get_moe_source_group() is created_groups["moe_source"][0]
        assert created_groups["moe_source"][1] == [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]

        destroy_ascend_model_parallel()
        assert parallel_state._MOE_TP is None
        assert parallel_state._MOE_SOURCE is None
