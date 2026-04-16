from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.ubatch_utils import UBatchSlice

from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import create_ascend_forward_context


class _FakeForwardContext:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestAscendForwardContext(TestBase):
    def test_only_first_ubatch_enables_first_layer_sync(self):
        vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(
                fast_moe_cold_start=False,
                static_forward_context={},
                static_all_moe_layers=None,
            ),
        )
        cur_forward_context = SimpleNamespace(
            flash_comm_v1_enabled=False,
            flashcomm_v2_enabled=False,
            moe_comm_type=None,
            in_profile_run=False,
            capturing=False,
            mmrs_fusion=False,
            is_first_layer=True,
            layer_idx=0,
            model_instance=None,
            prefetch_mlp_gate_up_proj=False,
            prefetch_mlp_down_proj=False,
            is_draft_model=False,
            mc2_mask=None,
        )
        ubatch_slices = [
            UBatchSlice(slice(0, 1), slice(0, 4)),
            UBatchSlice(slice(0, 1), slice(4, 8)),
        ]
        positions = torch.arange(8)

        with patch("vllm_ascend.ascend_forward_context.ForwardContext", _FakeForwardContext), patch(
            "vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size", return_value=1
        ), patch(
            "vllm_ascend.ascend_forward_context.get_dp_group", return_value=SimpleNamespace(world_size=1)
        ), patch(
            "vllm_ascend.ascend_forward_context.get_mc2_mask", return_value=None
        ), patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_comm_method", return_value=None
        ), patch(
            "vllm_ascend.ops.rotary_embedding.update_cos_sin"
        ), patch(
            "vllm_ascend.ops.rotary_embedding.get_cos_and_sin_slice", return_value=(None, None)
        ), patch(
            "vllm_ascend.ops.rotary_embedding.get_cos_and_sin_mla", return_value=(None, None)
        ):
            first_ctx = create_ascend_forward_context(
                cur_forward_context=cur_forward_context,
                attn_metadata=None,
                vllm_config=vllm_config,
                ubatch_slices=ubatch_slices,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                ubatch_num=0,
                positions=positions,
            )
            second_ctx = create_ascend_forward_context(
                cur_forward_context=cur_forward_context,
                attn_metadata=None,
                vllm_config=vllm_config,
                ubatch_slices=ubatch_slices,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                ubatch_num=1,
                positions=positions,
            )

        self.assertTrue(first_ctx.dbo_first_layer_sync)
        self.assertFalse(second_ctx.dbo_first_layer_sync)
