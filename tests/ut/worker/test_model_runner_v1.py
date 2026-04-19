import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner, _torch_cuda_wrapper


class TestNPUModelRunnerKVCache(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestTorchCudaWrapper(unittest.TestCase):

    def test_maps_stream_capture_check_to_npu(self):
        original = torch.cuda.is_current_stream_capturing
        try:
            with _torch_cuda_wrapper():
                self.assertIs(
                    torch.cuda.is_current_stream_capturing,
                    torch.npu.is_current_stream_capturing,
                )
        finally:
            torch.cuda.is_current_stream_capturing = original


class TestNPUModelRunnerProfileRun(unittest.TestCase):

    def _build_runner(self, cudagraph_mode):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.max_num_tokens = 8192
        runner.max_num_reqs = 1
        runner.pcp_size = 1
        runner.eplb_warmup = MagicMock()
        runner._dummy_run = MagicMock()
        runner.vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
        )
        return runner

    def test_profile_run_skips_mc2_dummy_run_for_graph_mode(self):
        runner = self._build_runner(CUDAGraphMode.FULL)
        with (
            patch("vllm_ascend.worker.model_runner_v1.get_mc2_tokens_capacity", return_value=4096),
            patch("vllm_ascend.worker.model_runner_v1.select_moe_comm_method", return_value=MoECommType.MC2),
            patch.object(GPUModelRunner, "profile_run", autospec=True) as super_profile_run,
        ):
            NPUModelRunner.profile_run(runner)

        runner._dummy_run.assert_not_called()
        super_profile_run.assert_called_once_with(runner)

    def test_profile_run_keeps_mc2_dummy_run_for_eager(self):
        runner = self._build_runner(CUDAGraphMode.NONE)
        with (
            patch("vllm_ascend.worker.model_runner_v1.get_mc2_tokens_capacity", return_value=4096),
            patch("vllm_ascend.worker.model_runner_v1.select_moe_comm_method", return_value=MoECommType.MC2),
            patch.object(GPUModelRunner, "profile_run", autospec=True) as super_profile_run,
        ):
            NPUModelRunner.profile_run(runner)

        runner._dummy_run.assert_called_once_with(4096, with_prefill=True, is_profile=True)
        super_profile_run.assert_called_once_with(runner)


if __name__ == "__main__":
    unittest.main()
