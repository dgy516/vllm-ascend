from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (AscendAttentionBackend,
                                                AscendC8AttentionBackendImpl,
                                                AscendAttentionBackendImpl,
                                                AscendAttentionMetadataBuilder,
                                                AscendAttentionState)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.utils import AscendDeviceType


class TestAscendAttentionBackend(TestBase):

    def setUp(self):
        self.mock_config = MagicMock()

        mock_parallel_config = MagicMock()
        mock_parallel_config.prefill_context_parallel_size = 1
        mock_parallel_config.decode_context_parallel_size = 1

        self.mock_config.parallel_config = mock_parallel_config

        self.utils_patcher = patch(
            'vllm_ascend.attention.utils.get_current_vllm_config',
            return_value=self.mock_config)
        self.utils_patcher.start()

        from vllm_ascend.attention.utils import enable_cp
        enable_cp.cache_clear()

    def test_get_name(self):
        self.assertEqual(AscendAttentionBackend.get_name(), "CUSTOM")

    def test_get_impl_cls(self):
        self.assertEqual(AscendAttentionBackend.get_impl_cls(),
                         AscendAttentionBackendImpl)

    def test_get_builder_cls(self):
        self.assertEqual(AscendAttentionBackend.get_builder_cls(),
                         AscendAttentionMetadataBuilder)

    def test_get_kv_cache_shape_not(self):
        result = AscendAttentionBackend.get_kv_cache_shape(10, 20, 30, 40)
        self.assertEqual(result, (2, 10, 20, 30, 40))

    def test_swap_blocks(self):
        src_kv_cache = [torch.zeros((10, 20)), torch.zeros((10, 20))]
        dst_kv_cache = [torch.zeros((10, 20)), torch.zeros((10, 20))]
        src_to_dst = torch.tensor([[0, 1], [2, 3]])
        AscendAttentionBackend.swap_blocks(src_kv_cache, dst_kv_cache,
                                           src_to_dst)
        self.assertTrue(torch.all(dst_kv_cache[0][1] == src_kv_cache[0][0]))
        self.assertTrue(torch.all(dst_kv_cache[1][3] == src_kv_cache[1][2]))

    def test_copy_blocks(self):
        kv_caches = [torch.zeros((10, 20)), torch.zeros((10, 20))]
        src_to_dists = torch.tensor([[0, 1], [2, 3]])
        AscendAttentionBackend.copy_blocks(kv_caches, src_to_dists)
        self.assertTrue(torch.all(kv_caches[0][1] == kv_caches[0][0]))
        self.assertTrue(torch.all(kv_caches[1][3] == kv_caches[1][2]))


class TestAscendAttentionMetadataBuilder(TestBase):

    def setUp(self):
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.speculative_config = None
        self.mock_vllm_config.model_config.max_model_len = 640
        self.mock_vllm_config.model_config.hf_text_config.sliding_window = None
        self.mock_vllm_config.cache_config.block_size = 64
        self.mock_vllm_config.compilation_config.cudagraph_mode = None
        self.mock_vllm_config.scheduler_config.max_num_seqs = 10
        self.mock_vllm_config.scheduler_config.decode_max_num_seqs = 10
        self.mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        self.mock_device = 'cpu:0'
        torch.Tensor.pin_memory = lambda x: x  # noqa
        self.builder = AscendAttentionMetadataBuilder(None, None,
                                                      self.mock_vllm_config,
                                                      self.mock_device)

    def test_reorder_batch(self):
        mock_input_batch = MagicMock()
        mock_scheduler_output = MagicMock()

        result = self.builder.reorder_batch(mock_input_batch,
                                            mock_scheduler_output)

        self.assertFalse(result)

    @patch('vllm_ascend.attention.attention_v1.AscendMetadata')
    def test_build(self, mock_ascend_metadata):
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 2, 5, 9]),
            query_start_loc_cpu=torch.tensor([0, 2, 5, 9]),
            seq_lens_cpu=torch.tensor([4, 5, 6]),
            num_reqs=3,
            num_actual_tokens=15,
            max_query_len=6,
            decode_token_per_req=torch.tensor([1, 1, 1]),
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping=torch.tensor(range(20)),
            actual_seq_lengths_q=torch.tensor([0, 1, 2]),
            positions=torch.tensor([10, 10]),
            attn_state=AscendAttentionState.ChunkedPrefill,
            num_computed_tokens_cpu=None,
            seq_lens=None,
            max_seq_len=6)
        mock_model = MagicMock()

        self.builder.build(1, common_attn_metadata, mock_model)


class TestAscendAttentionBackendImpl(TestBase):

    def setUp(self):
        self.mock_event = MagicMock()
        self.mock_event.record.return_value = None
        self.mock_event.wait.return_value = None

        self.mock_stream = MagicMock()
        self.event_patcher = patch('torch_npu.npu.Event',
                                   return_value=self.mock_event)
        self.stream_patcher = patch('torch_npu.npu.current_stream',
                                    return_value=self.mock_stream)

        self.event_patcher.start()
        self.stream_patcher.start()

        self.layer = MagicMock()
        self.layer.layer_name = "test_layer"
        self.layer._k_scale_float = 1.0
        self.layer._v_scale_float = 1.0
        self.attention_type = MagicMock()
        self.attention_type.DECODER = "decoder"
        self.attention_type.ENCODER = "encoder"
        self.attn_metadata = MagicMock()
        self.attn_metadata.return_value = "1"
        self.layer_no_quant = MagicMock(
            spec=['layer_name', '_k_scale_float', '_v_scale_float'])
        self.layer_no_quant.layer_name = "test_layer"
        self.layer_no_quant._k_scale_float = 1.0
        self.layer_no_quant._v_scale_float = 1.0
        self.mock_vllm_config = MagicMock()
        self.config_patcher = patch(
            'vllm_ascend.attention.attention_v1.get_current_vllm_config',
            return_value=self.mock_vllm_config)
        self.config_patcher.start()

        self.impl = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None)

        self.impl_192 = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=192,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None)

        self.impl_error = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=192,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None)

        self.impl_swa = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=1024,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None)

    def test_forward_no_attn_metadata(self):
        """Test forward pass when attn_metadata is None"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 0, 0, 8, 64)
        layer = self.layer_no_quant
        output = torch.empty_like(query)

        output = self.impl.forward(layer, query, key, value, kv_cache, None,
                                   output)

        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu.npu_fused_infer_attention_score')
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    def test_forward_fused_infer_attention(
            self, mock_get_forward_context,
            mock_npu_fused_infer_attention_score, mock_npu_reshape_and_cache):
        """Test forward pass in PrefillCacheHit state"""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 64)
        value = torch.randn(10, 8, 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.PrefillCacheHit
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.actual_seq_lengths_q = [10]
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.num_decode_tokens = 0
        metadata.num_decodes = 0
        metadata.num_prefills = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant

        mock_get_forward_context.return_value = MagicMock(capturing=False)
        mock_npu_fused_infer_attention_score.return_value = (torch.ones(
            10, 8, 64), torch.ones(10, 8, 64))
        output = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output)

        mock_npu_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8, 64)

    @patch('vllm_ascend.attention.attention_v1.using_paged_attention')
    @patch('torch_npu._npu_paged_attention')
    @patch('torch_npu._npu_reshape_and_cache')
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    def test_forward_paged_attention(self, mock_get_forward_context,
                                     mock_npu_reshape_and_cache,
                                     mock_paged_attention,
                                     mock_using_paged_attention):
        """Test forward pass in DecodeOnly state"""
        query = torch.randn(4, 8 * 64)
        key = torch.randn(4, 8 * 64)
        value = torch.randn(4, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([4])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 4
        metadata.slot_mapping = torch.zeros(4, dtype=torch.long)
        metadata.num_decodes = 4
        metadata.num_prefills = 0
        layer = self.layer_no_quant
        mock_using_paged_attention.return_value = True

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        output = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output)

        mock_paged_attention.assert_called_once()
        assert output.shape == (4, 8 * 64)

    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch('torch_npu.npu_fused_infer_attention_score')
    @patch('torch_npu._npu_reshape_and_cache')
    def test_forward_decode_only_swa(self, mock_npu_reshape_and_cache,
                                     mock_fused_infer_attention_score,
                                     mock_get_forward_context):
        """Test forward pass in DecodeOnly state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty(10, 8, 64)

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10] * 10)
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 100
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 10
        metadata.num_prefills = 0
        layer = self.layer_no_quant
        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8,
                                                                    64), 1)
        output = self.impl_swa.forward(layer, query, key, value, kv_cache,
                                       metadata, output)
        print(output.shape)
        mock_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8, 64)

    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch('torch_npu._npu_paged_attention')
    @patch('torch_npu.npu_fused_infer_attention_score')
    @patch('torch_npu._npu_reshape_and_cache')
    def test_forward_decode_only_swa_seq_len_mismatch(
            self, mock_npu_reshape_and_cache, mock_fused_infer_attention_score,
            mock_paged_attention, mock_get_forward_context):
        """Test forward pass in DecodeOnly state when seq)len_mismatch"""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 64)
        value = torch.randn(10, 8, 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10])  # len == 1 != query.size(0)==10
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant
        metadata.num_decodes = 10
        metadata.num_prefills = 0
        metadata.actual_seq_lengths_q = [10]

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8, 64),
                                                         torch.ones(10, 8, 64))

        output = self.impl_swa.forward(layer, query, key, value, kv_cache,
                                       metadata, output)

        mock_paged_attention.assert_not_called()
        mock_fused_infer_attention_score.assert_called_once()

        assert output.shape == (10, 8, 64)

    @patch('torch_npu.npu_fused_infer_attention_score')
    def test_c8_decode_passes_antiquant_scales_to_fia(self, mock_fia):
        impl = object.__new__(AscendC8AttentionBackendImpl)
        impl.num_heads = 2
        impl.num_kv_heads = 1
        impl.head_size = 8
        impl.scale = 0.125
        impl.key_cache = torch.zeros((4, 32, 1, 8), dtype=torch.int8)
        impl.value_cache = torch.zeros((4, 32, 1, 8), dtype=torch.int8)

        layer = MagicMock()
        layer._c8_k_aq_scale = torch.ones((1, 1, 1, 8), dtype=torch.float32)
        layer._c8_k_aq_offset = torch.zeros((1, 1, 1, 8), dtype=torch.float32)
        layer._c8_v_aq_scale = torch.ones((1, 1, 1, 8), dtype=torch.float32)
        layer._c8_v_aq_offset = torch.zeros((1, 1, 1, 8), dtype=torch.float32)

        metadata = MagicMock()
        metadata.actual_seq_lengths_q = [4, 8]
        metadata.seq_lens_list = [10, 20]
        metadata.block_tables = torch.zeros((2, 1), dtype=torch.int32)
        query = torch.zeros((8, 2, 8), dtype=torch.float32)
        output = torch.empty_like(query)
        mock_fia.return_value = (torch.ones((8, 2, 1, 8), dtype=torch.float32), None)

        impl._forward_c8_decode(query, metadata, output, layer)

        args = mock_fia.call_args.args
        kwargs = mock_fia.call_args.kwargs
        self.assertEqual(tuple(args[0].shape), (8, 2, 1, 8))
        self.assertEqual(tuple(kwargs["block_table"].shape), (8, 1))
        self.assertEqual(kwargs["actual_seq_lengths_kv"], [7, 8, 9, 10, 17, 18, 19, 20])
        self.assertIs(kwargs["key_antiquant_scale"], layer._c8_k_aq_scale)
        self.assertIs(kwargs["key_antiquant_offset"], layer._c8_k_aq_offset)
        self.assertIs(kwargs["value_antiquant_scale"], layer._c8_v_aq_scale)
        self.assertIs(kwargs["value_antiquant_offset"], layer._c8_v_aq_offset)
        self.assertEqual(kwargs["input_layout"], "BNSD")
        self.assertEqual(kwargs["key_antiquant_mode"], 0)
        self.assertEqual(kwargs["value_antiquant_mode"], 0)
        self.assertTrue(torch.all(output == 1))

    @patch('torch_npu.npu_fused_infer_attention_score')
    def test_c8_chunked_decode_expands_mtp_metadata(self, mock_fia):
        impl = object.__new__(AscendC8AttentionBackendImpl)
        impl.num_heads = 2
        impl.num_kv_heads = 1
        impl.head_size = 8
        impl.scale = 0.125
        impl.key_cache = torch.zeros((4, 32, 1, 8), dtype=torch.int8)
        impl.value_cache = torch.zeros((4, 32, 1, 8), dtype=torch.int8)

        layer = MagicMock()
        layer._c8_k_aq_scale = torch.ones((1, 1, 1, 8), dtype=torch.float32)
        layer._c8_k_aq_offset = torch.zeros((1, 1, 1, 8), dtype=torch.float32)
        layer._c8_v_aq_scale = torch.ones((1, 1, 1, 8), dtype=torch.float32)
        layer._c8_v_aq_offset = torch.zeros((1, 1, 1, 8), dtype=torch.float32)

        metadata = MagicMock()
        metadata.num_decode_tokens = 8
        metadata.num_decodes = 2
        metadata.num_prefills = 0
        metadata.actual_seq_lengths_q = [4, 8]
        metadata.seq_lens_list = [10, 20]
        metadata.block_tables = torch.zeros((2, 1), dtype=torch.int32)
        query = torch.zeros((8, 2, 8), dtype=torch.float32)
        output = torch.empty_like(query)
        mock_fia.return_value = (torch.ones((8, 2, 1, 8), dtype=torch.float32), None)

        impl._forward_c8_chunked_prefill(query, None, None, metadata, output, layer)

        args = mock_fia.call_args.args
        kwargs = mock_fia.call_args.kwargs
        self.assertEqual(tuple(args[0].shape), (8, 2, 1, 8))
        self.assertEqual(tuple(kwargs["block_table"].shape), (8, 1))
        self.assertEqual(kwargs["actual_seq_lengths_kv"], [7, 8, 9, 10, 17, 18, 19, 20])
        self.assertTrue(torch.all(output == 1))

    def test_c8_full_graph_uses_token_as_batch_bnsd_decode_shape(self):
        impl = object.__new__(AscendAttentionBackendImpl)
        impl.num_heads = 2
        impl.num_kv_heads = 1
        impl.head_size = 8
        impl.scale = 0.125

        key = torch.zeros((4, 32, 8), dtype=torch.int8)
        value = torch.zeros((4, 32, 8), dtype=torch.int8)
        block_table = torch.zeros((2, 1), dtype=torch.int32)
        impl._get_fia_params = MagicMock(return_value=(key, value, 32, block_table, [10, 20]))

        layer = MagicMock()
        layer._c8_k_aq_scale = torch.ones((1, 1, 1, 8), dtype=torch.float32)
        layer._c8_k_aq_offset = torch.zeros((1, 1, 1, 8), dtype=torch.float32)
        layer._c8_v_aq_scale = torch.ones((1, 1, 1, 8), dtype=torch.float32)
        layer._c8_v_aq_offset = torch.zeros((1, 1, 1, 8), dtype=torch.float32)

        metadata = MagicMock()
        metadata.actual_seq_lengths_q = [4, 8]
        metadata.attn_mask = torch.ones((1, 1, 8, 8), dtype=torch.bool)
        query = torch.zeros((8, 2, 8), dtype=torch.float32)
        output = torch.empty_like(query)

        graph_params = MagicMock()
        graph_params.workspaces = {8: None}
        graph_params.events = {8: []}
        graph_params.handles = {8: []}
        graph_params.attn_params = {8: []}
        mock_stream = MagicMock()
        mock_event = MagicMock()
        mock_fia = MagicMock()

        with patch('vllm_ascend.attention.attention_v1._EXTRA_CTX',
                   MagicMock(is_draft_model=False)), \
             patch('vllm_ascend.attention.attention_v1.get_graph_params', return_value=graph_params), \
             patch('vllm_ascend.attention.attention_v1.update_graph_params_workspaces'), \
             patch('vllm_ascend.attention.attention_v1.weak_ref_tensors', side_effect=lambda tensor: tensor), \
             patch('torch_npu._npu_fused_infer_attention_score_get_max_workspace',
                   return_value=torch.empty(1, dtype=torch.float32)), \
             patch('torch_npu.npu_fused_infer_attention_score', mock_fia), \
             patch('torch.npu.current_stream', return_value=mock_stream), \
             patch('torch.npu.ExternalEvent', return_value=mock_event), \
             patch('torch.npu.graph_task_group_begin'), \
             patch('torch.npu.graph_task_group_end', return_value='handle'):
            attn_output, num_tokens = impl.full_graph_fia(query, key, value, metadata, output, layer)

        kwargs = mock_fia.out.call_args.kwargs
        self.assertEqual(num_tokens, 8)
        self.assertEqual(tuple(kwargs["query"].shape), (8, 2, 1, 8))
        self.assertEqual(tuple(kwargs["block_table"].shape), (8, 1))
        self.assertEqual(kwargs["actual_seq_lengths"], None)
        self.assertEqual(kwargs["actual_seq_lengths_kv"], [7, 8, 9, 10, 17, 18, 19, 20])
        self.assertEqual(kwargs["input_layout"], "BNSD")
        self.assertEqual(kwargs["sparse_mode"], 0)
        self.assertIs(kwargs["key_antiquant_scale"], layer._c8_k_aq_scale)
        self.assertEqual(tuple(attn_output.shape), (8, 2, 8))

    def test_c8_bnsd_decode_metadata_clamps_padded_seq_lens(self):
        block_tables = torch.zeros((2, 1), dtype=torch.int32)

        expanded_block_tables, expanded_seq_lens = (
            AscendAttentionBackendImpl._expand_c8_bnsd_decode_metadata(
                block_tables,
                [0, 3],
                [4, 8],
                num_tokens=8,
                num_seqs=2,
            )
        )

        self.assertEqual(tuple(expanded_block_tables.shape), (8, 1))
        self.assertEqual(expanded_seq_lens, [0, 0, 0, 0, 0, 1, 2, 3])

    def test_c8_bnsd_decode_metadata_cache_reuses_expanded_metadata(self):
        metadata = MagicMock()
        metadata.c8_bnsd_decode_metadata_cache = None
        block_tables = torch.zeros((2, 1), dtype=torch.int32)
        original_expand = AscendAttentionBackendImpl._expand_c8_bnsd_decode_metadata

        with patch.object(
            AscendAttentionBackendImpl,
            "_expand_c8_bnsd_decode_metadata",
            side_effect=original_expand,
        ) as mock_expand:
            first_block_tables, first_seq_lens = AscendAttentionBackendImpl._get_or_create_c8_bnsd_decode_metadata(
                metadata,
                block_tables,
                [10, 20],
                [4, 8],
                num_tokens=8,
                num_seqs=2,
            )
            second_block_tables, second_seq_lens = AscendAttentionBackendImpl._get_or_create_c8_bnsd_decode_metadata(
                metadata,
                block_tables,
                [10, 20],
                [4, 8],
                num_tokens=8,
                num_seqs=2,
            )

        self.assertEqual(mock_expand.call_count, 1)
        self.assertIs(first_block_tables, second_block_tables)
        self.assertIs(first_seq_lens, second_seq_lens)
        self.assertEqual(tuple(first_block_tables.shape), (8, 1))
        self.assertEqual(first_seq_lens, [7, 8, 9, 10, 17, 18, 19, 20])

    def test_c8_prepare_scales_expands_scalar_dummy_scales(self):
        impl = object.__new__(AscendC8AttentionBackendImpl)
        impl.num_kv_heads = 1
        impl.head_size = 8

        class Layer:
            pass

        layer = Layer()
        layer.k_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        layer.k_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        layer.v_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        layer.v_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)

        impl._prepare_c8_scales(layer, torch.device("cpu"))

        self.assertEqual(tuple(layer._c8_k_scale.shape), (1, 1, 8))
        self.assertEqual(tuple(layer._c8_k_aq_scale.shape), (1, 1, 1, 8))
        self.assertTrue(torch.all(layer._c8_k_scale == 1))
        self.assertTrue(torch.all(layer._c8_v_offset == 0))
