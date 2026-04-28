import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.modelslim_config import (
    MODELSLIM_CONFIG_FILENAME,
    AscendModelSlimConfig,
)
from vllm_ascend.quantization.methods import get_scheme_class
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD

from vllm.model_executor.layers.attention import Attention


class TestAscendModelSlimConfig(TestBase):

    def setUp(self):
        self.sample_config = {
            "weight": "INT8",
            "fa_quant_type": "FAKQuant",
            "layers.1.fa_k.scale": "FLOAT",
            "layer1.weight": "INT8",
            "layer2.weight": "FLOAT",
            "fused_layer.weight": "FLOAT",
            "fused_layer.shard1.weight": "FLOAT",
            "fused_layer.shard2.weight": "FLOAT",
            "shard1.weight": "FLOAT",
            "shard2.weight": "FLOAT",
        }
        self.ascend_config = AscendModelSlimConfig(self.sample_config)
        self.ascend_config.packed_modules_mapping = None

    def test_init(self):
        self.assertEqual(self.ascend_config.quant_description,
                         self.sample_config)

    def test_repr(self):
        repr_str = repr(self.ascend_config)
        self.assertTrue(repr_str.startswith("AscendModelSlimConfig:\n"))

    def test_get_name(self):
        self.assertEqual(AscendModelSlimConfig.get_name(),
                         ASCEND_QUANTIZATION_METHOD)

    def test_get_supported_act_dtypes(self):
        supported_dtypes = AscendModelSlimConfig.get_supported_act_dtypes()
        self.assertEqual(len(supported_dtypes), 3)

    def test_get_min_capability(self):
        with self.assertRaises(NotImplementedError):
            AscendModelSlimConfig.get_min_capability()

    def test_get_config_filenames(self):
        filenames = AscendModelSlimConfig.get_config_filenames()
        self.assertEqual(filenames, [])

    def test_from_config(self):
        config = AscendModelSlimConfig.from_config(self.sample_config)
        self.assertIsInstance(config, AscendModelSlimConfig)
        self.assertEqual(config.quant_description, self.sample_config)

    @patch('torch.npu.is_available')
    def test_override_quantization_method(self, mock_is_available):
        # Test when NPU is available
        mock_is_available.return_value = True
        result = AscendModelSlimConfig.override_quantization_method(None, None)
        self.assertIsNone(result)
        hf_quant_cfg = {"quant_method": ""}
        result = AscendModelSlimConfig.override_quantization_method(
            hf_quant_cfg, None)
        self.assertEqual(result, "ascend")

        # Test when NPU is not available
        mock_is_available.return_value = False
        result = AscendModelSlimConfig.override_quantization_method(None, None)
        self.assertIsNone(result)
        hf_quant_cfg = {"quant_method": ""}
        result = AscendModelSlimConfig.override_quantization_method(
            hf_quant_cfg, None)
        self.assertIsNone(result)

    def test_get_quant_method_for_linear(self):
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        linear_layer = MagicMock(spec=LinearBase)
        # Test skipped layer
        with patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch.object(self.ascend_config, \
                          'is_layer_skipped_ascend',
                          return_value=True):
            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIsInstance(method, AscendUnquantizedLinearMethod)

        # Test quantized layer
        mock_scheme = MagicMock()
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
            patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch("vllm_ascend.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme), \
            patch('vllm_ascend.quantization.method_adapters.AscendLinearMethod', return_value=MagicMock()) as mock_ascend_linear:

            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIs(method, mock_ascend_linear.return_value)
            mock_ascend_linear.assert_called_once_with(mock_scheme)

    def test_get_quant_method_for_attention(self):
        attention_layer = MagicMock(spec=Attention)
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        mock_scheme = MagicMock()
        with patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch("vllm_ascend.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme), \
            patch('vllm_ascend.quantization.method_adapters.AscendKVCacheMethod', \
                   return_value=MagicMock()) as mock_ascend_kvcache:
            # Test with fa_quant_type
            method = self.ascend_config.get_quant_method(
                attention_layer, ".attn")
            self.assertIs(method, None)
            method = self.ascend_config.get_quant_method(
                attention_layer, "layers.1.attn")
            self.assertIs(method, mock_ascend_kvcache.return_value)

    def test_fa_quant_layer_prefix_matching_keeps_mtp_distinct(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "FAKQuant",
            "model.layers.3.self_attn.fa_k.scale": "FLOAT",
        })

        self.assertTrue(config.is_fa_quant_layer("model.layers.3.self_attn"))
        self.assertFalse(config.is_fa_quant_layer("mtp.layers.3.self_attn"))
        self.assertFalse(config.is_fa_quant_layer("model.layers.7.self_attn"))

    def test_fa_quant_layer_prefix_matching_supports_mtp_fakquant(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "FAKQuant",
            "mtp.layers.0.self_attn.fa_k.scale": "FLOAT",
        })

        self.assertTrue(config.is_fa_quant_layer("mtp.layers.0.self_attn"))
        self.assertFalse(config.is_fa_quant_layer("model.layers.0.self_attn"))

    def test_mtp_fa_quant_layer_uses_int8_kv_dtype(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "FAKQuant",
            "mtp.layers.0.self_attn.fa_k.scale": "FLOAT",
        })
        model_config = MagicMock()
        model_config.dtype = torch.bfloat16
        model_config.use_mla = False

        self.assertEqual(
            config.get_kv_quant_dtype("mtp.layers.0.self_attn",
                                      torch.bfloat16, model_config),
            (torch.int8, torch.int8))
        self.assertEqual(
            config.get_kv_quant_dtype("model.layers.0.self_attn",
                                      torch.bfloat16, model_config),
            (torch.bfloat16, torch.bfloat16))

    def test_fa_quant_layer_prefix_matching_supports_legacy_main_layers(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "FAKQuant",
            "layers.1.fa_k.scale": "FLOAT",
        })

        self.assertTrue(config.is_fa_quant_layer("layers.1.attn"))
        self.assertTrue(config.is_fa_quant_layer("model.layers.1.attn"))
        self.assertFalse(config.is_fa_quant_layer("mtp.layers.1.attn"))

    def test_fakquant_attention_scheme_is_registered(self):
        scheme_cls = get_scheme_class("FAKQuant", "attention")

        self.assertIsNotNone(scheme_cls)
        self.assertEqual(scheme_cls.__name__, "AscendFAQuantAttentionMethod")

    def test_get_quant_method_for_c8_kv_cache_attention(self):
        config = AscendModelSlimConfig({
            "weight": "INT8",
            "kv_cache_type": "C8",
        })
        attention_layer = MagicMock(spec=Attention)
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        mock_config.kv_transfer_config = None
        mock_method = MagicMock()

        with patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch("vllm_ascend.quantization.methods.kv_c8.get_current_vllm_config", return_value=mock_config), \
            patch(
                "vllm_ascend.quantization.method_adapters.AscendKVCacheMethod", return_value=mock_method
            ) as mock_kvcache:
            method = config.get_quant_method(attention_layer, "model.layers.0.self_attn.attn")

        self.assertTrue(config.enable_c8_quant)
        self.assertIs(method, mock_method)
        self.assertEqual(mock_kvcache.call_args.args[0].__class__.__name__, "AscendC8KVCacheAttentionMethod")

    def test_get_quant_method_for_fused_moe(self):
        fused_moe_layer = MagicMock(spec=FusedMoE)
        fused_moe_layer.moe = MagicMock(spec=FusedMoEConfig)
        fused_moe_layer.moe_config = MagicMock(spec=FusedMoEConfig)
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None

        # Test skipped layer
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=True), \
            patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch('vllm_ascend.ops.fused_moe.fused_moe.AscendUnquantizedFusedMoEMethod', return_value=MagicMock()) as mock_ascend_moe:
            method = self.ascend_config.get_quant_method(
                fused_moe_layer, "moe_layer")
            self.assertIs(method, mock_ascend_moe.return_value)

        # Test quantized layer
        mock_scheme = MagicMock()
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
            patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch("vllm_ascend.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme), \
            patch('vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod', return_value=MagicMock()) as mock_ascend_moe:
            method = self.ascend_config.get_quant_method(
                fused_moe_layer, "moe_layer")
            self.assertIs(method, mock_ascend_moe.return_value)

    def test_is_layer_skipped_ascend(self):
        # Test non-fused layer that should be quantized
        self.assertFalse(self.ascend_config.is_layer_skipped_ascend("layer1"))

        # Test non-fused layer that should be skipped
        self.assertTrue(self.ascend_config.is_layer_skipped_ascend("layer2"))

        # Test fused layer
        fused_mapping = {"fused_layer": ["shard1", "shard2"]}
        self.assertTrue(
            self.ascend_config.is_layer_skipped_ascend("fused_layer",
                                                       fused_mapping))

        # Test inconsistent fused layer shards
        bad_config = {"shard1.weight": "FLOAT", "shard2.weight": "INT8"}
        config = AscendModelSlimConfig(bad_config)
        with self.assertRaises(ValueError):
            config.is_layer_skipped_ascend("fused_layer", fused_mapping)

    def test_init_with_none_config(self):
        config = AscendModelSlimConfig(None)
        self.assertEqual(config.quant_description, {})

    def test_init_with_default_config(self):
        config = AscendModelSlimConfig()
        self.assertEqual(config.quant_description, {})

    def test_maybe_update_config_already_populated(self):
        # When quant_description is already populated, should be a no-op
        self.assertTrue(len(self.ascend_config.quant_description) > 0)
        self.ascend_config.maybe_update_config("/some/model/path")
        # quant_description should remain unchanged
        self.assertEqual(self.ascend_config.quant_description,
                         self.sample_config)

    def test_maybe_update_config_loads_from_file(self):
        config = AscendModelSlimConfig()
        self.assertEqual(config.quant_description, {})

        quant_data = {"layer1.weight": "INT8", "layer2.weight": "FLOAT"}
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, MODELSLIM_CONFIG_FILENAME)
            with open(config_path, "w") as f:
                json.dump(quant_data, f)

            config.maybe_update_config(tmpdir)

        self.assertEqual(config.quant_description, quant_data)

    def test_maybe_update_config_raises_when_file_missing(self):
        config = AscendModelSlimConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                config.maybe_update_config(tmpdir)

            error_msg = str(ctx.exception)
            self.assertIn("ModelSlim Quantization Config Not Found", error_msg)
            self.assertIn(MODELSLIM_CONFIG_FILENAME, error_msg)

    def test_maybe_update_config_raises_with_json_files_listed(self):
        config = AscendModelSlimConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy json file that is NOT the config file
            dummy_path = os.path.join(tmpdir, "config.json")
            with open(dummy_path, "w") as f:
                json.dump({"dummy": True}, f)

            with self.assertRaises(ValueError) as ctx:
                config.maybe_update_config(tmpdir)

            error_msg = str(ctx.exception)
            self.assertIn("config.json", error_msg)

    def test_maybe_update_config_non_directory_raises(self):
        config = AscendModelSlimConfig()

        with self.assertRaises(ValueError) as ctx:
            config.maybe_update_config("not_a_real_directory_path")

        error_msg = str(ctx.exception)
        self.assertIn("ModelSlim Quantization Config Not Found", error_msg)

    def test_apply_extra_quant_adaptations_shared_head(self):
        config = AscendModelSlimConfig()
        config.quant_description = {
            "model.layers.0.shared_head.weight": "INT8",
        }
        config._apply_extra_quant_adaptations()
        self.assertIn("model.layers.0.weight", config.quant_description)
        self.assertEqual(config.quant_description["model.layers.0.weight"],
                         "INT8")

    def test_apply_extra_quant_adaptations_weight_packed(self):
        config = AscendModelSlimConfig()
        config.quant_description = {
            "model.layers.0.weight_packed": "INT8",
        }
        config._apply_extra_quant_adaptations()
        self.assertIn("model.layers.0.weight", config.quant_description)
        self.assertEqual(config.quant_description["model.layers.0.weight"],
                         "INT8")
