from unittest.mock import MagicMock, patch

import torch
from tests.ut.base import TestBase
from vllm_ascend.ascend_config import WeightPrefetchConfig
from vllm_ascend.ops.weight_prefetch import WeightPrefetchMethod


class TestWeightPrefetchMethod(TestBase):

    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model")
    def test_explicit_moe_prefetch_ratio_keeps_moe_enabled(
        self, mock_is_moe_model, mock_get_current_vllm_config
    ):
        mock_is_moe_model.return_value = True
        mock_get_current_vllm_config.return_value = MagicMock()

        method = WeightPrefetchMethod(
            WeightPrefetchConfig({
                "enabled": True,
                "prefetch_ratio": {
                    "attn": {
                        "qkv": 1.0,
                        "o": 1.0,
                    },
                    "moe": {
                        "gate_up": 0.8,
                    },
                    "mlp": {
                        "gate_up": 1.0,
                        "down": 1.0,
                    },
                },
            })
        )

        self.assertTrue(method.attn.enable)
        self.assertTrue(method.moe.enable)
        self.assertFalse(method.mlp.enable)
        self.assertEqual(method.moe.prefetch_ratio, {"gate_up": 0.8})

    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model")
    def test_missing_prefetch_ratio_keeps_default_modules(
        self, mock_is_moe_model, mock_get_current_vllm_config
    ):
        mock_is_moe_model.return_value = True
        mock_get_current_vllm_config.return_value = MagicMock()

        method = WeightPrefetchMethod(WeightPrefetchConfig({"enabled": True}))

        self.assertTrue(method.attn.enable)
        self.assertTrue(method.moe.enable)
        self.assertFalse(method.mlp.enable)
        self.assertEqual(method.moe.prefetch_ratio, {"gate_up": 0.8})

    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model")
    def test_moe_prefetch_resolves_aclgraph_wrapped_language_model(
        self, mock_is_moe_model, mock_get_current_vllm_config
    ):
        mock_is_moe_model.return_value = True
        mock_get_current_vllm_config.return_value = MagicMock()

        class Experts:
            w13_weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))

        class Layer:
            mlp = type("Mlp", (), {"experts": Experts()})()

        class LanguageModel:
            model = type("InnerModel", (), {"layers": [Layer()]})()

        class Model:
            language_model = LanguageModel()

        class GraphWrapper:
            def unwrap(self):
                return Model()

        method = WeightPrefetchMethod(WeightPrefetchConfig({
            "enabled": True,
            "prefetch_ratio": {
                "moe": {
                    "gate_up": 0.8,
                },
            },
        }))

        extra_ctx = MagicMock()
        extra_ctx.model_instance = GraphWrapper()
        extra_ctx.layer_idx = 1
        with patch(
            "vllm_ascend.ops.weight_prefetch.get_forward_context",
            return_value=MagicMock(),
        ), patch(
            "vllm_ascend.ops.weight_prefetch._EXTRA_CTX", extra_ctx
        ), patch(
            "vllm_ascend.ops.weight_prefetch.torch.ops.vllm.prefetch_preprocess"
        ) as mock_prefetch:
            method.maybe_prefetch_moe_weight_preprocess(
                torch.ones((128, 4)), "gate_up"
            )

        mock_prefetch.assert_called_once()
        self.assertTrue(method.moe.is_active_this_forward)
