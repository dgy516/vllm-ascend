from types import SimpleNamespace
from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.dbo.overlap_templates.qwen3_dense import (
    QwenDenseAllgatherTemplate,
    QwenDenseAlltoallTemplate,
)
from vllm_ascend.dbo.overlap_templates.qwen3_moe import (
    QwenMoEAllgatherTemplate,
    QwenMoEAlltoallTemplate,
)
from vllm_ascend.dbo.utils import select_dbo_templates
from vllm_ascend.utils import AscendDeviceType


class TestDboUtils(TestBase):
    @staticmethod
    def _mock_vllm_config(architecture: str):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture]),
            ),
        )

    def test_select_qwen3_5_dense_templates(self):
        vllm_config = self._mock_vllm_config("Qwen3_5ForCausalLM")

        cases = (
            (AscendDeviceType.A2, QwenDenseAllgatherTemplate),
            (AscendDeviceType.A3, QwenDenseAlltoallTemplate),
        )
        for device_type, expected_cls in cases:
            with self.subTest(device_type=device_type, expected_cls=expected_cls.__name__):
                with patch("vllm_ascend.dbo.utils.get_ascend_device_type", return_value=device_type):
                    template = select_dbo_templates(vllm_config)
                self.assertIsInstance(template, expected_cls)

    def test_select_qwen3_5_moe_templates(self):
        vllm_config = self._mock_vllm_config("Qwen3_5MoeForCausalLM")

        cases = (
            (AscendDeviceType.A2, QwenMoEAllgatherTemplate),
            (AscendDeviceType.A3, QwenMoEAlltoallTemplate),
        )
        for device_type, expected_cls in cases:
            with self.subTest(device_type=device_type, expected_cls=expected_cls.__name__):
                with patch("vllm_ascend.dbo.utils.get_ascend_device_type", return_value=device_type):
                    template = select_dbo_templates(vllm_config)
                self.assertIsInstance(template, expected_cls)
