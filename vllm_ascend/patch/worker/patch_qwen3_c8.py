#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections.abc import Iterable

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

_orig_qwen3_causal_lm_load_weights = Qwen3ForCausalLM.load_weights
_orig_qwen3_5_causal_lm_base_load_weights = Qwen3_5ForCausalLMBase.load_weights
_orig_qwen3_5_conditional_generation_load_weights = (
    Qwen3_5ForConditionalGeneration.load_weights
)
_orig_qwen3_5_mtp_load_weights = Qwen3_5MTP.load_weights


def _get_quant_config(module):
    quant_config = getattr(module, "quant_config", None)
    if quant_config is None and hasattr(module, "language_model"):
        quant_config = getattr(module.language_model, "quant_config", None)
    if quant_config is None and hasattr(module, "model"):
        quant_config = getattr(module.model, "quant_config", None)
    if quant_config is None:
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            quant_config = vllm_config.quant_config
    return quant_config


def _unique_names(names: Iterable[str | None]) -> Iterable[str]:
    seen: set[str] = set()
    for name in names:
        if name is None or name in seen:
            continue
        seen.add(name)
        yield name


def _load_weights_with_c8_scale_intercept(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
    orig_load_weights,
    mapper=None,
) -> set[str]:
    quant_config = _get_quant_config(self)
    if quant_config is None or not callable(
        getattr(quant_config, "get_cache_scale", None)
    ):
        return orig_load_weights(self, weights)

    params_dict = dict(self.named_parameters())
    c8_loaded_params: set[str] = set()

    def _candidate_names(name: str) -> Iterable[str]:
        mapped_name = mapper._map_name(name) if mapper is not None else None
        names = [
            name,
            mapped_name,
        ]

        for candidate in tuple(_unique_names(names)):
            names.extend(
                [
                    candidate.removeprefix("language_model."),
                    candidate.removeprefix("model."),
                    candidate.replace(
                        "model.language_model.",
                        "language_model.model.",
                        1,
                    ),
                    candidate.replace("model.language_model.", "model.", 1),
                    candidate.replace("language_model.model.", "model.", 1),
                    candidate.replace("mtp.", "model.", 1),
                    candidate.replace("model.", "", 1),
                ]
            )

        if mapper is not None:
            for candidate in tuple(_unique_names(names)):
                names.append(mapper._map_name(candidate))
        yield from _unique_names(names)

    def _intercept_c8_scales(
        raw_weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, loaded_weight in raw_weights:
            c8_scale_seen = False
            loaded = False
            for candidate_name in _candidate_names(name):
                scale_name = quant_config.get_cache_scale(candidate_name)
                if scale_name is None:
                    continue
                c8_scale_seen = True
                if scale_name in params_dict:
                    param = params_dict[scale_name]
                    weight_loader = getattr(
                        param,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(param, loaded_weight.squeeze())
                    c8_loaded_params.add(scale_name)
                    loaded = True
                    break
            if not c8_scale_seen:
                yield name, loaded_weight
            elif not loaded:
                # C8 scale tensors are plugin-owned parameters. If the target
                # layer is not instantiated, do not forward them to the base
                # loader where they would be treated as unexpected weights.
                continue

    loaded_params = orig_load_weights(self, _intercept_c8_scales(weights))
    loaded_params.update(c8_loaded_params)
    return loaded_params


def _patched_qwen3_causal_lm_load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    return _load_weights_with_c8_scale_intercept(self, weights, _orig_qwen3_causal_lm_load_weights)


def _patched_qwen3_5_causal_lm_base_load_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    return _load_weights_with_c8_scale_intercept(
        self,
        weights,
        _orig_qwen3_5_causal_lm_base_load_weights,
    )


def _patched_qwen3_5_conditional_generation_load_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    return _load_weights_with_c8_scale_intercept(
        self,
        weights,
        _orig_qwen3_5_conditional_generation_load_weights,
        mapper=self.hf_to_vllm_mapper,
    )


def _patched_qwen3_5_mtp_load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    return _load_weights_with_c8_scale_intercept(self, weights, _orig_qwen3_5_mtp_load_weights)


Qwen3ForCausalLM.load_weights = _patched_qwen3_causal_lm_load_weights
Qwen3_5ForCausalLMBase.load_weights = _patched_qwen3_5_causal_lm_base_load_weights
Qwen3_5ForConditionalGeneration.load_weights = _patched_qwen3_5_conditional_generation_load_weights
Qwen3_5MTP.load_weights = _patched_qwen3_5_mtp_load_weights
