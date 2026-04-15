#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
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

from __future__ import annotations

import torch
import torch.nn as nn


def _get_direct_top_tokens(
    model: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    if hasattr(model, "get_top_tokens"):
        return model.get_top_tokens(hidden_states)

    # Some draft models remap draft vocab ids back to target vocab ids inside
    # compute_logits(). Bypassing that logic would return incorrect token ids.
    if getattr(model, "draft_id_to_target_id", None) is not None:
        return None

    if hasattr(model, "logits_processor") and hasattr(model, "lm_head"):
        embedding_bias = getattr(model.lm_head, "bias", None)
        return model.logits_processor.get_top_tokens(
            model.lm_head,
            hidden_states,
            embedding_bias,
        )

    return None


def _iter_language_model_candidates(model: nn.Module):
    seen = {id(model)}
    current = model

    while True:
        next_model = None
        if hasattr(current, "get_language_model"):
            next_model = current.get_language_model()
        elif hasattr(current, "language_model"):
            next_model = current.language_model

        if next_model is None or id(next_model) in seen:
            return

        seen.add(id(next_model))
        yield next_model
        current = next_model


def get_greedy_top_tokens(
    model: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    top_tokens = _get_direct_top_tokens(model, hidden_states)
    if top_tokens is not None:
        return top_tokens

    for candidate in _iter_language_model_candidates(model):
        top_tokens = _get_direct_top_tokens(candidate, hidden_states)
        if top_tokens is not None:
            return top_tokens

    logits = model.compute_logits(hidden_states)
    return logits.argmax(dim=-1)
