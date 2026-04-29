#!/usr/bin/env python3
"""Convert a Qwen3.5 checkpoint to a minimal ModelSlim W8A8+C8 layout.

This converter is intentionally narrow:
- Qwen3.5 dense checkpoints with safetensors weights.
- W8A8_DYNAMIC for language MLP and full-attention q/k/v projections.
- C8 static per-channel KV-cache scales for full-attention k/v projections.
- Linear-attention, MTP, embeddings, norms, lm_head, and vision weights stay
  in their original dtype.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


QUANT_CONFIG = "quant_model_description.json"
OUT_WEIGHTS = "model.safetensors"
OUT_INDEX = "model.safetensors.index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--c8-sigma", default=6.0, type=float)
    return parser.parse_args()


def load_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    return json.loads(index_path.read_text())


def iter_safetensor_keys(model_dir: Path, index: dict):
    files: dict[str, list[str]] = {}
    for key, filename in index["weight_map"].items():
        files.setdefault(filename, []).append(key)
    for filename, keys in files.items():
        path = model_dir / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in keys:
                yield key, handle.get_tensor(key)


def is_language_layer_weight(name: str) -> bool:
    return name.startswith("model.language_model.layers.") and name.endswith(".weight")


def is_mtp_weight(name: str) -> bool:
    return name.startswith("mtp.")


def is_full_attention_qkv(name: str) -> bool:
    if not is_language_layer_weight(name):
        return False
    return any(
        name.endswith(f".self_attn.{proj}.weight")
        for proj in ("q_proj", "k_proj", "v_proj")
    )


def is_language_mlp(name: str) -> bool:
    if not is_language_layer_weight(name):
        return False
    return any(
        name.endswith(f".mlp.{proj}.weight")
        for proj in ("gate_proj", "up_proj", "down_proj")
    )


def should_quantize_weight(name: str, tensor: torch.Tensor) -> bool:
    if is_mtp_weight(name) or tensor.ndim != 2:
        return False
    return is_full_attention_qkv(name) or is_language_mlp(name)


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_f32 = weight.to(torch.float32)
    scale = weight_f32.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6) / 127.0
    quant_weight = torch.round(weight_f32 / scale).clamp(-127, 127).to(torch.int8)
    offset = torch.zeros_like(scale, dtype=torch.float32)
    return quant_weight, scale.to(torch.float32), offset


def c8_scale_from_projection(weight: torch.Tensor, sigma: float) -> torch.Tensor:
    """Estimate static KV-cache scale from projection output range.

    For normalized hidden states with unit RMS, each output channel of
    ``x @ weight.T`` has RMS ``sqrt(in_features) * rms(weight_row)``.  Using the
    raw weight RMS directly underestimates the V-cache range by roughly
    ``sqrt(hidden_size)`` and causes severe INT8 saturation.
    """
    weight_f32 = weight.to(torch.float32)
    row_std = weight_f32.square().mean(dim=1).sqrt()
    output_std = row_std * math.sqrt(weight.shape[1])
    return (output_std * sigma / 127.0).clamp_min(1.0e-5).to(torch.float32)


def c8_key_scale(weight: torch.Tensor, sigma: float) -> torch.Tensor:
    # Qwen3.5 applies k_norm before caching, so a conservative RMS-based static
    # scale is more stable than deriving K cache scale from the k_proj weight.
    return torch.full((weight.shape[0],), sigma / 127.0, dtype=torch.float32)


def add_c8_cache_params(
    tensors: dict[str, torch.Tensor],
    name: str,
    weight: torch.Tensor,
    sigma: float,
) -> None:
    if not name.endswith(".weight"):
        return
    prefix = name[: -len(".weight")]
    if name.endswith(".self_attn.k_proj.weight"):
        scale = c8_key_scale(weight, sigma)
    elif name.endswith(".self_attn.v_proj.weight"):
        scale = c8_scale_from_projection(weight, sigma)
    else:
        return
    tensors[f"{prefix}.kv_cache_scale"] = scale
    tensors[f"{prefix}.kv_cache_offset"] = torch.zeros_like(scale, dtype=torch.float32)


def copy_sidecar_files(model_dir: Path, output_dir: Path) -> None:
    for src in model_dir.iterdir():
        if not src.is_file():
            continue
        if ".safetensors" in src.name:
            continue
        if src.name == QUANT_CONFIG:
            continue
        shutil.copy2(src, output_dir / src.name)


def write_index(output_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    weight_map = {key: OUT_WEIGHTS for key in sorted(tensors)}
    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    payload = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / OUT_INDEX).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    index = load_index(model_dir)
    tensors: dict[str, torch.Tensor] = {}
    quant_desc: dict[str, str | int | dict] = {
        "group_size": 0,
        "kv_cache_type": "C8",
        "metadata": {
            "converter": "tools/convert_qwen35_w8a8c8.py",
            "source": str(model_dir),
            "weight_quant": "W8A8_DYNAMIC",
        },
    }

    num_quantized = 0
    num_c8_layers = 0
    for name, tensor in iter_safetensor_keys(model_dir, index):
        if should_quantize_weight(name, tensor):
            qweight, scale, offset = quantize_weight(tensor)
            tensors[name] = qweight
            tensors[f"{name[:-len('.weight')]}.weight_scale"] = scale
            tensors[f"{name[:-len('.weight')]}.weight_offset"] = offset
            quant_desc[name] = "W8A8_DYNAMIC"
            quant_desc[f"{name[:-len('.weight')]}.weight_scale"] = "W8A8_DYNAMIC"
            quant_desc[f"{name[:-len('.weight')]}.weight_offset"] = "W8A8_DYNAMIC"
            num_quantized += 1
        else:
            tensors[name] = tensor
            if name.endswith(".weight"):
                quant_desc[name] = "FLOAT"

        if name.endswith(".self_attn.k_proj.weight") or name.endswith(".self_attn.v_proj.weight"):
            add_c8_cache_params(tensors, name, tensor, args.c8_sigma)
            prefix = name[: -len(".weight")]
            quant_desc[f"{prefix}.kv_cache_scale"] = "C8"
            quant_desc[f"{prefix}.kv_cache_offset"] = "C8"
            if name.endswith(".self_attn.k_proj.weight"):
                num_c8_layers += 1

    copy_sidecar_files(model_dir, output_dir)
    save_file(tensors, output_dir / OUT_WEIGHTS, metadata={"format": "pt"})
    write_index(output_dir, tensors)
    (output_dir / QUANT_CONFIG).write_text(json.dumps(quant_desc, indent=2, sort_keys=True) + "\n")

    print(f"output_dir={output_dir}")
    print(f"tensors={len(tensors)}")
    print(f"quantized_weights={num_quantized}")
    print(f"c8_attention_layers={num_c8_layers}")


if __name__ == "__main__":
    main()
