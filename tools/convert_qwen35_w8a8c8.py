#!/usr/bin/env python3
"""Convert a Qwen3.5 checkpoint to a ModelSlim W8A8+C8 layout.

This converter is intentionally narrow:
- Qwen3.5 dense and MoE checkpoints with safetensors weights.
- W8A8_DYNAMIC for supported language/MTP projections and MoE experts.
- C8 static per-channel KV-cache scales for full-attention k/v projections.
- Embeddings, norms, lm_head, routers, convolution weights, and vision weights
  stay in their original dtype.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


QUANT_CONFIG = "quant_model_description.json"
OUT_INDEX = "model.safetensors.index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--c8-sigma", default=6.0, type=float)
    parser.add_argument(
        "--quantize-mtp",
        action="store_true",
        help="Quantize supported MTP linear and MoE expert weights too.",
    )
    return parser.parse_args()


def load_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    return json.loads(index_path.read_text())


def group_safetensor_keys(index: dict) -> OrderedDict[str, list[str]]:
    files: OrderedDict[str, list[str]] = OrderedDict()
    for key, filename in index["weight_map"].items():
        files.setdefault(filename, []).append(key)
    return files


def is_language_layer_weight(name: str) -> bool:
    return name.startswith("model.language_model.layers.") and name.endswith(".weight")


def is_mtp_weight(name: str) -> bool:
    return name.startswith("mtp.")


def should_include_mtp(name: str, quantize_mtp: bool) -> bool:
    return quantize_mtp or not is_mtp_weight(name)


def is_full_attention_qkv(name: str) -> bool:
    if not (is_language_layer_weight(name) or is_mtp_weight(name)):
        return False
    return any(
        name.endswith(f".self_attn.{proj}.weight")
        for proj in ("q_proj", "k_proj", "v_proj")
    )


def is_full_attention_linear(name: str) -> bool:
    if not (is_language_layer_weight(name) or is_mtp_weight(name)):
        return False
    return any(
        name.endswith(f".self_attn.{proj}.weight")
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
    )


def is_linear_attention_linear(name: str) -> bool:
    if not is_language_layer_weight(name):
        return False
    # Qwen3.5 GDN is numerically sensitive. The vLLM Ascend ModelSlim
    # adaptation documents partial support for in_proj_qkvz, but the
    # lightweight PTQ path here does not calibrate the recurrent GDN path.
    # Keep linear_attn floating point to avoid NaNs on Qwen3.5 A10.
    return False


def is_mtp_fc(name: str) -> bool:
    return name == "mtp.fc.weight"


def is_dense_or_shared_mlp(name: str) -> bool:
    if not (is_language_layer_weight(name) or is_mtp_weight(name)):
        return False
    if any(
        name.endswith(f".mlp.{proj}.weight")
        for proj in ("gate_proj", "up_proj", "down_proj")
    ):
        return True
    return any(
        name.endswith(f".mlp.shared_expert.{proj}.weight")
        for proj in ("gate_proj", "up_proj", "down_proj")
    )


def is_fused_moe_expert(name: str) -> bool:
    return name.endswith(".mlp.experts.gate_up_proj") or name.endswith(
        ".mlp.experts.down_proj"
    )


def is_individual_moe_expert(name: str) -> bool:
    if ".mlp.experts." not in name:
        return False
    return any(
        name.endswith(f".{proj}.weight")
        for proj in ("gate_proj", "up_proj", "down_proj")
    )


def should_quantize_weight(name: str, tensor: torch.Tensor, quantize_mtp: bool) -> bool:
    if not should_include_mtp(name, quantize_mtp):
        return False
    if tensor.ndim == 2:
        return (
            is_full_attention_linear(name)
            or is_linear_attention_linear(name)
            or is_dense_or_shared_mlp(name)
            or is_individual_moe_expert(name)
            or is_mtp_fc(name)
        )
    if tensor.ndim == 3:
        return is_fused_moe_expert(name)
    return False


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_f32 = weight.to(torch.float32)
    scale = weight_f32.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / 127.0
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


def get_weight_scale_names(name: str) -> tuple[str, str]:
    if name.endswith(".weight"):
        prefix = name[: -len(".weight")]
        return f"{prefix}.weight_scale", f"{prefix}.weight_offset"
    return f"{name}_scale", f"{name}_offset"


def add_quant_description(
    quant_desc: dict[str, str | int | dict],
    name: str,
    quant_type: str,
) -> None:
    quant_desc[name] = quant_type
    if is_fused_moe_expert(name):
        layer_prefix = name[: name.rfind(".mlp.experts.")]
        if name.endswith(".mlp.experts.gate_up_proj"):
            quant_desc[f"{layer_prefix}.mlp.experts.0.gate_proj.weight"] = quant_type
            quant_desc[f"{layer_prefix}.mlp.experts.0.up_proj.weight"] = quant_type
        elif name.endswith(".mlp.experts.down_proj"):
            quant_desc[f"{layer_prefix}.mlp.experts.0.down_proj.weight"] = quant_type


def copy_sidecar_files(model_dir: Path, output_dir: Path) -> None:
    for src in model_dir.iterdir():
        if not src.is_file():
            continue
        if ".safetensors" in src.name:
            continue
        if src.name == QUANT_CONFIG:
            continue
        shutil.copy2(src, output_dir / src.name)


def write_index(output_dir: Path, weight_map: dict[str, str], total_size: int) -> None:
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
    weight_map: dict[str, str] = {}
    total_size = 0
    for filename, keys in group_safetensor_keys(index).items():
        shard_tensors: dict[str, torch.Tensor] = {}
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            for name in keys:
                tensor = handle.get_tensor(name)
                if should_quantize_weight(name, tensor, args.quantize_mtp):
                    qweight, scale, offset = quantize_weight(tensor)
                    scale_name, offset_name = get_weight_scale_names(name)
                    shard_tensors[name] = qweight
                    shard_tensors[scale_name] = scale
                    shard_tensors[offset_name] = offset
                    add_quant_description(quant_desc, name, "W8A8_DYNAMIC")
                    quant_desc[scale_name] = "W8A8_DYNAMIC"
                    quant_desc[offset_name] = "W8A8_DYNAMIC"
                    num_quantized += 1
                else:
                    shard_tensors[name] = tensor
                    if name.endswith(".weight"):
                        add_quant_description(quant_desc, name, "FLOAT")
                    elif is_fused_moe_expert(name):
                        add_quant_description(quant_desc, name, "FLOAT")

                if (
                    should_include_mtp(name, args.quantize_mtp)
                    and (
                        name.endswith(".self_attn.k_proj.weight")
                        or name.endswith(".self_attn.v_proj.weight")
                    )
                ):
                    add_c8_cache_params(shard_tensors, name, tensor, args.c8_sigma)
                    prefix = name[: -len(".weight")]
                    quant_desc[f"{prefix}.kv_cache_scale"] = "C8"
                    quant_desc[f"{prefix}.kv_cache_offset"] = "C8"
                    if name.endswith(".self_attn.k_proj.weight"):
                        num_c8_layers += 1

        save_file(shard_tensors, output_dir / filename, metadata={"format": "pt"})
        for tensor_name, tensor in shard_tensors.items():
            weight_map[tensor_name] = filename
            total_size += tensor.numel() * tensor.element_size()
        print(f"wrote {filename}: tensors={len(shard_tensors)}")

    copy_sidecar_files(model_dir, output_dir)
    write_index(output_dir, weight_map, total_size)
    (output_dir / QUANT_CONFIG).write_text(json.dumps(quant_desc, indent=2, sort_keys=True) + "\n")

    print(f"output_dir={output_dir}")
    print(f"tensors={len(weight_map)}")
    print(f"quantized_weights={num_quantized}")
    print(f"c8_attention_layers={num_c8_layers}")


if __name__ == "__main__":
    main()
