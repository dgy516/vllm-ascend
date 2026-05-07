#!/usr/bin/env python3
"""Shared helpers for Jenkins DeployCase scripts."""

from __future__ import annotations

import glob
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

API_VERSION = "llm-ci/v1"
KIND = "DeployCase"
ALLOWED_LEVELS = {"static", "smoke", "nightly", "release", "benchmark"}
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


class DeployCaseError(RuntimeError):
    """Raised for invalid DeployCase inputs."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    ensure_parent(output)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path) -> dict[str, Any]:
    case_path = Path(path)
    with case_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise DeployCaseError(f"{case_path}: expected YAML mapping at document root")
    return data


def load_case(path: str | Path) -> dict[str, Any]:
    data = load_yaml(path)
    data.setdefault("_path", str(path))
    return data


def expand_case_paths(patterns: list[str] | str) -> list[str]:
    if isinstance(patterns, str):
        raw_patterns = [patterns]
    else:
        raw_patterns = patterns

    expanded: list[str] = []
    for item in raw_patterns:
        for pattern in str(item).split(","):
            pattern = pattern.strip()
            if not pattern:
                continue
            matches = sorted(glob.glob(pattern))
            expanded.extend(matches if matches else [pattern])

    seen: set[str] = set()
    result: list[str] = []
    for path in expanded:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def read_case_list(path: str | Path) -> list[str]:
    case_list = Path(path)
    if not case_list.exists():
        raise DeployCaseError(f"case list does not exist: {case_list}")
    paths: list[str] = []
    for line in case_list.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            paths.append(stripped)
    return paths


def case_name(case: dict[str, Any]) -> str:
    return str(case.get("metadata", {}).get("name") or Path(case.get("_path", "case")).stem)


def case_level(case: dict[str, Any]) -> str:
    return str(case.get("metadata", {}).get("level", ""))


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return slug.strip("-") or "case"


def first_service(case: dict[str, Any]) -> dict[str, Any]:
    services = case.get("services") or []
    if not services:
        raise DeployCaseError(f"{case_name(case)}: services must not be empty")
    if not isinstance(services[0], dict):
        raise DeployCaseError(f"{case_name(case)}: first service must be a mapping")
    return services[0]


def served_model_name(service: dict[str, Any]) -> str:
    vllm = service.get("vllm") or {}
    return str(vllm.get("served_model_name") or vllm.get("model") or "")


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def runtime_env(case: dict[str, Any]) -> dict[str, str]:
    env = case.get("runtime", {}).get("env") or {}
    if not isinstance(env, dict):
        return {}
    return {str(k): str(v) for k, v in env.items()}


def expand_text(value: Any, env: dict[str, str] | None = None) -> str:
    merged_env = dict(os.environ)
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    text = str(value)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        return merged_env.get(key, match.group(0))

    return _ENV_PATTERN.sub(replace, text)


def _has_flag(args: list[str], flag: str) -> bool:
    return flag in args


def build_vllm_serve_command(
    case: dict[str, Any],
    service: dict[str, Any] | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    service = service or first_service(case)
    vllm = service.get("vllm") or {}
    env = runtime_env(case)
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

    model = expand_text(vllm.get("model", ""), env)
    model_root = env.get("MODEL_ROOT", "").strip()
    local_model_path = vllm.get("local_model_path")
    if model_root and local_model_path:
        model = expand_text(local_model_path, env)
    if not model:
        raise DeployCaseError(f"{case_name(case)}: vllm-serve service requires vllm.model")

    raw_args = vllm.get("args") or []
    if not isinstance(raw_args, list):
        raise DeployCaseError(f"{case_name(case)}: vllm.args must be a list")
    args = [expand_text(arg, env) for arg in raw_args]

    if service.get("host") is not None and not _has_flag(args, "--host"):
        args.extend(["--host", str(service["host"])])
    if service.get("port") is not None and not _has_flag(args, "--port"):
        args.extend(["--port", str(service["port"])])
    served_name = served_model_name(service)
    if served_name and served_name != model and not _has_flag(args, "--served-model-name"):
        args.extend(["--served-model-name", served_name])

    return ["vllm", "serve", model, *args]


def command_to_shell(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def get_path(data: Any, path: str) -> Any:
    cur = data
    for item in path.split("."):
        if isinstance(cur, dict) and item in cur:
            cur = cur[item]
        else:
            return None
    return cur


def _is_non_empty(value: Any) -> bool:
    return value is not None and value != "" and value != []


def validate_case(case: dict[str, Any], path: str | Path | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    label = str(path or case.get("_path") or case_name(case))

    if case.get("apiVersion") != API_VERSION:
        errors.append(f"{label}: apiVersion must be {API_VERSION}")
    if case.get("kind") != KIND:
        errors.append(f"{label}: kind must be {KIND}")

    required_paths = [
        "metadata.name",
        "metadata.title",
        "metadata.level",
        "metadata.owner",
        "metadata.description",
        "metadata.tags",
        "doc.output",
        "doc.audience",
        "doc.difficulty",
        "doc.generated_warning",
        "requirements.hardware",
        "requirements.software",
        "requirements.model",
        "runtime.image",
        "runtime.workdir",
        "runtime.env",
        "checks.readiness",
        "tests.smoke",
        "tests.benchmark",
        "tests.accuracy",
    ]
    for item in required_paths:
        if not _is_non_empty(get_path(case, item)):
            errors.append(f"{label}: missing required field {item}")

    level = get_path(case, "metadata.level")
    if level not in ALLOWED_LEVELS:
        errors.append(f"{label}: metadata.level must be one of {sorted(ALLOWED_LEVELS)}")

    services = case.get("services")
    if not isinstance(services, list) or not services:
        errors.append(f"{label}: services must be a non-empty list")
        return errors, warnings

    service_names: set[str] = set()
    ports: set[int] = set()
    for index, service in enumerate(services):
        if not isinstance(service, dict):
            errors.append(f"{label}: services[{index}] must be a mapping")
            continue
        name = service.get("name")
        if not name:
            errors.append(f"{label}: services[{index}].name is required")
        elif str(name) in service_names:
            errors.append(f"{label}: duplicate service name {name}")
        else:
            service_names.add(str(name))

        port = service.get("port")
        try:
            port_int = int(port)
            if not (1 <= port_int <= 65535):
                errors.append(f"{label}: service {name} port out of range: {port}")
            elif port_int in ports:
                errors.append(f"{label}: duplicate service port {port_int}")
            else:
                ports.add(port_int)
        except (TypeError, ValueError):
            errors.append(f"{label}: service {name} port must be an integer")

        if service.get("type") == "vllm-serve":
            vllm = service.get("vllm")
            if not isinstance(vllm, dict):
                errors.append(f"{label}: service {name} vllm config must be a mapping")
            elif not vllm.get("model"):
                errors.append(f"{label}: service {name} type vllm-serve requires vllm.model")
            elif not isinstance(vllm.get("args", []), list):
                errors.append(f"{label}: service {name} vllm.args must be a list")
        else:
            warnings.append(f"{label}: service {name} type {service.get('type')} is reserved for future runners")

    smoke = case.get("tests", {}).get("smoke") or {}
    if isinstance(smoke, dict) and smoke.get("enabled", True):
        payload = smoke.get("payload") or {}
        if not isinstance(payload, dict):
            errors.append(f"{label}: tests.smoke.payload must be a mapping")
        else:
            payload_model = payload.get("model")
            primary_model = served_model_name(services[0])
            if payload_model != primary_model:
                errors.append(
                    f"{label}: tests.smoke.payload.model ({payload_model}) "
                    f"must match served-model-name ({primary_model})"
                )

    return errors, warnings


def stage_result(status: str, **fields: Any) -> dict[str, Any]:
    result = {"status": status}
    result.update(fields)
    return result


def load_case_results(input_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(input_dir)
    if not directory.exists():
        return []
    results: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "case_name" in payload:
            payload.setdefault("_path", str(path))
            results.append(payload)
    return results
