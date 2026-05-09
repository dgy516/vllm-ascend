#!/usr/bin/env python3
"""Shared helpers for Jenkins DeployCase scripts."""

from __future__ import annotations

import csv
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
ALLOWED_SOCS = {"A2", "A3", "any"}
CONTAINER_WORKSPACE = "/home/ma-user/AscendCloud/jenkins"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
SMOKE_SUITE_DIR = ".ci/test_suites/smoke"
SMOKE_SUITE_COLUMNS = (
    "id",
    "enabled",
    "mode",
    "method",
    "endpoint",
    "model",
    "prompt",
    "messages_json",
    "max_tokens",
    "temperature",
    "extra_json",
    "expected_http_status",
    "min_output_tokens",
    "response_contains",
    "response_not_contains",
    "timeout_sec",
    "description",
)
ALLOWED_SMOKE_MODES = {"completion", "chat", "raw"}


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


def hardware_config(case: dict[str, Any]) -> dict[str, Any]:
    hardware = case.get("requirements", {}).get("hardware") or {}
    return hardware if isinstance(hardware, dict) else {}


def docker_config(case: dict[str, Any]) -> dict[str, Any]:
    docker = case.get("runtime", {}).get("docker") or {}
    return docker if isinstance(docker, dict) else {}


def service_card_count(service: dict[str, Any]) -> int | None:
    resources = service.get("resources") or {}
    if not isinstance(resources, dict) or resources.get("card_count") is None:
        return None
    try:
        return int(resources["card_count"])
    except (TypeError, ValueError):
        raise DeployCaseError(f"service {service.get('name')}: resources.card_count must be an integer") from None


def case_card_count(case: dict[str, Any]) -> int:
    value = hardware_config(case).get("card_count", 1)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise DeployCaseError(f"{case_name(case)}: requirements.hardware.card_count must be an integer") from None


def case_soc(case: dict[str, Any]) -> str:
    return str(hardware_config(case).get("soc", "any"))


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
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def set_cli_flag_value(args: list[str], flag: str, value: Any) -> None:
    value_text = str(value)
    for index, arg in enumerate(args):
        if arg == flag:
            if index + 1 < len(args) and not args[index + 1].startswith("--"):
                args[index + 1] = value_text
            else:
                args.insert(index + 1, value_text)
            return
        if arg.startswith(f"{flag}="):
            args[index] = f"{flag}={value_text}"
            return
    args.extend([flag, value_text])


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        items = value
    else:
        items = [value]
    result: list[int] = []
    for item in items:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            raise DeployCaseError(f"allocation value must be an integer, got {item!r}") from None
    return result


def allocation_cards(allocation: dict[str, Any] | None) -> list[int]:
    if not allocation:
        return []
    return _as_int_list(allocation.get("cards") or allocation.get("allocated_cards"))


def allocation_ports(allocation: dict[str, Any] | None) -> list[int]:
    if not allocation:
        return []
    return _as_int_list(allocation.get("ports") or allocation.get("allocated_ports"))


def apply_allocation(case: dict[str, Any], allocation: dict[str, Any] | None) -> None:
    ports = allocation_ports(allocation)
    if not ports:
        return
    services = case.get("services") or []
    if len(ports) < len(services):
        raise DeployCaseError(
            f"{case_name(case)}: allocation provides {len(ports)} port(s) "
            f"for {len(services)} service(s)"
        )
    for service, port in zip(services, ports, strict=False):
        if isinstance(service, dict):
            service["port"] = port


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

    if service.get("host") is not None:
        set_cli_flag_value(args, "--host", service["host"])
    if service.get("port") is not None:
        set_cli_flag_value(args, "--port", service["port"])
    served_name = served_model_name(service)
    if served_name and served_name != model:
        set_cli_flag_value(args, "--served-model-name", served_name)

    return ["vllm", "serve", model, *args]


def command_to_shell(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def parse_bool(value: Any, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise DeployCaseError(f"expected boolean value, got {value!r}")


def _as_str_list(value: Any, field_name: str) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise DeployCaseError(f"{field_name} must be a string or list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DeployCaseError(f"{field_name} must contain non-empty strings")
        result.append(item.strip())
    return result


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(expand_text(path))
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def smoke_enabled(case: dict[str, Any]) -> bool:
    smoke = case.get("tests", {}).get("smoke") or {}
    if not isinstance(smoke, dict):
        return False
    return parse_bool(smoke.get("enabled"), True)


def selected_smoke_suite_files(case: dict[str, Any]) -> list[str]:
    """Return smoke CSV suite files selected by DeployCase capabilities."""

    if not smoke_enabled(case):
        return []

    smoke = case.get("tests", {}).get("smoke") or {}
    suites = smoke.get("suites")

    # Backward compatibility: legacy inline payload-only cases remain runnable
    # until all downstream inventories migrate to CSV suites.
    if suites is None and smoke.get("payload") is not None:
        return []

    if suites is None:
        suites = {}
    if not isinstance(suites, dict):
        raise DeployCaseError(f"{case_name(case)}: tests.smoke.suites must be a mapping")

    files: list[str] = []
    if parse_bool(suites.get("include_common"), True):
        files.append(f"{SMOKE_SUITE_DIR}/common.csv")

    for capability in _as_str_list(suites.get("capabilities"), "tests.smoke.suites.capabilities"):
        files.append(f"{SMOKE_SUITE_DIR}/{safe_slug(capability)}.csv")

    files.extend(_as_str_list(suites.get("extra_suite_files"), "tests.smoke.suites.extra_suite_files"))
    files.extend(_as_str_list(smoke.get("suite_files"), "tests.smoke.suite_files"))

    seen: set[str] = set()
    result: list[str] = []
    for item in files:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _row_value(row: dict[str, Any], key: str, env: dict[str, str]) -> str:
    return expand_text((row.get(key) or "").strip(), env)


def _parse_json_object(value: str, field_name: str, test_id: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DeployCaseError(f"{test_id}: {field_name} must be valid JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise DeployCaseError(f"{test_id}: {field_name} must be a JSON object")
    return payload


def _parse_json_list(value: str, field_name: str, test_id: str) -> list[Any]:
    if not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DeployCaseError(f"{test_id}: {field_name} must be valid JSON: {exc}") from None
    if not isinstance(payload, list):
        raise DeployCaseError(f"{test_id}: {field_name} must be a JSON list")
    return payload


def _parse_optional_int(value: str, field_name: str, test_id: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        raise DeployCaseError(f"{test_id}: {field_name} must be an integer") from None


def _parse_optional_float(value: str, field_name: str, test_id: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise DeployCaseError(f"{test_id}: {field_name} must be a float") from None


def _smoke_template_env(case: dict[str, Any], service: dict[str, Any], model_root: str = "") -> dict[str, str]:
    return {
        "served_model_name": served_model_name(service),
        "case_name": case_name(case),
        "host": str(service.get("host", "127.0.0.1")),
        "port": str(service.get("port", 8000)),
        "model_root": model_root,
    }


def build_smoke_test_case(
    case: dict[str, Any],
    row: dict[str, Any],
    suite_file: str | Path,
    service: dict[str, Any] | None = None,
    model_root: str = "",
) -> dict[str, Any]:
    service = service or first_service(case)
    env = _smoke_template_env(case, service, model_root)
    test_id = _row_value(row, "id", env)
    if not test_id:
        raise DeployCaseError(f"{suite_file}: smoke test id is required")

    mode = (_row_value(row, "mode", env) or "completion").lower()
    if mode not in ALLOWED_SMOKE_MODES:
        raise DeployCaseError(f"{test_id}: mode must be one of {sorted(ALLOWED_SMOKE_MODES)}")

    enabled = parse_bool(_row_value(row, "enabled", env), True)
    method = (_row_value(row, "method", env) or "POST").upper()
    endpoint = _row_value(row, "endpoint", env)
    if not endpoint:
        endpoint = "/v1/chat/completions" if mode == "chat" else "/v1/completions"

    expected_http_status = _parse_optional_int(
        _row_value(row, "expected_http_status", env),
        "expected_http_status",
        test_id,
    )
    expected_http_status = 200 if expected_http_status is None else expected_http_status
    timeout_sec = _parse_optional_int(_row_value(row, "timeout_sec", env), "timeout_sec", test_id) or 120
    min_output_tokens = _parse_optional_int(
        _row_value(row, "min_output_tokens", env),
        "min_output_tokens",
        test_id,
    )
    max_tokens = _parse_optional_int(_row_value(row, "max_tokens", env), "max_tokens", test_id)
    temperature = _parse_optional_float(_row_value(row, "temperature", env), "temperature", test_id)
    extra_json = _parse_json_object(_row_value(row, "extra_json", env), "extra_json", test_id)
    model = _row_value(row, "model", env) or served_model_name(service)

    if mode == "raw":
        payload = dict(extra_json)
        if model and "model" not in payload:
            payload["model"] = model
    elif mode == "chat":
        messages = _parse_json_list(_row_value(row, "messages_json", env), "messages_json", test_id)
        if not messages:
            prompt = _row_value(row, "prompt", env)
            if not prompt:
                raise DeployCaseError(f"{test_id}: chat mode requires messages_json or prompt")
            messages = [{"role": "user", "content": prompt}]
        payload = {"model": model, "messages": messages}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(extra_json)
    else:
        prompt = _row_value(row, "prompt", env)
        if not prompt:
            raise DeployCaseError(f"{test_id}: completion mode requires prompt")
        payload = {"model": model, "prompt": prompt}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(extra_json)

    return {
        "id": test_id,
        "enabled": enabled,
        "suite_file": str(suite_file),
        "mode": mode,
        "method": method,
        "endpoint": endpoint,
        "payload": payload,
        "expected_http_status": expected_http_status,
        "min_output_tokens": min_output_tokens,
        "response_contains": _row_value(row, "response_contains", env),
        "response_not_contains": _row_value(row, "response_not_contains", env),
        "timeout_sec": timeout_sec,
        "description": _row_value(row, "description", env),
    }


def _legacy_smoke_test_case(case: dict[str, Any], service: dict[str, Any]) -> dict[str, Any]:
    smoke = case.get("tests", {}).get("smoke") or {}
    return {
        "id": "legacy-smoke",
        "enabled": True,
        "suite_file": "inline-payload",
        "mode": "raw",
        "method": str(smoke.get("method", "POST")).upper(),
        "endpoint": str(smoke.get("endpoint", "/v1/completions")),
        "payload": smoke.get("payload") or {},
        "expected_http_status": int(smoke.get("expected_http_status", 200)),
        "min_output_tokens": None,
        "response_contains": "",
        "response_not_contains": "",
        "timeout_sec": int(smoke.get("timeout_sec", 120)),
        "description": "legacy inline smoke payload",
    }


def load_smoke_test_cases(
    case: dict[str, Any],
    model_root: str = "",
    service: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not smoke_enabled(case):
        return []
    service = service or first_service(case)
    suite_files = selected_smoke_suite_files(case)
    if not suite_files:
        smoke = case.get("tests", {}).get("smoke") or {}
        if smoke.get("payload") is not None:
            return [_legacy_smoke_test_case(case, service)]
        return []

    tests: list[dict[str, Any]] = []
    for suite_file in suite_files:
        suite_path = _resolve_repo_path(suite_file)
        with suite_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tests.append(build_smoke_test_case(case, row, suite_file, service, model_root))
    return tests


def validate_smoke_suites(case: dict[str, Any], label: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not smoke_enabled(case):
        return errors, warnings

    smoke = case.get("tests", {}).get("smoke") or {}
    if smoke.get("payload") is not None:
        warnings.append(f"{label}: tests.smoke.payload is deprecated; use tests.smoke.suites")

    try:
        suite_files = selected_smoke_suite_files(case)
    except DeployCaseError as exc:
        return [f"{label}: {exc}"], warnings

    if not suite_files:
        if smoke.get("payload") is None:
            errors.append(f"{label}: tests.smoke.suites selects no CSV suite files")
        return errors, warnings

    seen_ids: set[str] = set()
    service = first_service(case)
    primary_model = served_model_name(service)
    for suite_file in suite_files:
        suite_path = _resolve_repo_path(suite_file)
        if not suite_path.exists():
            errors.append(f"{label}: smoke suite file does not exist: {suite_file}")
            continue
        try:
            with suite_path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                missing = [column for column in SMOKE_SUITE_COLUMNS if column not in fieldnames]
                if missing:
                    errors.append(f"{label}: {suite_file} missing CSV column(s): {missing}")
                    continue
                for row in reader:
                    try:
                        test_case = build_smoke_test_case(case, row, suite_file, service)
                    except DeployCaseError as exc:
                        errors.append(f"{label}: {suite_file}: {exc}")
                        continue
                    test_id = str(test_case["id"])
                    if test_id in seen_ids:
                        errors.append(f"{label}: duplicate smoke test id: {test_id}")
                    seen_ids.add(test_id)
                    if 200 <= int(test_case["expected_http_status"]) < 300:
                        payload = test_case.get("payload") or {}
                        payload_model = payload.get("model") if isinstance(payload, dict) else None
                        if payload_model != primary_model:
                            errors.append(
                                f"{label}: {suite_file}:{test_id} payload.model ({payload_model}) "
                                f"must match served-model-name ({primary_model})"
                            )
        except OSError as exc:
            errors.append(f"{label}: failed to read smoke suite {suite_file}: {exc}")

    return errors, warnings


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
        "requirements.hardware.soc",
        "requirements.hardware.card_count",
        "requirements.hardware.allow_parallel_on_host",
        "requirements.software",
        "requirements.model",
        "runtime.image",
        "runtime.workdir",
        "runtime.env",
        "runtime.docker",
        "runtime.docker.enabled",
        "runtime.docker.image",
        "runtime.docker.workspace",
        "runtime.docker.network",
        "runtime.docker.shm_size",
        "runtime.docker.mounts",
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

    if get_path(case, "runtime.workdir") != CONTAINER_WORKSPACE:
        errors.append(f"{label}: runtime.workdir must be {CONTAINER_WORKSPACE}")

    hardware = hardware_config(case)
    soc = hardware.get("soc")
    if soc not in ALLOWED_SOCS:
        errors.append(f"{label}: requirements.hardware.soc must be one of {sorted(ALLOWED_SOCS)}")
    try:
        card_count = int(hardware.get("card_count"))
        if card_count < 1:
            errors.append(f"{label}: requirements.hardware.card_count must be >= 1")
    except (TypeError, ValueError):
        errors.append(f"{label}: requirements.hardware.card_count must be an integer")
    allow_parallel = hardware.get("allow_parallel_on_host")
    if not isinstance(allow_parallel, bool):
        errors.append(f"{label}: requirements.hardware.allow_parallel_on_host must be boolean")

    docker = docker_config(case)
    if docker.get("enabled") is not True:
        errors.append(f"{label}: runtime.docker.enabled must be true for Jenkins runtime execution")
    if docker.get("workspace") != CONTAINER_WORKSPACE:
        errors.append(f"{label}: runtime.docker.workspace must be {CONTAINER_WORKSPACE}")
    if docker.get("network") != "host":
        errors.append(f"{label}: runtime.docker.network must be host in the MVP runner")
    mounts = docker.get("mounts")
    if not isinstance(mounts, list) or not mounts:
        errors.append(f"{label}: runtime.docker.mounts must be a non-empty list")
    elif any(not isinstance(mount, dict) for mount in mounts):
        errors.append(f"{label}: each runtime.docker.mounts entry must be a mapping")
    else:
        targets = {str(mount.get("target")) for mount in mounts}
        if "/workspace/vllm-ascend" in targets:
            errors.append(f"{label}: runtime.docker.mounts must not target /workspace/vllm-ascend")
        required_targets = {
            f"{CONTAINER_WORKSPACE}/.ci",
            f"{CONTAINER_WORKSPACE}/reports",
            f"{CONTAINER_WORKSPACE}/logs",
        }
        missing_targets = sorted(required_targets - targets)
        if missing_targets:
            errors.append(f"{label}: runtime.docker.mounts missing required target(s): {missing_targets}")

    services = case.get("services")
    if not isinstance(services, list) or not services:
        errors.append(f"{label}: services must be a non-empty list")
        return errors, warnings

    service_names: set[str] = set()
    ports: set[int] = set()
    service_card_total = 0
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
            if len(services) > 1:
                try:
                    service_cards = service_card_count(service)
                except DeployCaseError as exc:
                    errors.append(f"{label}: {exc}")
                    service_cards = None
                if service_cards is None:
                    errors.append(f"{label}: service {name} resources.card_count is required for multi-service cases")
                elif service_cards < 1:
                    errors.append(f"{label}: service {name} resources.card_count must be >= 1")
                else:
                    service_card_total += service_cards
        else:
            warnings.append(f"{label}: service {name} type {service.get('type')} is reserved for future runners")
    if len(services) > 1:
        try:
            card_count = case_card_count(case)
            if service_card_total > card_count:
                errors.append(
                    f"{label}: sum of services[].resources.card_count ({service_card_total}) "
                    f"must not exceed requirements.hardware.card_count ({card_count})"
                )
        except DeployCaseError as exc:
            errors.append(f"{label}: {exc}")

    smoke = case.get("tests", {}).get("smoke") or {}
    if not isinstance(smoke, dict):
        errors.append(f"{label}: tests.smoke must be a mapping")
    else:
        suite_errors, suite_warnings = validate_smoke_suites(case, label)
        errors.extend(suite_errors)
        warnings.extend(suite_warnings)

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
