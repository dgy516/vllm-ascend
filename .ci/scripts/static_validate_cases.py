#!/usr/bin/env python3
"""Static validation for selected DeployCase files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from deploy_case_lib import (
    build_vllm_serve_command,
    case_name,
    command_to_shell,
    first_service,
    load_case,
    read_case_list,
    validate_case,
    write_json,
)

INT_FLAGS = {
    "--port",
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--data-parallel-size",
    "--data-parallel-size-local",
    "--data-parallel-rank",
    "--max-model-len",
    "--max-num-batched-tokens",
    "--max-num-seqs",
    "--block-size",
}
FLOAT_FLAGS = {"--gpu-memory-utilization", "--request-rate", "--temperature", "--top-p"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statically validate selected DeployCase files.")
    parser.add_argument("--case-list", required=True, help="Path produced by select_deploy_cases.py")
    parser.add_argument("--output", default="reports/static_validate.json", help="JSON report path")
    parser.add_argument("--model-root", default="", help="Optional local model root")
    parser.add_argument("--check-model-path", action="store_true", help="Check local model paths when possible")
    return parser.parse_args()


def _validate_flag_types(command: list[str]) -> list[str]:
    errors: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in INT_FLAGS or token in FLOAT_FLAGS:
            if index + 1 >= len(command):
                errors.append(f"{token} requires a value")
                index += 1
                continue
            value = command[index + 1]
            try:
                int(value) if token in INT_FLAGS else float(value)
            except ValueError:
                expected = "integer" if token in INT_FLAGS else "float"
                errors.append(f"{token} expects {expected}, got {value!r}")
            index += 2
            continue
        index += 1
    return errors


def _check_model_path(case: dict[str, Any], model_root: str) -> list[str]:
    service = first_service(case)
    model = str((service.get("vllm") or {}).get("model") or "")
    if not model:
        return []

    candidates: list[Path] = []
    model_path = Path(model)
    if model_path.is_absolute() or model.startswith("."):
        candidates.append(model_path)
    elif model_root:
        candidates.append(Path(model_root) / model)
        candidates.append(Path(model_root) / model.split("/")[-1])

    if candidates and not any(path.exists() for path in candidates):
        rendered = ", ".join(str(path) for path in candidates)
        return [f"model path not found for {case_name(case)}; checked {rendered}"]
    return []


def main() -> int:
    args = parse_args()
    paths = read_case_list(args.case_list)
    report = {"status": "passed", "total": len(paths), "cases": [], "errors": [], "warnings": []}

    for path in paths:
        entry = {
            "path": path,
            "case_name": Path(path).stem,
            "status": "passed",
            "command": "",
            "errors": [],
            "warnings": [],
        }
        try:
            case = load_case(path)
            entry["case_name"] = case_name(case)
            errors, warnings = validate_case(case, path)
            service = first_service(case)
            if service.get("type") != "vllm-serve":
                errors.append(f"{case_name(case)}: only vllm-serve is supported in MVP")
            command = build_vllm_serve_command(case, service, {"MODEL_ROOT": args.model_root})
            entry["command"] = command_to_shell(command)
            errors.extend(_validate_flag_types(command))
            if args.check_model_path:
                errors.extend(_check_model_path(case, args.model_root))
        except Exception as exc:  # noqa: BLE001 - convert to clear JSON report
            errors = [f"{path}: {exc}"]
            warnings = []

        if errors:
            entry["status"] = "failed"
            report["status"] = "failed"
            report["errors"].extend(errors)
        entry["errors"] = errors
        entry["warnings"] = warnings
        report["warnings"].extend(warnings)
        report["cases"].append(entry)

    write_json(args.output, report)
    print(f"static validated {len(paths)} case(s); status={report['status']}; output={args.output}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
