#!/usr/bin/env python3
"""Prepare bounded log artifacts for Jenkins archiving."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ci_tool.deploy_case_lib import STATUS_FAILED, load_case_results, safe_slug, write_json

FULL_MODES = {"full", "failed-full"}
TAIL_MODES = {"tail", "failed-tail"}
FAILED_ONLY_MODES = {"failed-full", "failed-tail"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy or tail runtime logs into reports/logs for Jenkins archive.")
    parser.add_argument("--case-results", default="reports/nightly/case_results", help="DeployCase result JSON dir")
    parser.add_argument("--logs-dir", default="logs", help="Runtime logs root")
    parser.add_argument("--output-dir", default="reports/logs", help="Bounded log artifact output dir")
    parser.add_argument(
        "--manifest",
        default="reports/nightly/log_artifacts.json",
        help="JSON manifest for archived log artifacts",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "full", "tail", "failed-full", "failed-tail", "none"],
        default="auto",
        help="Log archive policy. auto uses full for nightly/release/benchmark and failed-tail otherwise.",
    )
    parser.add_argument("--ci-mode", default="", help="Jenkins CI_MODE used when --mode=auto")
    parser.add_argument("--tail-bytes", type=int, default=5 * 1024 * 1024, help="Bytes retained in tail modes")
    return parser.parse_args()


def _resolve_mode(mode: str, ci_mode: str) -> str:
    if mode != "auto":
        return mode
    if ci_mode in {"nightly", "release", "benchmark"}:
        return "full"
    return "failed-tail"


def _add_log_path(paths: dict[str, Path], value: Any) -> None:
    if not value:
        return
    path = Path(str(value))
    paths[str(path)] = path


def _case_log_paths(case: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    artifacts = case.get("artifacts") if isinstance(case.get("artifacts"), dict) else {}
    _add_log_path(paths, artifacts.get("server_log"))
    server_logs = artifacts.get("server_logs") if isinstance(artifacts.get("server_logs"), dict) else {}
    for value in server_logs.values():
        _add_log_path(paths, value)

    startup = case.get("startup") if isinstance(case.get("startup"), dict) else {}
    services = startup.get("services") if isinstance(startup.get("services"), list) else []
    for service in services:
        if isinstance(service, dict):
            _add_log_path(paths, service.get("log_file"))

    for stage in ("benchmark", "accuracy"):
        payload = case.get(stage) if isinstance(case.get(stage), dict) else {}
        _add_log_path(paths, payload.get("log_file"))
    return paths


def _artifact_path(source: Path, logs_dir: Path, output_dir: Path, case_name: str) -> Path:
    try:
        relative = source.resolve().relative_to(logs_dir.resolve())
    except (OSError, ValueError):
        try:
            relative = source.relative_to(logs_dir)
        except ValueError:
            relative = Path(safe_slug(case_name)) / safe_slug(str(source))
    return output_dir / relative


def _copy_tail(source: Path, target: Path, tail_bytes: int) -> tuple[bool, int]:
    size = source.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    if size <= tail_bytes:
        shutil.copy2(source, target)
        return False, size

    retained = max(tail_bytes, 0)
    marker = (
        f"[vllm-ascend-ci] log truncated for Jenkins archive; "
        f"original_size_bytes={size}; retained_tail_bytes={retained}\n"
    ).encode()
    with source.open("rb") as src, target.open("wb") as dst:
        dst.write(marker)
        if retained:
            src.seek(-retained, 2)
            shutil.copyfileobj(src, dst)
    return True, target.stat().st_size


def _archive_log(source: Path, target: Path, mode: str, tail_bytes: int) -> tuple[bool, int]:
    if mode in FULL_MODES:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return False, source.stat().st_size
    if mode in TAIL_MODES:
        return _copy_tail(source, target, tail_bytes)
    raise ValueError(f"unsupported archive mode: {mode}")


def _update_case_artifacts(
    case: dict[str, Any],
    case_entries: list[dict[str, Any]],
    result_path: Path,
) -> None:
    artifacts = case.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        case["artifacts"] = artifacts

    artifacts["log_archive"] = [
        {
            "source": entry["source"],
            "artifact": entry["artifact"],
            "truncated": entry["truncated"],
            "size_bytes": entry["size_bytes"],
            "archived_size_bytes": entry["archived_size_bytes"],
        }
        for entry in case_entries
    ]

    by_source = {entry["source"]: entry["artifact"] for entry in case_entries}
    server_log = artifacts.get("server_log")
    if server_log and str(server_log) in by_source:
        artifacts["archived_server_log"] = by_source[str(server_log)]

    server_logs = artifacts.get("server_logs") if isinstance(artifacts.get("server_logs"), dict) else {}
    archived_server_logs = {
        service: by_source[str(path)] for service, path in server_logs.items() if str(path) in by_source
    }
    if archived_server_logs:
        artifacts["archived_server_logs"] = archived_server_logs

    for stage in ("benchmark", "accuracy"):
        payload = case.get(stage) if isinstance(case.get(stage), dict) else {}
        log_file = payload.get("log_file")
        if log_file and str(log_file) in by_source:
            payload["archived_log_file"] = by_source[str(log_file)]

    write_json(result_path, case)


def main() -> int:
    args = parse_args()
    if args.tail_bytes < 0:
        print("--tail-bytes must be >= 0")
        return 1

    resolved_mode = _resolve_mode(args.mode, args.ci_mode)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.case_results)
    result_paths = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    results = load_case_results(results_dir)
    results_by_name = {str(item.get("case_name", "")): item for item in results}
    result_path_by_name = {}
    for path in result_paths:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            result_path_by_name[str(payload.get("case_name", ""))] = path

    entries: list[dict[str, Any]] = []
    if resolved_mode != "none":
        logs_dir = Path(args.logs_dir)
        for case_name, case in results_by_name.items():
            if resolved_mode in FAILED_ONLY_MODES and case.get("status") != STATUS_FAILED:
                continue
            case_entries: list[dict[str, Any]] = []
            for source in _case_log_paths(case).values():
                if not source.exists() or not source.is_file():
                    continue
                target = _artifact_path(source, logs_dir, output_dir, case_name)
                truncated, archived_size = _archive_log(source, target, resolved_mode, args.tail_bytes)
                entry = {
                    "case_name": case_name,
                    "case_status": case.get("status"),
                    "source": str(source),
                    "artifact": str(target),
                    "mode": resolved_mode,
                    "size_bytes": source.stat().st_size,
                    "archived_size_bytes": archived_size,
                    "truncated": truncated,
                }
                entries.append(entry)
                case_entries.append(entry)
            result_path = result_path_by_name.get(case_name)
            if result_path:
                _update_case_artifacts(case, case_entries, result_path)

    manifest = {
        "mode": resolved_mode,
        "requested_mode": args.mode,
        "ci_mode": args.ci_mode,
        "tail_bytes": args.tail_bytes,
        "total": len(entries),
        "entries": entries,
    }
    write_json(manifest_path, manifest)
    print(
        f"prepared {len(entries)} log artifact(s); mode={resolved_mode}; "
        f"output={output_dir}; manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
