#!/usr/bin/env python3
"""Run compiled benchmark tasks inside a runtime container."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from ci_tool.deploy_case_lib import STATUS_FAILED, STATUS_PASSED, STATUS_SKIPPED, read_json, safe_slug, stage_result, write_json

METRIC_ALIASES = {
    "failed_requests": ["failed_requests", "num_failed_requests", "failed"],
    "ttft": ["mean_ttft_ms", "ttft_ms", "mean_ttft", "Mean TTFT (ms)"],
    "tpot": ["mean_tpot_ms", "tpot_ms", "mean_tpot", "Mean TPOT (ms)"],
    "itl": ["mean_itl_ms", "itl_ms", "mean_itl", "Mean ITL (ms)"],
    "throughput": ["request_throughput", "requests_per_second", "throughput", "Output Token Throughput.total"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compiled benchmark tasks inside the runtime container.")
    parser.add_argument("--tasks", required=True, help="Node-local benchmark_tasks.json")
    parser.add_argument("--output-dir", default="reports/runtime_plan/benchmark_results")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value, next_prefix))
        return result
    if isinstance(payload, list):
        result = {}
        for index, value in enumerate(payload):
            next_prefix = f"{prefix}.{index}" if prefix else str(index)
            result.update(_flatten(value, next_prefix))
        return result
    return {prefix: payload}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _extract_metric(flattened: dict[str, Any], metric: str) -> float | None:
    aliases = METRIC_ALIASES[metric]
    for key in aliases:
        if key in flattened:
            return _number(flattened[key])
    for key, value in flattened.items():
        lower_key = key.lower().replace(" ", "_")
        if any(alias.lower().replace(" ", "_") in lower_key for alias in aliases):
            number = _number(value)
            if number is not None:
                return number
    return None


def _extract_metrics(path: str | Path) -> dict[str, float | None]:
    flattened = _flatten(read_json(path))
    return {metric: _extract_metric(flattened, metric) for metric in METRIC_ALIASES}


def _latest_json(result_dir: str | Path, started_at: float) -> str:
    directory = Path(result_dir)
    if not directory.exists():
        return ""
    candidates = [
        path
        for path in directory.rglob("*.json")
        if path.is_file() and path.stat().st_mtime >= started_at - 1
    ]
    if not candidates:
        return ""
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def _pct_delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline in (None, 0):
        return None
    return (current - baseline) / baseline * 100


def _compare_metrics(
    *,
    current_metrics: dict[str, float | None],
    baseline_path: str,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    if not baseline_path:
        return {"status": STATUS_SKIPPED, "reason": "baseline not configured"}
    baseline = Path(baseline_path)
    if not baseline.exists():
        return {"status": STATUS_FAILED, "failure_reason": f"baseline does not exist: {baseline}"}

    baseline_metrics = _extract_metrics(baseline)
    deltas = {metric: _pct_delta(current_metrics.get(metric), baseline_metrics.get(metric)) for metric in METRIC_ALIASES}
    failures: list[str] = []
    max_failed_requests = int(thresholds.get("max_failed_requests", 0))
    failed_requests = current_metrics.get("failed_requests")
    if failed_requests is not None and failed_requests > max_failed_requests:
        failures.append(f"failed_requests {failed_requests} > {max_failed_requests}")
    for metric, threshold_key in [
        ("ttft", "max_ttft_regression_pct"),
        ("tpot", "max_tpot_regression_pct"),
        ("itl", "max_itl_regression_pct"),
    ]:
        threshold = float(thresholds.get(threshold_key, 10.0))
        delta = deltas[metric]
        if delta is not None and delta > threshold:
            failures.append(f"{metric} regression {delta:.2f}% > {threshold}%")
    throughput_drop = None
    if deltas["throughput"] is not None:
        throughput_drop = -deltas["throughput"]
        threshold = float(thresholds.get("max_throughput_drop_pct", 10.0))
        if throughput_drop > threshold:
            failures.append(f"throughput drop {throughput_drop:.2f}% > {threshold}%")

    return {
        "status": STATUS_FAILED if failures else STATUS_PASSED,
        "baseline": str(baseline),
        "baseline_metrics": baseline_metrics,
        "delta_pct": deltas,
        "throughput_drop_pct": throughput_drop,
        "thresholds": thresholds,
        "failures": failures,
    }


def _run_task(task: dict[str, Any], output_dir: Path, dry_run: bool) -> dict[str, Any]:
    case_label = str(task.get("case_name") or "unknown")
    command = [str(item) for item in task.get("command") or []]
    log_file = Path(str(task.get("log_file") or f"logs/deploy/{safe_slug(case_label)}/benchmark.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    result_dir = str(task.get("result_dir") or f"reports/nightly/benchmark/{safe_slug(case_label)}")
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    if task.get("status") == STATUS_SKIPPED:
        return stage_result(
            STATUS_SKIPPED,
            reason=str(task.get("reason") or "benchmark skipped"),
            log_file=str(log_file),
            result_dir=result_dir,
        )
    if dry_run:
        return stage_result(
            STATUS_SKIPPED,
            reason="dry-run",
            command=" ".join(command),
            log_file=str(log_file),
            result_dir=result_dir,
        )
    if not command:
        return stage_result(STATUS_SKIPPED, reason="benchmark command is empty", log_file=str(log_file))

    started = time.time()
    started_monotonic = time.monotonic()
    with log_file.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        try:
            completed = subprocess.run(
                command,
                cwd=os.getcwd(),
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=int(task.get("timeout_sec", 3600)),
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            return stage_result(
                STATUS_FAILED,
                command=" ".join(command),
                log_file=str(log_file),
                result_dir=result_dir,
                duration_sec=round(time.monotonic() - started_monotonic, 3),
                failure_reason=f"benchmark timeout after {task.get('timeout_sec', 3600)}s",
            )

    result_file = _latest_json(result_dir, started)
    metrics = _extract_metrics(result_file) if result_file else {}
    status = STATUS_PASSED if returncode == 0 else STATUS_FAILED
    comparison = _compare_metrics(
        current_metrics=metrics,
        baseline_path=str(task.get("baseline") or ""),
        thresholds=task.get("thresholds") or {},
    )
    if comparison.get("status") == STATUS_FAILED:
        status = STATUS_FAILED
    failure_reason = ""
    if returncode != 0:
        failure_reason = f"benchmark command exited with {returncode}"
    elif comparison.get("status") == STATUS_FAILED:
        failure_reason = "; ".join(str(item) for item in comparison.get("failures") or []) or str(
            comparison.get("failure_reason") or "benchmark comparison failed"
        )

    return stage_result(
        status,
        command=" ".join(command),
        returncode=returncode,
        duration_sec=round(time.monotonic() - started_monotonic, 3),
        metrics=metrics,
        comparison=comparison,
        result_file=result_file,
        result_dir=result_dir,
        log_file=str(log_file),
        failure_reason=failure_reason,
    )


def main() -> int:
    args = parse_args()
    tasks_payload = read_json(args.tasks)
    tasks = tasks_payload if isinstance(tasks_payload, list) else []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        case_label = str(task.get("case_name") or "unknown")
        result = {
            "case_name": case_label,
            "node": task.get("node"),
            "container_name": task.get("container_name"),
            "endpoint": task.get("endpoint"),
            "benchmark": _run_task(task, output_dir, args.dry_run),
        }
        write_json(output_dir / f"{safe_slug(case_label)}.json", result)
    print(f"ran {len(tasks)} benchmark task(s); output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
