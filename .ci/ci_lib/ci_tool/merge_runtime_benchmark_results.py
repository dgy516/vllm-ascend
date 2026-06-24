#!/usr/bin/env python3
"""Merge runtime benchmark stage results into DeployCase case result JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ci_tool.deploy_case_lib import STATUS_FAILED, STATUS_PASSED, STATUS_SKIPPED, read_json, safe_slug, stage_result, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge runtime benchmark results into case result JSON files.")
    parser.add_argument("--case-results", default="reports/nightly/case_results")
    parser.add_argument("--benchmark-results", default="reports/runtime_plan/benchmark_results")
    return parser.parse_args()


def _case_result_path(case_results: Path, case_name: str) -> Path:
    return case_results / f"{safe_slug(case_name)}.json"


def _load_case_result(case_results: Path, case_name: str, benchmark_payload: dict[str, Any]) -> dict[str, Any]:
    path = _case_result_path(case_results, case_name)
    if path.exists():
        return read_json(path)
    return {
        "case_name": case_name,
        "level": "",
        "status": STATUS_SKIPPED,
        "failure_stage": None,
        "failure_reason": None,
        "allocated_cards": [],
        "allocated_ports": [],
        "container_name": benchmark_payload.get("container_name"),
        "host_node": benchmark_payload.get("node"),
        "startup": stage_result(STATUS_SKIPPED, reason="case result was not produced before benchmark merge"),
        "readiness": stage_result(STATUS_SKIPPED),
        "smoke": stage_result(STATUS_SKIPPED),
        "benchmark": stage_result(STATUS_SKIPPED),
        "accuracy": stage_result(STATUS_SKIPPED),
        "artifacts": {},
    }


def _merge_one(case_results: Path, benchmark_payload: dict[str, Any]) -> bool:
    case_name = str(benchmark_payload.get("case_name") or "")
    if not case_name:
        return False
    result = _load_case_result(case_results, case_name, benchmark_payload)
    benchmark = benchmark_payload.get("benchmark") if isinstance(benchmark_payload.get("benchmark"), dict) else {}
    if not benchmark:
        benchmark = stage_result(STATUS_SKIPPED, reason="empty benchmark result")
    result["benchmark"] = benchmark
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    if benchmark_payload.get("endpoint"):
        artifacts["benchmark_endpoint"] = benchmark_payload["endpoint"]
    if benchmark.get("result_file"):
        artifacts["benchmark_result_file"] = benchmark["result_file"]
    if benchmark.get("log_file"):
        artifacts["benchmark_log"] = benchmark["log_file"]
    result["artifacts"] = artifacts

    benchmark_status = str(benchmark.get("status", STATUS_SKIPPED))
    if benchmark_status == STATUS_FAILED:
        result["status"] = STATUS_FAILED
        result["failure_stage"] = "benchmark"
        result["failure_reason"] = str(benchmark.get("failure_reason") or "benchmark failed")
    elif benchmark_status == STATUS_PASSED and result.get("status") != STATUS_FAILED:
        smoke = result.get("smoke") if isinstance(result.get("smoke"), dict) else {}
        readiness = result.get("readiness") if isinstance(result.get("readiness"), dict) else {}
        if smoke.get("status") == STATUS_PASSED and readiness.get("status") == STATUS_PASSED:
            result["status"] = STATUS_PASSED
            result["failure_stage"] = None
            result["failure_reason"] = None

    write_json(_case_result_path(case_results, case_name), result)
    return benchmark_status == STATUS_FAILED


def main() -> int:
    args = parse_args()
    case_results = Path(args.case_results)
    benchmark_results = Path(args.benchmark_results)
    case_results.mkdir(parents=True, exist_ok=True)
    if not benchmark_results.exists():
        print(f"benchmark results directory does not exist, nothing to merge: {benchmark_results}")
        return 0
    failures = 0
    merged = 0
    for path in sorted(benchmark_results.glob("*.json")):
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        if _merge_one(case_results, payload):
            failures += 1
        merged += 1
    print(f"merged {merged} benchmark result(s) from {benchmark_results}; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
