#!/usr/bin/env python3
"""Compare current benchmark JSON against a baseline JSON."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from deploy_case_lib import read_json, write_json

METRIC_ALIASES = {
    "failed_requests": ["failed_requests", "num_failed_requests", "failed"],
    "ttft": ["mean_ttft_ms", "ttft_ms", "mean_ttft", "Mean TTFT (ms)"],
    "tpot": ["mean_tpot_ms", "tpot_ms", "mean_tpot", "Mean TPOT (ms)"],
    "itl": ["mean_itl_ms", "itl_ms", "mean_itl", "Mean ITL (ms)"],
    "throughput": ["request_throughput", "requests_per_second", "throughput", "Output Token Throughput.total"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare benchmark result JSON files.")
    parser.add_argument("--current", required=True, help="Current benchmark JSON")
    parser.add_argument("--baseline", required=True, help="Baseline benchmark JSON")
    parser.add_argument("--output", default="reports/benchmark_compare.json", help="Comparison JSON output")
    parser.add_argument("--max-failed-requests", type=int, default=0)
    parser.add_argument("--max-ttft-regression-pct", type=float, default=10.0)
    parser.add_argument("--max-tpot-regression-pct", type=float, default=10.0)
    parser.add_argument("--max-itl-regression-pct", type=float, default=10.0)
    parser.add_argument("--max-throughput-drop-pct", type=float, default=10.0)
    return parser.parse_args()


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
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


def _pct_delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline in (None, 0):
        return None
    return (current - baseline) / baseline * 100


def main() -> int:
    args = parse_args()
    current_path = Path(args.current)
    baseline_path = Path(args.baseline)
    if not current_path.exists():
        raise SystemExit(f"current benchmark JSON does not exist: {current_path}")
    if not baseline_path.exists():
        raise SystemExit(f"baseline benchmark JSON does not exist: {baseline_path}")

    current = _flatten(read_json(current_path))
    baseline = _flatten(read_json(baseline_path))
    current_metrics = {metric: _extract_metric(current, metric) for metric in METRIC_ALIASES}
    baseline_metrics = {metric: _extract_metric(baseline, metric) for metric in METRIC_ALIASES}

    deltas = {metric: _pct_delta(current_metrics[metric], baseline_metrics[metric]) for metric in METRIC_ALIASES}
    failures: list[str] = []
    failed_requests = current_metrics["failed_requests"]
    if failed_requests is not None and failed_requests > args.max_failed_requests:
        failures.append(f"failed_requests {failed_requests} > {args.max_failed_requests}")
    for metric, threshold in [
        ("ttft", args.max_ttft_regression_pct),
        ("tpot", args.max_tpot_regression_pct),
        ("itl", args.max_itl_regression_pct),
    ]:
        delta = deltas[metric]
        if delta is not None and delta > threshold:
            failures.append(f"{metric} regression {delta:.2f}% > {threshold}%")
    throughput_drop = None
    if deltas["throughput"] is not None:
        throughput_drop = -deltas["throughput"]
        if throughput_drop > args.max_throughput_drop_pct:
            failures.append(f"throughput drop {throughput_drop:.2f}% > {args.max_throughput_drop_pct}%")

    report = {
        "status": "failed" if failures else "passed",
        "current": str(current_path),
        "baseline": str(baseline_path),
        "current_metrics": current_metrics,
        "baseline_metrics": baseline_metrics,
        "delta_pct": deltas,
        "throughput_drop_pct": throughput_drop,
        "thresholds": {
            "max_failed_requests": args.max_failed_requests,
            "max_ttft_regression_pct": args.max_ttft_regression_pct,
            "max_tpot_regression_pct": args.max_tpot_regression_pct,
            "max_itl_regression_pct": args.max_itl_regression_pct,
            "max_throughput_drop_pct": args.max_throughput_drop_pct,
        },
        "failures": failures,
    }
    write_json(args.output, report)
    print(f"benchmark compare status={report['status']}; output={args.output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
