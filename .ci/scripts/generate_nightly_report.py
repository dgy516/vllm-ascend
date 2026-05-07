#!/usr/bin/env python3
"""Generate HTML, JSON, and CSV nightly reports for DeployCase results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deploy_case_lib import STATUS_FAILED, STATUS_PASSED, STATUS_SKIPPED, load_case_results, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DeployCase nightly HTML/JSON/CSV reports.")
    parser.add_argument("--input", default="reports/nightly/case_results", help="Directory with case result JSON files")
    parser.add_argument("--output", default="reports/nightly/index.html", help="HTML output path")
    parser.add_argument("--template", default=".ci/templates/nightly_report.html.j2", help="HTML template path")
    parser.add_argument("--environment", default="", help="Optional environment JSON to include")
    return parser.parse_args()


def _read_environment(path: str) -> dict[str, Any]:
    if path and Path(path).exists():
        with Path(path).open(encoding="utf-8") as f:
            payload = json.load(f)
            return payload if isinstance(payload, dict) else {"value": payload}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ci_mode": os.getenv("CI_MODE", ""),
        "case_level": os.getenv("CASE_LEVEL", ""),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _template_render(template_text: str, context: dict[str, str]) -> str:
    try:
        from jinja2 import Template

        return Template(template_text).render(**context)
    except ModuleNotFoundError:
        rendered = template_text
        for key, value in context.items():
            rendered = rendered.replace("{{ " + key + " }}", value)
            rendered = rendered.replace("{{" + key + "}}", value)
        return rendered


def _case_rows(results: list[dict[str, Any]]) -> str:
    rows = []
    for item in results:
        status = html.escape(str(item.get("status", "")))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('case_name', '')))}</td>"
            f"<td>{html.escape(str(item.get('level', '')))}</td>"
            f"<td class=\"{status}\">{status}</td>"
            f"<td>{html.escape(str(item.get('failure_stage') or ''))}</td>"
            f"<td>{html.escape(str(item.get('failure_reason') or ''))}</td>"
            f"<td>{html.escape(str((item.get('artifacts') or {}).get('server_log', '')))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _stage_rows(results: list[dict[str, Any]], stage: str, columns: list[str]) -> str:
    rows = []
    for item in results:
        payload = item.get(stage) if isinstance(item.get(stage), dict) else {}
        status = str(payload.get("status", STATUS_SKIPPED))
        cells = [
            html.escape(str(item.get("case_name", ""))),
            f"<span class=\"{html.escape(status)}\">{html.escape(status)}</span>",
        ]
        for column in columns:
            value = payload.get(column, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            cells.append(html.escape(str(value)))
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return "\n".join(rows)


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    out_dir = output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    results = load_case_results(input_dir)

    total = len(results)
    passed = sum(1 for item in results if item.get("status") == STATUS_PASSED)
    failed = sum(1 for item in results if item.get("status") == STATUS_FAILED)
    skipped = sum(1 for item in results if item.get("status") == STATUS_SKIPPED)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "failed_cases": [item.get("case_name") for item in results if item.get("status") == STATUS_FAILED],
        "skipped_cases": [item.get("case_name") for item in results if item.get("status") == STATUS_SKIPPED],
    }
    write_json(out_dir / "summary.json", summary)

    environment = _read_environment(args.environment)
    write_json(out_dir / "environment.json", environment)

    case_rows = [
        {
            "case_name": item.get("case_name", ""),
            "level": item.get("level", ""),
            "status": item.get("status", ""),
            "failure_stage": item.get("failure_stage") or "",
            "failure_reason": item.get("failure_reason") or "",
            "server_log": (item.get("artifacts") or {}).get("server_log", ""),
        }
        for item in results
    ]
    _write_csv(
        out_dir / "cases.csv",
        case_rows,
        ["case_name", "level", "status", "failure_stage", "failure_reason", "server_log"],
    )
    _write_csv(
        out_dir / "benchmark.csv",
        [
            {
                "case_name": item.get("case_name", ""),
                "status": (item.get("benchmark") or {}).get("status", STATUS_SKIPPED),
                "metrics": json.dumps((item.get("benchmark") or {}).get("metrics", {}), ensure_ascii=False),
                "log_file": (item.get("benchmark") or {}).get("log_file", ""),
            }
            for item in results
        ],
        ["case_name", "status", "metrics", "log_file"],
    )
    _write_csv(
        out_dir / "accuracy.csv",
        [
            {
                "case_name": item.get("case_name", ""),
                "status": (item.get("accuracy") or {}).get("status", STATUS_SKIPPED),
                "mode": (item.get("accuracy") or {}).get("mode", "execute_only"),
                "score": (item.get("accuracy") or {}).get("score", "N/A"),
            }
            for item in results
        ],
        ["case_name", "status", "mode", "score"],
    )

    target_case_results = out_dir / "case_results"
    if input_dir.resolve() != target_case_results.resolve():
        target_case_results.mkdir(parents=True, exist_ok=True)
        for path in input_dir.glob("*.json"):
            shutil.copy2(path, target_case_results / path.name)

    template = Path(args.template).read_text(encoding="utf-8")
    context = {
        "generated_at": summary["generated_at"],
        "total": str(total),
        "passed": str(passed),
        "failed": str(failed),
        "skipped": str(skipped),
        "case_rows": _case_rows(results),
        "benchmark_rows": _stage_rows(results, "benchmark", ["metrics", "log_file"]),
        "accuracy_rows": _stage_rows(results, "accuracy", ["mode", "score"]),
        "environment_json": html.escape(json.dumps(environment, indent=2, ensure_ascii=False)),
    }
    output.write_text(_template_render(template, context), encoding="utf-8")
    print(f"wrote nightly report: {output}; total={total} passed={passed} failed={failed} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
