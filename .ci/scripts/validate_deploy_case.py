#!/usr/bin/env python3
"""Validate vLLM Ascend DeployCase YAML files."""

from __future__ import annotations

import argparse
from pathlib import Path

from deploy_case_lib import expand_case_paths, load_case, validate_case, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one or more DeployCase YAML files.")
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="DeployCase YAML glob(s), e.g. .ci/deploy_cases/*.yaml",
    )
    parser.add_argument("--output", default="reports/validated_cases.json", help="JSON report path")
    parser.add_argument("--allow-empty", action="store_true", help="Do not fail when no case file matches")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = expand_case_paths(args.cases)
    existing_paths = [path for path in paths if Path(path).exists()]

    report = {
        "status": "passed",
        "total": len(existing_paths),
        "cases": [],
        "errors": [],
        "warnings": [],
    }

    if not existing_paths and not args.allow_empty:
        report["status"] = "failed"
        report["errors"].append(f"no DeployCase files matched: {args.cases}")
        write_json(args.output, report)
        print(report["errors"][0])
        return 1

    for path in existing_paths:
        entry = {"path": path, "status": "passed", "errors": [], "warnings": []}
        try:
            case = load_case(path)
            errors, warnings = validate_case(case, path)
        except Exception as exc:  # noqa: BLE001 - report every malformed case cleanly
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
    print(f"validated {report['total']} DeployCase file(s); status={report['status']}; output={args.output}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
