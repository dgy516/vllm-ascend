#!/usr/bin/env python3
"""Generate Jenkins-compatible JUnit XML from DeployCase result JSON files."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from deploy_case_lib import STATUS_FAILED, STATUS_SKIPPED, load_case_results

STAGES = ["startup", "readiness", "smoke", "benchmark", "accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate JUnit XML for DeployCase results.")
    parser.add_argument("--input", default="reports/nightly/case_results", help="Directory with case result JSON files")
    parser.add_argument("--output", default="reports/nightly/junit.xml", help="JUnit XML output path")
    return parser.parse_args()


def _stage_status(case: dict[str, Any], stage: str) -> str:
    value = case.get(stage)
    if isinstance(value, dict):
        return str(value.get("status", STATUS_SKIPPED))
    return STATUS_SKIPPED


def _stage_time(case: dict[str, Any], stage: str) -> str:
    value = case.get(stage)
    if isinstance(value, dict):
        return str(value.get("duration_sec", 0))
    return "0"


def main() -> int:
    args = parse_args()
    results = load_case_results(args.input)
    failures = 0
    skipped = 0
    tests = 0
    suite = ET.Element(
        "testsuite",
        name="vllm-ascend-deploy-cases",
        tests="0",
        failures="0",
        skipped="0",
    )

    for case in results:
        case_name = str(case.get("case_name", "unknown"))
        for stage in STAGES:
            status = _stage_status(case, stage)
            testcase = ET.SubElement(
                suite,
                "testcase",
                classname=case_name,
                name=f"{case_name}.{stage}",
                time=_stage_time(case, stage),
            )
            stage_payload = case.get(stage) if isinstance(case.get(stage), dict) else {}
            if status == STATUS_FAILED:
                failures += 1
                reason = stage_payload.get("failure_reason") or case.get("failure_reason") or f"{stage} failed"
                failure = ET.SubElement(testcase, "failure", message=str(reason))
                failure.text = str(reason)
            elif status == STATUS_SKIPPED:
                skipped += 1
                reason = stage_payload.get("reason") or "skipped"
                ET.SubElement(testcase, "skipped", message=str(reason))
            tests += 1

        smoke_payload = case.get("smoke") if isinstance(case.get("smoke"), dict) else {}
        for smoke_case in smoke_payload.get("cases", []) if isinstance(smoke_payload.get("cases"), list) else []:
            smoke_id = str(smoke_case.get("id", "unknown"))
            status = str(smoke_case.get("status", STATUS_SKIPPED))
            testcase = ET.SubElement(
                suite,
                "testcase",
                classname=case_name,
                name=f"{case_name}.smoke.{smoke_id}",
                time=str(smoke_case.get("duration_sec", 0)),
            )
            if status == STATUS_FAILED:
                failures += 1
                reason = smoke_case.get("failure_reason") or "smoke test failed"
                failure = ET.SubElement(testcase, "failure", message=str(reason))
                failure.text = str(reason)
            elif status == STATUS_SKIPPED:
                skipped += 1
                reason = smoke_case.get("reason") or "skipped"
                ET.SubElement(testcase, "skipped", message=str(reason))
            tests += 1

    suite.set("tests", str(tests))
    suite.set("failures", str(failures))
    suite.set("skipped", str(skipped))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(output, encoding="utf-8", xml_declaration=True)
    print(f"wrote JUnit XML: {output} (tests={suite.get('tests')} failures={failures} skipped={skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
