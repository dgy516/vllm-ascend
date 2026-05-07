#!/usr/bin/env python3
"""Collect environment details for DeployCase nightly reports."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from deploy_case_lib import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Jenkins and runtime environment information.")
    parser.add_argument("--output", default="reports/nightly/environment.json", help="Environment JSON output")
    return parser.parse_args()


def _run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return ""


def main() -> int:
    args = parse_args()
    keys = [
        "BUILD_URL",
        "JOB_NAME",
        "BUILD_NUMBER",
        "CI_MODE",
        "CASE_LEVEL",
        "RUN_ASCEND",
        "RUN_BENCHMARK",
        "MODEL_ROOT",
        "ASCEND_LABEL",
        "ASCEND_LOCK_LABEL",
        "IMAGE_TAG",
    ]
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "vllm_version": _version("vllm"),
        "vllm_ascend_version": _version("vllm_ascend"),
        "jenkins": {key: os.getenv(key, "") for key in keys},
        "npu_smi": _run(["npu-smi", "info"]),
    }
    write_json(args.output, payload)
    print(f"wrote environment JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
