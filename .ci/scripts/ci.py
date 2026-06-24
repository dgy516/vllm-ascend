#!/usr/bin/env python3
"""Unified CLI entry point for vLLM Ascend DeployCase CI.

Keep Jenkins and local usage stable through one command surface. The
implementation is intentionally split into internal modules so failures remain
direct and each domain can be maintained independently.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path


CI_ROOT = Path(__file__).resolve().parents[1]
CI_LIB = CI_ROOT / "ci_lib"
if str(CI_LIB) not in sys.path:
    sys.path.insert(0, str(CI_LIB))


COMMANDS = {
    "validate": ("ci_tool.validate_deploy_case", "Validate DeployCase YAML files"),
    "select": ("ci_tool.select_deploy_cases", "Select DeployCase files"),
    "static-validate": ("ci_tool.static_validate_cases", "Statically validate selected cases"),
    "render-docs": ("ci_tool.render_deploy_docs", "Render generated deployment docs"),
    "lock-inventory": (
        "ci_tool.jenkins_lock_env_to_ansible_inventory",
        "Convert Jenkins Lockable Resources env into runtime inventory",
    ),
    "compile-plan": ("ci_tool.compile_deployment_plan", "Compile DeployCase runtime plan"),
    "smoke": ("ci_tool.run_http_smoke_suites", "Run HTTP smoke suites against runtime plan"),
    "benchmark-tasks": ("ci_tool.run_benchmark_tasks", "Run compiled benchmark tasks inside runtime container"),
    "merge-benchmark": ("ci_tool.merge_runtime_benchmark_results", "Merge benchmark results into case results"),
    "compare-benchmark": ("ci_tool.compare_benchmark", "Compare benchmark output with a baseline"),
    "prepare-logs": ("ci_tool.prepare_log_artifacts", "Prepare bounded log artifacts for Jenkins"),
    "collect-env": ("ci_tool.collect_env", "Collect environment JSON for reports"),
    "junit": ("ci_tool.generate_junit_report", "Generate Jenkins JUnit XML"),
    "report": ("ci_tool.generate_nightly_report", "Generate HTML/CSV/JSON report bundle"),
}


def _print_help() -> None:
    print("Usage: python3 .ci/scripts/ci.py <command> [command args]")
    print()
    print("Commands:")
    width = max(len(name) for name in COMMANDS)
    for name, (_, description) in sorted(COMMANDS.items()):
        print(f"  {name:<{width}}  {description}")
    print()
    print("Run 'python3 .ci/scripts/ci.py <command> --help' for command-specific help.")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    command = args[0]
    if command not in COMMANDS:
        _print_help()
        print(f"\nERROR: unknown command: {command}", file=sys.stderr)
        return 2

    module_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    module_main = getattr(module, "main", None)
    if module_main is None:
        print(f"ERROR: internal command has no main(): {module_name}", file=sys.stderr)
        return 2

    sys.argv = [f"ci.py {command}", *args[1:]]
    return int(module_main())


if __name__ == "__main__":
    raise SystemExit(main())
