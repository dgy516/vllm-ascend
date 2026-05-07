#!/usr/bin/env python3
"""Select DeployCase YAML files by metadata.level."""

from __future__ import annotations

import argparse
from pathlib import Path

from deploy_case_lib import ALLOWED_LEVELS, case_level, case_name, expand_case_paths, load_case

AUTO_LEVEL_BY_CI_MODE = {
    "pr": "static",
    "static": "static",
    "smoke": "smoke",
    "nightly": "nightly",
    "release": "release",
    "benchmark": "benchmark",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select DeployCase files for a Jenkins run.")
    parser.add_argument("--cases", nargs="+", required=True, help="DeployCase YAML glob(s)")
    parser.add_argument("--level", default="auto", help="Case level or auto")
    parser.add_argument(
        "--ci-mode",
        default="static",
        choices=sorted(AUTO_LEVEL_BY_CI_MODE),
        help="CI mode for auto level",
    )
    parser.add_argument("--run-all", action="store_true", help="Select every matched case")
    parser.add_argument("--output", default="reports/selected_cases.txt", help="Selected case list output")
    return parser.parse_args()


def resolve_level(level: str, ci_mode: str) -> str:
    if level == "auto":
        return AUTO_LEVEL_BY_CI_MODE[ci_mode]
    if level not in ALLOWED_LEVELS:
        raise ValueError(f"--level must be auto or one of {sorted(ALLOWED_LEVELS)}")
    return level


def main() -> int:
    args = parse_args()
    selected_level = resolve_level(args.level, args.ci_mode)
    paths = [path for path in expand_case_paths(args.cases) if Path(path).exists()]

    selected: list[str] = []
    skipped: list[dict[str, str]] = []
    for path in paths:
        case = load_case(path)
        level = case_level(case)
        if args.run_all or level == selected_level:
            selected.append(path)
        else:
            skipped.append({"path": path, "case_name": case_name(case), "level": level})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

    print(f"matched={len(paths)} selected={len(selected)} level={selected_level} run_all={args.run_all}")
    print(f"selected case list: {args.output}")
    if not selected:
        print(
            "No DeployCase was selected. Check --level/--ci-mode, use --run-all, "
            f"or add a case with metadata.level={selected_level}."
        )
        if skipped:
            print("Available levels:")
            for item in skipped:
                print(f"  - {item['path']}: {item['level']}")
        return 1
    for path in selected:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
