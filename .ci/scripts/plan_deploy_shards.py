#!/usr/bin/env python3
"""Plan DeployCase runtime metadata for one Jenkins runtime container."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from deploy_case_lib import (
    case_card_count,
    case_level,
    case_name,
    case_soc,
    docker_config,
    first_service,
    load_case,
    read_case_list,
    safe_slug,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan DeployCase metadata for one Jenkins runtime container.")
    parser.add_argument("--case-list", required=True, help="Path produced by select_deploy_cases.py")
    parser.add_argument("--output", default="reports/runtime_shards.json", help="Runtime plan JSON path")
    parser.add_argument(
        "--shard-dir",
        default="reports/runtime_shards",
        help="Directory for compatibility per-case list files",
    )
    return parser.parse_args()


def _port_count(case: dict[str, Any]) -> int:
    return len(case.get("services") or [])


def main() -> int:
    args = parse_args()
    case_paths = read_case_list(args.case_list)
    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, Any]] = []
    for index, case_path in enumerate(case_paths):
        case = load_case(case_path)
        slug = safe_slug(case_name(case))
        shard_name = f"{index + 1:03d}-{slug}"
        shard_case_list = shard_dir / f"{shard_name}.txt"
        shard_case_list.write_text(f"{case_path}\n", encoding="utf-8")
        service = first_service(case)
        docker = docker_config(case)
        shards.append(
            {
                "name": shard_name,
                "case_name": case_name(case),
                "case_path": case_path,
                "case_list": str(shard_case_list),
                "level": case_level(case),
                "soc": case_soc(case),
                "card_count": case_card_count(case),
                "port_count": _port_count(case),
                "service_name": service.get("name"),
                "docker": {
                    "image": docker.get("image"),
                    "network": docker.get("network", "host"),
                    "ipc": docker.get("ipc", "host"),
                    "shm_size": docker.get("shm_size", "64g"),
                    "mounts": docker.get("mounts") or [],
                },
            }
        )

    payload = {
        "status": "passed",
        "total": len(shards),
        "runtime_model": "single-container",
        "container_card_count": 0,
        "total_card_demand": sum(int(shard["card_count"]) for shard in shards),
        "max_case_card_count": max((int(shard["card_count"]) for shard in shards), default=0),
        "total_port_count": sum(int(shard["port_count"]) for shard in shards),
        "shard_dir": str(shard_dir),
        "shards": shards,
    }
    write_json(args.output, payload)
    print(
        f"planned {len(shards)} case(s) for one runtime container; "
        f"ports={payload['total_port_count']}; output={args.output}"
    )
    if not shards:
        print("No runtime shards planned because the selected case list is empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
