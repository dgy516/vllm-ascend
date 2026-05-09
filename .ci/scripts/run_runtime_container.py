#!/usr/bin/env python3
"""Run one Jenkins DeployCase runtime container."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from deploy_case_lib import (
    CONTAINER_WORKSPACE,
    allocation_cards,
    allocation_ports,
    command_to_shell,
    write_json,
)
from with_runtime_allocation import (
    DEFAULT_PORT_END,
    DEFAULT_PORT_START,
    DEFAULT_TIMEOUT_SEC,
    allocate_runtime_resources,
)

FIXED_ASCEND_DEVICES = [
    "/dev/davinci_manager",
    "/dev/devmm_svm",
    "/dev/hisi_hdc",
]
FIXED_ASCEND_MOUNTS = [
    ("/usr/local/dcmi", "/usr/local/dcmi", ""),
    ("/usr/local/bin/npu-smi", "/usr/local/bin/npu-smi", ""),
    ("/usr/local/Ascend/driver/lib64/", "/usr/local/Ascend/driver/lib64/", ""),
    ("/usr/local/Ascend/driver/version.info", "/usr/local/Ascend/driver/version.info", ""),
    ("/etc/ascend_install.info", "/etc/ascend_install.info", ""),
    ("/root/.cache", "/root/.cache", ""),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single vLLM Ascend CI runtime Docker container.")
    parser.add_argument("--case-list", default="reports/selected_cases.txt", help="Selected DeployCase list")
    parser.add_argument(
        "--allocation-json",
        default="reports/runtime_container_allocation.json",
        help="Allocation JSON output path",
    )
    parser.add_argument("--card-count", type=int, default=0, help="Host card count to allocate; 0 means all cards")
    parser.add_argument("--port-count", type=int, default=1, help="Host port count to allocate")
    parser.add_argument("--npu-lock-dir", default="/tmp/vllm-ascend-ci/npu", help="Directory for card lock files")
    parser.add_argument("--port-lock-dir", default="/tmp/vllm-ascend-ci/ports", help="Directory for port lock files")
    parser.add_argument("--total-cards", type=int, default=0, help="Total host cards; auto-detect when omitted")
    parser.add_argument("--port-start", type=int, default=DEFAULT_PORT_START, help="First candidate host port")
    parser.add_argument("--port-end", type=int, default=DEFAULT_PORT_END, help="Last candidate host port")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC, help="Allocation timeout")
    parser.add_argument("--poll-sec", type=float, default=2.0, help="Seconds between allocation retries")
    parser.add_argument("--docker-image", required=True, help="Runtime Docker image")
    parser.add_argument("--container-name", required=True, help="Runtime container name")
    parser.add_argument("--workspace", default=os.getenv("WORKSPACE", os.getcwd()), help="Host Jenkins workspace")
    parser.add_argument("--container-workspace", default=CONTAINER_WORKSPACE, help="Container work directory")
    parser.add_argument("--model-root", default="", help="Optional host model root")
    parser.add_argument("--output-dir", default="reports/nightly/case_results", help="Container result directory")
    parser.add_argument("--logs-dir", default="logs/deploy", help="Container log directory")
    parser.add_argument("--parallelism", default="0", help="Concurrent cases inside the runtime container")
    parser.add_argument("--extra-docker-args", default="", help="Optional extra site-specific docker run arguments")
    parser.add_argument("--run-benchmark", action="store_true", help="Pass --run-benchmark to run_deploy_cases.py")
    parser.add_argument("--run-accuracy", action="store_true", help="Pass --run-accuracy to run_deploy_cases.py")
    parser.add_argument("--continue-on-error", action="store_true", help="Pass --continue-on-error to runner")
    parser.add_argument("--dry-run", action="store_true", help="Print the docker command without running it")
    parser.add_argument("--print-command", action="store_true", help="Print the docker command before execution")
    parser.add_argument(
        "--command-output",
        default="reports/runtime_docker_command.sh",
        help="Path to write the rendered docker command",
    )
    return parser.parse_args()


def _mount_arg(source: str, target: str, mode: str = "") -> list[str]:
    suffix = f":{mode}" if mode else ""
    return ["-v", f"{source}:{target}{suffix}"]


def _runner_command(args: argparse.Namespace) -> list[str]:
    command = [
        "python3",
        ".ci/scripts/run_deploy_cases.py",
        "--case-list",
        args.case_list,
        "--allocation-json",
        args.allocation_json,
        "--output-dir",
        args.output_dir,
        "--logs-dir",
        args.logs_dir,
        "--model-root",
        args.model_root,
        "--parallelism",
        str(args.parallelism),
    ]
    if args.run_benchmark:
        command.append("--run-benchmark")
    if args.run_accuracy:
        command.append("--run-accuracy")
    if args.continue_on_error:
        command.append("--continue-on-error")
    return command


def build_docker_command(args: argparse.Namespace, allocation: dict) -> list[str]:
    workspace = Path(args.workspace) if "$" in args.workspace else Path(args.workspace).resolve()
    cards = allocation_cards(allocation)
    ports = allocation_ports(allocation)
    if not cards:
        raise ValueError(f"allocation JSON does not contain any cards: {args.allocation_json}")
    if not ports:
        raise ValueError(f"allocation JSON does not contain any ports: {args.allocation_json}")

    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        args.container_name,
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size=1g",
    ]
    for card in cards:
        command.extend(["--device", f"/dev/davinci{card}"])
    for device in FIXED_ASCEND_DEVICES:
        command.extend(["--device", device])
    if args.extra_docker_args.strip():
        command.extend(shlex.split(args.extra_docker_args))
    for source, target, mode in FIXED_ASCEND_MOUNTS:
        command.extend(_mount_arg(source, target, mode))

    command.extend(_mount_arg(str(workspace / ".ci"), f"{args.container_workspace}/.ci", "ro"))
    command.extend(_mount_arg(str(workspace / "reports"), f"{args.container_workspace}/reports", "rw"))
    command.extend(_mount_arg(str(workspace / "logs"), f"{args.container_workspace}/logs", "rw"))
    if args.model_root:
        command.extend(["-e", f"MODEL_ROOT={args.model_root}"])
        command.extend(_mount_arg(args.model_root, args.model_root, "ro"))

    command.extend(
        [
            "-e",
            f"ASCEND_RT_VISIBLE_DEVICES={','.join(str(card) for card in cards)}",
            "-e",
            f"VLLM_CI_ALLOCATED_PORTS={','.join(str(port) for port in ports)}",
            "-e",
            f"VLLM_CI_ALLOCATION_JSON={args.allocation_json}",
            "-e",
            f"VLLM_CI_CONTAINER_NAME={args.container_name}",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-w",
            args.container_workspace,
            args.docker_image,
            "bash",
            "-lc",
            f"cd {shlex.quote(args.container_workspace)} && {command_to_shell(_runner_command(args))}",
        ]
    )
    return command


def docker_command_example() -> str:
    example_args = argparse.Namespace(
        allocation_json="reports/runtime_container_allocation.json",
        case_list="reports/selected_cases.txt",
        command_output="reports/runtime_docker_command.sh",
        container_name="vllm-ascend-ci-${BUILD_TAG}",
        container_workspace=CONTAINER_WORKSPACE,
        continue_on_error=True,
        docker_image="${ASCEND_DOCKER_IMAGE}",
        dry_run=True,
        extra_docker_args="",
        logs_dir="logs/deploy",
        model_root="${MODEL_ROOT}",
        output_dir="reports/nightly/case_results",
        parallelism="${RUNTIME_PARALLELISM}",
        print_command=True,
        run_accuracy=False,
        run_benchmark=False,
        workspace="${WORKSPACE}",
    )
    allocation = {"cards": [0], "ports": [20000]}
    rendered = command_to_shell(build_docker_command(example_args, allocation))
    return (
        rendered.replace("/dev/davinci0", "/dev/davinci${ASCEND_CARD_ID}")
        .replace("ASCEND_RT_VISIBLE_DEVICES=0", "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_CARD_ID}")
        .replace("VLLM_CI_ALLOCATED_PORTS=20000", "VLLM_CI_ALLOCATED_PORTS=${VLLM_PORT}")
    )


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace)
    (workspace / "reports/nightly/case_results").mkdir(parents=True, exist_ok=True)
    (workspace / "logs/deploy").mkdir(parents=True, exist_ok=True)

    allocation_args = argparse.Namespace(
        card_count=args.card_count,
        port_count=args.port_count,
        output=args.allocation_json,
        npu_lock_dir=args.npu_lock_dir,
        port_lock_dir=args.port_lock_dir,
        total_cards=args.total_cards,
        port_start=args.port_start,
        port_end=args.port_end,
        timeout_sec=args.timeout_sec,
        poll_sec=args.poll_sec,
        dry_run=args.dry_run,
        command=[],
    )
    with allocate_runtime_resources(allocation_args) as allocation:
        allocation["container_name"] = args.container_name
        command = build_docker_command(args, allocation)
        command_text = command_to_shell(command)
        allocation["command"] = command_text
        write_json(args.allocation_json, allocation)

        output = Path(args.command_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{command_text}\n", encoding="utf-8")

        print(f"ASCEND_RT_VISIBLE_DEVICES={','.join(str(card) for card in allocation_cards(allocation))}", flush=True)
        print(f"VLLM_CI_ALLOCATED_PORTS={','.join(str(port) for port in allocation_ports(allocation))}", flush=True)
        print(f"VLLM_CI_ALLOCATION_JSON={args.allocation_json}", flush=True)
        if args.print_command or args.dry_run:
            print(command_text, flush=True)
        if args.dry_run:
            return 0

        completed = subprocess.run(command, check=False)
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
