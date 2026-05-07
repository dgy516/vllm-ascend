#!/usr/bin/env python3
"""Allocate Ascend cards and host ports for one Jenkins runtime shard."""

from __future__ import annotations

import argparse
import fcntl
import glob
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from deploy_case_lib import write_json

DEFAULT_PORT_START = 20000
DEFAULT_PORT_END = 26000
DEFAULT_TIMEOUT_SEC = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Allocate host-local Ascend card ids and TCP ports with file locks, "
            "then optionally run a child command while holding the locks."
        )
    )
    parser.add_argument("--card-count", type=int, required=True, help="Number of Ascend cards to allocate")
    parser.add_argument("--port-count", type=int, required=True, help="Number of host ports to allocate")
    parser.add_argument("--output", required=True, help="Allocation JSON path")
    parser.add_argument("--npu-lock-dir", default="/tmp/vllm-ascend-ci/npu", help="Directory for card lock files")
    parser.add_argument("--port-lock-dir", default="/tmp/vllm-ascend-ci/ports", help="Directory for port lock files")
    parser.add_argument("--total-cards", type=int, default=0, help="Total cards on this host; auto-detect when omitted")
    parser.add_argument("--port-start", type=int, default=DEFAULT_PORT_START, help="First candidate port")
    parser.add_argument("--port-end", type=int, default=DEFAULT_PORT_END, help="Last candidate port")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC, help="Allocation timeout")
    parser.add_argument("--poll-sec", type=float, default=2.0, help="Seconds between allocation retries")
    parser.add_argument("--dry-run", action="store_true", help="Allocate and report, but do not run the child command")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Optional command after --")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def _run_quiet(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout


def _detect_total_cards() -> int:
    for key in ("VLLM_CI_TOTAL_CARDS", "ASCEND_TOTAL_CARDS"):
        value = os.getenv(key)
        if value:
            try:
                return int(value)
            except ValueError:
                pass

    output = _run_quiet(["npu-smi", "info", "-l"])
    match = re.search(r"Total\s+Count\s*:\s*(\d+)", output, re.IGNORECASE)
    if match:
        return int(match.group(1))

    device_ids: set[int] = set()
    for path in glob.glob("/dev/davinci[0-9]*"):
        match = re.search(r"davinci(\d+)$", path)
        if match:
            device_ids.add(int(match.group(1)))
    return len(device_ids)


def _lock_file(path: Path) -> TextIO | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} host={socket.gethostname()} time={datetime.now(timezone.utc).isoformat()}\n")
    handle.flush()
    return handle


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _release(handles: list[TextIO]) -> None:
    while handles:
        handle = handles.pop()
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _try_allocate(
    items: list[int],
    count: int,
    lock_dir: Path,
    prefix: str,
    check_port_available: bool = True,
) -> tuple[list[int], list[TextIO]]:
    allocated: list[int] = []
    handles: list[TextIO] = []
    for item in items:
        handle = _lock_file(lock_dir / f"{prefix}-{item}.lock")
        if handle is None:
            continue
        if check_port_available and prefix == "port" and not _port_is_available(item):
            _release([handle])
            continue
        allocated.append(item)
        handles.append(handle)
        if len(allocated) == count:
            return allocated, handles
    _release(handles)
    return [], []


def _allocate(args: argparse.Namespace) -> tuple[list[int], list[int], list[TextIO]]:
    if args.card_count < 1:
        raise ValueError("--card-count must be >= 1")
    if args.port_count < 1:
        raise ValueError("--port-count must be >= 1")
    if args.port_start > args.port_end:
        raise ValueError("--port-start must be <= --port-end")

    total_cards = args.total_cards
    if total_cards <= 0 and args.dry_run:
        total_cards = max(args.card_count, 8)
    if total_cards <= 0:
        total_cards = _detect_total_cards()
    if total_cards <= 0:
        raise RuntimeError(
            "could not detect Ascend card count; pass --total-cards or set VLLM_CI_TOTAL_CARDS"
        )
    if args.card_count > total_cards:
        raise RuntimeError(f"requested {args.card_count} card(s), but host exposes {total_cards}")

    card_items = list(range(total_cards))
    port_items = list(range(args.port_start, args.port_end + 1))
    deadline = time.monotonic() + args.timeout_sec
    all_handles: list[TextIO] = []

    while True:
        cards, card_handles = _try_allocate(card_items, args.card_count, Path(args.npu_lock_dir), "card")
        if cards:
            ports, port_handles = _try_allocate(
                port_items,
                args.port_count,
                Path(args.port_lock_dir),
                "port",
                check_port_available=not args.dry_run,
            )
            if ports:
                all_handles.extend(card_handles)
                all_handles.extend(port_handles)
                return cards, ports, all_handles
            _release(card_handles)

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out allocating {args.card_count} card(s) and {args.port_count} port(s)"
            )
        time.sleep(args.poll_sec)


def _allocation_payload(args: argparse.Namespace, cards: list[int], ports: list[int]) -> dict[str, Any]:
    return {
        "cards": cards,
        "ports": ports,
        "card_count": args.card_count,
        "port_count": args.port_count,
        "host_node": os.getenv("NODE_NAME") or socket.gethostname(),
        "npu_lock_dir": args.npu_lock_dir,
        "port_lock_dir": args.port_lock_dir,
        "dry_run": bool(args.dry_run),
        "command": shlex.join(args.command) if args.command else "",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    handles: list[TextIO] = []
    try:
        cards, ports, handles = _allocate(args)
        payload = _allocation_payload(args, cards, ports)
        write_json(args.output, payload)
        cards_text = ",".join(str(card) for card in cards)
        ports_text = ",".join(str(port) for port in ports)
        print(f"ASCEND_RT_VISIBLE_DEVICES={cards_text}", flush=True)
        print(f"VLLM_CI_ALLOCATED_PORTS={ports_text}", flush=True)
        print(f"VLLM_CI_ALLOCATION_JSON={args.output}", flush=True)

        if args.dry_run or not args.command:
            if args.command:
                print(f"dry-run command: {shlex.join(args.command)}", flush=True)
            return 0

        child_env = dict(os.environ)
        child_env.update(
            {
                "ASCEND_RT_VISIBLE_DEVICES": cards_text,
                "VLLM_CI_ALLOCATED_PORTS": ports_text,
                "VLLM_CI_ALLOCATION_JSON": str(args.output),
            }
        )
        return subprocess.run(args.command, env=child_env, check=False).returncode
    finally:
        _release(handles)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - convert allocator failures to clear Jenkins logs
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
