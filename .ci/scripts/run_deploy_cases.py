#!/usr/bin/env python3
"""Run selected DeployCase files against vLLM Ascend."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

from deploy_case_lib import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    allocation_cards,
    allocation_ports,
    apply_allocation,
    build_vllm_serve_command,
    case_card_count,
    case_level,
    case_name,
    command_to_shell,
    expand_text,
    first_service,
    load_case,
    load_smoke_test_cases,
    read_case_list,
    read_json,
    safe_slug,
    service_card_count,
    set_cli_flag_value,
    stage_result,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected vLLM Ascend DeployCase files.")
    parser.add_argument("--case-list", required=True, help="Path produced by select_deploy_cases.py")
    parser.add_argument("--output-dir", default="reports/nightly/case_results", help="Per-case JSON output directory")
    parser.add_argument("--logs-dir", default="logs/deploy", help="Runtime log directory")
    parser.add_argument("--model-root", default="", help="Optional local model root for env expansion")
    parser.add_argument(
        "--allocation-json",
        default="",
        help="Optional allocation JSON from with_runtime_allocation.py",
    )
    parser.add_argument("--dry-run", action="store_true", help="Render commands and results without launching vLLM")
    parser.add_argument("--run-benchmark", action="store_true", help="Run tests.benchmark when enabled")
    parser.add_argument("--run-accuracy", action="store_true", help="Run tests.accuracy when enabled")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after case failures")
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Concurrent cases inside one runtime container; 0 means one worker per selected case",
    )
    parser.add_argument(
        "--allocation-wait-sec",
        type=int,
        default=3600,
        help="Seconds a case waits for cards/ports inside the runtime container",
    )
    return parser.parse_args()


def _load_allocation(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    allocation_path = Path(path)
    if not allocation_path.exists():
        raise FileNotFoundError(f"allocation JSON does not exist: {allocation_path}")
    payload = read_json(allocation_path)
    if not isinstance(payload, dict):
        raise ValueError(f"allocation JSON must contain an object: {allocation_path}")
    return payload


def _env_int_list(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        result.append(int(stripped))
    return result


def _allocation_from_env() -> dict[str, Any] | None:
    cards = _env_int_list(os.getenv("ASCEND_RT_VISIBLE_DEVICES", ""))
    ports = _env_int_list(os.getenv("VLLM_CI_ALLOCATED_PORTS", ""))
    if not cards and not ports:
        return None
    return {
        "cards": cards,
        "ports": ports,
        "host_node": os.getenv("NODE_NAME"),
        "source": "environment",
    }


class RuntimeAllocator:
    """In-container case-level allocator backed by one host-level allocation."""

    def __init__(self, allocation: dict[str, Any], wait_sec: int) -> None:
        self._available_cards = sorted(allocation_cards(allocation))
        self._available_ports = sorted(allocation_ports(allocation))
        self._total_cards = len(self._available_cards)
        self._total_ports = len(self._available_ports)
        self._host_node = allocation.get("host_node")
        self._container_name = allocation.get("container_name")
        self._wait_sec = wait_sec
        self._condition = threading.Condition()

    def allocate(self, card_count: int, port_count: int, label: str) -> dict[str, Any] | None:
        if card_count > self._total_cards or port_count > self._total_ports:
            return None
        deadline = time.monotonic() + self._wait_sec
        with self._condition:
            while len(self._available_cards) < card_count or len(self._available_ports) < port_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=min(remaining, 5))

            cards = self._available_cards[:card_count]
            ports = self._available_ports[:port_count]
            del self._available_cards[:card_count]
            del self._available_ports[:port_count]
            return {
                "cards": cards,
                "ports": ports,
                "card_count": card_count,
                "port_count": port_count,
                "label": label,
                "host_node": self._host_node,
                "container_name": self._container_name,
            }

    def release(self, allocation: dict[str, Any] | None) -> None:
        if not allocation:
            return
        with self._condition:
            self._available_cards.extend(allocation_cards(allocation))
            self._available_ports.extend(allocation_ports(allocation))
            self._available_cards = sorted(set(self._available_cards))
            self._available_ports = sorted(set(self._available_ports))
            self._condition.notify_all()


def _result_skeleton(case: dict[str, Any], allocation: dict[str, Any] | None = None) -> dict[str, Any]:
    cards = allocation_cards(allocation)
    ports = allocation_ports(allocation)
    if not cards and os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        cards = _env_int_list(os.environ["ASCEND_RT_VISIBLE_DEVICES"])
    if not ports and os.getenv("VLLM_CI_ALLOCATED_PORTS"):
        ports = _env_int_list(os.environ["VLLM_CI_ALLOCATED_PORTS"])
    return {
        "case_name": case_name(case),
        "level": case_level(case),
        "status": STATUS_PASSED,
        "failure_stage": None,
        "failure_reason": None,
        "allocated_cards": cards,
        "allocated_ports": ports,
        "container_name": (allocation or {}).get("container_name")
        or os.getenv("VLLM_CI_CONTAINER_NAME")
        or os.getenv("HOSTNAME")
        or None,
        "host_node": (allocation or {}).get("host_node") or os.getenv("NODE_NAME") or None,
        "startup": stage_result(STATUS_SKIPPED),
        "readiness": stage_result(STATUS_SKIPPED),
        "smoke": stage_result(STATUS_SKIPPED),
        "benchmark": stage_result(STATUS_SKIPPED),
        "accuracy": stage_result(STATUS_SKIPPED),
        "artifacts": {},
    }


def _mark_failed(result: dict[str, Any], stage: str, reason: str) -> None:
    result["status"] = STATUS_FAILED
    result["failure_stage"] = stage
    result["failure_reason"] = reason
    stage_payload = result.get(stage) if isinstance(result.get(stage), dict) else {}
    stage_payload["status"] = STATUS_FAILED
    stage_payload["failure_reason"] = reason
    result[stage] = stage_payload


def _case_env(case: dict[str, Any], model_root: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (case.get("runtime", {}).get("env") or {}).items()})
    if model_root:
        env["MODEL_ROOT"] = model_root
    return env


def _log_path(case: dict[str, Any], service: dict[str, Any], logs_dir: Path) -> Path:
    configured = service.get("log_file")
    if configured:
        configured_path = Path(str(configured))
        if configured_path.is_absolute():
            return configured_path
        try:
            return logs_dir / configured_path.relative_to("logs/deploy")
        except ValueError:
            return configured_path
    return logs_dir / safe_slug(case_name(case)) / "server.log"


def _wait_for_readiness(host: str, port: int, readiness: dict[str, Any]) -> dict[str, Any]:
    path = readiness.get("path", "/health")
    timeout_sec = int(readiness.get("timeout_sec", 600))
    interval_sec = int(readiness.get("interval_sec", 5))
    url = f"http://{host}:{port}{path}"
    started = time.monotonic()
    last_error = ""

    while time.monotonic() - started < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=min(interval_sec, 10)) as response:
                if 200 <= response.status < 300:
                    return stage_result(
                        STATUS_PASSED,
                        url=url,
                        http_status=response.status,
                        duration_sec=round(time.monotonic() - started, 3),
                    )
                last_error = f"unexpected HTTP status {response.status}"
        except Exception as exc:  # noqa: BLE001 - keep retrying until timeout
            last_error = str(exc)
        time.sleep(interval_sec)

    return stage_result(
        STATUS_FAILED,
        url=url,
        duration_sec=round(time.monotonic() - started, 3),
        failure_reason=f"readiness timeout after {timeout_sec}s; last_error={last_error}",
    )


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any],
    expected_status: int,
    timeout_sec: int,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
            http_status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        http_status = exc.code
    except urllib.error.URLError as exc:
        return stage_result(
            STATUS_FAILED,
            url=url,
            http_status=None,
            output_tokens=None,
            duration_sec=round(time.monotonic() - started, 3),
            response_preview="",
            response_text="",
            failure_reason=str(exc),
        )

    status = STATUS_PASSED if http_status == expected_status else STATUS_FAILED
    parsed: dict[str, Any] = {}
    with suppress(json.JSONDecodeError):
        parsed = json.loads(body)

    output_tokens = None
    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    if isinstance(usage, dict):
        output_tokens = usage.get("completion_tokens")
    if output_tokens is None and isinstance(parsed, dict):
        choices = parsed.get("choices") or []
        if choices and isinstance(choices[0], dict):
            text = choices[0].get("text") or choices[0].get("message", {}).get("content", "")
            output_tokens = len(str(text).split())

    return stage_result(
        status,
        url=url,
        http_status=http_status,
        output_tokens=output_tokens,
        duration_sec=round(time.monotonic() - started, 3),
        response_preview=body[:1000],
        response_text=body,
    )


def _finalize_smoke_case(test_case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    result = {
        "id": test_case["id"],
        "suite_file": test_case["suite_file"],
        "mode": test_case["mode"],
        "status": response.get("status", STATUS_FAILED),
        "http_status": response.get("http_status"),
        "expected_http_status": test_case.get("expected_http_status"),
        "duration_sec": response.get("duration_sec", 0),
        "output_tokens": response.get("output_tokens"),
        "response_preview": response.get("response_preview", ""),
        "description": test_case.get("description", ""),
    }
    body = str(response.get("response_text", ""))
    if result["status"] == STATUS_PASSED:
        min_output_tokens = test_case.get("min_output_tokens")
        if min_output_tokens is not None:
            output_tokens = response.get("output_tokens")
            if output_tokens is None or int(output_tokens) < int(min_output_tokens):
                result["status"] = STATUS_FAILED
                result["failure_reason"] = (
                    f"output_tokens {output_tokens} is lower than min_output_tokens {min_output_tokens}"
                )
        expected_text = str(test_case.get("response_contains") or "")
        if result["status"] == STATUS_PASSED and expected_text and expected_text not in body:
            result["status"] = STATUS_FAILED
            result["failure_reason"] = f"response does not contain {expected_text!r}"
        forbidden_text = str(test_case.get("response_not_contains") or "")
        if result["status"] == STATUS_PASSED and forbidden_text and forbidden_text in body:
            result["status"] = STATUS_FAILED
            result["failure_reason"] = f"response contains forbidden text {forbidden_text!r}"
    else:
        result["failure_reason"] = response.get("failure_reason") or (
            f"HTTP status {response.get('http_status')} != {test_case.get('expected_http_status')}"
        )
    return result


def _run_smoke_suite(
    case: dict[str, Any],
    service: dict[str, Any],
    model_root: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    tests = load_smoke_test_cases(case, model_root=model_root, service=service)
    suite_files = sorted({str(item.get("suite_file")) for item in tests})
    if not tests:
        return stage_result(STATUS_SKIPPED, reason="no smoke tests selected", total=0, cases=[], suite_files=[])

    started = time.monotonic()
    case_results: list[dict[str, Any]] = []
    for test_case in tests:
        if not test_case.get("enabled", True):
            case_results.append(
                {
                    "id": test_case["id"],
                    "suite_file": test_case["suite_file"],
                    "status": STATUS_SKIPPED,
                    "reason": "disabled",
                    "description": test_case.get("description", ""),
                }
            )
            continue
        url = f"http://{service.get('host', '127.0.0.1')}:{service.get('port', 8000)}{test_case['endpoint']}"
        if dry_run:
            case_results.append(
                {
                    "id": test_case["id"],
                    "suite_file": test_case["suite_file"],
                    "mode": test_case["mode"],
                    "status": STATUS_SKIPPED,
                    "reason": "dry-run",
                    "method": test_case["method"],
                    "url": url,
                    "expected_http_status": test_case["expected_http_status"],
                    "description": test_case.get("description", ""),
                }
            )
            continue
        response = _request_json(
            str(test_case["method"]),
            url,
            test_case["payload"],
            int(test_case["expected_http_status"]),
            int(test_case["timeout_sec"]),
        )
        smoke_case = _finalize_smoke_case(test_case, response)
        smoke_case["method"] = test_case["method"]
        smoke_case["url"] = url
        case_results.append(smoke_case)

    passed = sum(1 for item in case_results if item.get("status") == STATUS_PASSED)
    failed = sum(1 for item in case_results if item.get("status") == STATUS_FAILED)
    skipped = sum(1 for item in case_results if item.get("status") == STATUS_SKIPPED)
    status = STATUS_FAILED if failed else (STATUS_SKIPPED if passed == 0 else STATUS_PASSED)
    return stage_result(
        status,
        total=len(case_results),
        passed=passed,
        failed=failed,
        skipped=skipped,
        suite_files=suite_files,
        cases=case_results,
        duration_sec=round(time.monotonic() - started, 3),
    )


def _run_command(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    cwd: str | None = None,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {command_to_shell(command)}\n")
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    status = STATUS_PASSED if completed.returncode == 0 else STATUS_FAILED
    return stage_result(
        status,
        command=command_to_shell(command),
        returncode=completed.returncode,
        duration_sec=round(time.monotonic() - started, 3),
        log_file=str(log_path),
    )


def _terminate_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except Exception:
        with suppress(Exception):
            os.killpg(process.pid, signal.SIGKILL)


def _terminate_processes(processes: list[subprocess.Popen[Any]]) -> None:
    for process in reversed(processes):
        _terminate_process(process)


def _service_card_allocations(
    case: dict[str, Any],
    services: list[dict[str, Any]],
    cards: list[int],
) -> list[list[int]]:
    if len(services) == 1:
        return [cards]

    allocations: list[list[int]] = []
    offset = 0
    for service in services:
        count = service_card_count(service)
        if count is None:
            raise ValueError(f"{case_name(case)}: service {service.get('name')} requires resources.card_count")
        if count < 1:
            raise ValueError(f"{case_name(case)}: service {service.get('name')} resources.card_count must be >= 1")
        next_offset = offset + count
        if next_offset > len(cards):
            raise ValueError(f"{case_name(case)}: service card counts exceed allocated cards {cards}")
        allocations.append(cards[offset:next_offset])
        offset = next_offset
    return allocations


def _run_one_case(
    case_path: str,
    args: argparse.Namespace,
    allocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case = load_case(case_path)
    apply_allocation(case, allocation)
    result = _result_skeleton(case, allocation)
    services = [service for service in case.get("services", []) if isinstance(service, dict)]
    service = first_service(case)
    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)
    log_path = _log_path(case, service, logs_dir)
    result["artifacts"]["server_log"] = str(log_path)
    result["artifacts"]["server_logs"] = {
        str(item.get("name")): str(_log_path(case, item, logs_dir)) for item in services
    }
    if args.allocation_json:
        result["artifacts"]["allocation_json"] = args.allocation_json
    processes: list[subprocess.Popen[Any]] = []

    try:
        if not services or any(item.get("type") != "vllm-serve" for item in services):
            result["status"] = STATUS_SKIPPED
            reason = "runner supports vllm-serve services only"
            result["failure_stage"] = "topology"
            result["failure_reason"] = reason
            result["startup"] = stage_result(STATUS_SKIPPED, reason=reason)
            return result

        env = _case_env(case, args.model_root)
        service_cards = _service_card_allocations(case, services, allocation_cards(allocation))
        service_commands = [
            build_vllm_serve_command(case, item, {"MODEL_ROOT": args.model_root}) for item in services
        ]
        result["artifacts"]["serve_command"] = command_to_shell(service_commands[0])
        result["artifacts"]["serve_commands"] = {
            str(item.get("name")): command_to_shell(command)
            for item, command in zip(services, service_commands, strict=False)
        }

        if args.dry_run:
            result["status"] = STATUS_SKIPPED
            result["startup"] = stage_result(
                STATUS_SKIPPED,
                reason="dry-run",
                command=command_to_shell(service_commands[0]),
                services=[
                    {
                        "name": item.get("name"),
                        "cards": cards,
                        "command": command_to_shell(command),
                    }
                    for item, cards, command in zip(services, service_cards, service_commands, strict=False)
                ],
            )
            result["smoke"] = _run_smoke_suite(case, service, args.model_root, dry_run=True)
            return result

        workdir = expand_text(case.get("runtime", {}).get("workdir", "."), env)
        cwd = workdir if Path(workdir).exists() else "."
        started = time.monotonic()
        started_services: list[dict[str, Any]] = []
        for item, cards, command in zip(services, service_cards, service_commands, strict=False):
            item_log_path = _log_path(case, item, logs_dir)
            item_log_path.parent.mkdir(parents=True, exist_ok=True)
            item_env = dict(env)
            if cards:
                item_env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(card) for card in cards)
            with item_log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"$ {command_to_shell(command)}\n")
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=item_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            processes.append(process)
            pid_path = item_log_path.with_suffix(".pid")
            pid_path.write_text(str(process.pid), encoding="utf-8")
            started_services.append(
                {
                    "name": item.get("name"),
                    "pid": process.pid,
                    "cards": cards,
                    "command": command_to_shell(command),
                    "log_file": str(item_log_path),
                    "pid_file": str(pid_path),
                }
            )
        result["startup"] = stage_result(
            STATUS_PASSED,
            services=started_services,
            duration_sec=round(time.monotonic() - started, 3),
        )

        readiness_results = []
        for item in services:
            host = str(item.get("host", "127.0.0.1"))
            port = int(item.get("port", 8000))
            item_readiness = _wait_for_readiness(host, port, case.get("checks", {}).get("readiness") or {})
            item_readiness["service"] = item.get("name")
            readiness_results.append(item_readiness)
        readiness_status = (
            STATUS_PASSED
            if all(item.get("status") == STATUS_PASSED for item in readiness_results)
            else STATUS_FAILED
        )
        result["readiness"] = stage_result(readiness_status, services=readiness_results)
        if readiness_status != STATUS_PASSED:
            failed_reason = next(
                (item.get("failure_reason") for item in readiness_results if item.get("status") != STATUS_PASSED),
                "readiness failed",
            )
            _mark_failed(result, "readiness", str(failed_reason))
            return result

        host = str(service.get("host", "127.0.0.1"))
        port = int(service.get("port", 8000))
        smoke = case.get("tests", {}).get("smoke") or {}
        if smoke.get("enabled", True):
            result["smoke"] = _run_smoke_suite(case, service, args.model_root)
            if result["smoke"]["status"] != STATUS_PASSED:
                failed_case = next(
                    (item for item in result["smoke"].get("cases", []) if item.get("status") == STATUS_FAILED),
                    {},
                )
                reason = failed_case.get("failure_reason") or "smoke suite failed"
                _mark_failed(result, "smoke", str(reason))
                return result

        benchmark = case.get("tests", {}).get("benchmark") or {}
        if benchmark.get("enabled") and args.run_benchmark:
            command = [str(item) for item in benchmark.get("command") or []]
            if command:
                set_cli_flag_value(command, "--host", host)
                set_cli_flag_value(command, "--port", port)
                result["benchmark"] = _run_command(
                    command,
                    env,
                    log_path.with_name("benchmark.log"),
                    cwd=cwd,
                    timeout_sec=int(benchmark.get("timeout_sec", 3600)),
                )
                if result["benchmark"]["status"] != STATUS_PASSED:
                    _mark_failed(result, "benchmark", "benchmark command failed")
                    return result
            else:
                result["benchmark"] = stage_result(STATUS_SKIPPED, reason="benchmark command is empty")
        elif benchmark.get("enabled"):
            result["benchmark"] = stage_result(STATUS_SKIPPED, reason="--run-benchmark not set")

        accuracy = case.get("tests", {}).get("accuracy") or {}
        if accuracy.get("enabled") and args.run_accuracy:
            command = [str(item) for item in accuracy.get("command") or []]
            if accuracy.get("mode") != "execute_only":
                result["accuracy"] = stage_result(STATUS_SKIPPED, reason="only execute_only is supported")
            elif command:
                result["accuracy"] = _run_command(
                    command,
                    env,
                    log_path.with_name("accuracy.log"),
                    cwd=cwd,
                    timeout_sec=int(accuracy.get("timeout_sec", 3600)),
                )
                result["accuracy"]["score"] = "N/A"
                if result["accuracy"]["status"] != STATUS_PASSED:
                    _mark_failed(result, "accuracy", "accuracy command failed")
                    return result
            else:
                result["accuracy"] = stage_result(
                    STATUS_SKIPPED,
                    mode="execute_only",
                    score="N/A",
                    reason="empty command",
                )
        elif accuracy.get("enabled"):
            result["accuracy"] = stage_result(STATUS_SKIPPED, mode=accuracy.get("mode", "execute_only"), score="N/A")

        return result
    except Exception as exc:  # noqa: BLE001 - every case must produce a JSON result
        _mark_failed(result, result.get("failure_stage") or "runtime", str(exc))
        return result
    finally:
        _terminate_processes(processes)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / f"{safe_slug(case_name(case))}.json", result)


def _allocation_failure_result(case_path: str, args: argparse.Namespace, reason: str) -> dict[str, Any]:
    case = load_case(case_path)
    result = _result_skeleton(case)
    _mark_failed(result, "allocation", reason)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{safe_slug(case_name(case))}.json", result)
    return result


def _case_port_count(case: dict[str, Any]) -> int:
    return len(case.get("services") or [])


def _run_one_case_with_allocator(
    case_path: str,
    args: argparse.Namespace,
    allocator: RuntimeAllocator | None,
    fallback_allocation: dict[str, Any] | None,
) -> dict[str, Any]:
    if allocator is None:
        return _run_one_case(case_path, args, fallback_allocation)

    case = load_case(case_path)
    card_count = case_card_count(case)
    port_count = _case_port_count(case)
    allocation = allocator.allocate(card_count, port_count, case_name(case))
    if allocation is None:
        return _allocation_failure_result(
            case_path,
            args,
            f"timed out waiting for {card_count} card(s) and {port_count} port(s)",
        )

    try:
        return _run_one_case(case_path, args, allocation)
    finally:
        allocator.release(allocation)


def main() -> int:
    args = parse_args()
    paths = read_case_list(args.case_list)
    if not paths:
        print(f"case list is empty: {args.case_list}")
        return 1

    base_allocation = _load_allocation(args.allocation_json) or _allocation_from_env()
    allocator = RuntimeAllocator(base_allocation, args.allocation_wait_sec) if base_allocation else None
    if args.parallelism < 0:
        print("--parallelism must be >= 0")
        return 1
    workers = len(paths) if args.parallelism == 0 else max(args.parallelism, 1)
    workers = max(1, min(workers, len(paths)))

    failures = 0
    if workers == 1:
        for path in paths:
            print(f"running DeployCase: {path}")
            result = _run_one_case_with_allocator(path, args, allocator, base_allocation)
            print(f"{result['case_name']}: {result['status']}")
            if result["status"] == STATUS_FAILED:
                failures += 1
                if not args.continue_on_error:
                    break
    else:
        print(f"running {len(paths)} DeployCase(s) with in-container parallelism={workers}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_one_case_with_allocator, path, args, allocator, base_allocation): path
                for path in paths
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(f"{result['case_name']}: {result['status']}")
                if result["status"] == STATUS_FAILED:
                    failures += 1

    if failures:
        print(f"{failures} DeployCase run(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
