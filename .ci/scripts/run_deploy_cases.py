#!/usr/bin/env python3
"""Run selected DeployCase files against vLLM Ascend."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
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
    case_level,
    case_name,
    command_to_shell,
    expand_text,
    first_service,
    load_case,
    read_case_list,
    read_json,
    safe_slug,
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
        "container_name": os.getenv("VLLM_CI_CONTAINER_NAME") or os.getenv("HOSTNAME") or None,
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
    result[stage] = stage_result(STATUS_FAILED, failure_reason=reason)


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


def _post_json(url: str, payload: dict[str, Any], expected_status: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8", errors="replace")
            http_status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        http_status = exc.code

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
        response_preview=body[:500],
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


def _run_one_case(case_path: str, args: argparse.Namespace) -> dict[str, Any]:
    case = load_case(case_path)
    allocation = _load_allocation(args.allocation_json)
    apply_allocation(case, allocation)
    result = _result_skeleton(case, allocation)
    service = first_service(case)
    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)
    log_path = _log_path(case, service, logs_dir)
    result["artifacts"]["server_log"] = str(log_path)
    if args.allocation_json:
        result["artifacts"]["allocation_json"] = args.allocation_json
    process: subprocess.Popen[Any] | None = None

    try:
        if len(case.get("services") or []) != 1 or service.get("type") != "vllm-serve":
            result["status"] = STATUS_SKIPPED
            reason = "MVP runner supports exactly one vllm-serve service"
            result["failure_stage"] = "topology"
            result["failure_reason"] = reason
            result["startup"] = stage_result(STATUS_SKIPPED, reason=reason)
            return result

        env = _case_env(case, args.model_root)
        command = build_vllm_serve_command(case, service, {"MODEL_ROOT": args.model_root})
        result["artifacts"]["serve_command"] = command_to_shell(command)

        if args.dry_run:
            result["status"] = STATUS_SKIPPED
            result["startup"] = stage_result(STATUS_SKIPPED, reason="dry-run", command=command_to_shell(command))
            return result

        workdir = expand_text(case.get("runtime", {}).get("workdir", "."), env)
        cwd = workdir if Path(workdir).exists() else "."
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {command_to_shell(command)}\n")
            started = time.monotonic()
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        result["startup"] = stage_result(
            STATUS_PASSED,
            pid=process.pid,
            command=command_to_shell(command),
            duration_sec=round(time.monotonic() - started, 3),
        )
        pid_path = log_path.with_suffix(".pid")
        pid_path.write_text(str(process.pid), encoding="utf-8")
        result["artifacts"]["pid_file"] = str(pid_path)

        host = str(service.get("host", "127.0.0.1"))
        port = int(service.get("port", 8000))
        result["readiness"] = _wait_for_readiness(host, port, case.get("checks", {}).get("readiness") or {})
        if result["readiness"]["status"] != STATUS_PASSED:
            _mark_failed(result, "readiness", result["readiness"].get("failure_reason", "readiness failed"))
            return result

        smoke = case.get("tests", {}).get("smoke") or {}
        if smoke.get("enabled", True):
            endpoint = smoke.get("endpoint", "/v1/completions")
            url = f"http://{host}:{port}{endpoint}"
            expected = int(smoke.get("expected_http_status", 200))
            result["smoke"] = _post_json(url, smoke.get("payload") or {}, expected)
            if result["smoke"]["status"] != STATUS_PASSED:
                _mark_failed(result, "smoke", f"smoke HTTP status {result['smoke'].get('http_status')}")
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
        _terminate_process(process)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / f"{safe_slug(case_name(case))}.json", result)


def main() -> int:
    args = parse_args()
    paths = read_case_list(args.case_list)
    if not paths:
        print(f"case list is empty: {args.case_list}")
        return 1

    failures = 0
    for path in paths:
        print(f"running DeployCase: {path}")
        result = _run_one_case(path, args)
        print(f"{result['case_name']}: {result['status']}")
        if result["status"] == STATUS_FAILED:
            failures += 1
            if not args.continue_on_error:
                break

    if failures:
        print(f"{failures} DeployCase run(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
