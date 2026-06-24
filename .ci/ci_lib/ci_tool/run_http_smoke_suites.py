#!/usr/bin/env python3
"""Run DeployCase CSV smoke suites against services deployed by Ansible."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

from ci_tool.deploy_case_lib import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    case_level,
    case_name,
    first_service,
    load_case,
    load_smoke_test_cases,
    read_case_list,
    read_json,
    safe_slug,
    served_model_name,
    service_host,
    stage_result,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CSV smoke suites against an existing Ansible DeployCase runtime."
    )
    parser.add_argument("--case-list", required=True, help="Selected DeployCase list")
    parser.add_argument(
        "--deployment-plan",
        default="reports/runtime_plan/deployment_plan.json",
        help="Compiled deployment plan JSON",
    )
    parser.add_argument("--output-dir", default="reports/nightly/case_results", help="Per-case result directory")
    parser.add_argument("--model-root", default="", help="Model root used for CSV template variables")
    parser.add_argument("--readiness-timeout-sec", type=int, default=1800)
    parser.add_argument("--readiness-interval-sec", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Render selected smoke suite without HTTP requests")
    parser.add_argument("--continue-on-error", action="store_true", help="Run all cases before returning failure")
    return parser.parse_args()


def _result_skeleton(
    case: dict[str, Any],
    instances: list[dict[str, Any]],
    *,
    deployment_plan: str,
) -> dict[str, Any]:
    cards = sorted({card for instance in instances for card in instance.get("cards", [])})
    ports = sorted(
        {
            int(port)
            for instance in instances
            for port in [instance.get("port"), *(instance.get("extra_ports") or [])]
            if port is not None
        }
    )
    nodes = sorted({str(instance.get("node")) for instance in instances if instance.get("node")})
    return {
        "case_name": case_name(case),
        "level": case_level(case),
        "status": STATUS_PASSED,
        "failure_stage": None,
        "failure_reason": None,
        "allocated_cards": cards,
        "allocated_ports": ports,
        "container_name": ",".join(sorted({f"vllm-ascend-ci-{safe_slug(node)}" for node in nodes})) or None,
        "host_node": ",".join(nodes) or None,
        "startup": stage_result(STATUS_PASSED, reason="deployed by Ansible"),
        "readiness": stage_result(STATUS_SKIPPED),
        "smoke": stage_result(STATUS_SKIPPED),
        "benchmark": stage_result(STATUS_SKIPPED),
        "accuracy": stage_result(STATUS_SKIPPED),
        "artifacts": {
            "deployment_plan": deployment_plan,
            "server_logs": {
                str(instance.get("name")): str(instance.get("log_file"))
                for instance in instances
                if instance.get("log_file")
            },
        },
    }


def _mark_failed(result: dict[str, Any], stage: str, reason: str) -> None:
    result["status"] = STATUS_FAILED
    result["failure_stage"] = stage
    result["failure_reason"] = reason
    stage_payload = result.get(stage) if isinstance(result.get(stage), dict) else {}
    stage_payload["status"] = STATUS_FAILED
    stage_payload["failure_reason"] = reason
    result[stage] = stage_payload


def _case_instances(plan: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        instance
        for instance in plan.get("instances", [])
        if isinstance(instance, dict) and str(instance.get("case_name")) == name
    ]


def _endpoint_instance(instances: list[dict[str, Any]]) -> dict[str, Any] | None:
    heads = [item for item in instances if item.get("head", True) and not item.get("headless")]
    if not heads:
        return None
    role_priority = {"proxy": 0, "server": 1, "decode": 2, "prefill": 3}
    return sorted(heads, key=lambda item: role_priority.get(str(item.get("role")), 100))[0]


def _profile_served_model_name(case: dict[str, Any]) -> str:
    profile = case.get("profile") if isinstance(case.get("profile"), dict) else {}
    model = profile.get("model") if isinstance(profile.get("model"), dict) else {}
    if model.get("served_model_name"):
        return str(model["served_model_name"])
    return served_model_name(first_service(case))


def _endpoint_service(case: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(instance.get("name") or "runtime-endpoint"),
        "role": str(instance.get("role") or "server"),
        "type": "vllm-serve",
        "host": str(instance.get("node_ip") or "127.0.0.1"),
        "port": int(instance.get("port")),
        "served_model_name": _profile_served_model_name(case),
    }


def _wait_for_readiness(
    *,
    host: str,
    port: int,
    timeout_sec: int,
    interval_sec: int,
) -> dict[str, Any]:
    url = f"http://{host}:{port}/health"
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
        except Exception as exc:  # noqa: BLE001 - readiness probes are retried until timeout
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
            message = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
            text = choices[0].get("text") or message.get("content", "")
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


def _json_path_value(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _is_non_empty_json_value(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


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
        json_path_expr = str(test_case.get("response_json_non_empty_path") or "")
        json_paths = [item.strip() for item in json_path_expr.split("|") if item.strip()]
        if result["status"] == STATUS_PASSED and json_paths:
            try:
                parsed_body = json.loads(body)
            except json.JSONDecodeError:
                result["status"] = STATUS_FAILED
                result["failure_reason"] = "response is not valid JSON for response_json_non_empty_path assertion"
            else:
                matched_path = next(
                    (
                        path
                        for path in json_paths
                        if _is_non_empty_json_value(_json_path_value(parsed_body, path))
                    ),
                    None,
                )
                if matched_path is None:
                    result["status"] = STATUS_FAILED
                    result["failure_reason"] = (
                        "response JSON has no non-empty value at any expected path: "
                        f"{', '.join(json_paths)}"
                    )
                else:
                    result["response_json_non_empty_path"] = matched_path
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
        url = f"http://{service_host(case, service)}:{service.get('port', 8000)}{test_case['endpoint']}"
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


def _run_one_case(case_path: str, plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    case = load_case(case_path)
    instances = _case_instances(plan, case_name(case))
    result = _result_skeleton(case, instances, deployment_plan=args.deployment_plan)
    if not instances:
        _mark_failed(result, "startup", "deployment plan contains no instances for this case")
        return result

    endpoint = _endpoint_instance(instances)
    if endpoint is None:
        _mark_failed(result, "startup", "deployment plan contains no head endpoint for this case")
        return result

    service = _endpoint_service(case, endpoint)
    result["artifacts"]["endpoint"] = f"http://{service['host']}:{service['port']}"

    if args.dry_run:
        result["status"] = STATUS_SKIPPED
        result["startup"] = stage_result(STATUS_SKIPPED, reason="dry-run")
        result["readiness"] = stage_result(STATUS_SKIPPED, reason="dry-run", endpoint=result["artifacts"]["endpoint"])
        result["smoke"] = _run_smoke_suite(case, service, args.model_root, dry_run=True)
        return result

    result["readiness"] = _wait_for_readiness(
        host=str(service["host"]),
        port=int(service["port"]),
        timeout_sec=args.readiness_timeout_sec,
        interval_sec=args.readiness_interval_sec,
    )
    if result["readiness"].get("status") != STATUS_PASSED:
        _mark_failed(result, "readiness", str(result["readiness"].get("failure_reason") or "readiness failed"))
        return result

    result["smoke"] = _run_smoke_suite(case, service, args.model_root, dry_run=False)
    if result["smoke"].get("status") == STATUS_FAILED:
        _mark_failed(result, "smoke", "one or more smoke suite cases failed")
    elif result["smoke"].get("status") == STATUS_SKIPPED:
        result["status"] = STATUS_SKIPPED
        result["failure_stage"] = "smoke"
        result["failure_reason"] = str(result["smoke"].get("reason") or "smoke suite skipped")
    return result


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.deployment_plan)
    failed = False

    for case_path in read_case_list(args.case_list):
        try:
            result = _run_one_case(case_path, plan, args)
        except Exception as exc:  # noqa: BLE001 - keep Jenkins error readable and write a result if possible
            case = load_case(case_path)
            result = _result_skeleton(case, [], deployment_plan=args.deployment_plan)
            _mark_failed(result, "smoke", str(exc))
        write_json(output_dir / f"{safe_slug(result['case_name'])}.json", result)
        if result.get("status") == STATUS_FAILED:
            failed = True
            if not args.continue_on_error:
                break

    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
