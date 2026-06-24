#!/usr/bin/env python3
"""Compile DeployCase profile inputs into a node-scoped runtime deployment plan."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import yaml
from ci_tool.deploy_case_lib import (
    CONTAINER_WORKSPACE,
    build_command_service_command,
    build_vllm_serve_command,
    case_card_count,
    case_name,
    command_to_shell,
    expand_text,
    first_service,
    load_case,
    read_case_list,
    read_json,
    safe_slug,
    service_card_count,
    served_model_name,
    set_cli_flag_value,
    write_json,
)

ALLOWED_QUANTIZATION = {"ascend", "none"}
DEFAULT_PORT_START = 8100
DEFAULT_EXTRA_PORT_START = 36000
DEFAULT_BENCHMARK_RESULT_DIR = "reports/nightly/benchmark"
NETWORK_INTERFACE_ENV_KEYS = (
    "NETWORK_CARD_NAME",
    "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "HCCL_SOCKET_IFNAME",
)


class PlanError(RuntimeError):
    """Raised when a deployment plan cannot be compiled safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile DeployCase profiles into an Ansible-consumable runtime plan.")
    parser.add_argument("--case-list", required=True, help="Selected DeployCase list")
    parser.add_argument("--inventory-json", required=True, help="Cluster JSON from Jenkins Lockable Resources")
    parser.add_argument("--output-dir", default="reports/runtime_plan", help="Runtime plan output directory")
    parser.add_argument("--model-root", default="", help="Optional model root for local_path expansion")
    parser.add_argument("--docker-image", required=True, help="Runtime Docker image")
    parser.add_argument("--extra-docker-args", default="", help="Optional extra site-specific docker run arguments")
    parser.add_argument(
        "--host-workspace",
        default=CONTAINER_WORKSPACE,
        help="Remote host workspace mounted to container",
    )
    parser.add_argument("--container-workspace", default=CONTAINER_WORKSPACE)
    parser.add_argument("--port-start", type=int, default=DEFAULT_PORT_START)
    parser.add_argument("--extra-port-start", type=int, default=DEFAULT_EXTRA_PORT_START)
    parser.add_argument("--dry-run", action="store_true", help="Mark generated plan as dry-run")
    return parser.parse_args()


def _positive_int(value: Any, field: str, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise PlanError(f"{field} must be an integer") from None
    if parsed < minimum:
        raise PlanError(f"{field} must be >= {minimum}")
    return parsed


def _profile(case: dict[str, Any]) -> dict[str, Any] | None:
    profile = case.get("profile")
    return profile if isinstance(profile, dict) else None


def _case_served_model_name(case: dict[str, Any]) -> str:
    profile = _profile(case)
    if profile:
        model = profile.get("model") or {}
        if model.get("served_model_name"):
            return str(model["served_model_name"])
    return served_model_name(first_service(case))


def _case_model_path(case: dict[str, Any], model_root: str) -> str:
    profile = _profile(case)
    if profile:
        model = profile.get("model") or {}
        return _model_path(model, model_root)

    service = first_service(case)
    vllm = service.get("vllm") or {}
    env = _runtime_env(case, model_root)
    model = str(vllm.get("model") or "")
    local_model_path = str(vllm.get("local_model_path") or "")
    if model_root and local_model_path:
        model = local_model_path
    return expand_text(model, env) if model else ""


def _model_path(model: dict[str, Any], model_root: str) -> str:
    env = {"MODEL_ROOT": model_root} if model_root else {}
    local_path = model.get("local_path")
    if local_path:
        return expand_text(local_path, env)
    return str(model.get("name") or "")


def _runtime_env(case: dict[str, Any], model_root: str) -> dict[str, str]:
    env = {str(k): str(v) for k, v in (case.get("runtime", {}).get("env") or {}).items()}
    if model_root:
        env["MODEL_ROOT"] = model_root
    return env


def _load_inventory(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(nodes, list) or not nodes:
        raise PlanError(f"inventory JSON must contain non-empty nodes: {path}")
    result: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        name = str(node.get("jenkins_node") or node.get("name") or f"node-{index}")
        ip = str(node.get("ip") or "")
        if not ip:
            raise PlanError(f"inventory node {name} is missing IP")
        ansible_host = str(node.get("ansible_host") or ip)
        cards = _positive_int(node.get("cards"), f"inventory node {name}.cards")
        result.append(
            {
                "name": name,
                "ip": ip,
                "ansible_host": ansible_host,
                "network_interface": str(node.get("network_interface") or "").strip(),
                "cards_total": cards,
                "available_cards": list(range(cards)),
                "containers": [],
                "instances": [],
                "used_ports": set(),
            }
        )
    return result


def _next_port(used: set[int], cursor: int) -> tuple[int, int]:
    port = cursor
    while port in used:
        port += 1
    used.add(port)
    return port, port + 1


def _network_interface_for_layerwise(
    *,
    case_label: str,
    node: dict[str, Any],
    runtime_env: dict[str, str],
) -> str:
    network_interface = str(node.get("network_interface") or "").strip()
    if not network_interface:
        for key in NETWORK_INTERFACE_ENV_KEYS:
            value = str(runtime_env.get(key) or "").strip()
            if value:
                network_interface = value
                break
    if not network_interface:
        raise PlanError(
            f"{case_label}: MooncakeLayerwiseConnector requires a Lockable Resources "
            "NETWORK_INTERFACE/NIC/IFNAME/NETWORK_CARD_NAME property or runtime.env NETWORK_CARD_NAME"
        )
    return network_interface


def _allocate_ranks(nodes: list[dict[str, Any]], *, dp: int, tp: int, label: str) -> list[dict[str, Any]]:
    if not any(len(node["available_cards"]) >= tp for node in nodes):
        raise PlanError(f"{label}: no inventory node can host one TP group of {tp} card(s)")

    remaining_ranks = list(range(dp))
    assignments: list[dict[str, Any]] = []

    # Prefer single-node placement when possible.
    for node in nodes:
        local_capacity = len(node["available_cards"]) // tp
        if local_capacity >= dp:
            cards = [node["available_cards"].pop(0) for _ in range(dp * tp)]
            return [{"node": node, "ranks": remaining_ranks, "cards": cards}]

    for node in nodes:
        local_capacity = len(node["available_cards"]) // tp
        if local_capacity <= 0:
            continue
        ranks = remaining_ranks[:local_capacity]
        if not ranks:
            continue
        cards = [node["available_cards"].pop(0) for _ in range(len(ranks) * tp)]
        assignments.append({"node": node, "ranks": ranks, "cards": cards})
        del remaining_ranks[: len(ranks)]
        if not remaining_ranks:
            return assignments

    raise PlanError(f"{label}: required {dp * tp} card(s), but inventory has insufficient allocatable cards")


def _base_vllm_command(
    *,
    model_path: str,
    served_model_name: str,
    host: str,
    port: int,
    common: dict[str, Any],
    parallel: dict[str, int],
    quantization: str,
    headless: bool,
    data_parallel_start_rank: int,
    data_parallel_size_local: int,
    master_ip: str,
    rpc_port: int,
    extra_args: list[str],
) -> list[str]:
    command = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(parallel["tp"]),
    ]
    if headless:
        command.append("--headless")
    if parallel["dp"] > 1:
        command.extend(
            [
                "--data-parallel-size",
                str(parallel["dp"]),
                "--data-parallel-size-local",
                str(data_parallel_size_local),
                "--data-parallel-start-rank",
                str(data_parallel_start_rank),
                "--data-parallel-address",
                master_ip,
                "--data-parallel-rpc-port",
                str(rpc_port),
            ]
        )
    if parallel.get("ep", 1) > 1:
        command.append("--enable-expert-parallel")
    if quantization == "ascend":
        command.extend(["--quantization", "ascend"])
    if common.get("prefix_caching") is False:
        command.append("--no-enable-prefix-caching")

    simple_fields = {
        "seed": "--seed",
        "max_model_len": "--max-model-len",
        "max_num_batched_tokens": "--max-num-batched-tokens",
        "max_num_seqs": "--max-num-seqs",
        "gpu_memory_utilization": "--gpu-memory-utilization",
    }
    for key, flag in simple_fields.items():
        if common.get(key) is not None:
            command.extend([flag, str(common[key])])
    command.extend(str(item) for item in extra_args)
    return command


def _kv_transfer_config(
    *,
    connector: str,
    role: str,
    kv_port: int,
    engine_id: int,
    service_profiles: dict[str, Any],
) -> str:
    prefill_parallel = service_profiles["prefill"]["parallel"]
    decode_parallel = service_profiles["decode"]["parallel"]
    return json.dumps(
        {
            "kv_connector": connector,
            "kv_role": f"kv_{role}",
            "kv_port": str(kv_port),
            "engine_id": str(engine_id),
            "kv_connector_extra_config": {
                "prefill": {"dp_size": prefill_parallel["dp"], "tp_size": prefill_parallel["tp"]},
                "decode": {"dp_size": decode_parallel["dp"], "tp_size": decode_parallel["tp"]},
            },
        },
        separators=(",", ":"),
    )


def _profile_service_entries(
    case: dict[str, Any],
    nodes: list[dict[str, Any]],
    *,
    model_root: str,
    port_cursor: int,
    extra_port_cursor: int,
) -> tuple[list[dict[str, Any]], int, int]:
    profile = _profile(case)
    if not profile:
        return _legacy_service_entries(case, nodes, model_root=model_root, port_cursor=port_cursor)

    model = profile.get("model") or {}
    quantization = model.get("quantization")
    if quantization not in ALLOWED_QUANTIZATION:
        raise PlanError(f"{case_name(case)}: profile.model.quantization must be one of {sorted(ALLOWED_QUANTIZATION)}")
    local_path = str(model.get("local_path") or "")
    if "${MODEL_ROOT}" in local_path and not model_root:
        raise PlanError(f"{case_name(case)}: profile.model.local_path uses MODEL_ROOT but --model-root is empty")
    model_path = _model_path(model, model_root)
    if not model_path:
        raise PlanError(f"{case_name(case)}: profile.model.name or local_path is required")
    served_model_name = str(model.get("served_model_name") or model.get("name") or "")
    if not served_model_name:
        raise PlanError(f"{case_name(case)}: profile.model.served_model_name is required")

    deployment = profile.get("deployment") or {}
    mode = str(deployment.get("mode", "standalone"))
    if mode not in {"standalone", "pd"}:
        raise PlanError(f"{case_name(case)}: unsupported profile.deployment.mode={mode!r}")

    service_profiles = profile.get("services") or {}
    if not isinstance(service_profiles, dict) or not service_profiles:
        raise PlanError(f"{case_name(case)}: profile.services must be a non-empty mapping")

    common = profile.get("common") or {}
    runtime_env = _runtime_env(case, model_root)
    host = expand_text(common.get("host", "0.0.0.0"), runtime_env)
    used_ports: set[int] = set()
    entries: list[dict[str, Any]] = []
    service_heads: dict[str, list[str]] = {}
    engine_id = 0
    pd_connectors: set[str] = set()

    normalized_services: dict[str, Any] = {}
    for role, raw_service in service_profiles.items():
        if role == "proxy":
            continue
        parallel_raw = raw_service.get("parallel") or {}
        parallel = {
            "dp": _positive_int(parallel_raw.get("dp", 1), f"{role}.parallel.dp"),
            "tp": _positive_int(parallel_raw.get("tp", 1), f"{role}.parallel.tp"),
            "ep": _positive_int(parallel_raw.get("ep", 1), f"{role}.parallel.ep"),
        }
        normalized_services[role] = {**raw_service, "parallel": parallel}

    for role, service in normalized_services.items():
        replicas = _positive_int(service.get("replicas", 1), f"{role}.replicas")
        parallel = service["parallel"]
        for replica_index in range(replicas):
            label = f"{case_name(case)}:{role}-{replica_index}"
            assignments = _allocate_ranks(nodes, dp=parallel["dp"], tp=parallel["tp"], label=label)
            master_ip = assignments[0]["node"]["ip"]
            rpc_port, port_cursor = _next_port(used_ports, port_cursor)
            head_endpoint: str | None = None
            for assignment_index, assignment in enumerate(assignments):
                node = assignment["node"]
                port, port_cursor = _next_port(used_ports, port_cursor)
                extra_ports = []
                extra_port_count = _positive_int(
                    service.get("extra_port_count", 0),
                    f"{role}.extra_port_count",
                    minimum=0,
                )
                for _ in range(extra_port_count):
                    extra_port, extra_port_cursor = _next_port(used_ports, extra_port_cursor)
                    extra_ports.append(extra_port)
                headless = assignment_index > 0
                kv_connector = str(service.get("kv_connector") or "MooncakeConnectorV1")
                uses_layerwise = mode == "pd" and role in {"prefill", "decode"} and (
                    kv_connector == "MooncakeLayerwiseConnector"
                )
                env = {
                    **runtime_env,
                    "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(card) for card in assignment["cards"]),
                    "LOCAL_IP": node["ip"],
                    "MASTER_IP": master_ip,
                    "HOST_IP": host,
                    "VLLM_HOST_IP": node["ip"],
                    "PORT": str(port),
                }
                if uses_layerwise:
                    network_interface = _network_interface_for_layerwise(
                        case_label=label,
                        node=node,
                        runtime_env=runtime_env,
                    )
                    env.update(
                        {
                            "HCCL_IF_IP": node["ip"],
                            "NETWORK_CARD_NAME": network_interface,
                            "GLOO_SOCKET_IFNAME": network_interface,
                            "TP_SOCKET_IFNAME": network_interface,
                            "HCCL_SOCKET_IFNAME": network_interface,
                        }
                    )
                command = _base_vllm_command(
                    model_path=model_path,
                    served_model_name=served_model_name,
                    host=host,
                    port=port,
                    common=common,
                    parallel=parallel,
                    quantization=str(quantization),
                    headless=headless,
                    data_parallel_start_rank=assignment["ranks"][0],
                    data_parallel_size_local=len(assignment["ranks"]),
                    master_ip=master_ip,
                    rpc_port=rpc_port,
                    extra_args=list(service.get("extra_args") or []),
                )
                if mode == "pd" and role in {"prefill", "decode"}:
                    if not extra_ports:
                        raise PlanError(f"{label}: PD service requires at least one extra port for kv_transfer_config")
                    kv_role = str(service.get("kv_role") or ("producer" if role == "prefill" else "consumer"))
                    pd_connectors.add(kv_connector)
                    command.extend(
                        [
                            "--kv-transfer-config",
                            _kv_transfer_config(
                                connector=kv_connector,
                                role=kv_role,
                                kv_port=extra_ports[0],
                                engine_id=engine_id,
                                service_profiles=normalized_services,
                            ),
                        ]
                    )
                name = f"{safe_slug(case_name(case))}-{role}-{replica_index}-rank{assignment['ranks'][0]}"
                entry = {
                    "name": name,
                    "case_name": case_name(case),
                    "role": role,
                    "node": node["name"],
                    "node_ip": node["ip"],
                    "headless": headless,
                    "head": not headless,
                    "cards": assignment["cards"],
                    "port": port,
                    "extra_ports": extra_ports,
                    "model_path": model_path,
                    "env": env,
                    "command": command,
                    "command_shell": command_to_shell(command),
                    "log_file": f"logs/deploy/{safe_slug(case_name(case))}/{name}.log",
                }
                node["instances"].append(entry)
                entries.append(entry)
                if not headless:
                    head_endpoint = f"{node['ip']}:{port}"
                    service_heads.setdefault(role, []).append(head_endpoint)
            if head_endpoint is None:
                raise PlanError(f"{label}: logical service has no head endpoint")
            engine_id += 1

    if mode == "pd":
        prefill_servers = service_heads.get("prefill") or []
        decode_servers = service_heads.get("decode") or []
        if not prefill_servers or not decode_servers:
            raise PlanError(f"{case_name(case)}: PD plan requires non-empty PREFILL_SERVERS and DECODE_SERVERS")
        proxy_service = service_profiles.get("proxy") or {}
        proxy_node = nodes[0]
        proxy_port, port_cursor = _next_port(used_ports, port_cursor)
        proxy_mode = "layerwise" if "MooncakeLayerwiseConnector" in pd_connectors else "v1"
        proxy_env = {
            **runtime_env,
            "HOST_IP": host,
            "VLLM_HOST_IP": proxy_node["ip"],
            "PORT": str(proxy_port),
            "PREFILL_SERVERS": ",".join(prefill_servers),
            "DECODE_SERVERS": ",".join(decode_servers),
            "VLLM_CI_PD_PROXY_MODE": proxy_mode,
            "VLLM_CI_PD_PROXY_ENDPOINT": f"{proxy_node['ip']}:{proxy_port}",
        }
        proxy_command = [
            "python3",
            ".ci/scripts/pd_proxy.py",
            "--host",
            host,
            "--port",
            str(proxy_port),
        ]
        proxy_command.extend(str(item) for item in proxy_service.get("extra_args") or [])
        proxy_name = f"{safe_slug(case_name(case))}-proxy"
        proxy_entry = {
            "name": proxy_name,
            "case_name": case_name(case),
            "role": "proxy",
            "node": proxy_node["name"],
            "node_ip": proxy_node["ip"],
            "headless": False,
            "head": True,
            "cards": [],
            "port": proxy_port,
            "extra_ports": [],
            "model_path": "",
            "env": proxy_env,
            "command": proxy_command,
            "command_shell": command_to_shell(proxy_command),
            "log_file": f"logs/deploy/{safe_slug(case_name(case))}/{proxy_name}.log",
        }
        proxy_node["instances"].append(proxy_entry)
        entries.append(proxy_entry)

    return entries, port_cursor, extra_port_cursor


def _legacy_service_entries(
    case: dict[str, Any],
    nodes: list[dict[str, Any]],
    *,
    model_root: str,
    port_cursor: int,
) -> tuple[list[dict[str, Any]], int, int]:
    runtime_env = _runtime_env(case, model_root)
    entries: list[dict[str, Any]] = []
    used_ports: set[int] = set()
    services = case.get("services") or []
    for service in case.get("services") or []:
        service_type = str(service.get("type") or "")
        configured_card_count = service_card_count(service)
        if configured_card_count is None and service_type == "vllm-serve":
            configured_card_count = case_card_count(case) if len(services) == 1 else None
        card_count = configured_card_count or 0
        if service_type == "vllm-serve" and card_count < 1:
            raise PlanError(f"{case_name(case)}:{service.get('name')}: vLLM service requires at least one card")
        node = next((item for item in nodes if len(item["available_cards"]) >= card_count), None)
        if node is None:
            raise PlanError(f"{case_name(case)}:{service.get('name')}: insufficient cards for v1 service")
        cards = [node["available_cards"].pop(0) for _ in range(card_count)]
        port, port_cursor = _next_port(used_ports, port_cursor)
        service = dict(service)
        service["port"] = port
        env = {**runtime_env, "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(card) for card in cards), "PORT": str(port)}
        if service_type == "vllm-serve":
            command = build_vllm_serve_command(case, service, {"MODEL_ROOT": model_root})
        elif service_type == "command":
            command = build_command_service_command(case, service, {"MODEL_ROOT": model_root})
        else:
            raise PlanError(f"{case_name(case)}:{service.get('name')}: unsupported service type {service.get('type')}")
        name = f"{safe_slug(case_name(case))}-{safe_slug(str(service.get('name') or service.get('role') or 'service'))}"
        entry = {
            "name": name,
            "case_name": case_name(case),
            "role": str(service.get("role") or ""),
            "node": node["name"],
            "node_ip": node["ip"],
            "headless": False,
            "head": True,
            "cards": cards,
            "port": port,
            "extra_ports": [],
            "model_path": str(command[2]) if service_type == "vllm-serve" and len(command) > 2 else "",
            "env": env,
            "command": command,
            "command_shell": command_to_shell(command),
            "log_file": f"logs/deploy/{safe_slug(case_name(case))}/{name}.log",
        }
        node["instances"].append(entry)
        entries.append(entry)
    return entries, port_cursor, DEFAULT_EXTRA_PORT_START


def _container_command(
    node: dict[str, Any],
    *,
    docker_image: str,
    host_workspace: str,
    model_root: str,
    extra_docker_args: str,
) -> list[str]:
    devices = []
    used_cards = sorted({card for instance in node["instances"] for card in instance["cards"]})
    for card in used_cards:
        devices.extend(["--device", f"/dev/davinci{card}"])
    devices.extend(
        [
            "--device",
            "/dev/davinci_manager",
            "--device",
            "/dev/devmm_svm",
            "--device",
            "/dev/hisi_hdc",
        ]
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"vllm-ascend-ci-{safe_slug(node['name'])}",
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size=1g",
        *devices,
        "-v",
        "/usr/local/dcmi:/usr/local/dcmi",
        "-v",
        "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi",
        "-v",
        "/usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/",
        "-v",
        "/usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info",
        "-v",
        "/etc/ascend_install.info:/etc/ascend_install.info",
        "-v",
        f"{host_workspace}/.ci:{CONTAINER_WORKSPACE}/.ci:ro",
        "-v",
        f"{host_workspace}/reports:{CONTAINER_WORKSPACE}/reports:rw",
        "-v",
        f"{host_workspace}/logs:{CONTAINER_WORKSPACE}/logs:rw",
        "-v",
        "/root/.cache:/root/.cache",
        "-w",
        CONTAINER_WORKSPACE,
    ]
    if model_root:
        command.extend(["-v", f"{model_root}:{model_root}:ro"])
    if extra_docker_args.strip():
        command.extend(shlex.split(extra_docker_args))
    command.extend(
        [
            docker_image,
            "bash",
            "-lc",
            f"bash {CONTAINER_WORKSPACE}/reports/runtime_plan/{node['name']}/start_instances.sh",
        ]
    )
    return command


def _endpoint_instance(entries: list[dict[str, Any]], case_label: str) -> dict[str, Any] | None:
    heads = [
        entry
        for entry in entries
        if entry.get("case_name") == case_label and entry.get("head", True) and not entry.get("headless")
    ]
    if not heads:
        return None
    role_priority = {"proxy": 0, "server": 1, "serve": 1, "decode": 2, "prefill": 3}
    return sorted(heads, key=lambda item: role_priority.get(str(item.get("role")), 100))[0]


def _has_cli_flag(command: list[str], flag: str) -> bool:
    return any(item == flag or item.startswith(f"{flag}=") for item in command)


def _benchmark_result_dir(command: list[str], case_label: str) -> str:
    for index, item in enumerate(command):
        if item == "--result-dir" and index + 1 < len(command):
            return str(command[index + 1])
        if item.startswith("--result-dir="):
            return item.split("=", 1)[1]
    return f"{DEFAULT_BENCHMARK_RESULT_DIR}/{safe_slug(case_label)}"


def _benchmark_tasks(
    cases: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    node_container_names: dict[str, str],
    *,
    model_root: str,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for case in cases:
        case_label = case_name(case)
        benchmark = case.get("tests", {}).get("benchmark") or {}
        if not benchmark.get("enabled"):
            continue
        raw_command = benchmark.get("command") or []
        if not isinstance(raw_command, list) or not raw_command:
            tasks.append(
                {
                    "case_name": case_label,
                    "status": "skipped",
                    "reason": "benchmark command is empty",
                }
            )
            continue
        endpoint = _endpoint_instance(entries, case_label)
        if endpoint is None:
            raise PlanError(f"{case_label}: benchmark enabled but no head endpoint is available")

        env = _runtime_env(case, model_root)
        command = [expand_text(str(item), env) for item in raw_command]
        set_cli_flag_value(command, "--host", str(endpoint["node_ip"]))
        set_cli_flag_value(command, "--port", str(endpoint["port"]))
        served_name = _case_served_model_name(case)
        if served_name:
            set_cli_flag_value(command, "--model", served_name)

        result_dir = str(benchmark.get("result_dir") or _benchmark_result_dir(command, case_label))
        if command[:3] == ["vllm", "bench", "serve"]:
            tokenizer_path = _case_model_path(case, model_root)
            if tokenizer_path and not _has_cli_flag(command, "--tokenizer"):
                set_cli_flag_value(command, "--tokenizer", tokenizer_path)
            if not _has_cli_flag(command, "--save-result"):
                command.append("--save-result")
            set_cli_flag_value(command, "--result-dir", result_dir)

        task = {
            "case_name": case_label,
            "node": endpoint["node"],
            "node_ip": endpoint["node_ip"],
            "container_name": node_container_names.get(str(endpoint["node"]), ""),
            "endpoint": f"http://{endpoint['node_ip']}:{endpoint['port']}",
            "command": command,
            "command_shell": command_to_shell(command),
            "timeout_sec": int(benchmark.get("timeout_sec", 3600)),
            "log_file": f"logs/deploy/{safe_slug(case_label)}/benchmark.log",
            "result_dir": result_dir,
            "baseline": str(benchmark.get("baseline") or ""),
            "thresholds": benchmark.get("thresholds") or {},
        }
        tasks.append(task)
    return tasks


def _write_node_files(output_dir: Path, nodes: list[dict[str, Any]]) -> None:
    for node in nodes:
        node_dir = output_dir / node["name"]
        env_dir = node_dir / "env"
        env_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            'export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"',
            "mkdir -p logs/deploy",
        ]
        for instance in node["instances"]:
            env_path = env_dir / f"{instance['name']}.env"
            container_env_path = Path("reports/runtime_plan") / node["name"] / "env" / f"{instance['name']}.env"
            env_path.write_text(
                "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in sorted(instance["env"].items())) + "\n",
                encoding="utf-8",
            )
            lines.append(f"mkdir -p {shlex.quote(str(Path(instance['log_file']).parent))}")
            lines.append(
                "set -a; "
                f". {shlex.quote(str(container_env_path))}; "
                "set +a; "
                f"{instance['command_shell']} > {shlex.quote(instance['log_file'])} 2>&1 &"
            )
        lines.append("wait")
        script = node_dir / "start_instances.sh"
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o755)
        run_container = node_dir / "run_container.sh"
        run_container.write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            f"cd {shlex.quote(CONTAINER_WORKSPACE)}\n"
            f"{node['container']['command_shell']}\n",
            encoding="utf-8",
        )
        run_container.chmod(0o755)
        required_model_paths = sorted(
            {
                str(instance.get("model_path"))
                for instance in node["instances"]
                if str(instance.get("model_path") or "").startswith("/")
            }
        )
        (node_dir / "required_model_paths.txt").write_text(
            "\n".join(required_model_paths) + ("\n" if required_model_paths else ""),
            encoding="utf-8",
        )
        write_json(node_dir / "benchmark_tasks.json", node.get("benchmark_tasks") or [])


def compile_plan(args: argparse.Namespace) -> dict[str, Any]:
    if not args.docker_image:
        raise PlanError("docker image must not be empty")
    if args.container_workspace != CONTAINER_WORKSPACE:
        raise PlanError(f"container workspace must be {CONTAINER_WORKSPACE}")

    nodes = _load_inventory(args.inventory_json)
    paths = read_case_list(args.case_list)
    cases = [load_case(path) for path in paths]
    port_cursor = args.port_start
    extra_port_cursor = args.extra_port_start
    case_entries: list[dict[str, Any]] = []
    for case in cases:
        entries, port_cursor, extra_port_cursor = _profile_service_entries(
            case,
            nodes,
            model_root=args.model_root,
            port_cursor=port_cursor,
            extra_port_cursor=extra_port_cursor,
        )
        case_entries.extend(entries)

    output_nodes = []
    for node in nodes:
        if not node["instances"]:
            continue
        command = _container_command(
            node,
            docker_image=args.docker_image,
            host_workspace=args.host_workspace,
            model_root=args.model_root,
            extra_docker_args=args.extra_docker_args,
        )
        output_nodes.append(
            {
                "name": node["name"],
                "ip": node["ip"],
                "ansible_host": node["ansible_host"],
                "network_interface": node.get("network_interface") or "",
                "cards_total": node["cards_total"],
                "container": {
                    "name": f"vllm-ascend-ci-{safe_slug(node['name'])}",
                    "command": command,
                    "command_shell": command_to_shell(command),
                },
                "instances": node["instances"],
            }
        )

    if not output_nodes:
        raise PlanError("deployment plan has no node instances")

    node_container_names = {node["name"]: node["container"]["name"] for node in output_nodes}
    benchmark_tasks = _benchmark_tasks(cases, case_entries, node_container_names, model_root=args.model_root)
    for node in output_nodes:
        node["benchmark_tasks"] = [task for task in benchmark_tasks if task.get("node") == node["name"]]

    return {
        "schema_version": "deploy-plan/v1",
        "dry_run": bool(args.dry_run),
        "container_per_node": True,
        "selected_cases_file": str(Path(args.case_list).resolve()),
        "nodes": output_nodes,
        "instances": case_entries,
        "benchmark_tasks": benchmark_tasks,
    }


def _write_ansible_inventory(plan: dict[str, Any], output_dir: Path) -> None:
    hosts = {
        node["name"]: {
            "ansible_host": node.get("ansible_host") or node["ip"],
            "node_ip": node["ip"],
            "network_interface": node.get("network_interface") or "",
            "runtime_plan_node_dir": str((output_dir / node["name"]).resolve()),
            "runtime_container_name": node["container"]["name"],
            "runtime_container_command": node["container"]["command_shell"],
            "runtime_readiness_urls": [
                f"http://{instance['node_ip']}:{instance['port']}/health"
                for instance in node.get("instances", [])
                if instance.get("head", True) and not instance.get("headless")
            ],
            "runtime_container_log": f"logs/deploy/{node['container']['name']}.container.log",
            "runtime_benchmark_task_count": len(node.get("benchmark_tasks") or []),
        }
        for node in plan["nodes"]
    }
    inventory = {
        "all": {
            "hosts": hosts,
            "vars": {
                "runtime_plan_file": str((output_dir / "deployment_plan.json").resolve()),
                "runtime_selected_cases_file": str(plan.get("selected_cases_file") or ""),
                "container_workspace": CONTAINER_WORKSPACE,
            },
        }
    }
    (output_dir / "ansible_inventory.yml").write_text(yaml.safe_dump(inventory, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plan = compile_plan(args)
        write_json(output_dir / "deployment_plan.json", plan)
        _write_ansible_inventory(plan, output_dir)
        _write_node_files(output_dir, plan["nodes"])
    except Exception as exc:  # noqa: BLE001 - emit one clear failure line for Jenkins
        raise SystemExit(f"ERROR: failed to compile deployment plan: {exc}") from exc

    print(f"compiled deployment plan: {output_dir / 'deployment_plan.json'}")
    print(f"compiled Ansible inventory: {output_dir / 'ansible_inventory.yml'}")
    print(f"planned {len(plan['nodes'])} node container(s), {len(plan['instances'])} instance process(es)")
    print(f"planned {len(plan.get('benchmark_tasks') or [])} benchmark task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
