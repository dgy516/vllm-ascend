#!/usr/bin/env python3
"""Convert Jenkins Lockable Resources variables into Ansible inventory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml
from ci_tool.deploy_case_lib import write_json

IP_PROPERTY_CANDIDATES = ("IP", "HOST_IP", "ADVERTISE_IP", "ADDR", "HOST")
ANSIBLE_HOST_PROPERTY_CANDIDATES = ("ANSIBLE_HOST", "SSH_HOST", "SSH_IP", "HOSTNAME")
CARD_PROPERTY_CANDIDATES = ("CARDS", "CARD_COUNT", "ASCEND_CARDS", "NPU_COUNT")
NETWORK_INTERFACE_PROPERTY_CANDIDATES = ("NETWORK_INTERFACE", "NIC", "IFNAME", "NETWORK_CARD_NAME")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build runtime cluster JSON and Ansible inventory from Jenkins Lockable Resources variables."
    )
    parser.add_argument("--variable", default="LOCKED_ASCEND_NODES", help="Lock step variable name")
    parser.add_argument("--output-json", default="reports/runtime_plan/runtime_cluster_nodes.json")
    parser.add_argument("--output-inventory", default="reports/runtime_plan/ansible_inventory.yml")
    parser.add_argument("--default-cards", type=int, default=8, help="Default cards per locked node")
    parser.add_argument("--remote-workdir", default="/home/ma-user/AscendCloud/jenkins")
    parser.add_argument("--ansible-user", default="", help="Optional Ansible SSH user")
    parser.add_argument("--allow-missing-ip", action="store_true")
    return parser.parse_args()


def _env_value(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return ""


def _resource_names(variable: str) -> list[str]:
    raw = os.environ.get(variable, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resource_property(variable: str, index: int, candidates: tuple[str, ...]) -> str:
    names = [f"{variable}{index}_PROP_{candidate}" for candidate in candidates]
    return _env_value(*names)


def _int_or_default(value: str, default: int) -> int:
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise SystemExit(f"invalid integer resource property: {value!r}") from None
    if parsed < 1:
        raise SystemExit(f"resource card count must be >= 1, got {parsed}")
    return parsed


def _cluster_inventory(args: argparse.Namespace) -> dict[str, Any]:
    names = _resource_names(args.variable)
    if not names:
        raise SystemExit(f"Jenkins lock variable {args.variable} is empty; no locked Ascend nodes found")

    nodes: list[dict[str, Any]] = []
    for index, fallback_name in enumerate(names):
        name = _env_value(f"{args.variable}{index}") or fallback_name
        ip = _resource_property(args.variable, index, IP_PROPERTY_CANDIDATES)
        if not ip and args.allow_missing_ip:
            ip = name
        if not ip:
            raise SystemExit(
                f"locked resource {name} is missing an IP property; configure one of "
                f"{', '.join(IP_PROPERTY_CANDIDATES)} in Lockable Resources"
            )
        ansible_host = _resource_property(args.variable, index, ANSIBLE_HOST_PROPERTY_CANDIDATES) or ip
        cards = _int_or_default(_resource_property(args.variable, index, CARD_PROPERTY_CANDIDATES), args.default_cards)
        node = {
            "name": name,
            "jenkins_node": name,
            "ip": ip,
            "ansible_host": ansible_host,
            "cards": cards,
            "lock_variable": args.variable,
            "lock_index": index,
        }
        network_interface = _resource_property(args.variable, index, NETWORK_INTERFACE_PROPERTY_CANDIDATES)
        if network_interface:
            node["network_interface"] = network_interface
        nodes.append(node)

    return {
        "allocation": "jenkins-lockable",
        "lock_variable": args.variable,
        "node_count": len(nodes),
        "nodes": nodes,
    }


def _ansible_inventory(cluster: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    hosts: dict[str, Any] = {}
    for node in cluster["nodes"]:
        host_vars = {
            "ansible_host": node.get("ansible_host") or node["ip"],
            "node_ip": node["ip"],
            "cards": int(node["cards"]),
            "lock_resource": node["name"],
        }
        if node.get("network_interface"):
            host_vars["network_interface"] = node["network_interface"]
        if args.ansible_user:
            host_vars["ansible_user"] = args.ansible_user
        hosts[node["jenkins_node"]] = host_vars

    return {
        "all": {
            "hosts": hosts,
            "vars": {
                "container_workspace": args.remote_workdir,
                "runtime_plan_dir": "reports/runtime_plan",
            },
        }
    }


def main() -> int:
    args = parse_args()
    cluster = _cluster_inventory(args)
    inventory = _ansible_inventory(cluster, args)

    write_json(args.output_json, cluster)
    inventory_path = Path(args.output_inventory)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(yaml.safe_dump(inventory, sort_keys=False), encoding="utf-8")

    print(f"wrote runtime cluster JSON: {args.output_json}")
    print(f"wrote Ansible inventory: {args.output_inventory}")
    for node in cluster["nodes"]:
        network_interface = f" network_interface={node['network_interface']}" if node.get("network_interface") else ""
        print(
            f"- {node['jenkins_node']} ansible_host={node.get('ansible_host') or node['ip']} "
            f"node_ip={node['ip']} cards={node['cards']}{network_interface}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
