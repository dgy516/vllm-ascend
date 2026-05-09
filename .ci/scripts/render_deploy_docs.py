#!/usr/bin/env python3
"""Render DeployCase YAML files into deployment markdown documents."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import yaml
from deploy_case_lib import (
    ALLOWED_LEVELS,
    build_vllm_serve_command,
    case_card_count,
    case_level,
    case_name,
    command_to_shell,
    docker_config,
    expand_case_paths,
    first_service,
    load_case,
    served_model_name,
)
from run_runtime_container import docker_command_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render generated deployment docs from DeployCase YAML files.")
    parser.add_argument("--cases", nargs="+", required=True, help="DeployCase YAML glob(s)")
    parser.add_argument("--level", default="all", help="Render all or one metadata.level")
    parser.add_argument("--output-dir", default="docs/deploy/generated", help="Generated markdown output directory")
    parser.add_argument("--template", default=".ci/templates/model_deploy.md.j2", help="Jinja2 markdown template")
    return parser.parse_args()


def _format_mapping(value: Any) -> str:
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"- `{key}`:")
                for sub_key, sub_value in item.items():
                    lines.append(f"  - `{sub_key}`: {sub_value}")
            elif isinstance(item, list):
                lines.append(f"- `{key}`:")
                for index, entry in enumerate(item, start=1):
                    if isinstance(entry, dict):
                        lines.append(f"  - item {index}:")
                        for sub_key, sub_value in entry.items():
                            lines.append(f"    - `{sub_key}`: {sub_value}")
                    else:
                        lines.append(f"  - {entry}")
            else:
                lines.append(f"- `{key}`: {item}")
        return "\n".join(lines) if lines else "- N/A"
    return str(value)


def _env_exports(env: dict[str, Any]) -> str:
    if not env:
        return "# no runtime environment variables"
    return "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in env.items())


def _parameter_table(args: list[str]) -> str:
    if not args:
        return "No extra vLLM parameters."
    rows = ["| Parameter | Value |", "| --- | --- |"]
    index = 0
    while index < len(args):
        token = str(args[index])
        if token.startswith("--") and index + 1 < len(args) and not str(args[index + 1]).startswith("--"):
            rows.append(f"| `{token}` | `{args[index + 1]}` |")
            index += 2
        else:
            rows.append(f"| `{token}` | enabled |")
            index += 1
    return "\n".join(rows)


def _render_template(template_text: str, context: dict[str, str]) -> str:
    try:
        from jinja2 import Template

        return Template(template_text).render(**context)
    except ModuleNotFoundError:
        rendered = template_text
        for key, value in context.items():
            rendered = rendered.replace("{{ " + key + " }}", value)
            rendered = rendered.replace("{{" + key + "}}", value)
        return rendered


def _build_context(case: dict[str, Any]) -> dict[str, str]:
    metadata = case.get("metadata", {})
    doc = case.get("doc", {})
    requirements = case.get("requirements", {})
    runtime = case.get("runtime", {})
    docker = docker_config(case)
    service = first_service(case)
    vllm = service.get("vllm") or {}
    command = build_vllm_serve_command(case, service)
    serve_shell = command_to_shell(command)
    host = service.get("host", "127.0.0.1")
    port = service.get("port", 8000)
    readiness = case.get("checks", {}).get("readiness", {})
    readiness_path = readiness.get("path", "/health")
    benchmark = case.get("tests", {}).get("benchmark", {})
    accuracy = case.get("tests", {}).get("accuracy", {})
    smoke_payload = json.dumps(
        {
            "model": served_model_name(service),
            "prompt": "San Francisco is a",
            "max_tokens": 8,
            "temperature": 0,
        },
        indent=2,
        ensure_ascii=False,
    )

    benchmark_command = "# benchmark disabled"
    if benchmark.get("enabled") and benchmark.get("command"):
        benchmark_command = command_to_shell([str(item) for item in benchmark["command"]])

    accuracy_command = "# accuracy disabled"
    if accuracy.get("enabled") and accuracy.get("command"):
        accuracy_command = command_to_shell([str(item) for item in accuracy["command"]])

    topology = (
        f"- Service `{service.get('name')}` runs as `{service.get('type')}` on "
        f"`{host}:{port}` with role `{service.get('role')}`."
    )
    docker_command = docker_command_example()

    return {
        "generated_warning": str(doc.get("generated_warning", "")),
        "title": str(metadata.get("title", case_name(case))),
        "description": str(metadata.get("description", "")),
        "name": case_name(case),
        "level": case_level(case),
        "owner": str(metadata.get("owner", "")),
        "audience": str(doc.get("audience", "")),
        "difficulty": str(doc.get("difficulty", "")),
        "tags": ", ".join(str(tag) for tag in metadata.get("tags", [])),
        "hardware": _format_mapping(requirements.get("hardware", {})),
        "card_count": str(case_card_count(case)),
        "software": _format_mapping(requirements.get("software", {})),
        "model_requirements": _format_mapping(requirements.get("model", {})),
        "topology": topology,
        "docker_runtime": _format_mapping(docker),
        "docker_command": docker_command,
        "env_exports": _env_exports(runtime.get("env") or {}),
        "serve_command": serve_shell,
        "vllm_config": yaml.safe_dump(vllm, allow_unicode=True, sort_keys=False).rstrip(),
        "readiness_command": f"curl -fsS http://{host}:{port}{readiness_path}",
        "smoke_command": (
            f"curl -sS -X POST http://{host}:{port}/v1/completions \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            f"  -d {shlex.quote(smoke_payload)} | python3 -m json.tool"
        ),
        "benchmark_command": benchmark_command,
        "accuracy_command": accuracy_command,
        "parameter_table": _parameter_table([str(arg) for arg in vllm.get("args", [])]),
        "stop_command": f"pkill -f {shlex.quote(' '.join(command[:3]))} || true",
    }


def _should_render(case: dict[str, Any], level: str) -> bool:
    if level == "all":
        return True
    if level not in ALLOWED_LEVELS:
        raise ValueError(f"--level must be all or one of {sorted(ALLOWED_LEVELS)}")
    return case_level(case) == level


def main() -> int:
    args = parse_args()
    template_path = Path(args.template)
    template_text = template_path.read_text(encoding="utf-8")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[str] = []
    for path in [p for p in expand_case_paths(args.cases) if Path(p).exists()]:
        case = load_case(path)
        if not _should_render(case, args.level):
            continue
        doc_output = Path(str(case.get("doc", {}).get("output") or f"{case_name(case)}.md"))
        output_path = output_dir / doc_output.name
        output_path.write_text(_render_template(template_text, _build_context(case)), encoding="utf-8")
        rendered.append(str(output_path))
        print(f"rendered {path} -> {output_path}")

    if not rendered:
        print(f"No DeployCase docs rendered for level={args.level}.")
        return 1
    print(f"rendered {len(rendered)} generated deployment doc(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
