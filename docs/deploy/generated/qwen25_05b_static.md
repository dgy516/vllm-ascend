本文档由 .ci/deploy_cases/*.yaml 自动生成，请不要直接手工修改。

# Qwen2.5 0.5B TP2 Static DeployCase Contract Check

## 1. 文档概述

Static Jenkins case used by PR/static flows to validate DeployCase rendering, Docker metadata, and CLI generation.

- Case: `qwen25-05b-static`
- Level: `static`
- Owner: `vllm-ascend-ci`
- Audience: CI maintainers
- Difficulty: basic
- Tags: static, single-service, qwen

## 2. 环境要求

### Hardware

- `accelerator`: Ascend NPU
- `soc`: any
- `min_cards`: 2
- `card_count`: 2
- `allow_parallel_on_host`: True
- `memory`: 16 GB or higher recommended

### Software

- `python`: >=3.10
- `cann`: Compatible with the checked-out vLLM Ascend branch
- `vllm_ascend`: Installed from this repository

## 3. 模型信息

- `name`: Qwen/Qwen2.5-0.5B-Instruct
- `source`: Hugging Face or ModelScope
- `path_hint`: Set MODEL_ROOT to force a local model path during runtime execution.

## 4. 部署拓扑

- Service `qwen25-static` runs as `vllm-serve` on `127.0.0.1:8001` with role `serve` and card_count=`2`.

Jenkins runtime 通过 Lockable Resources 获取 Ascend 节点，`.ci/scripts/ci.py compile-plan`
根据 DeployCase 和节点 inventory 编译物理部署计划。每台物理节点最多启动一个 runtime Docker 容器；
容器内可按编译结果启动多个 vLLM 实例，每个实例使用独立 `ASCEND_RT_VISIBLE_DEVICES` 和端口。

## 5. 环境变量

```bash
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

## 6. 启动服务命令

### vLLM 命令

```bash
# qwen25-static (serve)
vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name qwen25-05b-static --host 127.0.0.1 --port 8001 --tensor-parallel-size 2 --max-model-len 2048 --max-num-batched-tokens 2048 --trust-remote-code
```

### Jenkins / Ansible runtime 示例

```bash
python3 .ci/scripts/ci.py lock-inventory \
  --variable LOCKED_ASCEND_NODES \
  --output-json reports/runtime_plan/runtime_cluster_nodes.json \
  --output-inventory reports/runtime_plan/locked_ansible_inventory.yml

python3 .ci/scripts/ci.py compile-plan \
  --case-list reports/selected_cases.txt \
  --inventory-json reports/runtime_plan/runtime_cluster_nodes.json \
  --output-dir reports/runtime_plan \
  --model-root "${MODEL_ROOT}" \
  --docker-image "${ASCEND_DOCKER_IMAGE}" \
  --host-workspace /home/ma-user/AscendCloud/jenkins

ANSIBLE_CONFIG=.ci/ansible/ansible.cfg ansible-playbook \
  -i reports/runtime_plan/ansible_inventory.yml \
  .ci/ansible/playbooks/deploy_cases.yml \
  -e dry_run_runtime=false

# The compiled per-node Docker command is written to:
# reports/runtime_plan/<node>/run_container.sh
```

Docker 配置：

- `enabled`: True
- `image`: ${ASCEND_DOCKER_IMAGE}
- `workspace`: /home/ma-user/AscendCloud/jenkins
- `network`: host
- `ipc`: host
- `shm_size`: 1g
- `mounts`:
  - item 1:
    - `source`: ${WORKSPACE}/.ci
    - `target`: /home/ma-user/AscendCloud/jenkins/.ci
    - `mode`: ro
  - item 2:
    - `source`: ${WORKSPACE}/reports
    - `target`: /home/ma-user/AscendCloud/jenkins/reports
    - `mode`: rw
  - item 3:
    - `source`: ${WORKSPACE}/logs
    - `target`: /home/ma-user/AscendCloud/jenkins/logs
    - `mode`: rw
  - item 4:
    - `source`: ${MODEL_ROOT}
    - `target`: ${MODEL_ROOT}
    - `mode`: ro

## 7. vLLM config 示例

```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
local_model_path: ${MODEL_ROOT}/Qwen2.5-0.5B-Instruct
served_model_name: qwen25-05b-static
args:
- --served-model-name
- qwen25-05b-static
- --host
- 127.0.0.1
- --port
- '8001'
- --tensor-parallel-size
- '2'
- --max-model-len
- '2048'
- --max-num-batched-tokens
- '2048'
- --trust-remote-code
```

## 8. 服务验证

```bash
curl -fsS http://127.0.0.1:8001/health
```

```bash
curl -sS -X POST http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "qwen25-05b-static",
  "prompt": "San Francisco is a",
  "max_tokens": 8,
  "temperature": 0
}' | python3 -m json.tool
```

## 9. Benchmark 验证

```bash
# benchmark disabled
```

## 10. Accuracy 验证

```bash
# accuracy disabled
```

## 11. 参数说明

| Parameter | Value |
| --- | --- |
| `--served-model-name` | `qwen25-05b-static` |
| `--host` | `127.0.0.1` |
| `--port` | `8001` |
| `--tensor-parallel-size` | `2` |
| `--max-model-len` | `2048` |
| `--max-num-batched-tokens` | `2048` |
| `--trust-remote-code` | enabled |

## 12. 停止服务

```bash
pkill -f 'vllm serve Qwen/Qwen2.5-0.5B-Instruct' || true
```

## 13. 注意事项

- Jenkins 默认 `RUN_ASCEND=false`，不会在静态流程中启动真实模型。
- A2/A3 runtime 均采用单容器内并发，不在同一节点上启动多个 Ascend workload 容器。
- 大模型 case 会按 `requirements.hardware.card_count` 申请更多卡；`ASCEND_LOCK_LABEL` 可作为整机外层保护。
- `reports/` 和 `logs/` 是运行时产物目录，不应提交到 Git。
- PD 分离可在同一容器内通过多个 `services[]` 进程表达；多机拓扑仍预留。