本文档由 .ci/deploy_cases/*.yaml 自动生成，请不要直接手工修改。

# Qwen3.5 0.8B P/D Smoke Deployment

## 1. 文档概述

Qwen3.5-0.8B 1P1D disaggregated deployment for PR smoke validation with MTP decode graph and tool-call checks.

- Case: `qwen35-08b-pd-smoke`
- Level: `smoke`
- Owner: `vllm-ascend-ci`
- Audience: CI maintainers and vLLM Ascend deployment engineers
- Difficulty: intermediate
- Tags: smoke, benchmark, pd, qwen35, mtp, tool-call

## 2. 环境要求

### Hardware

- `accelerator`: Ascend NPU
- `soc`: any
- `min_cards`: 2
- `card_count`: 2
- `allow_parallel_on_host`: True
- `memory`: Two Ascend cards are required; 32 GB or higher per card is recommended.

### Software

- `python`: >=3.10
- `cann`: Compatible with the checked-out vLLM Ascend branch
- `vllm_ascend`: Installed in the runtime Docker image
- `extra`: MooncakeLayerwiseConnector, Qwen3.5 MTP, and qwen3_xml tool parser are required.

## 3. 模型信息

- `name`: Qwen/Qwen3.5-0.8B
- `source`: Local mirror
- `path_hint`: Set MODEL_ROOT to the directory containing Qwen3.5-0.8B.

## 4. 部署拓扑

- Service `qwen35-08b-proxy` runs as `command` on `0.0.0.0:8240` with role `proxy` and card_count=`0`.
- Service `qwen35-08b-prefill` runs as `vllm-serve` on `0.0.0.0:8241` with role `prefill` and card_count=`1`.
- Service `qwen35-08b-decode` runs as `vllm-serve` on `0.0.0.0:8242` with role `decode` and card_count=`1`.

Jenkins runtime 通过 Lockable Resources 获取 Ascend 节点，`.ci/scripts/ci.py compile-plan`
根据 DeployCase 和节点 inventory 编译物理部署计划。每台物理节点最多启动一个 runtime Docker 容器；
容器内可按编译结果启动多个 vLLM 实例，每个实例使用独立 `ASCEND_RT_VISIBLE_DEVICES` 和端口。

## 5. 环境变量

```bash
export VLLM_USE_MODELSCOPE=true
export PYTHONHASHSEED=0
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export ASCEND_BUFFER_POOL=4:8
export VLLM_USE_V1=1
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
export VLLM_ASCEND_ENABLE_NZ=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1536
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export TASK_QUEUE_ENABLE=1
export VLLM_TORCH_PROFILER_WITH_STACK=0
export PREFILL_EXTRA_PORT_0=36441
export DECODE_EXTRA_PORT_0=36442
export PREFILL_SERVERS=0.0.0.0:8241
export DECODE_SERVERS=0.0.0.0:8242
```

## 6. 启动服务命令

### vLLM 命令

```bash
# qwen35-08b-proxy (proxy)
python3 .ci/scripts/pd_proxy.py --host 0.0.0.0 --port 8240

# qwen35-08b-prefill (prefill)
vllm serve Qwen/Qwen3.5-0.8B --served-model-name qwen35-08b-pd-smoke --host 0.0.0.0 --port 8241 --tensor-parallel-size 1 --trust-remote-code --no-enable-prefix-caching --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml

# qwen35-08b-decode (decode)
vllm serve Qwen/Qwen3.5-0.8B --served-model-name qwen35-08b-pd-smoke --host 0.0.0.0 --port 8242 --tensor-parallel-size 1 --trust-remote-code --no-enable-prefix-caching --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}' --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
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
model: Qwen/Qwen3.5-0.8B
local_model_path: ${MODEL_ROOT}/Qwen3.5-0.8B
served_model_name: qwen35-08b-pd-smoke
args:
- --served-model-name
- qwen35-08b-pd-smoke
- --host
- 0.0.0.0
- --port
- '8241'
- --tensor-parallel-size
- '1'
- --trust-remote-code
- --no-enable-prefix-caching
- --reasoning-parser
- qwen3
- --enable-auto-tool-choice
- --tool-call-parser
- qwen3_xml
```

## 8. 服务验证

```bash
curl -fsS http://0.0.0.0:8240/health
```

```bash
curl -sS -X POST http://0.0.0.0:8240/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "qwen35-08b-pd-smoke",
  "messages": [
    {
      "role": "user",
      "content": "Describe vLLM Ascend in one sentence."
    }
  ],
  "max_tokens": 16,
  "temperature": 0
}' | python3 -m json.tool
```

## 9. Benchmark 验证

```bash
vllm bench serve --model qwen35-08b-pd-smoke --host 127.0.0.1 --port 8240 --dataset-name random --random-input-len 128 --random-output-len 32 --num-prompts 16 --request-rate 1 --save-result --result-dir reports/nightly/benchmark/qwen35-08b-pd-smoke
```

## 10. Accuracy 验证

```bash
# accuracy disabled
```

## 11. 参数说明

| Parameter | Value |
| --- | --- |
| `--served-model-name` | `qwen35-08b-pd-smoke` |
| `--host` | `0.0.0.0` |
| `--port` | `8241` |
| `--tensor-parallel-size` | `1` |
| `--trust-remote-code` | enabled |
| `--no-enable-prefix-caching` | enabled |
| `--reasoning-parser` | `qwen3` |
| `--enable-auto-tool-choice` | enabled |
| `--tool-call-parser` | `qwen3_xml` |

## 12. 停止服务

```bash
pkill -f 'python3 .ci/scripts/pd_proxy.py --host' || true
pkill -f 'vllm serve Qwen/Qwen3.5-0.8B' || true
```

## 13. 注意事项

- Jenkins 默认 `RUN_ASCEND=false`，不会在静态流程中启动真实模型。
- A2/A3 runtime 均采用单容器内并发，不在同一节点上启动多个 Ascend workload 容器。
- 大模型 case 会按 `requirements.hardware.card_count` 申请更多卡；`ASCEND_LOCK_LABEL` 可作为整机外层保护。
- `reports/` 和 `logs/` 是运行时产物目录，不应提交到 Git。
- PD 分离可在同一容器内通过多个 `services[]` 进程表达；多机拓扑仍预留。