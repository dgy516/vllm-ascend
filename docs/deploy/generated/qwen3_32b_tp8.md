本文档由 .ci/deploy_cases/*.yaml 自动生成，请不要直接手工修改。

# Qwen3 32B TP8 Nightly Deployment

## 1. 文档概述

Single-service Qwen3 32B deployment case for 8-card Ascend nightly validation.

- Case: `qwen3-32b-tp8`
- Level: `nightly`
- Owner: `vllm-ascend-ci`
- Audience: vLLM Ascend deployment engineers
- Difficulty: advanced
- Tags: nightly, single-service, qwen3, tp8

## 2. 环境要求

### Hardware

- `accelerator`: Ascend 910B or compatible
- `soc`: any
- `min_cards`: 8
- `card_count`: 8
- `allow_parallel_on_host`: True
- `memory`: 64 GB per card recommended

### Software

- `python`: >=3.10
- `cann`: Compatible with the checked-out vLLM Ascend branch
- `vllm_ascend`: Installed from this repository

## 3. 模型信息

- `name`: Qwen/Qwen3-32B
- `source`: Hugging Face or ModelScope
- `path_hint`: Use MODEL_ROOT or a pre-populated model cache on Jenkins agents.

## 4. 部署拓扑

- Service `qwen3-32b` runs as `vllm-serve` on `127.0.0.1:8113` with role `serve`.

Jenkins runtime 每次 build 只启动一个 Docker 容器。容器级卡池和端口池由 `.ci/scripts/run_runtime_container.py` 在宿主机分配并持锁；容器内 runner 再按 `requirements.hardware.card_count=8` 为本 case 分配子卡集和端口。

## 5. 环境变量

```bash
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export HCCL_OP_EXPANSION_MODE=AIV
```

## 6. 启动服务命令

### vLLM 命令

```bash
vllm serve Qwen/Qwen3-32B --served-model-name qwen3-32b-tp8 --host 127.0.0.1 --port 8113 --tensor-parallel-size 8 --max-model-len 32768 --max-num-batched-tokens 32768 --max-num-seqs 16 --gpu-memory-utilization 0.9 --trust-remote-code --no-enable-prefix-caching
```

### Docker runtime 示例

```bash
docker run --rm --name 'vllm-ascend-ci-${BUILD_TAG}' --network host --ipc host --shm-size=1g --device /dev/davinci${ASCEND_CARD_ID} --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -v /root/.cache:/root/.cache -v '${WORKSPACE}/.ci:/home/ma-user/AscendCloud/jenkins/.ci:ro' -v '${WORKSPACE}/reports:/home/ma-user/AscendCloud/jenkins/reports:rw' -v '${WORKSPACE}/logs:/home/ma-user/AscendCloud/jenkins/logs:rw' -e 'MODEL_ROOT=${MODEL_ROOT}' -v '${MODEL_ROOT}:${MODEL_ROOT}:ro' -e ASCEND_RT_VISIBLE_DEVICES=${ASCEND_CARD_ID} -e VLLM_CI_ALLOCATED_PORTS=${VLLM_PORT} -e VLLM_CI_ALLOCATION_JSON=reports/runtime_container_allocation.json -e 'VLLM_CI_CONTAINER_NAME=vllm-ascend-ci-${BUILD_TAG}' -e PYTHONUNBUFFERED=1 -w /home/ma-user/AscendCloud/jenkins '${ASCEND_DOCKER_IMAGE}' bash -lc 'cd /home/ma-user/AscendCloud/jenkins && python3 .ci/scripts/run_deploy_cases.py --case-list reports/selected_cases.txt --allocation-json reports/runtime_container_allocation.json --output-dir reports/nightly/case_results --logs-dir logs/deploy --model-root '"'"'${MODEL_ROOT}'"'"' --parallelism '"'"'${RUNTIME_PARALLELISM}'"'"' --continue-on-error'
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
model: Qwen/Qwen3-32B
local_model_path: ${MODEL_ROOT}/Qwen3-32B
served_model_name: qwen3-32b-tp8
args:
- --served-model-name
- qwen3-32b-tp8
- --host
- 127.0.0.1
- --port
- '8113'
- --tensor-parallel-size
- '8'
- --max-model-len
- '32768'
- --max-num-batched-tokens
- '32768'
- --max-num-seqs
- '16'
- --gpu-memory-utilization
- '0.9'
- --trust-remote-code
- --no-enable-prefix-caching
```

## 8. 服务验证

```bash
curl -fsS http://127.0.0.1:8113/health
```

```bash
curl -sS -X POST http://127.0.0.1:8113/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "qwen3-32b-tp8",
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
| `--served-model-name` | `qwen3-32b-tp8` |
| `--host` | `127.0.0.1` |
| `--port` | `8113` |
| `--tensor-parallel-size` | `8` |
| `--max-model-len` | `32768` |
| `--max-num-batched-tokens` | `32768` |
| `--max-num-seqs` | `16` |
| `--gpu-memory-utilization` | `0.9` |
| `--trust-remote-code` | enabled |
| `--no-enable-prefix-caching` | enabled |

## 12. 停止服务

```bash
pkill -f 'vllm serve Qwen/Qwen3-32B' || true
```

## 13. 注意事项

- Jenkins 默认 `RUN_ASCEND=false`，不会在静态流程中启动真实模型。
- A2/A3 runtime 均采用单容器内并发，不在同一节点上启动多个 Ascend workload 容器。
- 大模型 case 会按 `requirements.hardware.card_count` 申请更多卡；`ASCEND_LOCK_LABEL` 可作为整机外层保护。
- `reports/` 和 `logs/` 是运行时产物目录，不应提交到 Git。
- PD 分离可在同一容器内通过多个 `services[]` 进程表达；多机拓扑仍预留。