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

本 case 在 Jenkins runtime 中申请 `8` 张 Ascend 卡。卡号和端口由 `.ci/scripts/with_runtime_allocation.py` 在宿主机分配，并通过 `ASCEND_RT_VISIBLE_DEVICES` 和 `VLLM_CI_ALLOCATED_PORTS` 注入容器。

## 5. 环境变量

```bash
export VLLM_USE_MODELSCOPE=true
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
docker run --rm \
  --name vllm-ascend-ci-qwen3-32b-tp8 \
  --network host \
  --ipc host \
  --shm-size 128g \
  ${ASCEND_DOCKER_DEVICE_ARGS} \
  -e ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} \
  -e VLLM_CI_ALLOCATED_PORTS=${VLLM_CI_ALLOCATED_PORTS} \
  -e MODEL_ROOT=${MODEL_ROOT} \
  -v ${WORKSPACE}:/workspace/vllm-ascend:rw \
  -v ${MODEL_ROOT}:${MODEL_ROOT}:ro \
  -w /workspace/vllm-ascend \
  ${ASCEND_DOCKER_IMAGE} \
  vllm serve Qwen/Qwen3-32B --served-model-name qwen3-32b-tp8 --host 127.0.0.1 --port 8113 --tensor-parallel-size 8 --max-model-len 32768 --max-num-batched-tokens 32768 --max-num-seqs 16 --gpu-memory-utilization 0.9 --trust-remote-code --no-enable-prefix-caching
```

Docker 配置：

- `enabled`: True
- `image`: ${ASCEND_DOCKER_IMAGE}
- `network`: host
- `ipc`: host
- `shm_size`: 128g
- `mounts`:
  - item 1:
    - `source`: ${WORKSPACE}
    - `target`: /workspace/vllm-ascend
    - `mode`: rw
  - item 2:
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
  "prompt": "The future of AI is",
  "max_tokens": 16,
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
- 一个 Jenkins parallel branch 对应一个 Docker 容器，容器只看到分配到的 Ascend 卡。
- 大模型 case 会按 `requirements.hardware.card_count` 申请更多卡；`ASCEND_LOCK_LABEL` 只作为可选外层保护。
- `reports/` 和 `logs/` 是运行时产物目录，不应提交到 Git。
- PD、多服务和多机场景在第一版中只预留结构，runner 暂不执行。