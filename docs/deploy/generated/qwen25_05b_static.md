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

- Service `qwen25-static` runs as `vllm-serve` on `127.0.0.1:8001` with role `serve`.

本 case 在 Jenkins runtime 中申请 `2` 张 Ascend 卡。卡号和端口由 `.ci/scripts/with_runtime_allocation.py` 在宿主机分配，并通过 `ASCEND_RT_VISIBLE_DEVICES` 和 `VLLM_CI_ALLOCATED_PORTS` 注入容器。

## 5. 环境变量

```bash
export VLLM_USE_MODELSCOPE=true
```

## 6. 启动服务命令

### vLLM 命令

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name qwen25-05b-static --host 127.0.0.1 --port 8001 --tensor-parallel-size 2 --max-model-len 2048 --max-num-batched-tokens 2048 --trust-remote-code
```

### Docker runtime 示例

```bash
docker run --rm \
  --name vllm-ascend-ci-qwen25-05b-static \
  --network host \
  --ipc host \
  --shm-size 64g \
  ${ASCEND_DOCKER_DEVICE_ARGS} \
  -e ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} \
  -e VLLM_CI_ALLOCATED_PORTS=${VLLM_CI_ALLOCATED_PORTS} \
  -e MODEL_ROOT=${MODEL_ROOT} \
  -v ${WORKSPACE}:/workspace/vllm-ascend:rw \
  -v ${MODEL_ROOT}:${MODEL_ROOT}:ro \
  -w /workspace/vllm-ascend \
  ${ASCEND_DOCKER_IMAGE} \
  vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name qwen25-05b-static --host 127.0.0.1 --port 8001 --tensor-parallel-size 2 --max-model-len 2048 --max-num-batched-tokens 2048 --trust-remote-code
```

Docker 配置：

- `enabled`: True
- `image`: ${ASCEND_DOCKER_IMAGE}
- `network`: host
- `ipc`: host
- `shm_size`: 64g
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
- 一个 Jenkins parallel branch 对应一个 Docker 容器，容器只看到分配到的 Ascend 卡。
- 大模型 case 会按 `requirements.hardware.card_count` 申请更多卡；`ASCEND_LOCK_LABEL` 只作为可选外层保护。
- `reports/` 和 `logs/` 是运行时产物目录，不应提交到 Git。
- PD、多服务和多机场景在第一版中只预留结构，runner 暂不执行。