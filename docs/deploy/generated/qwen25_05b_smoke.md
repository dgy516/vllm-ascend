本文档由 .ci/deploy_cases/*.yaml 自动生成，请不要直接手工修改。

# Qwen2.5 0.5B Single-Card Smoke Deployment

## 1. 文档概述

Minimal single-service vLLM Ascend smoke case for Jenkins validation.

- Case: `qwen25-05b-smoke`
- Level: `smoke`
- Owner: `vllm-ascend-ci`
- Audience: CI maintainers and deployment engineers
- Difficulty: basic
- Tags: smoke, single-service, qwen

## 2. 环境要求

### Hardware

- `accelerator`: Ascend NPU
- `min_cards`: 1
- `memory`: 16 GB or higher recommended

### Software

- `python`: >=3.10
- `cann`: Compatible with the checked-out vLLM Ascend branch
- `vllm_ascend`: Installed from this repository

## 3. 模型信息

- `name`: Qwen/Qwen2.5-0.5B-Instruct
- `source`: Hugging Face or ModelScope
- `path_hint`: Set MODEL_ROOT only when using a local mirror.

## 4. 部署拓扑

- Service `qwen25-smoke` runs as `vllm-serve` on `127.0.0.1:8000` with role `serve`.

## 5. 环境变量

```bash
export VLLM_USE_MODELSCOPE=true
```

## 6. 启动服务命令

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name qwen25-05b-smoke --host 127.0.0.1 --port 8000 --max-model-len 2048 --max-num-batched-tokens 2048 --trust-remote-code
```

## 7. vLLM config 示例

```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
local_model_path: ${MODEL_ROOT}/Qwen2.5-0.5B-Instruct
served_model_name: qwen25-05b-smoke
args:
- --served-model-name
- qwen25-05b-smoke
- --host
- 127.0.0.1
- --port
- '8000'
- --max-model-len
- '2048'
- --max-num-batched-tokens
- '2048'
- --trust-remote-code
```

## 8. 服务验证

```bash
curl -fsS http://127.0.0.1:8000/health
```

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "qwen25-05b-smoke",
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
| `--served-model-name` | `qwen25-05b-smoke` |
| `--host` | `127.0.0.1` |
| `--port` | `8000` |
| `--max-model-len` | `2048` |
| `--max-num-batched-tokens` | `2048` |
| `--trust-remote-code` | enabled |

## 12. 停止服务

```bash
pkill -f 'vllm serve Qwen/Qwen2.5-0.5B-Instruct' || true
```

## 13. 注意事项

- Jenkins 默认 `RUN_ASCEND=false`，不会在静态流程中启动真实模型。
- 大模型 case 需要独占 Ascend 资源，建议配置 `ASCEND_LOCK_LABEL`。
- `reports/` 和 `logs/` 是运行时产物目录，不应提交到 Git。
- PD、多服务和多机场景在第一版中只预留结构，runner 暂不执行。