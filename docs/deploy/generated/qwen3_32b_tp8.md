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
- `min_cards`: 8
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

## 5. 环境变量

```bash
export VLLM_USE_MODELSCOPE=true
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export HCCL_OP_EXPANSION_MODE=AIV
```

## 6. 启动服务命令

```bash
vllm serve Qwen/Qwen3-32B --served-model-name qwen3-32b-tp8 --host 127.0.0.1 --port 8113 --tensor-parallel-size 8 --max-model-len 32768 --max-num-batched-tokens 32768 --max-num-seqs 16 --gpu-memory-utilization 0.9 --trust-remote-code --no-enable-prefix-caching
```

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
- 大模型 case 需要独占 Ascend 资源，建议配置 `ASCEND_LOCK_LABEL`。
- `reports/` 和 `logs/` 是运行时产物目录，不应提交到 Git。
- PD、多服务和多机场景在第一版中只预留结构，runner 暂不执行。