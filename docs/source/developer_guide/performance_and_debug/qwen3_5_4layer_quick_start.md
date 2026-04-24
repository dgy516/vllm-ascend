# Qwen3.5 4-Layer 启动与验证指南

这份文档面向当前机器上的本地优化验证环境，目标很单一：用 4-layer 裁剪模型快速跑通 `Qwen3.5-122B-A10B-w8a8-mtp` 的 hybrid 路径，并沉淀精度与 profiling 基线。

## 目标与范围

- 只跑 4-layer 裁剪模型，不占用全量卡。
- 只跑 hybrid 模式，不跑 PD 分离。
- 默认使用设备 `2,3`，对应 `tensor_parallel_size=2`。
- 优先保证运行路径稳定，可在此基础上做优化前后对齐。

## 当前假设

- 代码目录：`/data/dong/workspace/fork_vllm_ascend/vllm-ascend-model-multistream-opt`
- 4-layer 模型目录：`/data/models/debug_views/Qwen3.5-122B-A10B-w8a8-mtp-hybrid-4layers`
- baseline 辅助脚本目录：`/data/dong/workspace/fork_vllm_ascend/vllm-ascend-model-multistream-baseline/tools/qwen3_5_multistream`

如果你的本地路径不同，改环境变量，不要重装包。

## 1. 准备环境

```bash
export REPO_ROOT=/data/dong/workspace/fork_vllm_ascend/vllm-ascend-model-multistream-opt
export MODEL_PATH=/data/models/debug_views/Qwen3.5-122B-A10B-w8a8-mtp-hybrid-4layers
export BASELINE_TOOLS=/data/dong/workspace/fork_vllm_ascend/vllm-ascend-model-multistream-baseline/tools/qwen3_5_multistream

cd "${REPO_ROOT}"

# 只调整自己的 PYTHONPATH，不要重装环境
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export ASCEND_RT_VISIBLE_DEVICES=2,3
export ASCEND_VISIBLE_DEVICES=2,3
export HCCL_IF_IP=$(hostname -I | awk '{print $1}')
export VLLM_HOST_IP="${HCCL_IF_IP}"
export GLOO_SOCKET_IFNAME=$(awk '$2=="00000000" {print $1; exit}' /proc/net/route)
export TP_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
export HCCL_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=512
export HF_HUB_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
```

## 2. 确认当前模型确实是 4-layer

```bash
python - <<'PY'
import json
import os
from pathlib import Path

config = json.loads(Path(os.environ["MODEL_PATH"]).joinpath("config.json").read_text())
text_config = config["text_config"]
print("num_hidden_layers =", text_config["num_hidden_layers"])
print("layer_types       =", text_config["layer_types"])
assert text_config["num_hidden_layers"] == 4
assert text_config["layer_types"] == [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
]
PY
```

如果这里失败，先不要继续做优化验证。

## 3. 最小跑通

这一步的目标只是确认当前 worktree、当前 `PYTHONPATH`、当前模型目录能一起正常工作。

```bash
python - <<'PY'
import os
from vllm import LLM, SamplingParams

llm = LLM(
    model=os.environ["MODEL_PATH"],
    trust_remote_code=True,
    tensor_parallel_size=2,
    quantization="ascend",
    gpu_memory_utilization=0.90,
    max_model_len=32768,
    max_num_seqs=16,
    max_num_batched_tokens=4096,
    enable_expert_parallel=True,
    additional_config={
        "recompute_scheduler_enable": True,
        "enable_cpu_binding": True,
    },
    compilation_config={
        "cudagraph_mode": "FULL_DECODE_ONLY",
    },
    disable_hybrid_kv_cache_manager=False,
    enable_prefix_caching=False,
)

outputs = llm.generate(
    ["请用一句话介绍 vLLM。"],
    SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16),
    use_tqdm=False,
)
print(outputs[0].outputs[0].text)

shutdown = getattr(llm, "shutdown", None)
if callable(shutdown):
    shutdown()
PY
```

如果这一步能返回文本，说明 4-layer 模型已经跑起来了。

## 4. 生成精度留档

推荐直接复用已有离线精度 harness。它会把 prompt、输出文本、token ids 和环境快照保存到单独目录，方便后续和 baseline / candidate 做对比。

```bash
export RUN_TAG=$(date -u +%Y%m%d-%H%M%S)
export RUN_DIR=/data/dong/vllm_accuracy_qwen3_5_multistream_opt_4l_offline/${RUN_TAG}

python "${BASELINE_TOOLS}/run_offline_accuracy.py" run \
  --model-path "${MODEL_PATH}" \
  --run-dir "${RUN_DIR}" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --quantization ascend \
  --enable-expert-parallel \
  --max-tokens 16 \
  --additional-config-json '{"recompute_scheduler_enable": true, "enable_cpu_binding": true}'
```

输出目录里最重要的文件是：

- `run_config.json`
- `accuracy_results.json`

如果你要做 baseline / candidate 对齐，流程是：

1. 在 baseline worktree 下跑一份。
2. 在 candidate worktree 下跑一份。
3. 用同一个 compare 脚本比对。

```bash
python "${BASELINE_TOOLS}/run_offline_accuracy.py" compare \
  --baseline-run-dir /path/to/baseline_run \
  --candidate-run-dir /path/to/candidate_run \
  --compare-out /tmp/qwen3_5_4l_accuracy_compare.json
```

## 5. 生成 profiling 留档

同样建议复用已有离线 profiling harness。它会完成两件事：

- `capture`：跑 warmup 和 profile iteration，导出原始 trace
- `analyze`：把 raw trace 解析成 `kernel_details.csv`、`op_statistic.csv` 等文本结果

```bash
export RUN_TAG=$(date -u +%Y%m%d-%H%M%S)
export RUN_DIR=/data/dong/vllm_profile_qwen3_5_multistream_opt_4l_offline/${RUN_TAG}

python "${BASELINE_TOOLS}/run_offline_profile.py" capture \
  --model-path "${MODEL_PATH}" \
  --run-dir "${RUN_DIR}" \
  --warmup-iters 1 \
  --profile-iters 8 \
  --prompt-repeat 128 \
  --prompt-count 1 \
  --max-tokens 16 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --quantization ascend \
  --enable-expert-parallel \
  --additional-config-json '{"recompute_scheduler_enable": true, "enable_cpu_binding": true}'

python "${BASELINE_TOOLS}/run_offline_profile.py" analyze \
  --model-path "${MODEL_PATH}" \
  --run-dir "${RUN_DIR}" \
  --max-process-number 8
```

分析完成后，重点查看：

- `${RUN_DIR}/capture_results.json`
- `${RUN_DIR}/analysis_summary.json`
- `${RUN_DIR}/profile_raw/*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`
- `${RUN_DIR}/profile_raw/*_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv`

## 6. 关于 `cudagraph_mode`

这里要明确区分两类用法：

- 最小跑通命令默认使用 `FULL_DECODE_ONLY`，因为它更接近日常服务配置。
- 现有离线 harness 内部默认写死的是 `FULL`。这是历史基线的一部分。

因此，如果你继续用现有 harness 做 baseline / candidate 对齐，最重要的不是立刻改它，而是让两边保持一致。否则你对比到的可能是配置差异，不是优化收益。

## 7. 常见问题

### `PYTHONPATH` 指到了错误版本

现象通常是：

- 修改的代码没有生效
- import 到了别的 worktree
- 行为和你当前分支不一致

处理方式：

```bash
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python - <<'PY'
import vllm_ascend
print(vllm_ascend.__file__)
PY
```

只改 `PYTHONPATH`，不要重装包。

### 进程或端口被占用

这个环境上可能还有别的实验，不要用强杀。优先正常退出当前脚本，必要时只结束自己启动的进程。

### 模型目录不存在

如果 `/data/models/debug_views/Qwen3.5-122B-A10B-w8a8-mtp-hybrid-4layers` 不存在，可以用 sample worktree 里的切片工具生成：

```bash
export SAMPLE_REPO=/data/dong/workspace/fork_vllm_ascend/vllm-ascend-sample
export SRC_MODEL=/data/models/Qwen3.5-122B-A10B-w8a8-mtp
export DST_MODEL=/data/models/debug_views/Qwen3.5-122B-A10B-w8a8-mtp-hybrid-4layers

PYTHONPATH="${SAMPLE_REPO}${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
import os
from vllm_ascend.debug_utils.model_view import create_symlinked_model_view

create_symlinked_model_view(
    os.environ["SRC_MODEL"],
    os.environ["DST_MODEL"],
    num_layers=4,
    force=False,
)
print(os.environ["DST_MODEL"])
PY
```

这个工具会改写 `config.json` 和权重索引，只保留前 4 层，其他大文件以软链接方式复用。

## 8. 建议的验证顺序

1. 先做第 3 步，确认当前分支能直接跑通。
2. 再做第 4 步，固化精度结果。
3. 再做第 5 步，固化 profiling 结果。
4. 所有 4-layer 验证完成后，再决定是否切回全量模型做最终复验。
