本文档由 .ci/deploy_cases/*.yaml 自动生成，请不要直接手工修改。

# Qwen2.5-VL 7B P/D Smoke Deployment

## 1. 文档概述

Single-node Qwen2.5-VL-7B-Instruct prefiller/decoder disaggregated deployment for PR contract checks and smoke validation.

- Case: `qwen25-vl-7b-pd-smoke`
- Level: `smoke`
- Owner: `vllm-ascend-ci`
- Audience: CI maintainers and vLLM Ascend deployment engineers
- Difficulty: intermediate
- Tags: static, smoke, vl, pd, pd-disaggregation, qwen

## 2. 环境要求

### Hardware

- `accelerator`: Ascend NPU
- `soc`: any
- `min_cards`: 2
- `card_count`: 2
- `allow_parallel_on_host`: True
- `memory`: 32 GB or higher per card recommended

### Software

- `python`: >=3.10
- `cann`: Compatible with the checked-out vLLM Ascend branch
- `vllm_ascend`: Installed in the runtime Docker image
- `extra`: MooncakeConnectorV1 support is required for P/D KV transfer.

## 3. 模型信息

- `name`: Qwen/Qwen2.5-VL-7B-Instruct
- `source`: Hugging Face or ModelScope
- `path_hint`: Set MODEL_ROOT to a local model mirror when Jenkins agents cannot download models.

## 4. 部署拓扑

- Service `qwen25-vl-proxy` runs as `command` on `127.0.0.1:8090` with role `proxy` and card_count=`0`.
- Service `qwen25-vl-prefill` runs as `vllm-serve` on `127.0.0.1:8091` with role `prefill` and card_count=`1`.
- Service `qwen25-vl-decode` runs as `vllm-serve` on `127.0.0.1:8092` with role `decode` and card_count=`1`.

Jenkins runtime 每次 build 只启动一个 Docker 容器。容器级卡池和端口池由 `.ci/scripts/run_runtime_container.py` 在宿主机分配并持锁；容器内 runner 再按 `requirements.hardware.card_count=2` 为本 case 分配子卡集和端口。

## 5. 环境变量

```bash
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export HCCL_IF_IP=localhost
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
```

## 6. 启动服务命令

### vLLM 命令

```bash
# qwen25-vl-proxy (proxy)
python3 .ci/scripts/pd_proxy.py --host 127.0.0.1 --port 8090 --prefiller-urls http://127.0.0.1:8091 --decoder-urls http://127.0.0.1:8092

# qwen25-vl-prefill (prefill)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --served-model-name qwen25-vl-7b-pd --host 127.0.0.1 --port 8091 --no-enable-prefix-caching --tensor-parallel-size 1 --seed 1024 --max-model-len 10000 --max-num-batched-tokens 10000 --trust-remote-code --gpu-memory-utilization 0.90 --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}'

# qwen25-vl-decode (decode)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --served-model-name qwen25-vl-7b-pd --host 127.0.0.1 --port 8092 --no-enable-prefix-caching --tensor-parallel-size 1 --seed 1024 --max-model-len 10000 --max-num-batched-tokens 10000 --trust-remote-code --gpu-memory-utilization 0.90 --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"30100","engine_id":"1","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}'
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
model: Qwen/Qwen2.5-VL-7B-Instruct
local_model_path: ${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct
served_model_name: qwen25-vl-7b-pd
args:
- --served-model-name
- qwen25-vl-7b-pd
- --host
- 127.0.0.1
- --port
- '8091'
- --no-enable-prefix-caching
- --tensor-parallel-size
- '1'
- --seed
- '1024'
- --max-model-len
- '10000'
- --max-num-batched-tokens
- '10000'
- --trust-remote-code
- --gpu-memory-utilization
- '0.90'
- --kv-transfer-config
- '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"${SERVICE_QWEN25_VL_PREFILL_EXTRA_PORT_0}","engine_id":"0","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}'
```

## 8. 服务验证

```bash
curl -fsS http://127.0.0.1:8090/health
```

```bash
curl -sS -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "qwen25-vl-7b-pd",
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
# benchmark disabled
```

## 10. Accuracy 验证

```bash
# accuracy disabled
```

## 11. 参数说明

| Parameter | Value |
| --- | --- |
| `--served-model-name` | `qwen25-vl-7b-pd` |
| `--host` | `127.0.0.1` |
| `--port` | `8091` |
| `--no-enable-prefix-caching` | enabled |
| `--tensor-parallel-size` | `1` |
| `--seed` | `1024` |
| `--max-model-len` | `10000` |
| `--max-num-batched-tokens` | `10000` |
| `--trust-remote-code` | enabled |
| `--gpu-memory-utilization` | `0.90` |
| `--kv-transfer-config` | `{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"${SERVICE_QWEN25_VL_PREFILL_EXTRA_PORT_0}","engine_id":"0","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}` |

## 12. 停止服务

```bash
pkill -f 'python3 .ci/scripts/pd_proxy.py --host' || true
pkill -f 'vllm serve Qwen/Qwen2.5-VL-7B-Instruct' || true
```

## 13. 注意事项

- Jenkins 默认 `RUN_ASCEND=false`，不会在静态流程中启动真实模型。
- A2/A3 runtime 均采用单容器内并发，不在同一节点上启动多个 Ascend workload 容器。
- 大模型 case 会按 `requirements.hardware.card_count` 申请更多卡；`ASCEND_LOCK_LABEL` 可作为整机外层保护。
- `reports/` 和 `logs/` 是运行时产物目录，不应提交到 Git。
- PD 分离可在同一容器内通过多个 `services[]` 进程表达；多机拓扑仍预留。