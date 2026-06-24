# Jenkins Ansible DeployCase Runtime Design

本文档定义 vLLM Ascend DeployCase CI 的 Ansible runtime 设计。目标是把多节点部署从 Jenkinsfile
和 Python runner 中抽离出来：Jenkins 只负责调度和报告，DeployCase/profile 只描述模型和拓扑，
编译器生成物理部署计划，Ansible 按计划执行。

## 1. 责任边界

### Jenkins

Jenkins 负责：

- checkout、case 选择、静态校验、文档生成和报告归档。
- 通过 Lockable Resources 分配 Ascend 节点。
- 把锁到的节点转换为 Ansible inventory。
- 调用部署计划编译器和通用 Ansible playbook。

Jenkins 不负责：

- 拼接 `vllm serve` 命令。
- 判断 DP/TP/EP 如何落到节点。
- 判断 PD 的 head/headless。
- 生成 `PREFILL_SERVERS` 或 `DECODE_SERVERS`。

### DeployCase/Profile

DeployCase/profile 是模型部署事实源，负责描述：

- 模型权重、served model name 和量化方式。
- 部署模式，例如 `standalone` 或 `pd`。
- PD 拓扑，例如 `1P1D`。
- 每个 logical service 的 replica 和 DP/TP/EP。
- 模型族或场景相关的通用参数。

DeployCase/profile 不写每台机器的最终 placement。节点数、每节点用卡、head/headless 由编译器根据
Jenkins inventory 计算。

### Compile Deployment Plan

`.ci/scripts/ci.py compile-plan` 是编译层。它不启动容器、不启动模型，只把高层配置编译为：

- `deployment_plan.json`
- `ansible_inventory.yml`
- 每个节点的 `start_instances.sh`
- 每个实例的环境变量文件

Ansible 只消费编译结果，不重新推导 DP/TP/EP 或 PD endpoint。

### Ansible

Ansible 负责：

- 在每台节点创建 runtime 目录。
- 同步或引用 `.ci`、`reports`、`logs`。
- 每台物理机器最多启动一个 Docker 容器。
- 在容器内启动该节点计划中的多个实例进程。
- 收集日志和结果。

Ansible 不负责 case 选择、拓扑推导、报告生成和复杂兜底。

## 2. Runtime 硬约束

- 一台物理机器最多启动一个 runtime Docker 容器。
- 一个容器内可以启动多个 vLLM 实例。
- 每个实例必须显式设置进程级 `ASCEND_RT_VISIBLE_DEVICES`。
- 每个实例必须使用唯一端口并写独立日志。
- 容器级可暴露该节点被计划使用的全部卡；实例级再切分卡。
- logical service endpoint 只指向非 headless 的 head 实例。

## 3. Profile v2

现有 v1 DeployCase 保持可读。v2 结构放在可选 `profile` 字段下，避免破坏已有 `services[]`。

示例：

```yaml
profile:
  model:
    name: Qwen/Qwen2.5-VL-7B-Instruct
    local_path: ${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct
    served_model_name: qwen25-vl-7b-pd
    quantization: none

  deployment:
    mode: pd
    topology: 1P1D

  common:
    host: ${HOST_IP}
    seed: 1024
    max_model_len: 10000
    max_num_batched_tokens: 10000
    gpu_memory_utilization: 0.90
    prefix_caching: false

  services:
    prefill:
      replicas: 1
      parallel:
        dp: 1
        tp: 2
        ep: 1
      kv_role: producer
      extra_port_count: 1

    decode:
      replicas: 1
      parallel:
        dp: 1
        tp: 2
        ep: 1
      kv_role: consumer
      extra_port_count: 1
```

规则：

- `trust_remote_code` MVP 固定开启，不开放配置。
- `model.quantization` 必须显式为 `ascend` 或 `none`。
- `quantization: ascend` 生成 `--quantization ascend`。
- `quantization: none` 不生成 `--quantization`。
- `parallel.ep > 1` 生成 `--enable-expert-parallel`。
- required cards 只由 `dp * tp` 决定，`ep` 不参与卡数乘法。

## 4. Placement 计算

用户不配置 `node_count` 或 `cards_per_node`。编译器根据 Jenkins inventory 自动计算。

示例：

```yaml
parallel:
  dp: 2
  tp: 8
  ep: 16
```

计算：

```text
required_cards = dp * tp = 16
```

- A2 两台 8 卡节点：每台放一个 DP rank，生成一个 head 和一个 headless rank。
- A3 一台 16 卡节点：同一节点放两个 DP rank，生成一个实例进程，`data_parallel_size_local=2`。

如果没有任何节点能容纳一个 `tp` group，直接失败。例如 `TP8` 不允许落到 4 卡节点。

## 5. PD Endpoint

PD 拓扑由 logical service 计算 endpoint：

- `PREFILL_SERVERS` 只包含 prefill logical instance 的 head endpoint。
- `DECODE_SERVERS` 只包含 decode logical instance 的 head endpoint。
- headless worker 不进入 servers 列表。

跨节点 Decode 仍是一个 logical decode instance。rank0/head 暴露 endpoint，其他节点为 headless worker。

## 6. MTP 与图模式策略

支持 MTP 的模型应优先在 Decode 侧启用 MTP。PD 场景下 Prefill 负责 KV 构造，不默认启用 MTP，避免扩大
Prefill 图和显存压力。

MTP 启用规则：

- 权重或 config 明确包含 MTP 能力时，Decode service 应配置 `--speculative-config`。
- Qwen3.5 系列使用 `{"method":"qwen3_5_mtp","num_speculative_tokens":3}`。
- 其它模型族使用其模型文档或测试中验证过的方法名，例如 `mtp`、`deepseek_mtp`、`qwen3_next_mtp`。
- 不允许仅根据模型名猜测 MTP 方法；无法确认方法时直接标记为未启用，而不是静默回退。

图模式规则：

- Decode 侧优先使用 `FULL_DECODE_ONLY`。
- `cudagraph_capture_sizes` 必须按 `(num_speculative_tokens + 1)` 的整数倍配置。
- PR/smoke 只捕获最小必要图。例如 Qwen3.5 MTP=3 时使用 `[4]`，避免启动阶段生成多档大图。
- Nightly/benchmark 可以按真实并发扩展，例如 `[4,8,12,16]`，但需要明确记录图显存和启动耗时。
- 不默认给 MTP Decode 加 `--enforce-eager`。只有在图模式验证失败时，才把 eager 作为隔离或临时 fallback，并在报告中标明。

功能验收规则：

- MTP text smoke 必须返回非空文本。
- 如果模型同时支持 VL，VL + MTP 需要单独验证；VL 请求 HTTP 200 但 content 为空不能算通过。
- 报告需要区分 `text+MTP+graph`、`VL+MTP+graph` 和 `VL without MTP`，避免把不同路径混在一起。

## 7. Function Call 与 Reasoning 策略

Function call 和 reasoning 不是全局默认能力，必须由模型族显式声明 parser。错误 parser 可能把正常输出解析错，
所以编译器不根据模型名静默推断。

Qwen3.5 系列验证路径：

- Prefill 和 Decode service 都增加 `--reasoning-parser qwen3`。
- Prefill 和 Decode service 都增加 `--enable-auto-tool-choice --tool-call-parser qwen3_xml`。PD proxy
  会先把同一 OpenAI 请求发给 Prefill，Prefill 也必须能通过 tool 请求校验和 chat template 渲染。
- smoke case 在 `tests.smoke.suites.capabilities` 中声明 `thinking` 和 `tool_call`。
- `thinking` capability 加载 `.ci/test_suites/smoke/thinking.csv`，必须在响应 JSON 中看到
  `reasoning_content`。
- `tool_call` capability 加载 `.ci/test_suites/smoke/tool_call.csv`，必须在响应 JSON 中看到 `tool_calls`。

验收规则：

- HTTP 200 但没有 `reasoning_content`，reasoning 不算通过。
- HTTP 200 但没有 `tool_calls`，function call 不算通过。
- tool call 用例不强制 `min_output_tokens`，因为成功响应通常是结构化 `tool_calls` 而不是普通文本。
- tool call 用例显式设置 `chat_template_kwargs.enable_thinking=false`。Qwen3.5 同时开启 thinking 和 tool
  时，工具 JSON 可能被 reasoning parser 消费，导致 OpenAI `tool_calls` 为空。
- reasoning 用例使用 `response_json_non_empty_path` 断言结构化字段非空，允许兼容
  `choices.0.message.reasoning_content` 和 `choices.0.message.reasoning` 两种字段名。
- tool/reasoning parser 在 P/D 两侧保持一致；Prefill 不承担最终响应解析，但承担请求校验和 prompt 构造。

## 8. Runtime Inventory

`IP` 是服务通信 endpoint，不一定等于 Jenkins controller 可 SSH 的地址。Lockable Resources 可以额外配置
`ANSIBLE_HOST`、`SSH_HOST`、`SSH_IP` 或 `HOSTNAME` 作为 Ansible SSH 目标；未配置时默认使用 `IP`。
编译器必须保持这两个概念分离，避免把 Jenkins hostname 写入 PD endpoint，或用不可达的服务 IP 做 SSH。

使用 `MooncakeLayerwiseConnector` 的 PD case 还必须提供通信网卡。推荐在 Lockable Resources 中配置
`NETWORK_INTERFACE`，也兼容 `NIC`、`IFNAME` 或 `NETWORK_CARD_NAME`。编译器会基于该属性为每个实例生成
`NETWORK_CARD_NAME`、`GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME`，并把
`HCCL_IF_IP` 设置为当前节点的 `IP`。如果缺少通信网卡，Layerwise PD 计划必须直接失败。

## 9. 必须失败的场景

以下场景必须直接失败：

- Jenkins 锁到的节点数不足。
- Lockable resource 缺少 `IP` 或 `CARDS`。
- 使用 `MooncakeLayerwiseConnector` 时 Lockable resource 缺少通信网卡属性。
- 当 SSH 目标不同于服务 endpoint 时，缺少可达的 `ANSIBLE_HOST`/`SSH_HOST`/`SSH_IP`/`HOSTNAME`。
- Docker image 为空。
- `model.quantization` 缺失或不是 `ascend|none`。
- 模型要求本地路径但路径不存在。
- `dp * tp` 超过 inventory 总卡数。
- 任意节点卡数不足以容纳一个 `tp` group。
- 同一节点计划启动多个容器。
- 两个实例分到同一张卡。
- 两个实例分到同一端口。
- 实例级 `ASCEND_RT_VISIBLE_DEVICES` 为空。
- logical service 没有唯一 head endpoint。
- `PREFILL_SERVERS` 或 `DECODE_SERVERS` 为空或包含 headless rank。
- readiness timeout。
- enabled smoke 正向用例失败。

允许 skip 的场景仅限：

- case level 未选中。
- benchmark/accuracy 显式 disabled。
- capability suite 不适用于当前模型。

## 10. Jenkins 工作流集成

`RUN_ASCEND=true` 时，Jenkins runtime stage 的执行顺序固定为：

1. Jenkins 通过 Lockable Resources 分配节点。
2. `.ci/scripts/ci.py lock-inventory` 把锁到的节点转换为 cluster JSON。
3. `.ci/scripts/ci.py compile-plan` 生成物理部署计划和 Ansible inventory。
4. `.ci/ansible/playbooks/deploy_cases.yml` 在每个节点启动一个 runtime container，并等待编译出的
   head endpoint readiness。
5. `.ci/scripts/ci.py smoke` 在 Jenkins 控制侧读取 `deployment_plan.json`，对每个 case 的
   proxy/head endpoint 执行 CSV smoke suite，并写入 `reports/nightly/case_results/*.json`。
6. `CI_MODE=nightly|benchmark|release` 或 `RUN_BENCHMARK=true` 时，调用
   `.ci/ansible/playbooks/run_benchmarks.yml`，在已启动的 runtime container 内执行编译出的 benchmark task。
7. `finally` 中调用 `.ci/ansible/playbooks/cleanup_runtime.yml` 停止远端容器，拉取远端 runtime artifacts，
   再执行本地日志裁剪和归档。
8. `.ci/scripts/ci.py merge-benchmark` 把远端 benchmark stage JSON 合并回
   `reports/nightly/case_results/*.json`。

这个边界是有意的：

- Ansible 只负责远端状态变更：目录、文件、容器、readiness。
- HTTP suite 不放进 Ansible role，避免把测试结果、JUnit/HTML 报告和响应体处理混入远端部署层。
- `ci.py smoke` 不启动模型，只验证已经由 Ansible 部署好的服务。
- `ci.py benchmark-tasks` 在 runtime container 内执行 benchmark command，保证 benchmark 工具链来自
  运行镜像而不是 Jenkins controller。
- `ci.py merge-benchmark` 只做结果合并，不重新判断拓扑或 endpoint。
- 容器以前台 `docker run` 的形式由远端 `nohup` 后台托管，避免 Ansible 被容器内 `wait` 长时间阻塞。
- cleanup playbook 是强制清理路径；即使 smoke 失败，也必须尝试停止容器并拉回日志。

## 11. 当前 MVP 边界

已经纳入 Jenkins 工作流：

- Lockable Resources 到 Ansible inventory 的转换。
- DeployCase/profile 到物理部署计划的 dry-run/真实编译。
- 每节点单容器启动。
- 每容器多实例启动。
- Ansible readiness。
- 控制侧 CSV smoke suite。
- 容器内 benchmark task 执行。
- benchmark metrics/comparison 合并到 nightly/JUnit/HTML 输入。
- 远端 artifact fetch 和 Jenkins 报告归档。

仍然保持后续演进：

- accuracy 仍沿用现有结构，尚未迁移为 Ansible 部署后的控制侧执行器。
- benchmark baseline 缺省不强制；只有 DeployCase 显式配置 `tests.benchmark.baseline` 时才执行阈值比较。
- 多节点真实 vLLM 的失败定位依赖远端日志；当前只归档压缩包和裁剪后的日志，不做日志语义分析。
- `ci.py smoke` 复用现有 CSV suite 和报告结构，后续可以把公共 smoke 执行逻辑抽成显式库函数。
