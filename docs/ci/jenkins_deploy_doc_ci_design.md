# 需求 & 设计文档

````markdown
# 基于 Jenkins 的 vLLM Ascend 可执行部署文档与 Nightly CI 报告系统需求 & 设计文档

## 1. 背景

当前 vLLM / vLLM Ascend 相关模型部署存在以下问题：

1. 模型启动参数复杂，容易变更  
   例如 TP、DP、EP、PD 分离、MTP、量化、max-model-len、max-num-batched-tokens、KV transfer 等参数组合较多。

2. 部署文档容易过期  
   文档中的启动命令经常和实际可运行命令不一致，尤其在 vLLM / vLLM Ascend 版本升级后，参数名、默认行为、支持特性可能变化。

3. CI 只验证代码，不验证文档  
   即使代码能跑，文档中的部署指导也可能已经不可用。

4. Nightly 缺少体系化报告  
   需要知道：
   - 哪些模型部署 case 跑过？
   - 哪些通过？
   - 哪些失败？
   - 哪些跳过？
   - 性能是否退化？
   - 精度是否正常？
   - 失败日志在哪里？

5. Jenkins 是当前唯一可用 CI 平台  
   当前环境没有 Buildkite / GitHub Actions GPU CI 能力，因此需要基于 Jenkins 构建完整流水线。

基于上述背景，设计一套基于 Jenkins 的 CI 系统，将结构化部署配置作为唯一源头，同时用于：

- 生成模型部署指导文档
- Jenkins CI 启动模型并验证
- Nightly 生成测试报告
- 性能和精度结果归档
- 支持后续接入 Grafana / PostgreSQL 做趋势分析

---

## 2. 目标

### 2.1 总体目标

构建一套“结构化部署配置驱动”的 CI 与文档系统。

核心思路：

```text
.ci/deploy_cases/*.yaml
        |
        |-- 生成模型部署指导文档
        |-- Jenkins 根据配置启动模型
        |-- 执行 smoke / benchmark / accuracy
        |-- 生成 nightly 报告
        |-- 输出 JUnit / HTML / JSON / CSV
````

### 2.2 具体目标

1. 使用 `.ci/` 隐藏目录统一管理 CI 内部配置、脚本、模板和 schema。
    
2. 使用结构化 YAML 描述每个模型部署 case。
    
3. 使用 Jenkinsfile 编排 CI 流程。
    
4. 支持生成类似 vLLM Ascend 风格的模型部署指导文档。
    
5. 支持 Jenkins nightly job 生成完整报告。
    
6. 支持 case 级别的启动、smoke、benchmark、accuracy 验证。
    
7. 支持性能基线对比。
    
8. 支持精度结果记录。
    
9. 支持用例失败后继续执行其他 case，最终统一汇总。
    
10. 支持按 CI 模式选择执行范围：
    
    - static
        
    - smoke
        
    - nightly
        
    - release
        
    - benchmark
        

---

## 3. 非目标

第一阶段暂不实现以下能力：

1. 不做完整的 CI 平台替换。
    
2. 不引入 Buildkite。
    
3. 不强依赖 GitHub Actions。
    
4. 不自动更新性能基线。
    
5. 不强制接入 PostgreSQL / Grafana。
    
6. 不在 PR 阶段跑所有大模型。
    
7. 不把 Jenkins 凭据、Harbor 密码、SSH 私钥写入仓库。
    
8. 不要求一开始支持所有 PD 多机复杂部署，可分阶段实现。
    

---

## 4. 目录结构设计

推荐仓库结构如下：

```text
repo-root/
  Jenkinsfile

  .ci/
    deploy_cases/
      qwen3_32b_tp8.yaml
      qwen35_122b_a10b_dp4tp2ep8.yaml
      deepseek_v3_pd.yaml

    schemas/
      deploy_case.schema.json

    templates/
      model_deploy.md.j2
      nightly_report.html.j2

    scripts/
      ci.py
      pd_proxy.py
      run_precheck.sh
      cleanup_processes.sh

    ci_lib/
      ci_tool/
        deploy_case_lib.py
        # Internal modules; do not call these directly from Jenkins.

    test_suites/
      smoke/
        common.csv
        thinking.csv

    docker/
      Dockerfile.ci

    baselines/
      benchmark/
        qwen3_32b_tp8.json
        qwen35_122b_a10b_dp4tp2ep8.json

    prompts/
      review_and_implement_ci.md

  docs/
    deploy/
      generated/
        qwen3_32b_tp8.md
        qwen35_122b_a10b_dp4tp2ep8.md

    ci/
      jenkins_deploy_doc_ci_design.md

  reports/
    # CI 运行时生成，不提交

  logs/
    # CI 运行时生成，不提交
```

### 4.1 为什么使用 `.ci/`

`.ci/` 用于隐藏 CI 内部实现，优点：

1. 仓库根目录更干净。
    
2. CI 配置、脚本、模板集中管理。
    
3. 与 `.github/`、`.gitlab/`、`.devcontainer/` 风格一致。
    
4. 不影响 Jenkins 执行。
    

注意：

`.ci/` 不是安全机制，不能存放密钥、token、密码、私钥。

---

## 5. Jenkins Job 设计

建议至少规划以下 Jenkins Job：

```text
vllm-ascend-pr-ci
  用于 PR / MR 触发
  运行 precheck + 配置校验 + 文档生成校验 + 少量 smoke

vllm-ascend-nightly-ci
  每日定时触发
  运行 nightly 级别 case
  生成完整报告

vllm-ascend-benchmark-ci
  手动或定时触发
  运行 benchmark 级别 case
  聚焦性能回归

vllm-ascend-release-ci
  发版前手动触发
  运行 release 级别 case
  作为发版强门禁
```

### 5.1 Jenkinsfile 与 Jenkins Job 的关系

`Jenkinsfile` 负责定义流水线内容。

Jenkins Job 负责：

1. 拉取代码仓库。
    
2. 找到 Jenkinsfile。
    
3. 触发流水线。
    
4. 调度 Jenkins agent。
    
5. 管理凭据。
    
6. 配置定时触发或 webhook。
    

---

## 6. Jenkins Agent 设计

推荐 Jenkins agent label：

```text
cpu
  普通 CPU 节点，用于 checkout、schema 校验、文档生成、报告生成

docker-builder
  用于构建 CI 镜像

ascend a2
  A2 8 卡 Ascend 节点，runtime job 建议 executor=1

ascend a3
  A3 16 卡 Ascend 节点，runtime job 建议 executor=1

ascend multinode
  多机 Ascend 节点，用于 PD 分离、多机多卡
```

Jenkinsfile 中通过参数指定：

```text
CPU_LABEL=cpu
DOCKER_BUILDER_LABEL=docker-builder
ASCEND_LABEL=ascend
ASCEND_LOCK_LABEL=ascend-runtime
```

### 6.1 资源锁

多节点和 PD runtime 已收敛到 Ansible inventory 路径，权威设计见
`docs/ci/jenkins_ansible_deploycase_runtime_design.md`。旧的“Jenkins-managed cluster
dry-run”只作为过渡说明保留，不再作为后续实现方向。

当前资源模型：

1. Jenkins 通过 Lockable Resources 一次性锁定 runtime 所需 Ascend 节点。
2. `.ci/scripts/ci.py lock-inventory` 将锁到的节点转换为 cluster JSON 和 Ansible inventory。
3. `.ci/scripts/ci.py compile-plan` 根据 DeployCase/profile、节点卡数和 Jenkins 参数编译物理部署计划。
4. `.ci/ansible/playbooks/deploy_cases.yml` 只消费编译结果，不重新推导 DP/TP/EP 或 P/D endpoint。
5. 每台物理机器最多启动一个 Docker 容器。
6. 一个容器内可以启动多个 vLLM 实例进程，每个实例必须使用独立 `ASCEND_RT_VISIBLE_DEVICES` 和端口。
7. 约束不满足时直接失败，不通过 fallback 隐藏错误。

如果 Jenkins 节点 executor 大于 1，建议配置 `ASCEND_LOCK_LABEL` 做整机外层保护，避免同一节点同时启动两个 runtime 容器。

仍建议对以下场景使用 Jenkins Lockable Resources Plugin 做外层保护：

1. benchmark
    
2. PD 分离
    
3. 多机测试
    
4. 大模型启动
    
5. accuracy
    

推荐使用 Jenkins Lockable Resources Plugin。

示例：

```groovy
lock(label: 'ascend-maintenance', quantity: 1) {
    sh 'python3 .ci/scripts/ci.py lock-inventory ...'
    sh 'python3 .ci/scripts/ci.py compile-plan ...'
    sh 'ANSIBLE_CONFIG=.ci/ansible/ansible.cfg ansible-playbook -i reports/runtime_plan/ansible_inventory.yml .ci/ansible/playbooks/deploy_cases.yml'
}
```

### 6.2 Jenkins 分配多节点资源

多节点 PD 或跨节点 Decode 不使用手工维护的节点列表作为默认方案。推荐让 Jenkins 通过
Lockable Resources Plugin 原子分配一组 Ascend 节点：

```groovy
lock(label: 'ascend && a3 && vllm-runtime',
     quantity: 2,
     variable: 'LOCKED_ASCEND_NODES') {
    echo env.LOCKED_ASCEND_NODES
}
```

Lockable resource 的 `name` 建议与 Jenkins node name 保持一致。每个 resource 需要配置：

| 属性 | 用途 |
| --- | --- |
| `IP` | 分布式通信和 logical service endpoint 使用的节点 IP |
| `ANSIBLE_HOST` | 可选。Jenkins controller/Ansible 可 SSH 的地址；不配置时默认使用 `IP` |
| `CARDS` | 该节点可用于 runtime 的 NPU 卡数，例如 `8` 或 `16` |
| `NETWORK_INTERFACE` | Layerwise PD 通信用网卡，例如 `enp23s0f3`；也兼容 `NIC`、`IFNAME`、`NETWORK_CARD_NAME` |

Jenkins lock step 会把锁到的资源写入环境变量：

```text
LOCKED_ASCEND_NODES=a3-node-01,a3-node-02
LOCKED_ASCEND_NODES0=a3-node-01
LOCKED_ASCEND_NODES0_PROP_IP=10.0.0.11
LOCKED_ASCEND_NODES0_PROP_ANSIBLE_HOST=a3-node-01
LOCKED_ASCEND_NODES0_PROP_CARDS=16
LOCKED_ASCEND_NODES0_PROP_NETWORK_INTERFACE=enp23s0f3
LOCKED_ASCEND_NODES1=a3-node-02
LOCKED_ASCEND_NODES1_PROP_IP=10.0.0.12
LOCKED_ASCEND_NODES1_PROP_ANSIBLE_HOST=a3-node-02
LOCKED_ASCEND_NODES1_PROP_CARDS=16
LOCKED_ASCEND_NODES1_PROP_NETWORK_INTERFACE=enp23s0f3
```

`.ci/scripts/ci.py lock-inventory` 会把这些变量转换为标准 cluster JSON 和 Ansible inventory：

```json
{
  "allocation": "jenkins-lockable",
  "node_count": 2,
  "nodes": [
    {
      "name": "a3-node-01",
      "jenkins_node": "a3-node-01",
      "ip": "10.0.0.11",
      "ansible_host": "a3-node-01",
      "cards": 16,
      "network_interface": "enp23s0f3"
    },
    {
      "name": "a3-node-02",
      "jenkins_node": "a3-node-02",
      "ip": "10.0.0.12",
      "ansible_host": "a3-node-02",
      "cards": 16,
      "network_interface": "enp23s0f3"
    }
  ]
}
```

Jenkinsfile 通过 `RUNTIME_NODE_COUNT` 控制锁定节点数量。无论 `RUNTIME_NODE_COUNT=1`
还是大于 1，都走同一套 Ansible runtime：

1. 锁定 `ASCEND_LOCK_LABEL` 对应的 `RUNTIME_NODE_COUNT` 个资源。
2. 生成 `reports/runtime_plan/runtime_cluster_nodes.json`。
3. 编译 `reports/runtime_plan/deployment_plan.json` 和
   `reports/runtime_plan/ansible_inventory.yml`。
4. 执行 `.ci/ansible/playbooks/deploy_cases.yml`。

### 6.3 Docker 单容器运行约定

Docker runtime 命令由 `.ci/scripts/ci.py compile-plan` 生成，并写入
`reports/runtime_plan/deployment_plan.json` 的 `container.command_shell` 字段。容器参数参考
`docs/source/tutorials/models/Qwen3-30B-A3B.md`。基础 device 和 driver mount 由编译器生成，
`ASCEND_DOCKER_DEVICE_ARGS` 只保留给站点追加额外参数；该参数允许为空，空字符串表示不追加任何额外 Docker 参数。

容器启动约定：

```bash
docker run --rm \
  --name vllm-ascend-ci-${BUILD_TAG} \
  --network host \
  --ipc host \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -v ${WORKSPACE}/.ci:/home/ma-user/AscendCloud/jenkins/.ci:ro \
  -v ${WORKSPACE}/reports:/home/ma-user/AscendCloud/jenkins/reports:rw \
  -v ${WORKSPACE}/logs:/home/ma-user/AscendCloud/jenkins/logs:rw \
  -v ${MODEL_ROOT}:${MODEL_ROOT}:ro \
  -w /home/ma-user/AscendCloud/jenkins \
  ${ASCEND_DOCKER_IMAGE} \
  bash -lc 'bash /home/ma-user/AscendCloud/jenkins/reports/runtime_plan/${NODE_NAME}/start_instances.sh'
```

如果站点需要额外 Docker 参数，可以通过 Jenkins 参数 `ASCEND_DOCKER_DEVICE_ARGS` 注入，例如 `--ulimit memlock=-1:-1`。为空时不会在 `docker run` 中追加占位参数。

`DRY_RUN_RUNTIME=true` 时 Jenkins 仍会锁定资源、生成 Ansible inventory、编译 deployment plan 并以
Ansible check mode 打印每个节点的 Docker 命令，但不会启动真实容器或模型。

### 6.4 PD 多服务运行约定

PD 分离不再按“一个服务一个 Jenkins task”拆分。一个 DeployCase/profile 表达一个完整拓扑，编译器把 logical service 转换为容器内多个进程：

- `type: vllm-serve`：启动真实 vLLM worker，例如 prefill、decode、prefill-decode。
- `type: command`：启动 proxy/router 等控制面进程，例如 `.ci/scripts/pd_proxy.py`。
- `profile.services.<role>.parallel` 表达 DP/TP/EP；`required_cards = dp * tp`。
- `profile.services.<role>.extra_port_count` 表达 KV transfer 等附加端口需求；编译器会和 HTTP 服务端口一起分配，避免冲突。
- `metadata.levels` 可让同一个复杂 case 同时进入 `static` 和 `smoke` 选择，避免维护两份重复 YAML。
- 编译器会为 proxy 注入模型无关的 `PREFILL_SERVERS`、`DECODE_SERVERS`。它们的值只包含 `IP:port`，不带 `http://` 前缀；proxy 从环境变量读取 P/D backend，不通过 CLI 参数传入。DeployCase 不应依赖包含具体模型名的服务变量。
- 使用 `MooncakeLayerwiseConnector` 时，编译器会从 Lockable Resources 的 `NETWORK_INTERFACE` 生成实例级 `NETWORK_CARD_NAME`、`GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME`，并把 `HCCL_IF_IP` 设置为当前节点 `IP`。缺少该属性时直接失败，不回退到不确定网卡。

当前 Qwen2.5-VL-7B-Instruct 使用 P/D 形态作为 PR/smoke 示例：prefill 和 decode 各使用 2 张卡，顺带覆盖 TP=2 的多卡并行通信路径；proxy 接收 OpenAI chat 请求，先请求 prefiller 获取 `kv_transfer_params`，再把原请求转发给 decoder。后续多 P 多 D 拓扑仍沿用同一个 `profile.services` 模型。

### 6.5 Decode 跨节点语义

一个 Decode 实例可能需要两个或更多节点。此时不要把它拆成多个 Decode service；应表达为一个
logical Decode service，包含多个 node rank：

```yaml
profile:
  services:
    decode:
      replicas: 1
      parallel:
        dp: 2
        tp: 8
        ep: 1
```

语义约定：

1. Decode head/rank0 暴露 HTTP endpoint。
2. Decode worker rank 只参与分布式推理，不加入 `DECODE_SERVERS`。
3. `DECODE_SERVERS` 只包含 logical Decode endpoint，例如 `10.0.0.11:8092`。
4. 如果需要多个 Decode 副本，才定义多个 logical Decode service，每个 service 贡献一个 endpoint。
5. 分布式启动方式不在 Jenkinsfile 中硬编码；编译器按 profile 和 inventory 注入
   `--data-parallel-address`、`--data-parallel-rpc-port`、`--data-parallel-start-rank`、
   `--data-parallel-size-local` 和实例级卡分配。

### 6.6 MTP 与图模式约定

支持 MTP 的模型应尽量在 Decode 侧启用 MTP，并在能通过验证的情况下打开 Decode 图模式。PD 场景下
Prefill 不默认启用 MTP；Prefill 负责 KV 构造，默认保持更稳定的启动路径，避免扩大 Prefill 图和显存压力。

推荐约定：

- Qwen3.5 MTP 使用 `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'`。
- Decode 侧配合 `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'` 做 PR/smoke。
- `[4]` 来自 `num_speculative_tokens + 1`，只捕获单请求 smoke 所需最小图，避免默认生成多档大图。
- Nightly 或 benchmark 可以扩展为 `[4,8,12,16]` 等真实并发档位，但必须记录图显存和启动耗时。
- 不默认给 MTP Decode 加 `--enforce-eager`；图模式失败时才能作为隔离或临时 fallback。

验收时需要分开记录：

- `text + MTP + FULL_DECODE_ONLY graph`
- `VL + MTP + FULL_DECODE_ONLY graph`
- `VL without MTP`

如果 VL 请求 HTTP 200 但返回 content 为空，不能算 VL + MTP 通过。

### 6.7 Function Call 与 Reasoning 约定

Function call 和 reasoning parser 不做全局默认开启。它们是模型族协议，必须由 DeployCase/profile
显式声明启动参数，避免错误 parser 改写普通响应。

Qwen3.5 系列推荐在 Prefill 和 Decode service 上都开启：

```text
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_xml
```

PD proxy 会先把同一 OpenAI 请求发给 Prefill。Prefill 不负责最终响应解析，但仍要通过 tool 请求校验和
chat template 渲染，因此 P/D 两侧 parser 配置必须保持一致。

Smoke suite 通过 capability 选择：

- `thinking` 加载 `.ci/test_suites/smoke/thinking.csv`，要求响应包含 `reasoning_content`。
- `tool_call` 加载 `.ci/test_suites/smoke/tool_call.csv`，要求响应包含 `tool_calls`。

HTTP 200 但缺少上述结构化字段时，CI 必须判定该 capability 失败。`tool_call.csv` 会显式关闭
`chat_template_kwargs.enable_thinking`，避免工具 JSON 被 reasoning parser 消费后无法形成 OpenAI `tool_calls`。
对于需要验证结构化响应字段非空的 suite，可使用可选列 `response_json_non_empty_path`，支持用 `|`
配置多个候选 JSON path，例如 `choices.0.message.reasoning_content|choices.0.message.reasoning`。

---

## 7. CI 模式设计

### 7.1 static

只做静态检查，不启动模型。

包括：

1. YAML 格式检查
    
2. schema 校验
    
3. vLLM 参数合法性检查
    
4. 文档生成
    
5. 文档 diff 检查
    
6. case 选择逻辑检查
    

### 7.2 smoke

适合 PR 或小规模验证。

包括：

1. static 全部检查
    
2. 启动少量模型
    
3. 访问 `/v1/models`
    
4. 发送 `/v1/chat/completions`
    
5. 检查 HTTP 200
    
6. 检查返回内容非空
    
7. 清理服务
    

### 7.3 nightly

适合每日定时任务。

包括：

1. 多模型
    
2. 多精度
    
3. 多部署形态
    
4. smoke
    
5. benchmark
    
6. accuracy
    
7. 生成完整 nightly 报告
    
8. 失败后继续执行其他 case
    
9. 最后统一汇总结果
    

### 7.4 release

发版强门禁。

包括：

1. 核心模型部署
    
2. 核心文档验证
    
3. smoke
    
4. benchmark
    
5. accuracy
    
6. 镜像 / wheel / YAML / Helm 验证
    
7. 任意 blocking case 失败则 release 失败
    

### 7.5 benchmark

性能回归专用模式。

包括：

1. 启动模型
    
2. 执行 benchmark
    
3. 对比基线
    
4. 生成性能报告
    
5. 输出 CSV / JSON
    

---

## 8. DeployCase 配置设计

### 8.1 基本结构

每个部署 case 使用 YAML 描述。

最小结构：

```yaml
apiVersion: llm-ci/v1
kind: DeployCase

metadata: {}
requirements: {}
runtime: {}
services: []
checks: {}
tests: {}
```

### 8.2 字段说明

#### metadata

描述 case 的基本信息：

```yaml
metadata:
  name: qwen3-32b-tp8
  title: Qwen3-32B 单机 8 卡部署指导
  level: nightly
  owner: llm-infer-team
  description: >
    用于验证 Qwen3-32B 在 Ascend 单机 8 卡环境下的启动和推理。
  tags:
    - qwen
    - ascend
    - tp8
```

`level` 可选：

```text
static
smoke
nightly
release
benchmark
```

#### doc

描述文档生成信息：

```yaml
doc:
  output: docs/deploy/generated/qwen3_32b_tp8.md
  audience: 推理框架开发、部署运维、性能测试人员
  difficulty: 中等
  generated_warning: true
```

#### requirements

描述硬件、软件、模型要求：

```yaml
requirements:
  hardware:
    accelerator: Ascend NPU
    product: Atlas 800 A2 / A3
    nodes: 1
    cards_per_node: 8
    jenkins_label: "ascend && 8card"
    lock_label: "ascend-8card"
    exclusive: true

  software:
    os: Linux
    python: ">=3.10,<3.12"
    cann: "按当前工程要求"
    torch: "与 vLLM Ascend 匹配"
    torch_npu: "与 vLLM Ascend 匹配"
    vllm: "当前工程版本"
    vllm_ascend: "当前工程版本"

  model:
    name: Qwen3-32B
    path: /data/models/Qwen3-32B
    served_model_name: qwen3-32b
    max_model_len: 32768
```

#### runtime

描述运行环境：

```yaml
runtime:
  image: harbor.example.com/llm/vllm-ascend-ci:${GIT_COMMIT}
  workdir: /workspace/vllm
  env:
    VLLM_PLUGINS: ascend
    PYTHONUNBUFFERED: "1"
```

#### services

描述需要启动的服务。

单机 vLLM：

```yaml
services:
  - name: server
    role: standalone
    type: vllm-serve
    host: 0.0.0.0
    port: 8000
    log_file: logs/server.log

    vllm:
      model: /data/models/Qwen3-32B
      served-model-name: qwen3-32b
      trust-remote-code: true
      dtype: bfloat16

      tensor-parallel-size: 8
      data-parallel-size: 1

      max-model-len: 32768
      max-num-batched-tokens: 32768
      max-num-seqs: 64
      gpu-memory-utilization: 0.9
```

PD 分离：

```yaml
services:
  - name: prefill
    role: prefill
    type: vllm-serve
    port: 8100
    vllm:
      model: /data/models/DeepSeek-V3
      served-model-name: deepseek-v3
      tensor-parallel-size: 8
      data-parallel-size: 1
      enable-expert-parallel: true
      kv-transfer-config:
        kv_connector: mooncake
        kv_role: producer
        engine_id: prefill-0

  - name: decode
    role: decode
    type: vllm-serve
    port: 8200
    vllm:
      model: /data/models/DeepSeek-V3
      served-model-name: deepseek-v3
      tensor-parallel-size: 2
      data-parallel-size: 4
      enable-expert-parallel: true
      kv-transfer-config:
        kv_connector: mooncake
        kv_role: consumer
        engine_id: decode-0

  - name: router
    role: router
    type: python
    port: 8000
    command:
      - python3
      - -m
      - ci_tools.pd_router
      - --prefill
      - http://127.0.0.1:8100
      - --decode
      - http://127.0.0.1:8200
```

#### checks

描述服务 ready 条件：

```yaml
checks:
  readiness:
    timeout_sec: 1800
    interval_sec: 5
    endpoints:
      - service: server
        name: list-models
        method: GET
        path: /v1/models
        expect_status: 200

      - service: server
        name: chat-completions
        method: POST
        path: /v1/chat/completions
        expect_status: 200
```

#### tests

描述 smoke、benchmark、accuracy。

```yaml
tests:
  smoke:
    enabled: true
    suites:
      include_common: true
      capabilities:
        - thinking
        - tool_call
      extra_suite_files: []

  benchmark:
    enabled: true
    tool: vllm-bench-serve
    args:
      backend: openai-chat
      endpoint: /v1/chat/completions
      dataset-name: synthetic
      input-len: 1024
      output-len: 128
      num-prompts: 32
      max-concurrency: 4
    thresholds:
      max_failed_requests: 0
      max_avg_ttft_ms: 1000
      max_avg_tpot_ms: 50
      max_tpot_regression_pct: 10
      max_ttft_regression_pct: 15
      max_throughput_drop_pct: 10

  accuracy:
    enabled: false
    mode: execute_only
    dataset: /data/acc/dataset
    thresholds:
      min_score: null
```

Smoke API 用例不再写入 DeployCase。通用用例放在 `.ci/test_suites/smoke/common.csv`，对所有模型生效；专用能力用例按 capability 拆分，例如 `.ci/test_suites/smoke/thinking.csv` 只对声明 `capabilities: [thinking]` 的 DeployCase 生效，`.ci/test_suites/smoke/tool_call.csv` 只对声明 `capabilities: [tool_call]` 的 DeployCase 生效。CSV 用例支持正向和负向请求，使用 `expected_http_status` 判定是否通过。

---

## 9. 文档生成设计

### 9.1 输入

输入为：

```text
.ci/deploy_cases/*.yaml
.ci/templates/model_deploy.md.j2
```

### 9.2 输出

输出为：

```text
docs/deploy/generated/*.md
```

### 9.3 生成内容

每篇部署文档包含：

1. 文档概述
    
2. 环境要求
    
3. 模型信息
    
4. 部署拓扑
    
5. 环境变量
    
6. 启动服务命令
    
7. vLLM config 示例
    
8. 服务验证
    
9. benchmark 验证
    
10. accuracy 验证
    
11. 参数说明
    
12. 停止服务
    
13. 常见问题
    
14. 注意事项
    

### 9.4 文档一致性检查

Jenkins 中执行：

```bash
python3 .ci/scripts/ci.py render-docs \
  --cases ".ci/deploy_cases/*.yaml" \
  --level nightly \
  --output-dir docs/deploy/generated

git diff --exit-code docs/deploy/generated
```

如果有 diff，说明：

1. 配置改了但文档没更新。
    
2. 模板改了但文档没重新生成。
    
3. 文档被手工改动导致与配置不一致。
    

此时 CI 应失败。

---

## 10. Jenkinsfile 流程设计

推荐 Jenkinsfile stages：

```text
Checkout
  拉取代码，确定 commit、image tag、case level

Precheck
  运行基础 lint / format / unit 检查

Validate Deploy Case Schema
  校验 .ci/deploy_cases/*.yaml

Generate Deploy Docs
  根据 deploy case 生成文档
  可选 git diff 检查

Build CI Image
  可选构建 CI 镜像

Select Deploy Cases
  按 CI_MODE / CASE_LEVEL / RUN_ALL 选择 case

Static Validate Selected Cases
  对选中的 case 做静态检查

Run vLLM Ascend Deploy Cases
  按 case 启动服务，执行 smoke

Run Benchmark
  可选执行 benchmark

Run Accuracy
  可选执行 accuracy

Generate Nightly Report
  生成 JUnit / HTML / JSON / CSV 报告

Archive
  归档 logs / reports / generated docs
```

---

## 11. CI CLI 职责设计

Jenkins 和本地人工验证只调用统一入口：

```bash
python3 .ci/scripts/ci.py <command> [args]
```

`ci_tool/` 下的 Python 文件是内部实现模块，不作为稳定对外接口。这样后续重构实现时只需要保持
`ci.py` 的命令面稳定，Jenkinsfile 不需要跟着多个脚本文件名变化。

### 11.1 ci.py validate

职责：

1. 读取 YAML。
    
2. 校验必填字段。
    
3. 校验 metadata.level。
    
4. 校验 services 不为空。
    
5. 校验 service name 不重复。
    
6. 校验端口不重复。
    
7. 校验 vllm-serve 类型必须有 vllm.model。
    
8. 校验 tests.smoke.suites 选择的 CSV 文件存在、header 合法、id 唯一。
    
9. 校验 2xx smoke 用例的 payload.model 和 served-model-name 一致。
10. 输出校验结果 JSON。
    

输入：

```bash
python3 .ci/scripts/ci.py validate \
  --cases ".ci/deploy_cases/*.yaml" \
  --level nightly \
  --output reports/validated_cases.json
```

### 11.2 ci.py render-docs

职责：

1. 读取 deploy case。
    
2. 使用 Jinja2 模板生成 markdown。
    
3. 输出到 docs/deploy/generated。
    
4. 支持按 level 过滤。
    

### 11.3 ci.py select

职责：

1. 按 level 选择 case。
    
2. 支持 RUN_ALL。
    
3. 支持 tag include / exclude，后续可扩展。
    
4. 输出 selected_cases.txt。
    

### 11.4 ci.py static-validate

职责：

1. 检查模型路径是否存在，可配置是否强制。
    
2. 检查 vLLM 参数是否能渲染。
    
3. 检查布尔、字典、列表参数是否可序列化。
    
4. 检查 TP * DP 是否不明显超过卡数。
    
5. 输出 static_validate.json。
    

### 11.5 Ansible runtime execution scripts

职责：

1. `ci.py smoke` 读取 selected case 和 `deployment_plan.json`。
    
2. 对每个 case 选择 proxy/head endpoint，等待 readiness。
    
3. 根据 `tests.smoke.suites` 加载 `.ci/test_suites/smoke/*.csv` 并执行正向/负向 smoke 用例。
    
4. `ci.py benchmark-tasks` 在 runtime container 内执行编译出的 benchmark command。
    
5. `ci.py merge-benchmark` 将远端 benchmark stage JSON 合并回 case result。
    
6. 所有 case result 写入 `reports/nightly/case_results/*.json`。
    
7. nightly/benchmark 模式不 fail-fast，最后统一汇总报告。

### 11.6 ci.py compile-plan

职责：

1. 读取 selected DeployCase 和 Jenkins Lockable Resources 生成的 cluster inventory。
2. 根据 DeployCase/profile 的模型、PD 拓扑和 DP/TP/EP 编译物理部署计划。
3. 生成 `reports/runtime_plan/deployment_plan.json`。
4. 生成 `reports/runtime_plan/ansible_inventory.yml`。
5. 生成每个节点的 `start_instances.sh` 和实例级 env 文件。
6. 为每个节点生成唯一 Docker runtime 命令。
7. 追加 Ascend 基础 device、driver mount、`.ci/reports/logs/MODEL_ROOT` mount。
8. 仅当 `--extra-docker-args` 非空时追加站点额外 Docker 参数。
9. 直接失败不满足约束的 profile，例如空 Docker image、缺模型路径、TP 无法落到节点、卡/端口冲突。
    

关键要求：

nightly 模式不能 fail-fast。

一个 case 失败后，要继续执行后续 case，最终统一汇总。

### 11.7 Ansible playbook

职责：

1. 使用 `.ci/ansible/playbooks/deploy_cases.yml`。
2. 只消费编译后的 `ansible_inventory.yml` 和节点 plan。
3. 在每个 Ascend 节点创建 `/home/ma-user/AscendCloud/jenkins`。
4. 复制 `.ci` 和当前节点的 runtime plan。
5. `DRY_RUN_RUNTIME=true` 时只打印 Docker 命令。
6. `DRY_RUN_RUNTIME=false` 时执行编译器生成的 Docker 命令。

Ansible 不负责重新推导 DP/TP/EP、P/D endpoint 或卡分配。

### 11.8 ci.py prepare-logs

职责：

1. 读取 `reports/nightly/case_results/*.json` 中引用到的日志路径。
2. 按 Jenkins 参数 `LOG_ARCHIVE_MODE` 生成有界日志副本到 `reports/logs/`。
3. `failed-tail` 模式只保留失败 case 日志 tail，避免 PR/smoke 归档过大。
4. `full` 模式复制完整引用日志，供 nightly/release/benchmark 使用。
5. 写出 `reports/nightly/log_artifacts.json`，并回写 case result 中的 `artifacts.archived_server_log` / `artifacts.archived_server_logs`。

### 11.8 ci.py junit

职责：

1. 读取 case_results/*.json。
    
2. 生成 JUnit XML。
    
3. 每个 case 拆成多个 testcase：
    
    - startup
        
    - readiness
        
    - smoke
        
    - benchmark
        
    - accuracy
        

### 11.9 ci.py report

职责：

1. 读取 case_results/*.json。
    
2. 生成 HTML 报告。
    
3. 生成 summary.json。
    
4. 生成 cases.csv。
    
5. 生成 benchmark.csv。
    
6. 生成 accuracy.csv。
    

### 11.10 ci.py compare-benchmark

职责：

1. 读取当前 benchmark 结果。
    
2. 读取 baseline。
    
3. 计算 delta。
    
4. 判断是否超过阈值。
    
5. 输出比较结果。
    

---

## 12. Nightly 报告设计

### 12.1 报告产物

```text
reports/
  nightly/
    index.html
    summary.json
    junit.xml
    cases.csv
    benchmark.csv
    accuracy.csv
    environment.json
    log_artifacts.json

    case_results/
      qwen3_32b_tp8.json
      qwen35_122b_a10b_dp4tp2ep8.json
      deepseek_v3_pd.json

  logs/
    # Jenkins artifact 中的有界日志副本，由 ci.py prepare-logs 生成
```

### 12.2 HTML 报告内容

首页包含：

1. 任务信息
    
    - job name
        
    - build number
        
    - commit
        
    - branch
        
    - image
        
    - date
        
    - CI mode
        
2. 环境信息
    
    - OS
        
    - Python
        
    - CANN
        
    - torch
        
    - torch-npu
        
    - vLLM
        
    - vLLM Ascend
        
    - NPU 信息
        
3. 总览
    
    - total cases
        
    - passed
        
    - failed
        
    - skipped
        
    - unstable
        
4. 失败列表
    
    - case name
        
    - stage
        
    - reason
        
    - log path
        
5. 性能退化列表
    
    - case name
        
    - metric
        
    - current
        
    - baseline
        
    - delta
        
    - threshold
        
6. 精度列表
    
    - case name
        
    - dataset
        
    - mode
        
    - score
        
    - baseline
        
    - result
        
7. case 明细表
    
    - model
        
    - topology
        
    - precision
        
    - smoke
        
    - benchmark
        
    - accuracy
        
    - TTFT
        
    - TPOT
        
    - throughput
        

### 12.3 日志归档策略

运行时服务日志仍写入本地 `logs/`，便于在 Ascend 节点现场排障；Jenkins artifact 不直接归档原始 `logs/**/*`，而是归档 `.ci/scripts/ci.py prepare-logs` 生成的有界副本：

```text
reports/
  logs/
    deploy/
      <case>/
        server.log
  nightly/
    log_artifacts.json
```

默认策略由 `LOG_ARCHIVE_MODE=auto` 控制：

- `CI_MODE=nightly` / `release` / `benchmark`：归档当前 case result 引用到的完整日志。
- `CI_MODE=pr` / `static` / `smoke`：只归档失败 case 的日志 tail，默认每个文件保留最后 `LOG_ARCHIVE_TAIL_BYTES=5242880` 字节。
- `LOG_ARCHIVE_MODE=none`：不归档日志，仅生成空 manifest。

该策略避免把大模型长日志直接压入 Jenkins，同时保证失败 PR 有足够 tail 信息定位问题。完整原始日志只作为当前 build workspace 的临时运行产物，不作为长期 artifact 依赖。

### 12.4 JUnit 设计

每个 case 对应一个 testsuite。

每个阶段对应一个 testcase：

```text
qwen3-32b-tp8.startup
qwen3-32b-tp8.readiness
qwen3-32b-tp8.smoke
qwen3-32b-tp8.benchmark
qwen3-32b-tp8.accuracy
```

状态：

```text
passed
failed
skipped
```

---

## 13. 性能基线设计

### 13.1 基线位置

第一阶段使用文件存储：

```text
.ci/baselines/benchmark/
  qwen3_32b_tp8.json
  qwen35_122b_a10b_dp4tp2ep8.json
```

也可以使用 NFS：

```text
/data/ci_baselines/benchmark/
```

### 13.2 基线格式

```json
{
  "case_name": "qwen3-32b-tp8",
  "commit": "baseline_commit",
  "metrics": {
    "avg_ttft_ms": 420.0,
    "avg_tpot_ms": 18.0,
    "output_throughput_tokens_per_s": 2300.0
  }
}
```

### 13.3 对比规则

```text
TTFT 越低越好
TPOT 越低越好
ITL 越低越好
throughput 越高越好
failed_requests 必须等于 0
```

阈值：

```yaml
thresholds:
  max_failed_requests: 0
  max_ttft_regression_pct: 15
  max_tpot_regression_pct: 10
  max_itl_regression_pct: 10
  max_throughput_drop_pct: 10
```

### 13.4 基线更新

第一阶段不自动更新基线。

后续可增加独立 job：

```text
vllm-ascend-update-baseline
```

只允许手动触发。

---

## 14. 精度结果设计

### 14.1 execute_only 模式

只验证 accuracy 脚本可执行，模型可正常响应，不自动判断分数。

报告中展示：

```text
mode = execute_only
score = N/A
failed_samples = 0
result = PASS
```

PASS 的含义是“精度任务执行成功”，不是“精度达到目标”。

### 14.2 score_based 模式

后续可扩展自动判分：

```yaml
accuracy:
  enabled: true
  mode: score_based
  dataset: /data/acc/dataset
  metric: accuracy
  thresholds:
    min_score: 0.75
    max_score_drop: 0.01
```

---

## 15. 错误处理设计

### 15.1 case 级别错误

每个 case 需要记录：

1. status
    
2. failure_stage
    
3. failure_reason
    
4. logs
    
5. artifacts
    

示例：

```json
{
  "case_name": "deepseek-v3-pd",
  "status": "failed",
  "failure_stage": "startup",
  "failure_reason": "decode service readiness timeout",
  "artifacts": {
    "decode_log": "logs/deploy/deepseek-v3-pd/decode.log"
  }
}
```

### 15.2 nightly 继续执行

nightly 模式要求：

1. case1 失败后继续 case2。
    
2. 所有 case 跑完后生成报告。
    
3. 如果存在 blocking failure，最后 exit 1。
    
4. Jenkins job 标记失败或 unstable。
    

### 15.3 清理机制

每个 case 执行结束必须清理：

1. vLLM 进程
    
2. router 进程
    
3. benchmark 进程
    
4. 临时文件
    
5. 占用端口
    

建议：

```bash
pkill -f "vllm serve" || true
pkill -f "pd_router" || true
```

更稳妥的是记录 pid 文件，只 kill 当前 case 启动的进程。

---

## 16. 安全设计

不得将以下内容提交到仓库：

1. Harbor 密码
    
2. Git token
    
3. SSH 私钥
    
4. 模型仓库 token
    
5. AK/SK
    
6. Jenkins API token
    

这些应放入 Jenkins Credentials。

`.ci/` 不是安全边界，仅用于组织目录。

---

## 17. `.gitignore` 建议

```gitignore
reports/
logs/
*.pyc
__pycache__/
.pytest_cache/
.coverage
```

如果 Docker 需要 COPY `.ci/`，要注意 `.dockerignore`：

```dockerignore
.*
!.ci/
!.ci/**
```

---

## 18. 验收标准

### 18.1 static 模式

执行：

```bash
python3 .ci/scripts/ci.py validate --cases ".ci/deploy_cases/*.yaml"
python3 .ci/scripts/ci.py render-docs --cases ".ci/deploy_cases/*.yaml"
python3 .ci/scripts/ci.py select --cases ".ci/deploy_cases/*.yaml"
```

要求：

1. 配置校验通过。
    
2. 文档成功生成。
    
3. selected_cases.txt 正确生成。
    
4. 无 Python 异常。
    

### 18.2 smoke 模式

要求：

1. 至少一个 smoke case 被选中。
    
2. vLLM 服务可以启动。
    
3. `/v1/models` 返回 200。
    
4. `/v1/chat/completions` 返回 200。
    
5. 返回内容非空。
    
6. 日志被归档。
    
7. case result JSON 被生成。
    

### 18.3 nightly 模式

要求：

1. 多个 nightly case 被执行。
    
2. 一个 case 失败不影响后续 case。
    
3. 生成 JUnit XML。
    
4. 生成 HTML 报告。
    
5. 生成 summary.json。
    
6. 生成 benchmark.csv。
    
7. 生成 accuracy.csv。
    
8. Jenkins 页面可以查看测试结果和 HTML 报告。
    

### 18.4 benchmark 模式

要求：

1. benchmark 能执行。
    
2. 结果 JSON 被解析。
    
3. 能对比 baseline。
    
4. 超过阈值时 case 标记失败。
    
5. 报告中展示当前值、基线、变化百分比和阈值。
    

### 18.5 文档生成

要求：

1. 文档由 deploy case 自动生成。
    
2. 文档包含启动命令。
    
3. 文档包含服务验证命令。
    
4. 文档包含 benchmark 命令。
    
5. 文档包含参数说明。
    
6. Jenkins 可检查 generated docs 是否有未提交 diff。
    

---

## 19. 分阶段开发计划

### 第一阶段：基础框架

目标：跑通静态检查和文档生成。

交付：

1. `.ci/` 目录结构
    
2. DeployCase YAML 示例
    
3. schema 校验脚本
    
4. 文档生成模板
    
5. 文档生成脚本
    
6. Jenkinsfile static 流程
    

### 第二阶段：smoke 启动验证

目标：跑通一个小模型或基础模型 smoke。

交付：

1. `ci.py compile-plan` 支持单服务 vLLM 和 PD profile
    
2. Ansible role 在每节点启动一个 runtime container 并 wait readiness
    
3. `ci.py smoke` 执行 smoke request
    
4. `cleanup_runtime.yml` 清理远端容器
    
5. case result JSON
    

### 第三阶段：nightly 报告

目标：跑完多个 case 并生成报告。

交付：

1. continue-on-error
    
2. ci.py junit
    
3. ci.py report
    
4. HTML 报告
    
5. JSON / CSV 输出
    

### 第四阶段：benchmark

目标：支持性能回归检查。

交付：

1. benchmark 执行
    
2. baseline 读取
    
3. delta 计算
    
4. 阈值判断
    
5. benchmark.csv
    

### 第五阶段：accuracy

目标：支持精度任务执行和结果归档。

交付：

1. execute_only 模式
    
2. score_based 预留
    
3. accuracy.csv
    
4. 报告展示
    

### 第六阶段：PD 分离和多服务

目标：支持复杂部署。

交付：

1. 多 service 启动
    
2. startup_order
    
3. router 支持
    
4. prefill / decode 日志归档
    
5. PD 文档模板增强
    

---

## 20. 风险与规避

### 风险 1：Jenkins agent label 不匹配

表现：

```text
Still waiting to schedule task
```

规避：

1. Jenkins 节点正确配置 label。
    
2. Jenkinsfile 参数化 label。
    
3. 文档说明节点要求。
    

### 风险 2：多个任务抢同一台 Ascend 机器

规避：

1. 使用 Lockable Resources。
    
2. benchmark 和大模型测试强制加锁。
    

### 风险 3：文档生成依赖过多

规避：

1. Python 依赖尽量少。
    
2. 第一阶段只依赖 PyYAML、Jinja2。
    
3. requirements 单独放 `.ci/requirements.txt`。
    

### 风险 4：nightly 失败后报告不完整

规避：

1. `ci.py smoke` 使用 continue-on-error。
    
2. 每个 case 都写独立 JSON。
    
3. 最终统一生成报告。
    

### 风险 5：性能波动导致误报

规避：

1. benchmark 节点独占。
    
2. 固定机器、镜像、CANN、模型路径。
    
3. 设置合理阈值。
    
4. 支持多次运行取中位数，后续扩展。
    

---

## 21. 总结

本方案的核心是：

```text
结构化部署配置是唯一源头
Jenkins 根据配置执行验证
文档由配置自动生成
nightly 生成完整报告
性能和精度结果可追溯
```

最终希望达到：

1. 文档不再手工维护启动命令。
    
2. CI 可以反向验证文档可用。
    
3. Nightly 能看清每个模型、每种部署形态的健康状态。
    
4. 性能退化可以被及时发现。
    
5. 后续可以平滑扩展到 PD 分离、多机、多精度、多模型。

---

## 22. MVP 落地说明

当前第一版实现按最小可用闭环落地，边界如下：

1. CI 内部实现位于 `.ci/`，根目录只保留 `Jenkinsfile`。
2. 默认 `RUN_ASCEND=false`，静态和 PR 流程会在 Ascend 节点执行全量 pytest UT、YAML 校验、文档生成、case 选择和静态 CLI 校验。
3. Runner 支持单服务 `vllm-serve`，也支持同一容器内的多服务 `vllm-serve` + `command` 拓扑；Qwen2.5-VL-7B-Instruct P/D 是当前 PR/smoke 示例。
4. Accuracy 只支持 `execute_only`，命令返回 0 即通过，报告中 `score=N/A`。
5. Runtime 产物统一写入 `reports/` 和 `logs/`，两者已加入 `.gitignore`。
6. `static`、`smoke`、`nightly` 至少各有一个 case，保证 Jenkins 默认 `CASE_LEVEL=auto` 不会选空。
7. 如果 Jenkins 设置 `MODEL_ROOT`，runner 会优先使用 case 中的 `vllm.local_model_path`，避免误触发远程大模型下载。
8. Jenkinsfile 通过 `stash`/`unstash` 显式传递 `reports/`、`logs/` 和生成文档，避免不同 agent/workspace 间产物丢失。
9. PR 默认在 `ASCEND_LABEL` 节点运行全量 `tests/ut`；`tests/ut/conftest.py` 的 `npu_test` decorator 会自动加 `pytest.mark.npu`，便于必要时用 `UT_MARK_EXPR` 做临时过滤。

### Jenkins Job 配置

建议创建 Pipeline job，使用仓库根目录 `Jenkinsfile`。常用参数：

| 参数 | 建议值 | 说明 |
| --- | --- | --- |
| `CI_MODE` | `static` / `smoke` / `nightly` | 控制默认 case level |
| `CASE_LEVEL` | `auto` | 由 `CI_MODE` 推导 case level |
| `DEPLOY_CASE_GLOB` | `.ci/deploy_cases/*.yaml` | DeployCase 输入 |
| `RUN_ASCEND` | `false` for PR, `true` for smoke/nightly | 是否启动真实 vLLM 服务 |
| `RUN_BENCHMARK` | `false` by default | 只在 benchmark/nightly 节点上开启 |
| `RUN_UT` | `true` | 是否运行 pytest UT |
| `UT_TEST_PATH` | `tests/ut` | pytest UT 路径 |
| `UT_MARK_EXPR` | 空 | pytest marker 表达式；默认空表示全量 UT，包括 NPU UT |
| `UT_PYTEST_ARGS` | `-q` | pytest 附加参数；建议只放简单无空格参数 |
| `MODEL_ROOT` | Jenkins agent 本地模型根目录 | 可为空，或用于本地模型镜像 |
| `ASCEND_LABEL` | Ascend agent label | 例如 `ascend && 8card` |
| `ASCEND_LOCK_LABEL` | Lockable Resources label | 单节点时可做整机锁；多节点时作为 Jenkins 分配资源池 |
| `RUNTIME_NODE_COUNT` | runtime 需要的 Ascend 节点数 | 默认 `1`；大于 `1` 时通过 Lockable Resources 一次性锁定多台节点 |
| `ASCEND_DOCKER_IMAGE` | Ascend runtime Docker image | 为空时使用 `REGISTRY/IMAGE_TAG` 推导 |
| `ASCEND_DOCKER_DEVICE_ARGS` | 可选额外 Docker 参数 | 可为空；基础 Ascend device/mount 已由 `ci.py compile-plan` 生成 |
| `DRY_RUN_RUNTIME` | Docker runtime dry-run | 默认 `true`，验证命令但不启动模型 |
| `LOG_ARCHIVE_MODE` | 日志归档策略 | 默认 `auto`；PR/smoke 只归档失败日志 tail，nightly/release/benchmark 归档完整引用日志 |
| `LOG_ARCHIVE_TAIL_BYTES` | tail 模式每个日志保留字节数 | 默认 `5242880` |

UT stage 运行在 `ASCEND_LABEL` 节点，并复用 `ASCEND_LOCK_LABEL` 作为整机锁；该节点需要预装与当前分支匹配的 vLLM / vLLM Ascend / torch-npu Python 环境。

### 本地静态验证

```bash
python3 .ci/scripts/ci.py validate --cases ".ci/deploy_cases/*.yaml" --output reports/validated_cases.json
python3 -m pytest -q tests/ut --junitxml=reports/pytest/ut.xml
python3 .ci/scripts/ci.py select --cases ".ci/deploy_cases/*.yaml" --level smoke --output reports/selected_cases.txt
python3 .ci/scripts/ci.py render-docs --cases ".ci/deploy_cases/*.yaml" --level smoke --output-dir docs/deploy/generated
python3 .ci/scripts/ci.py static-validate --case-list reports/selected_cases.txt --output reports/static_validate.json
LOCKED_ASCEND_NODES=a2-node-01 \
LOCKED_ASCEND_NODES0=a2-node-01 \
LOCKED_ASCEND_NODES0_PROP_IP=10.0.0.11 \
LOCKED_ASCEND_NODES0_PROP_CARDS=8 \
python3 .ci/scripts/ci.py lock-inventory \
  --variable LOCKED_ASCEND_NODES \
  --output-json reports/runtime_plan/runtime_cluster_nodes.json \
  --output-inventory reports/runtime_plan/locked_ansible_inventory.yml
python3 .ci/scripts/ci.py compile-plan \
  --case-list reports/selected_cases.txt \
  --inventory-json reports/runtime_plan/runtime_cluster_nodes.json \
  --output-dir reports/runtime_plan \
  --model-root /path/to/model/root \
  --docker-image '${ASCEND_DOCKER_IMAGE}' \
  --extra-docker-args "" \
  --dry-run
python3 .ci/scripts/ci.py prepare-logs --case-results reports/nightly/case_results --logs-dir logs --output-dir reports/logs --mode failed-tail --tail-bytes 5242880
python3 .ci/scripts/ci.py junit --input reports/nightly/case_results --output reports/nightly/junit.xml
python3 .ci/scripts/ci.py report --input reports/nightly/case_results --output reports/nightly/index.html
```
