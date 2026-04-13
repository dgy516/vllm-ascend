# MoE Tensor Parallel over `DP x TP` for Qwen3.5-122B-A10B

## Background

For `Qwen3.5-122B-A10B`, the current `DP4 + TP2 + EP8` deployment fits the routed expert weights, but decode throughput can be limited by expert load imbalance. In this workload, online EPLB is not yet stable enough to be treated as a dependable production solution.

The obvious fallback, disabling expert parallelism, is not viable for this model. With global `TP2`, routed expert weights remain too large to fit on each rank even with quantization. Therefore the practical design space is reduced to:

- keep `EP8` and improve expert placement/load balancing, or
- keep attention on `TP2` and move routed MoE to a separate `TP8` execution mode.

This document defines the second option.

## Goals

- Support `DP4 + TP2 + MoE-TP8` for large routed MoE models, starting with `Qwen3.5-122B-A10B`.
- Preserve the existing attention path and global topology for non-MoE modules.
- Eliminate dependence on dynamic EPLB for the routed MoE path.
- Reuse the existing `ALLGATHER`-style MoE flow where it is structurally correct.
- Preserve `NZ` weight layout support for routed expert kernels. The new mode must not regress to an `ND`-only implementation.
- Beat the current `DP4 + TP2 + EP8` deployment with EPLB disabled on the target decode workload. If it does not outperform that baseline, the feature is not worth landing.

## Non-goals

- Replacing the existing `EP` path for other MoE models.
- Supporting `MC2`, `ALLTOALL`, or `FUSED_MC2` in the new mode.
- Supporting `FLASHCOMM1` for routed MoE in this mode.
- Supporting dynamic EPLB, shared-expert DP, or mix placement in the first version.
- Covering every runtime variant in the first patch set. The first target is the main v1 runner path for decode and standard graph mode.
- Accepting a functionally correct but slower implementation as the final result. A slower mode may exist as an intermediate debug-only step, but it is not an acceptable delivery target.

## Why the Existing `ALLGATHER` Path Is Not Enough

The current `ALLGATHER` MoE path is the right starting point because it already implements a gather -> local routed expert compute -> reduce style pipeline. However, it assumes one of two ownership models:

- `EP`: each rank owns a subset of experts, or
- `no-EP`: routed MoE follows the existing MoE TP behavior from upstream assumptions.

For `DP4 + TP2 + MoE-TP8`, neither assumption is sufficient:

1. Routed expert weights must be sharded over `8` ranks, not over the global `TP2` group.
2. In the normal `TP2` transformer path, TP peers may hold logically duplicated MoE inputs. Naively expanding the current `dp_group` all-gather to `dp x tp` would duplicate tokens and double traffic.
3. The current final reduction only understands the original tensor-parallel group. `MoE-TP8` must reduce over a dedicated `8`-rank group and then return to the outer `DP4 + TP2` layout.

Therefore the implementation must introduce explicit `MoE-TP` semantics instead of only changing one communication group.

## Proposed Topology

For each pipeline stage, define three new process-group concepts:

- `moe_tp_group`: all ranks in the stage participating in routed MoE tensor parallelism. In phase 1 this is exactly `data_parallel_size * tensor_parallel_size = 8`.
- `moe_source_group`: one source rank per DP replica, used to collect unique MoE inputs. In phase 1 this contains all ranks with `tp_rank == 0`, so its size is `4`.
- `moe_peer_group`: the existing local `TP2` pair inside one DP replica, used to fan out gathered MoE inputs and fan in local outputs.

Phase 1 intentionally constrains:

- `moe_tensor_parallel_size == data_parallel_size * tensor_parallel_size`
- `tensor_parallel_size == 2`
- one source rank per DP replica

These constraints keep the first implementation simple and avoid introducing a second layer of routing policy.

## Configuration

Add a dedicated routed-MoE parallel configuration under `additional_config`:

```json
{
  "moe_parallel_config": {
    "mode": "tensor_parallel",
    "moe_tensor_parallel_size": 8,
    "source_tp_rank": 0
  }
}
```

Validation rules:

- `mode == "tensor_parallel"` enables the new path.
- `moe_tensor_parallel_size` must equal `data_parallel_size * tensor_parallel_size` in phase 1.
- `enable_expert_parallel` must be `False`.
- `dynamic_eplb`, `enable_shared_expert_dp`, and `mix_placement` must be disabled.
- `FLASHCOMM1` must be disabled for routed MoE in this mode.
- The model must be a routed MoE model with no dependency on EP-only custom dispatch/combine kernels.

## Weight Layout

In `MoE-TP8` mode, routed experts are no longer partitioned by expert ownership. Every rank sees the full logical expert set, and each expert is tensor-sharded over the `moe_tp_group`.

For expert FFN weights:

- `w13_weight`: shard on the intermediate output dimension
- `w2_weight`: shard on the intermediate input dimension

For a model with:

- number of experts: `E`
- hidden size: `H`
- expert intermediate size: `I`
- `moe_tp_size = 8`

local shard shapes become:

- `w13_weight`: `[E, 2 * I / 8, H]`
- `w2_weight`: `[E, H, I / 8]`

Quantization metadata follows the local shard shape. For `W8A8`, this means all scale tensors are also sharded per expert per local intermediate shard. No EPLB expert list or `log2phy` remapping is used in this mode.

### NZ Requirement

`MoE-TP8` must preserve the routed-expert kernel assumptions already used in the current implementation:

- unquantized routed MoE weights should continue to use `maybe_trans_nz` where the current path relies on NZ-backed grouped matmul performance
- `W8A8` routed MoE weights and scales must continue to support `FRACTAL_NZ`
- the first implementation must not introduce a correctness-only fallback that stores routed expert weights in plain `ND` and leaves NZ optimization for later

This is a hard requirement because the feature only has value if it improves throughput over the no-EPLB `EP8` baseline.

## Routed MoE Forward Path

### 1. Input Canonicalization

Attention and other non-MoE modules continue to run under the existing `DP4 + TP2` semantics.

Before routed MoE execution:

- only `source_tp_rank` within each DP replica contributes unique MoE inputs to `moe_source_group`
- `hidden_states` and `router_logits` are all-gathered across `moe_source_group`
- the gathered global batch is broadcast within each local `moe_peer_group`

This produces one logical global MoE batch shared by all `8` ranks without double-counting TP peers.

### 2. Routing

All ranks compute identical `topk_ids` and `topk_weights` from the gathered `router_logits`.

The dispatcher reuses the current `ALLGATHER` reorder/unpermute structure, but it must not apply EP-local expert masking. In `MoE-TP8`, every rank participates in every logical expert, because each rank owns a shard of every expert.

### 3. Expert Compute

Each rank runs local grouped matmul on its tensor shard:

- local `gate_up_proj`
- activation
- local `down_proj`

The result at this point is a partial contribution of the full expert output.

All local expert compute paths in phase 1 must retain NZ-compatible weight layout. The implementation should reuse the current routed MoE weight post-processing pattern instead of creating a new weight storage format for `MoE-TP8`.

### 4. MoE-TP Reduction

Partial expert outputs are reduced across `moe_tp_group`.

Phase 1 uses correctness-first behavior:

- `all_reduce` over `moe_tp_group`
- source ranks slice out their local DP batch
- sliced local outputs are broadcast to the paired TP rank in `moe_peer_group`

This returns the tensor layout to the original `DP4 + TP2` execution model expected by the next transformer block.

The first version should optimize for correctness and fit. More aggressive `reduce_scatter`-based return paths can be added later.

## Required Code Changes

### Distributed Groups

Add `moe_tp_group` and `moe_source_group` construction to `vllm_ascend/distributed/parallel_state.py`.

Phase 1 group construction rules:

- `moe_tp_group`: all ranks in one pipeline stage
- `moe_source_group`: one rank per DP replica at `source_tp_rank`

Also add public getters:

- `get_moe_tp_group()`
- `get_moe_source_group()`
- `get_moe_peer_group()`

`moe_peer_group` may map to the existing local `TP2` relation internally, but the MoE path should access it through a dedicated helper instead of implicitly reusing unrelated TP code.

### Configuration and Validation

Update:

- `vllm_ascend/ascend_config.py`
- `vllm_ascend/platform.py`

to parse and validate `moe_parallel_config`, and to reject incompatible features early.

Proposed runtime interface:

```python
additional_config = {
    "moe_parallel_config": {
        "mode": "tensor_parallel",
        "moe_tensor_parallel_size": 8,
        "source_tp_rank": 0,
    }
}
```

Implementation rule:

- add a dedicated config object under `AscendConfig`
- expose helpers such as `moe_tp_mode_enabled()`
- do not leave string-based config reads scattered across call sites

### Forward-Context Selection

Extend `MoECommType` with a new mode, for example `MOE_TP_ALLGATHER`.

Update `select_moe_comm_method` to select this mode when `moe_parallel_config.mode == "tensor_parallel"`.

This new mode must bypass the current EP-based `ALLGATHER / MC2 / ALLTOALL` selection logic.

### Prepare/Finalize

Add a new prepare/finalize implementation instead of overloading the existing DP all-gather path:

- gather unique inputs on `moe_source_group`
- local peer broadcast to `TP2`
- reduce on `moe_tp_group`
- source-rank slice back to local DP batch
- local peer broadcast of the sliced result

### Token Dispatcher

Implement a dedicated dispatcher for `MoE-TP8`, reusing the current all-gather reorder logic but with:

- no EP-local active expert range
- no `expert_map` masking
- no EPLB remapping

### Weight Creation and Loading

Update routed MoE weight creation/loading so that `num_experts` is global and intermediate dimensions are sharded on `moe_tp_group`.

This affects:

- unquantized routed MoE
- `W8A8`
- `W4A8` if it is intended to be supported in phase 1

For each supported quantization mode, the implementation must explicitly define:

- shard shape before NZ transform
- NZ transform point during `process_weights_after_loading`
- local scale tensor shape after sharding
- whether grouped matmul kernels consume the sharded tensor directly or require a per-rank layout fixup

### Final Reduction Hook

The custom `maybe_all_reduce_tensor_model_parallel` helper must treat `MOE_TP_ALLGATHER` like the current `MC2/ALLTOALL` family, because the final MoE output has already been reduced on `moe_tp_group`.

Without this change, the runtime would incorrectly perform an extra all-reduce on the original `TP2` group.

## Implementation Sequence

The implementation should follow this order. Do not start quantized kernel adaptation before the unquantized path is numerically correct and the group topology is stable.

### Milestone 0: Config and Guards

Files:

- `vllm_ascend/ascend_config.py`
- `vllm_ascend/platform.py`

Deliverables:

- parse `moe_parallel_config`
- validate phase-1 constraints
- reject unsupported combinations at startup

Exit criteria:

- supported configs start cleanly
- unsupported configs fail before model loading

### Milestone 1: Process Groups

Files:

- `vllm_ascend/distributed/parallel_state.py`

Deliverables:

- initialize `moe_tp_group`
- initialize `moe_source_group`
- add getter APIs
- add destroy/reset logic

Exit criteria:

- deterministic rank membership
- unit tests cover group shapes and rank assignment

### Milestone 2: BF16 Forward Skeleton

Files:

- `vllm_ascend/ascend_forward_context.py`
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`
- `vllm_ascend/ops/fused_moe/prepare_finalize.py`
- `vllm_ascend/ops/fused_moe/token_dispatcher.py`
- `vllm_ascend/ops/register_custom_ops.py`

Deliverables:

- add `MOE_TP_ALLGATHER`
- add dedicated prepare/finalize implementation
- add dispatcher variant without EP-local ownership logic
- skip the old TP2 all-reduce after MoE finalize

Exit criteria:

- BF16 routed MoE path is numerically correct in multi-rank decode tests

### Milestone 3: Weight Sharding and NZ

Files:

- `vllm_ascend/ops/fused_moe/fused_moe.py`
- `vllm_ascend/quantization/methods/w8a8_dynamic.py`
- `vllm_ascend/quantization/methods/w4a8.py` if kept in phase 1

Deliverables:

- shard routed expert weights over `moe_tp_group`
- keep NZ conversion points explicit
- ensure local grouped matmul consumes sharded NZ weights directly

Exit criteria:

- target model loads successfully in BF16
- no ND-only fallback remains on the supported path

### Milestone 4: W8A8

Files:

- `vllm_ascend/quantization/methods/w8a8_dynamic.py`
- helper utilities touched by grouped matmul layout assumptions

Deliverables:

- shard W8A8 scales consistently with local weight shards
- preserve `FRACTAL_NZ` compatibility
- match routed MoE numerics with the current baseline

Exit criteria:

- target model loads and runs in W8A8
- multi-rank decode correctness passes

### Milestone 5: Performance Gate

Files:

- benchmarks and tests only

Deliverables:

- reproducible benchmark script for the target workload
- comparison against `DP4 + TP2 + EP8` with EPLB disabled

Exit criteria:

- median decode throughput over `5` runs exceeds baseline by at least `5%`
- memory remains within deployment limits

The `5%` threshold is a chosen engineering default to avoid merging a statistically noisy improvement that does not justify long-term maintenance.

## Rollout Plan

### Phase 1

- Main v1 runner
- routed experts only
- `BF16` and `W8A8`
- decode path first
- correctness-first all-reduce return path
- NZ-enabled weight layout from day one
- performance gate against the current `EP8` without EPLB

### Phase 2

- graph capture hardening
- prefill path validation
- shared-expert support if needed
- reduce-scatter based return path
- broader quantization coverage

## Testing

Add tests for:

- process-group construction for `moe_tp_group` and `moe_source_group`
- weight shard shapes and loader behavior
- dispatcher correctness without EP-local expert ownership
- routed MoE numerical parity against a dense reference or a trusted MoE baseline
- end-to-end multi-rank correctness for `Qwen3.5-122B-A10B`
- regression coverage for `W8A8`

Performance validation should compare:

- current `DP4 + TP2 + EP8` with EPLB disabled
- new `DP4 + TP2 + MoE-TP8`

with the same decode workload:

- batch size `64`
- speculative depth `3`
- total routed MoE batch of `256` tokens

Metrics:

- decode tokens per second
- per-rank MoE latency
- communication time split by gather / broadcast / reduce
- max memory per rank
- grouped matmul kernel time
- NZ transform overhead during load and warmup

Benchmark protocol:

- run at least `5` repeated decode measurements per configuration
- compare medians, not best-case runs
- discard a pure warmup-only run if graph capture or lazy kernel init skews it materially
- record both end-to-end throughput and routed MoE layer latency

## Acceptance Criteria

The first production-ready version must satisfy all of the following on the target workload:

- numerical correctness matches the existing routed MoE path within the same tolerance used by current multi-rank tests
- routed expert weights remain NZ-compatible for all supported phase-1 kernels
- peak memory fits the `Qwen3.5-122B-A10B` deployment target
- end-to-end decode throughput is higher than the current `DP4 + TP2 + EP8` deployment with EPLB disabled
- median end-to-end decode throughput over `5` runs is at least `5%` above that baseline
- there is no hidden correctness-only fallback path left enabled in production code for the supported model and quantization set

If the implementation is correct but fails the last condition, it should be treated as an exploratory branch, not as a feature ready for merge.

## Risks

- Communication cost may dominate if the gather-broadcast-return path is not tightened after phase 1.
- Graph capture and speculative decode may expose assumptions that are currently EP-specific.
- Quantized grouped matmul kernels may need additional shape/layout validation under the new shard pattern, especially for NZ-backed weights.
- If TP peers are not correctly canonicalized before gather, the mode will silently duplicate tokens and distort routing.

## Decision

The implementation should proceed by introducing a dedicated `MoE-TP8` mode built on the existing `ALLGATHER` skeleton, not by trying to reinterpret `EP8` or by expanding the current DP all-gather domain in place.
