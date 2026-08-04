# Bridge Phase 2 Execution Pipeline

**Status: pipeline implemented and synthetically validated. The Bridge training
run was never executed, and the direction it served is now closed.**

No Bridge stage has run: no parameter was trained, no Bridge checkpoint written,
and no held-out partition opened. What exists is the complete execution pipeline,
verified end to end against fake components.

Provider requests and Kronos asset downloads are no longer zero — the three
completed diagnostic and benchmark studies performed both while training nothing
(`trainable_parameter_count = 0`, unchanged parameter hashes,
`optimizer_constructed = false`). Those studies closed the structural-validity
direction this pipeline was built to serve; see
[`../research/reports/`](../research/reports/).

## Why local execution stopped

The Windows development machine has no CUDA accelerator and no Torch
installation. The locked experiment sets `stage_c_cpu_execution: prohibited` and
`accelerator_count_maximum: 1`, so Stage C cannot run here at all.

Running Stage A and Stage B locally and Stage C elsewhere would split the
experiment across two environments with different numerics, different library
versions, and two separate retrievals. The locked run requires one consistent
accelerator environment, so all three stages are deferred to a single GPU host.

Preparing the pipeline first also means the code that will touch the held-out
partition is reviewable *before* it can ever read it.

## Optional dependency boundary

Importing `openalpha_bridge` or `openalpha_bridge.phase2` pulls in no heavy
dependency. This is asserted by a subprocess test and by base CI.

| Extra | Provides | Needed for |
|---|---|---|
| `bridge-core` | `yfinance==1.5.2` | provider retrieval |
| `bridge-kronos` | `huggingface-hub`, `safetensors` | pinned asset resolution |
| `bridge-training` | `torch` | Stage C training |
| `bridge-gpu` | all three | a single GPU host |

A missing dependency raises `MISSING_OPTIONAL_DEPENDENCY` carrying the module
name, the required extra, the exact install command, and the current stage.

## State machine

States advance in one order. Stages cannot be skipped, and every non-terminal
state may also fail or block.

```
CREATED
  -> PREFLIGHT_PASSED
  -> DATA_RETRIEVED
  -> DATA_VALIDATED
  -> WINDOWS_BUILT
  -> COVERAGE_PASSED
  -> ASSETS_RESOLVED
  -> STAGE_A_PASSED
  -> STAGE_B_PASSED
  -> STAGE_C_TRAINED
  -> CHECKPOINT_FROZEN
  -> TEST_OPENED          <- explicit operator transition only
  -> TEST_EVALUATED
  -> EXTERNAL_EVALUATED
  -> FINALIZED            (terminal)

any non-terminal -> FAILED | BLOCKED
FAILED -> resume at the last verified stage
FINALIZED, BLOCKED       (terminal)
```

`TEST_OPENED` is reachable only from `CHECKPOINT_FROZEN`. The state machine is
versioned as `openalpha.bridge.phase2.states.v1`; the transition table is tested
directly.

## Run identity and resume

A run identity is derived from the experiment hash, both amendment hashes, and
the canonical hash of the resolved configuration. Changing any locked input
produces a different `run_id`, and reusing a run directory under a changed
configuration fails with `RUN_IDENTITY_CHANGED` rather than silently continuing.

Resume reads the persisted journal and continues from the last recorded state.
Completed stages are content-addressed and are not rerun unnecessarily.

## Windowing

The pipeline uses the committed 448-prefix / 64-scored-suffix contract from
Amendment 2 and no other interpretation. The score mask is immutable, and its
SHA-256 `2fe5b1b3c66dfd7c7e8af2612d69d3c2337a4a0f6dd3896a4d7c2f69911f3711` is
asserted in both the amendment and the code.

Every sequence artifact records symbol, partition, interval, prefix start and
end, target start and end, target-overlap status, and a deterministic sequence
ID.

## Metrics

All reconstruction metrics are computed on the 64 scored suffix candles only.
`scored_view` reduces a 512-row example through the locked mask before any
arithmetic, and an invariant test proves that arbitrarily corrupting all 448
warm-up rows leaves every reported metric byte-identical.

## Paired bootstrap

The resampling unit is the scored sequence or a declared instrument-time block,
never the individual candle. Replicates (10,000), seed (20260731), and
confidence level (0.95) are locked and never chosen from observed results.
Synthetic tests cover clear improvement, clear degradation, exact ties,
insufficient samples, missing pairs, and duplicate units.

## Final gate evaluator

Every locked threshold is encoded as a machine-readable gate with an ID, scope,
threshold, comparison, measured value, outcome, evidence artifact, and
explanation. The terminal conclusion is derived from that table:

- any integrity or structural failure -> `OPERATIONALLY_BLOCKED`
- no failures -> `BRIDGE_2K_FEASIBLE`
- every quality gate failing -> `TOKENS_INSUFFICIENT_FOR_COMPETITIVE_RECONSTRUCTION`
- some quality gates failing -> `BRIDGE_2K_PARTIALLY_FEASIBLE`
- nothing evaluated -> `OPERATIONALLY_BLOCKED`

No narrative can override a failed gate; the report is generated from the
artifact.

## Artifact locations

| Artifact | Location |
|---|---|
| State journal | `<run-dir>/journal.json` |
| Gate table | `<run-dir>/gate_table.json` |
| Test-opening record | `<run-dir>/test_opening_record.json` |
| Feature-cache shards | `<cache-dir>/<partition>/<sequence-id>.npz` |
| Preserved blocker record | `research/bridge-v0/phase2/` |

Run and cache directories must live outside the Git worktree. Preflight and the
cache constructor both refuse otherwise.

## Commands

```
python -m openalpha_bridge.phase2 preflight --run-dir DIR --cache-dir DIR
python -m openalpha_bridge.phase2 stage-a   --run-dir DIR --cache-dir DIR
python -m openalpha_bridge.phase2 stage-b   --run-dir DIR --cache-dir DIR
python -m openalpha_bridge.phase2 stage-c   --run-dir DIR --cache-dir DIR
python -m openalpha_bridge.phase2 verify    --run-dir DIR --cache-dir DIR
python -m openalpha_bridge.phase2 run       --run-dir DIR --cache-dir DIR --open-test-partition
```

Every command supports `--experiment`, `--run-dir`, `--cache-dir`,
`--repository-root`, `--device`, `--resume`, `--dry-run`, `--json`, `--verbose`,
`--provider-mode`, `--kronos-mode`, and `--evidence-class`. Exit codes are 0 for
success, 1 for failure, 2 for blocked, and 64 for usage errors.

`run` refuses to start without `--open-test-partition`; the test partition never
opens implicitly.

## What remains unproven

Everything empirical. Specifically:

- whether Kronos tokens carry enough wick and range information for the Bridge
  head to beat terminal projection
- every reconstruction metric, every gate outcome, and the terminal conclusion
- whether all 20 declared ETFs have continuous XNYS history from 2010-01-01, an
  assumption verifiable only at retrieval
- latency, memory, and deterministic-replay behaviour on real hardware
- external and volatility-regime slice quality

The synthetic run validates orchestration only. It is labelled
**SYNTHETIC PIPELINE VALIDATION - NOT EMPIRICAL EVIDENCE** and its artifacts
never enter the real evidence namespace.
