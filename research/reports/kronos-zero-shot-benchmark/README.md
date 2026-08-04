# Kronos Zero-Shot Forecasting Benchmark — Research Report

Can frozen Kronos-base zero-shot forecasts beat simple persistence under
shorter, paper-style horizons across multiple chronological origins?

**No.** Under both preregistered temperatures, on all four assets, at every level
of aggregation.

| | |
| --- | --- |
| Experiment | `openalpha-kronos-zero-shot-benchmark-v1` |
| Run | `zsb_25e0256eefb2b07a` (one run, no retries) |
| Outcome | `KRONOS_ZERO_SHOT_BENCHMARK_COMPLETED` |
| Finding | `NO_ZERO_SHOT_SKILL` (rule Z4) |
| Decision | `PROCEED_TO_FROZEN_REPRESENTATION_PROBE` |
| Direction | `STOP_KRONOS_ZERO_SHOT_DIRECTION` |
| Evidence class | `development_compatibility_canary` |

## Result at a glance

| | Config A (T=0.6) | Config B (T=1.0) | Threshold |
| --- | --- | --- | --- |
| Median relative skill | −0.041743 | −0.083634 | > 0 |
| Origins beating persistence | 0.31 | 0.26 | ≥ 0.60 |
| Supporting assets | 0 of 4 | 0 of 4 | ≥ 3 of 4 |
| Moving-block 95% interval | [−0.000645, −0.000223] | [−0.001577, −0.000346] | excludes 0 favorably |

All four Z1 conditions failed under both configurations. Both intervals exclude
zero on the *unfavorable* side.

## Contents

| File | Purpose |
| --- | --- |
| `report.md` | The technical report. Evidence, interpretation, limitations and non-claims are separated. |
| `results-summary.json` | Complete machine-readable results — per-asset, per-origin, per-step, all 25 bootstrap clusters. Generated from the artifact payload, not transcribed. |
| `artifact-manifest.json` | Immutable object key, digest, run ID, specification and commit provenance. |
| `reproduction.md` | Exactly how the run was produced and how to re-derive it. |
| `limitations.md` | Full limitation and non-claim inventory. |

## Scale

```
4 assets × 25 origins × 2 temperatures × 8 seeds = 1,600 generations
100 asset-origins · 40-session context · 12-session horizon
4 provider requests · 257.5 s on one Tesla T4
```

## Design integrity

- Specification SHA-256 sealed before execution and re-verified inside the container
- Origins are integer index offsets fixed in advance, never calendar dates chosen later
- Cross-asset calendar alignment **proved** from retrieved sessions at all 25 ordinals, not assumed
- Moving-block bootstrap over 4 consecutive origin clusters; block length `ceil(40/12)` fixed by geometry
- Both superseded bootstrap estimators deleted from the codebase, so neither can reach the decision layer
- Context-only normalization refit per origin; target rows structurally unreachable by function signature
- Zero clipped scalars of 24,000 — the standardization clip never engaged
- Parameter hash identical before and after; 0 trainable parameters

## Boundary

Development benchmark. Not holdout evidence, not trading evidence, not a
profitability claim. Every authorization field in the artifact is `false`. No
holdout was opened, no optimizer constructed, no backtest performed.

Structural-validity observations are recorded but are excluded from every
decision rule by construction — see `report.md` §8.

## Related

- [`../kronos-structural-validity/`](../kronos-structural-validity/) — the two
  completed structural diagnostics that motivated this benchmark
- [`../../bridge-v0/kronos-zero-shot-benchmark-v1.yaml`](../../bridge-v0/kronos-zero-shot-benchmark-v1.yaml)
  — the preregistration this run was bound to
- [`../../bridge-v0/kronos-frozen-representation-probe-design.md`](../../bridge-v0/kronos-frozen-representation-probe-design.md)
  — design note for the successor study (not implemented, not authorized)
