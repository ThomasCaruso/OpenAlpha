# openalpha-bridge

The research engine. Every completed study's measurement, decision and
publication logic lives here, and every Modal function is a thin shell around it.

## Layout

| Module | Purpose |
| --- | --- |
| `diagnostic/` | Shared measurement layer: normalization, structural validity, anchored-return metrics, persistence comparison, official backend, runtime probe, sanitized logging |
| `base_study/` | Kronos-base structural replication — its own spec, invocation, runner, artifact and cache inventory |
| `zero_shot/` | Zero-shot forecasting benchmark — origins, baselines, aggregation, moving-block bootstrap, decision rules Z1–Z5 |
| `phase2/` | Earlier Bridge pipeline: provider, asset pinning, evidence classes, GPU measurement |
| `cloud/` | Object store, append-only journal, run lease, secret redaction, control service |
| top level | Financial transforms, calendars, numerics, windowing, typed errors |

## Design rules

**Identity is separate; measurement is shared.** Each study owns its experiment
id, specification digest, run-ID pattern, artifact namespace and schemas. All
three compute error the same way, so two studies cannot silently disagree on
arithmetic.

**Separation is structural, not conventional.** `canary_`, `base_` and `zsb_` run
IDs are pairwise disjoint — no string validates under two patterns, so no study
can name another's objects. Each invocation validator rejects the others' IDs by
name.

**The decision layer cannot see what it must not use.** Nothing in `decide()`'s
signature can carry a structural-validity quantity, so structural numbers cannot
influence an outcome even by accident. Superseded bootstrap estimators were
deleted rather than deprecated, so no weaker confidence interval can reach Z1.

**Nothing imports Torch at module scope.** The whole suite runs offline with no
GPU and no model weights; Torch is imported inside the functions that need it.

## Tests

`packages/bridge/tests/` — the largest suite in the repository. Deterministic
doubles for the tokenizer, forecast model, provider and object store; no network,
no official assets, no Torch.

```bash
uv run pytest packages/bridge/tests -q
```
