# OpenAlpha Bridge v0 Phase 2 Amendment 3: scale-feature formulas and the Stage A canary

**DEVELOPMENT COMPATIBILITY RESEARCH - NO RECONSTRUCTION RESULT**

- Amendment date: 2026-08-02
- Amendment 3 SHA-256: `d9484020a22df42c93edc19941374c628ea891f08c8bee8151b82264c00bb12b`
- Preserved unchanged: experiment `d52a9be7…f47b2`, Amendment 1 `4c60846e…eeef3c8`, Amendment 2 `66f3c817…ab10c1`
- Execution state: zero provider requests, zero retrieved candles, zero checkpoint loads, zero optimizer steps, no held-out access

## Why this amendment exists

The original lock **names** the 13 causal scale features but never gives their
formulas. Any implementation must pick exact equations, so the choice is recorded
here, before a single real feature is extracted, rather than left implicit in
code where it could drift.

These formulas were derived from the locked feature names alone. **No empirical
reconstruction result influenced them**, because none exists.

## Definitions

The anchor is the final close of the 448-candle prefix. Every statistic is a
population statistic over the prefix only, with `ddof = 0`.

| Feature | Formula |
|---|---|
| `log_anchor_close` | `log(anchor_close)` |
| `open_mean_relative_to_anchor` | `prefix_population_mean(open) / anchor_close` |
| `high_mean_relative_to_anchor` | `prefix_population_mean(high) / anchor_close` |
| `low_mean_relative_to_anchor` | `prefix_population_mean(low) / anchor_close` |
| `close_mean_relative_to_anchor` | `prefix_population_mean(close) / anchor_close` |
| `log1p_open_std_over_anchor` | `log1p(prefix_population_std(open) / anchor_close)` |
| `log1p_high_std_over_anchor` | `log1p(prefix_population_std(high) / anchor_close)` |
| `log1p_low_std_over_anchor` | `log1p(prefix_population_std(low) / anchor_close)` |
| `log1p_close_std_over_anchor` | `log1p(prefix_population_std(close) / anchor_close)` |
| `log1p_volume_mean` | `log1p(prefix_population_mean(volume))` |
| `log1p_volume_std` | `log1p(prefix_population_std(volume))` |
| `log1p_amount_mean` | `log1p(prefix_population_mean(amount))` |
| `log1p_amount_std` | `log1p(prefix_population_std(amount))` |

The 13 values are computed once and broadcast to all 512 positions as float32.

## What "causal" means here, precisely

It means **no scored-suffix candle influences these features**. Every statistic is
taken over the 448-candle prefix alone, so the 64 scored candles cannot reach them.

It does **not** mean a later prefix candle cannot affect an earlier prefix row.
These are prefix-wide statistics broadcast to every position, so changing any
prefix candle changes all 512 rows, earlier ones included. Position-level
causality within the window is a property of the frozen Kronos decoder trunk, not
of these 13 features. Conflating the two would overstate what is proven.

## The Stage A official canary

One isolated compatibility check on real data. Selection rule: the first locked
stride-64 training window whose complete 512 candles fall inside the Amendment 1
Stage A request period.

| Field | Value |
|---|---|
| Symbol / interval / partition | SPY / `1d` / train |
| Sequence ID | `ecd7fd5797a9147a` |
| Prefix | 2015-05-07 to 2017-02-14 (448 candles) |
| Scored suffix | 2017-02-15 to 2017-05-17 (64 candles) |
| Retrieval | SPY, 2015-05-07 inclusive to 2017-05-18 exclusive, 512 candles, one request |

The canary retrieves **no** validation, reconstruction-test, external-period, or
unseen-symbol data. It performs no training, creates no checkpoint, evaluates no
metric or gate, and cannot open the test partition.

Evidence class is `development_compatibility_canary`, deliberately distinct from
`real_phase2`.

## What success does and does not mean

Success is exactly one code: `STAGE_A_OFFICIAL_CANARY_PASSED`.

It means the official frozen path executes and matches the locked contract on one
real training window: verified source and tokenizer hashes, official identifiers
in range, `[512,20]`, `[512,256]`, and `[512,269]` tensors at the locked dtypes,
byte-identical repeated extraction, scored-suffix causality holding, and an
identity-bound cache shard written and verified on read.

It says **nothing** about Bridge reconstruction quality, range-MAE improvement, or
feasibility, and it does **not** authorize Stage B or a real run. A failure of
official loading, dimensionality, causality, cache identity, or determinism stops
work and the negative result is preserved.

## Preserved

The experiment file and both prior amendments remain byte-identical. Architecture,
the 17,605 parameters, the 269 feature dimension, the score mask, periods,
symbols, partitions, training configuration, metrics, bootstrap, thresholds,
gates, one-time test-opening semantics, and the real and synthetic evidence
classes are all unchanged.
