# OpenAlpha Bridge-2K Phase 2 Report

**Terminal conclusion: `OPERATIONALLY_BLOCKED`**

This is not a Bridge reconstruction-quality result. No market data was retrieved.
No Kronos checkpoint was loaded. No tokenizer encode was run, no Bridge parameter
was initialized or trained, and no validation, reconstruction-test, external, or
untouched-holdout metric was opened.

## Pre-data finding

The frozen experiment requires a complete 512-candle daily sequence wholly inside
each chronological partition: 448 causal prefix candles and 64 supervised suffix
candles. Raw series must be partitioned before window construction, a sequence may
not cross a boundary, and at least 512 candles must be purged at time boundaries.

The validation period is one calendar year. It has an absolute upper bound of 365
calendar days and 260 weekdays, so it cannot contain one 512-candle sequence. The
reconstruction-test period has at most 390 weekdays before exchange holidays, also
below 512. It therefore cannot contain one test sequence. The required 5,000 unique
test target candles and 500 unseen-symbol target candles are unreachable.

These bounds were derived from the committed half-open dates before any provider
access. Exchange holidays would only reduce the counts.

## Decision

Stage A was not run because a successful smoke test could not repair the decisive
validation/test contradiction. Making the experiment executable would require
changing at least one locked methodological decision: the 512-candle window, the
periods, the daily frequency, the purge, or the rule forbidding cross-partition
context. None was changed.

The result is `OPERATIONALLY_BLOCKED`, not `BRIDGE_2K_FEASIBLE`,
`BRIDGE_2K_PARTIALLY_FEASIBLE`, or
`TOKENS_INSUFFICIENT_FOR_COMPETITIVE_RECONSTRUCTION`. There is no checkpoint hash,
range-MAE improvement, bootstrap interval, structural-invalidity measurement, or
per-slice quality result to report.

## Preservation and hashes

- Original experiment SHA-256:
  `d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2`
- Phase 2 amendment SHA-256:
  `4c60846e8cb791d922c29a5e704fe3a473740171dd2bbb7b3390796ceeeef3c8`
- Expected trainable architecture remains
  `Linear(269,64) -> SiLU -> Linear(64,5)` with 17,605 parameters.
- Observed trainable parameters: not measured because assets were not loaded.
- Frozen-weight verification: not run; no frozen weight entered the process.
- Retrieved candles: 0.
- Training runtime and GPU use: 0.
- Checkpoint size: 0 bytes; no checkpoint exists.

Phase 0, Phase 1, Candidate C, the exact mathematical contract, every Sentinel
artifact, and the untouched forecasting holdout remain preserved. No forecasting
integration is authorized.

## Next justified task

There is no Phase 2 training or forecast-integration task under the current lock.
Any attempt to make reconstruction feasibility executable would be a new explicit
methodological amendment that changes a frozen period/window/leakage decision; this
run does not authorize one.
