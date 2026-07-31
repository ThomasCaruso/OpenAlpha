# OpenAlpha Bridge v0 Phase 2 Preregistration Amendment

**DEVELOPMENT COMPATIBILITY RESEARCH - NO RECONSTRUCTION RESULT**

- Amendment date: 2026-07-31
- Original experiment SHA-256: `d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2`
- Phase 2 amended experiment SHA-256: `4c60846e8cb791d922c29a5e704fe3a473740171dd2bbb7b3390796ceeeef3c8`
- Execution state: zero provider requests, zero retrieved candles, zero source or checkpoint downloads, zero checkpoint loads, zero tokenizer encodes, zero optimizer steps, and no validation, reconstruction-test, external-slice, forecast-holdout, or untouched-holdout access
- Terminal conclusion: `OPERATIONALLY_BLOCKED`

## Why one amendment was required

The original lock fixes the provider, symbols, daily frequency, dates, architecture,
loss, optimizer, seed, bootstrap, and decision thresholds. It did not explicitly
state that the learning rate has no scheduler, how validation-loss ties are broken,
which exact subsets Stage A and Stage B use, the representation-coverage gate, the
CPU wall-clock gate, or the latency warmup and repetition policy. The YAML amendment
locks those details without modifying the original experiment bytes.

The current Phase 2 protocol also makes the split semantics unambiguous: partition
raw chronological series before constructing windows; do not let any sequence cross
a partition boundary; purge at least the maximum consumed sequence length at time
boundaries. The amendment records the literal, leakage-conservative interpretation.
It does not borrow a prefix from another partition.

The original Yahoo Finance/yfinance source remains controlling. Binance data remains
reserved for Phase 4 after the original continuation and data-policy gates; it is not
substituted into Phase 2.

## Provider-free feasibility proof

Every example requires 512 complete daily candles: a 448-candle prefix and a
64-candle supervised suffix. Before exchange holidays are removed, the locked
partitions have these absolute upper bounds:

| Partition | Half-open period | Calendar days | Weekdays | Can contain 512 daily candles? |
|---|---|---:|---:|---|
| Train | 2010-01-01 to 2022-01-01 | 4,383 | 3,131 | yes |
| Validation | 2022-01-01 to 2023-01-01 | 365 | 260 | no |
| Reconstruction test | 2023-01-01 to 2024-06-29 | 545 | 390 | no |
| Later external | 2024-07-01 to 2025-07-01 | 365 | 261 | no |

Validation cannot contain even one complete example because its calendar-day upper
bound is below 512. The equity reconstruction test cannot contain one because its
weekday upper bound is 390, before subtracting market holidays. The required purge
cannot create observations. Consequently, the maximum possible validation and test
sequence counts are both zero, and the locked requirements for 5,000 unique test
targets and 500 unseen-symbol targets cannot be met.

This proof uses only committed dates and elementary calendar counting. No provider
response, market observation, checkpoint, or outcome metric was inspected.

## Why the experiment stops

Making Phase 2 executable would require at least one prohibited methodological
change: shortening the 512-candle window, moving or lengthening chronological
partitions, borrowing context across a boundary, weakening the purge, or changing
frequency. None is an implementation-error correction, and the current instruction
explicitly prohibits modifying the locked data splits or architecture after this
audit.

The result is therefore `OPERATIONALLY_BLOCKED`, not evidence that Bridge succeeds,
partially succeeds, or that Kronos tokens are insufficient. Stage A is not run
because it cannot cure the decisive validation/test contract failure. No learned
metric, checkpoint, or scientific reconstruction claim will be created.

## Preservation

The original experiment file and hash remain byte-identical. Phase 0, Phase 1, all
Sentinel v0/v1/v1.1 research, the Candidate C architecture, source hashes, and the
untouched forecasting holdout remain unchanged and unaccessed.
