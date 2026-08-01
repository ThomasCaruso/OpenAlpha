# OpenAlpha Bridge v0 Phase 2 Amendment 2: context prefix and scored suffix

**DEVELOPMENT COMPATIBILITY RESEARCH - NO RECONSTRUCTION RESULT**

- Amendment date: 2026-07-31
- Original experiment SHA-256: `d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2`
- Amendment 1 SHA-256: `4c60846e8cb791d922c29a5e704fe3a473740171dd2bbb7b3390796ceeeef3c8`
- Blocker record commit: `e684c72`
- Execution state: zero provider requests, zero retrieved candles, zero source or
  checkpoint downloads, zero checkpoint loads, zero tokenizer encodes, zero optimizer
  steps, and no validation, reconstruction-test, external-slice, forecast-holdout, or
  untouched-holdout access

## What was wrong

Amendment 1 recorded a `literal_split_contract` requiring a complete 512-candle
example to lie wholly inside one chronological partition, with
`context_prefix_borrowing_from_another_partition: false`.

Under that interpretation the locked one-year validation period (251 XNYS sessions)
and the daily reconstruction-test period (374 XNYS sessions) are both shorter than
512 sessions. Neither could contain a single example, so the maximum possible
validation and test sequence counts were zero and the locked minimum sample counts
were unreachable. Phase 2 terminated `OPERATIONALLY_BLOCKED`.

That interpretation is a protocol-design error, not a data, compute, or Kronos
limitation. The original lock never required it. `experiment.yaml` already declares
`causal_prefix_length: 448` and `supervised_suffix_length: 64` as distinct
quantities, and its `split_crossing: prohibited` rule governs the scored target, not
the read-only warm-up context.

## The corrected reading

Each example remains exactly 512 candles: a context prefix at positions 0-447 and a
scored suffix at positions 448-511. Only the 64-candle scored suffix must lie wholly
inside its assigned partition. The preceding 448 candles are causal historical
warm-up context and may come from earlier partitions.

This mirrors deployment. A model evaluated at the start of a new period legitimately
reads every observation available before that period.

Partition isolation constrains trainable targets, losses, model fitting,
preprocessing fitting, checkpoint selection, threshold selection, and final metrics.
It does not prohibit earlier chronological observations serving as read-only causal
context for later targets.

Allowed: training history -> validation target, where the training history is
read-only context and the scored target lies wholly inside the validation period.

Forbidden: validation target -> training loss; test target -> validation selection;
future target candle -> earlier causal hidden state.

## Score mask

The score mask is immutable: `false` for positions 0-447, `true` for positions
448-511. Encoded as a 512-character ASCII string of `0` and `1` in position order,
its SHA-256 is
`2fe5b1b3c66dfd7c7e8af2612d69d3c2337a4a0f6dd3896a4d7c2f69911f3711`.

The preregistered training loss is already suffix-only: `preregistration.md` states
the example "is supervised only on the 64 target candles." The training mask and the
evaluation mask are therefore identical, and the original training contract is
retained exactly. No separate training/evaluation mask distinction arises.

All reconstruction losses and reported metrics for validation and test are computed
on the 64 scored suffix candles only. The official decoder, terminal projection, and
residual projection baselines are scored on the identical suffix. Warm-up context
enters no metric. Every artifact records prefix length, suffix length, and score-mask
hash.

## Calendar

The locked primary provider is Yahoo Finance via `yfinance 1.5.2` at `1d` interval
over exchange-traded US ETFs. The applicable calendar is **XNYS**, not a 24/7
cryptocurrency calendar. Weekends are not sessions. Binance remains Phase 4 only
after data-policy review, and Amendment 1's `binance_phase_2_access: prohibited`
stands.

Session counts use half-open ranges and the standard XNYS holiday rules, including
Good Friday, Juneteenth from 2022, weekend observance shifts, and the ad-hoc closures
2012-10-29, 2012-10-30, 2018-12-05, and 2025-01-09.

## Interpretation declared before data access

`preregistration.md` requires "at least 5,000 unique held-out target candles and at
least 500 unique unseen-symbol target candles."

Declared interpretation: the 5,000 minimum counts the entire reconstruction-test
scored corpus, comprising both the 14 training symbols and the 6 unseen symbols; the
500 minimum is a composition floor on the unseen-symbol subset of that same corpus.

This follows from the original lock placing `minimum_unique_unseen_symbol_test_candles`
inside the Phase 2 `windowing` block, which puts unseen symbols inside the Phase 2
reconstruction-test pass rather than only inside Phase 4.

The interpretation is declared here, before any provider access and before any
outcome is observable. No threshold value is changed.

## Pre-retrieval feasibility table

Derived only from committed dates and the XNYS calendar. Identical for every declared
symbol at `1d`. No provider response or market observation was used.

| Partition | Period | Sessions | Prefix available | Suffixes/symbol | Scored/symbol | 448 warm-up | Status |
|---|---|---:|---:|---:|---:|---|---|
| Train | 2010-01-01 to 2022-01-01 | 3,021 | 0 (in-partition) | 40 | 2,560 | yes | FEASIBLE |
| Validation | 2022-01-01 to 2023-01-01 | 251 | 3,021 | 3 | 192 | yes | FEASIBLE |
| Reconstruction test | 2023-01-01 to 2024-06-29 | 374 | 3,272 | 5 | 320 | yes | FEASIBLE |
| Later external | 2024-07-01 to 2025-07-01 | 250 | 3,646 | 3 | 192 | yes | FEASIBLE |

Eligible target suffix boundaries:

| Partition | First target | Last target | Unused tail |
|---|---|---|---:|
| Train | 2011-10-12 to 2012-01-12 | 2021-09-14 to 2021-12-13 | 13 |
| Validation | 2022-01-03 to 2022-04-04 | 2022-07-08 to 2022-10-06 | 59 |
| Reconstruction test | 2023-01-03 to 2023-04-04 | 2024-01-10 to 2024-04-11 | 54 |
| Later external | 2024-07-01 to 2024-09-30 | 2025-01-02 to 2025-04-04 | 58 |

Train warm-up is drawn from inside the training partition itself, so its first
scored target begins at training session 448 and its first prefix starts 2010-01-04.

## Pooled counts against the locked minima

| Quantity | Value | Required | Result |
|---|---:|---:|---|
| Train scored sequences / candles | 560 / 35,840 | - | - |
| Validation scored sequences / candles | 42 / 2,688 | - | - |
| Test scored candles, training symbols | 4,480 | - | - |
| Test scored candles, unseen symbols | 1,920 | >= 500 | PASS |
| Test scored candles, total | 6,400 | >= 5,000 | PASS |
| External scored sequences / candles | 60 / 3,840 | - | - |

Per-regime external counts depend on pooled-training quantile thresholds and cannot
be derived before retrieval; an even split yields about 1,280 per regime against a
500 minimum. That gate belongs to Phase 4.

The 512-candle compatibility window was not reduced to make the split fit. Periods,
symbols, architecture, and thresholds are unchanged.

## Assumption to verify at Stage A

Every declared symbol is assumed to have continuous XNYS history from 2010-01-01. All
20 declared ETFs listed before that date, but this is verifiable only at retrieval. It
is a Stage A gate, not a claim.

## Preservation

The original experiment file, Amendment 1, and the `OPERATIONALLY_BLOCKED` Phase 2
record under `research/bridge-v0/phase2/` remain byte-identical and are not rewritten
or discarded. The blocker record stands as the terminal outcome of the prior
interpretation. Phase 0, Phase 1, Candidate C, all Sentinel artifacts, and the
untouched forecasting holdout remain unchanged and unaccessed.

No market data, model asset, training run, validation outcome, or test outcome was
accessed before this correction. This amendment does not change the Bridge
architecture, the 17,605 trainable parameters, the frozen trunk, representation
formulas, numerical caps, loss, optimizer, seed, metrics, paired-bootstrap procedure,
success or failure thresholds, source universe, chronological periods, compute or
corpus limits, or any test outcome.
