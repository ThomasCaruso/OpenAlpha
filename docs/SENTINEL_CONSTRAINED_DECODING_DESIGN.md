# Sentinel v1: Financial Grammar-Constrained Decoding

## Research question

> Can hard financial-domain constraints be enforced inside the autoregressive
> generation loop of a pretrained financial time-series foundation model, without
> retraining, while preserving or improving forecast fidelity, distributional
> diversity, and computational practicality?

This is a separately versioned research track. It does not refit or reinterpret the
retired Sentinel v0 reliability model, and it does not access the v0 holdout.

## Pinned source trace

Phase 3B uses official Kronos source revision
`67b630e67f6a18c9e9be918d9b4337c960db1e9a`, Kronos-mini revision
`f4e68697d9d5aed55cef5c96aabc3376bcad9f81`, and Tokenizer-2k revision
`26966d0035065a0cae0ebad7af8ece35bc1fb51c`.

The executed numerical path is:

1. `KronosPredictor.predict` selects named `open`, `high`, `low`, `close`,
   `volume`, and internally derived `amount`, then computes per-context feature means
   and standard deviations and normalizes the six features.
2. `KronosTokenizer.encode` applies a learned embedding, four transformer encoder
   layers, and binary spherical quantization. With `half=True`, `BSQuantizer` splits
   the 20-bit code into a 10-bit coarse ID and a 10-bit fine ID. Each vocabulary
   therefore has 1,024 entries.
3. `auto_regressive_inference` maintains coarse and fine token buffers. At each step,
   `Kronos.decode_s1` emits coarse logits. `sample_from_logits` applies temperature
   and nucleus filtering and samples with `torch.multinomial`.
4. `Kronos.decode_s2` conditions the fine-token logits on both the transformer
   context and the selected coarse token. The sampled pair is appended at lines
   444-454 of the pinned `model/kronos.py` loop.
5. After all five steps, `KronosTokenizer.decode` decodes the complete historical
   and generated token sequence through four transformer decoder layers. The
   predictor then applies the inverse context normalization.

The smallest safe interception point is after both logits are available and before
the token pair is appended. No cached official source or model weight is modified.
The experimental worker reimplements only this five-step loop and imports the
pinned official model, tokenizer, filtering, time-stamp, and normalization behavior.

## Consequences of the trace

- A K-line is represented by a hierarchical coarse/fine token pair, but the
  tokenizer decoder is sequence-dependent. A pair cannot be assumed to decode to an
  invariant candle independently of its surrounding tokens.
- Projection occurs in inverse-normalized price space. A projected candle must be
  normalized with the original causal context statistics before re-encoding.
- Re-encoding is also sequence-dependent. Method C therefore re-encodes the full
  current continuous context and audits the resulting final token pair. It never
  pretends that a direct continuous-to-single-token map exists.
- Appending later tokens can alter the decoder reconstruction of earlier generated
  positions. Methods C and D validate every step when chosen and validate the entire
  final path again. A final invalid path becomes an explicit hard failure.
- The joint candidate score is
  `log P(coarse | context) + log P(fine | context, coarse)`. This is available from
  `decode_s1` and `decode_s2`; full-vocabulary enumeration is unnecessary.

## Financial K-line grammar v1

The grammar is model-independent and versioned separately from the decoder.

For each candle, `open`, `high`, `low`, and `close` must be finite and positive;
`high >= max(open, close, low)`; `low <= min(open, close, high)`; and generated
volume must be finite and nonnegative. A sequence must contain exactly five unique,
strictly increasing, expected XNYS timestamps with no missing step.

Validation never mutates raw model output. Violations include code, step, timestamp,
observed gap, and severity normalized by the causal cutoff close.

## Four paired methods

### A. RAW_AUTOREGRESSIVE

This is the pinned official sampling loop with `sample_count=1`, temperature 1.0,
top-p 0.9, context 512, and one declared seed. It is the source trajectory.

### B. TERMINAL_PROJECTION

The entire raw rollout completes, then `CONSTRAINT_PROJECTION_V0` adjusts high and
low only. Open and close, tokens, autoregressive conditioning, and implied return
remain unchanged. It is a validated comparison baseline, not in-loop prevention.

### C. STEPWISE_PROJECT_REENCODE

The method uses the same primary random draws as A until an invalid decoded step is
encountered. It projects that candle in price space, re-normalizes it with the
original context statistics, re-encodes the full current continuous sequence, and
uses the resulting final coarse/fine pair for subsequent conditioning. It records
both tokens, both candles, all adjustments, token changes, round-trip error, and the
effect on later steps. If round-trip or final-sequence validation fails after one
declared re-encode attempt, the path hard-fails; it is never returned as valid.

### D. VALID_CANDIDATE_RESAMPLING

The method first draws the same raw pair as A. If it decodes validly, it is appended
and the paths remain paired. If invalid, the decoder evaluates up to 16 ranked joint
candidate pairs: the four highest filtered coarse probabilities crossed with the
four highest conditional fine probabilities for each coarse candidate. If none is
valid, it expands once to at most 64 pairs using eight by eight. Valid candidates
are sampled after renormalizing their joint probabilities, using a deterministic
auxiliary generator derived from origin, seed, and step so the primary generator is
not perturbed. If the budget finds no valid candidate, Method C is the explicit
fallback. A failed fallback produces a hard failure. Every candidate ID, log
probability, validity decision, rank, rejection, expansion, fallback, and validation
latency is recorded.

## Paired sample

The six cutoff positions are fixed indices 0, 10, 20, 30, 40, and 50 in the existing
52-cutoff development calendar. Each is used for SPY and QQQ:

| Cutoff | Horizon end |
|---|---|
| 2024-07-05 | 2024-07-12 |
| 2024-09-13 | 2024-09-20 |
| 2024-11-22 | 2024-12-02 |
| 2025-01-31 | 2025-02-07 |
| 2025-04-11 | 2025-04-21 |
| 2025-06-20 | 2025-06-27 |

Each origin uses seeds 1729, 2027, and 7919. All outcomes resolve before the
2025-07-01 holdout boundary. Raw Yahoo responses, reusable histories, and model
weights remain outside Git.

## Metrics

All price errors are divided by the causal cutoff close before aggregation. The
study reports normalized full-OHLC, open, high, low, close, high-low range,
candle-body, upper-wick, lower-wick, and per-step MAE. Path-shape distance is the
root mean squared distance over the 20 normalized OHLC coordinates. Range-direction
accuracy compares the sign of each predicted versus cutoff-candle range change with
the realized change. The declared range-volatility estimator is Parkinson's
five-session estimator. Cumulative range is the sum of daily high-low ranges.

Barrier events use the cutoff close and fixed symmetric levels 0.5%, 1.0%, and 2.0%.
Positive touch uses forecast high, negative touch uses forecast low, and either-touch
is their union. Thresholds are not tuned.

Diversity is mean pairwise normalized OHLC path distance across the three seeds,
variance of final close returns, and exact repeated-path rate. Operational metrics
include interventions, severity, token changes, rejection and fallback rates,
selected rank, memory, runtime, and latency multiplier.

## Locked continuation gates

An in-loop method (C or D) qualifies only if every gate passes on the 36 mini paths:

1. Every returned final path is structurally valid; an invalid return is forbidden.
2. At most one hard failure is allowed (`hard_failure_rate <= 1/36`, or 2.78%).
3. Mean normalized close MAE is no more than both 110% of A and A plus 0.001.
4. Mean normalized full-OHLC MAE and high-low range MAE are each no more than 105%
   of the better of A and B for that metric.
5. Mean pairwise path diversity is at least 50% of A, final-return variance is
   positive, and repeated-path rate is at most 10%.
6. Two identical-seed synthetic replays and two identical-seed real canary replays
   have identical canonical output and intervention hashes.
7. Median latency is at most 5x A and peak process memory is at most 2x A plus
   1 GiB.
8. No retraining, fine-tuning, or model-weight modification occurs.

These are feasibility gates, not claims of statistical superiority. Phase 3B may
conclude that terminal projection is the only practical validity gateway.

## Model-size canary gate

Kronos-small and Kronos-base are downloaded and tested only if C or D passes all
mini continuation gates. The canary then uses the 2024-07-05 and 2025-04-11 cutoffs
for each asset and all three seeds. Otherwise it is recorded as not run because its
prerequisite failed.
