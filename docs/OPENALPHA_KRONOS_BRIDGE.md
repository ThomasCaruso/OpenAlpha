# OpenAlpha for Kronos Bridge

## Research question

> Can a learned, constraint-preserving decoder reconstruct the original
> hierarchical token sequences produced by Kronos into valid financial
> candlesticks, without retraining the Kronos forecasting backbone and without
> materially degrading reconstruction or forecast quality?

Bridge-2K is a compatibility experiment, not a new tokenizer, forecasting model, or
trading system.

## Phase 1 implementation status

The mathematical/runtime boundary is implemented in the internal
`openalpha-bridge` workspace package. It provides immutable configuration, honest
target encoding, versioned five-channel activation mapping, causal float32/float64
reconstruction, all three volume modes, a Sentinel audit adapter, deterministic
serialization, and numerical round-trip audit.

Phase 1 retrieved no market data, loaded no Kronos source or checkpoint at runtime,
ran no forecast, trained no head, created no checkpoint, and accessed no untouched
holdout. It proves the hard output contract only; learned reconstruction and
forecast quality remain entirely unmeasured.

## Product contract

Kronos remains responsible for predicting hierarchical market-token sequences.
OpenAlpha is responsible for preserving those tokens, reconstructing a separately
labeled safe path, validating both paths, and recording provenance.

For every supported forecast, the eventual integration must be able to state:

> Kronos generated these exact hierarchical forecast tokens. Its official decoder
> produced this raw forecast. OpenAlpha detected these structural violations and
> reconstructed the same tokens through a constraint-preserving compatibility
> decoder, producing this separately labeled valid forecast without changing the
> Kronos forecasting backbone.

The raw official path, exact violations, generated tokens, and official revisions
are mandatory outputs. Bridge never silently substitutes its path for the official
one.

## Source-backed architecture decision

The pinned `KronosTokenizer.decode` is sequence-based. It reconstructs the full
ordered latent window through causal self-attention before its final unrestricted
six-feature head. The selected architecture is therefore **Candidate C: a
sequence-level constrained reconstruction head**, with the smallest practical
trainable surface.

```text
official coarse_ids [B,T] + fine_ids [B,T]
    |
    v
frozen official indices_to_bits -> quantized_bits [B,T,20]
    |
    v
frozen official Linear(20,256) + three causal decoder blocks
    |
    v
decoder_hidden [B,T,256]
    |  + causal scale_features [B,T,13]
    v
OpenAlpha Linear(269,64) -> SiLU -> Linear(64,5)
    |
    v
bounded gap/body/upper/lower/volume representation
    |
    v
causal hard financial inverse
    |
    v
separately labeled valid OHLC(V) suffix
```

Only the 17,605-parameter OpenAlpha head is trainable in Bridge v0. The official
encoder, implicit BSQ codebook, `post_quant_embed`, causal decoder blocks,
forecast-model embeddings, forecasting transformer, and all official heads remain
frozen and hash-verified.

This choice retains the source architecture's temporal dependency and reuses its
learned latent geometry. A new per-candle lookup decoder is prohibited. A complete
replacement tokenizer is prohibited because it would change the identifiers
predicted by the pretrained forecasting transformer.

## Exact Bridge inputs

The public Bridge call accepts:

- unchanged coarse IDs `int64[B,T]` in `0..1023`;
- unchanged fine IDs `int64[B,T]` in `0..1023`;
- ordered sequence positions for a maximum 512-token decoder window;
- the exact historical six-feature population mean and standard deviation;
- normalization epsilon `1e-5` and clip bounds `[-5,5]`;
- the last observed close immediately before the output suffix;
- the number and positions of suffix candles to reconstruct; and
- volume presence and amount-derivation provenance.

The frozen trunk deterministically constructs the 20-dimensional bipolar latent and
256-dimensional causal state. The head also receives 13 causal scale features:

1. log anchor close;
2. four historical price means relative to the anchor;
3. four `log1p(price standard deviation / anchor)` values; and
4. `log1p(mean)` and `log1p(standard deviation)` for volume and amount.

The head receives no future observation, outcome, future normalization statistic,
trading label, or realized forecast error.

## Outputs and guarantees

The five head outputs parameterize:

- bounded gap return;
- bounded candle-body return;
- bounded nonnegative upper-wick log ratio;
- bounded nonnegative lower-wick log ratio; and
- bounded nonnegative log-volume when volume is supported.

The inverse mapping is recursive from the last observed close. For every successful
supported output it guarantees finite positive OHLC, high at or above open and
close, low at or below open and close, high at or above low, and nonnegative volume
when present. The representation, caps, missing-volume behavior, numerical guards,
proof, and Phase 1 test obligations are specified in
`BRIDGE_MATHEMATICAL_REPRESENTATION.md`.

The explicit dtype guards, failure classes, round-trip tolerances, tensor encoding,
and property evidence are specified in `BRIDGE_NUMERICAL_CONTRACT.md`.

No structural-validity penalty substitutes for this construction. Sentinel still
validates the result independently and fails closed if the implementation violates
the contract.

## Training example

Each reconstruction example uses a valid real sequence with a causal prefix:

```text
448 valid prefix candles
    -> calculate six-feature mean/std from prefix only
448 prefix + 64 valid target candles
    -> normalize with frozen prefix state
    -> frozen official Tokenizer-2k encoder
    -> unchanged coarse/fine sequence of length 512
    -> frozen official decoder trunk
    -> trainable OpenAlpha head on final 64 positions
    -> hard constrained reconstruction anchored at prefix close
    -> compare with the original 64 valid target candles
```

Training real-token reconstruction before forecast integration isolates the first
question: whether the official identifiers contain enough information for a better,
safe reconstruction.

## Objective

The loss is intentionally small and locked:

- mean Huber loss for standardized gap and body coordinates;
- mean Huber loss for standardized upper and lower wick coordinates;
- Huber loss on derived log high-low range
  `abs(body) + upper + lower`;
- optional masked log-volume Huber loss at weight 0.25.

Return, wick, and range groups each have weight 1.0. Coordinate scales are median
absolute deviations calculated on the training split only, floored at `1e-4`,
serialized, hashed, and frozen. Huber delta is 1.0 standardized unit. There is no
validity loss and no direct mixture of redundant OHLC level losses.

## Baselines

Phase 2 compares identical test tokens and normalization state under:

1. the frozen official Tokenizer-2k decoder;
2. deterministic terminal projection of official output;
3. a fixed linear residual baseline followed by explicit terminal projection; and
4. OpenAlpha Bridge-2K with no post-output projection.

The linear baseline fits ridge regression with alpha `1e-3` from the official
normalized six-vector plus 20 latent bits to a six-vector residual. It then applies
the same explicit terminal projection. It is included to distinguish the value of
hard representation learning from a cheap learned correction.

## Data plan

The initial corpus is deliberately smaller than a foundation-model corpus.

- Primary feasibility data: unadjusted daily public Yahoo Finance observations
  through pinned `yfinance==1.5.2`, using the repository's existing provider
  boundary and request options.
- Training symbols: `SPY`, `QQQ`, `XLF`, `XLK`, `XLE`, `XLI`, `XLV`, `XLP`, `XLY`,
  `XLU`, `TLT`, `HYG`, `EFA`, and `VNQ`.
- Symbol-generalization set: `IWM`, `DIA`, `GLD`, `EEM`, `SLV`, and `USO`.
- Training period: 2010-01-01 through 2021-12-31.
- Validation period: 2022-01-01 through 2022-12-31.
- Reconstruction test period: 2023-01-01 through 2024-06-28.
- Later external period, only after Phase 2 succeeds: 2024-07-01 through
  2025-06-30.
- Frequency generalization, only after Phase 3 succeeds: public Binance Vision
  `BTCUSDT` and `ETHUSDT` one-day and one-hour archives for the declared external
  period, subject to a recorded license/data-policy review.

No data on or after the Sentinel untouched-holdout start of 2025-07-01 may enter
Bridge development, training, validation, testing, scaling, or model selection.

The entire first Bridge-2K program is capped at 250,000 complete retrieved candles.
Raw responses stream through bounded batches and are discarded after validation and
hashing. Temporary raw and derived caches live outside Git under a configurable
cache root, are limited to 10 GiB, and support automatic cleanup. Git receives only
compact request/manifests, hashes, aggregate results, and synthetic fixtures.
Checkpoints are release or model-hub artifacts and are never committed.

## Chronology and leakage controls

- No 512-candle example crosses a chronological split.
- Normalization uses only the 448-candle causal prefix.
- Evaluation target suffixes do not overlap, even when their causal prefixes do.
- At least 5,000 unique test candles and 500 unique unseen-symbol candles are
  required for the locked reconstruction decision.
- Training windows may use stride 64; validation and test target suffixes are
  nonoverlapping.
- Train-only robust loss scales are frozen before validation or test metrics.
- Test results remain unread until the selected checkpoint, scalers, and manifest
  are sealed from training and validation evidence.

## Phased gates

### Phase 1: mathematical contract

Complete. The representation, inverse, exact batch/anchor checks, independent
Sentinel validator adapter, canonical serialization, dtype-specific numerical
audits, missing-volume policy, causal chaining, future-perturbation invariance, and
10,500 deterministic property examples are implemented and tested. There is no
post-output projection. No training data was loaded and no decoder head was fitted.

### Phase 2: Bridge-2K reconstruction feasibility

Freeze the official Tokenizer-2k encoder and decoder trunk. Train the fixed small
head. Compare paired held-out reconstruction under the four decoders. Stop unless
all Phase 2 gates in `experiment.yaml` pass.

### Phase 3: fixed Kronos forecast integration

Only after Phase 2 success, reuse the exact 12 preserved Sentinel development
origins and three seeds. Capture each unchanged generated token sequence and decode
it through the official decoder, terminal projection, and Bridge. Do not access the
untouched holdout.

### Phase 4: external generalization

Only after Phase 3 success, run the declared unseen-symbol, later-period,
volatility-regime, and non-dominant-frequency slices. No replacements are selected
after results are visible.

### Phase 5: Bridge-base

Only after Bridge-2K passes every gate, create a separate compatibility manifest and
preregistration for Tokenizer-base with Kronos-small and, when practical,
Kronos-base. Bridge-2K weights are not assumed compatible.

### Phase 6: public package

Only after reproducible Bridge evidence exists, package the paired raw/safe
predictor, immutable audits, compatibility rejection, example, benchmark, and
provenance surface. Prepare a factual upstream issue only after the result and
licenses are ready.

## Locked success summary

The numerical source of truth is `research/bridge-v0/experiment.yaml`. In summary,
Bridge must:

- produce zero invalid reconstruction and forecast candles;
- preserve every input token ID and every frozen official weight hash;
- apply no hidden terminal projection to Bridge output;
- reduce paired high-low range MAE by at least 10% versus terminal projection in
  reconstruction and forecast tests, with paired-bootstrap evidence;
- keep every OHLC MAE within 5% plus a small frozen additive margin of the official
  decoder;
- keep close and close-return error within the frozen non-inferiority margins;
- preserve at least 75% of raw forecast path diversity, positive return variance,
  and no more than 10% repeated paths;
- replay to the same canonical artifact hash twice on the same locked runtime;
- stay within one million trainable parameters, 25 MiB, two times official median
  decode latency, 50 ms p95 incremental latency, and 512 MiB incremental memory;
  and
- pass the declared unseen-symbol, period, regime, and frequency slices.

Thresholds do not move after test results are visible.

## Stop conditions

Stop Bridge v0 and preserve the result if any of the following is established under
the locked experiment:

- the Phase 1 hard contract cannot guarantee validity without a repair pass;
- future-token perturbation changes an earlier Bridge candle;
- Phase 2 cannot beat terminal projection on range while meeting OHLC and
  close/return non-inferiority;
- performance exists only on training symbols or periods;
- forecast integration materially degrades the locked important metrics;
- compatibility requires retraining or modifying Kronos;
- the head, checkpoint, data, cache, memory, latency, or compute cap is exceeded;
  or
- the approach needs an undeclared corpus or post-test threshold change.

On failure, OpenAlpha retains Sentinel validation, immutable audit, deterministic
projection, compatibility profiling, and the complete Bridge evidence. It does not
begin another conceptual pivot.

## Remaining risks after Phase 1

- The frozen token sequence may not retain enough range or wick information.
- Encoded real tokens and autoregressively generated tokens may have different
  distributions even when they share identifiers.
- Recursive previous-close reconstruction may accumulate path drift on learned
  suffix distributions outside the deterministic Phase 1 stress fixtures.
- The fixed numerical caps exclude rare but valid extreme candles.
- The selected boundary depends on pinned Python module internals rather than a
  formal upstream decoder-trunk API.
- GPU kernels may require deterministic-mode restrictions and may not be
  byte-identical across hardware families.
- Provider adjustments, corporate actions, volume semantics, and crypto/equity
  differences may limit generalization.
- Upstream source/model licenses and data-provider terms must permit checkpoint
  distribution before a public release.

These are evaluation questions and release gates, not reasons to weaken the
structural contract.
