# OpenAlpha Bridge v0 Preregistration

**DEVELOPMENT COMPATIBILITY RESEARCH - NOT HOLDOUT EVIDENCE**

## Lock and scope

This preregistration governs the first OpenAlpha Bridge-2K feasibility program. It
is committed before Phase 1 implementation, data retrieval, model fitting, Bridge
test evaluation, forecast integration, or untouched-holdout access.

The locked `experiment.yaml` SHA-256 is
`d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2`; the same
value is stored in `experiment.sha256`.

The numerical source of truth is `experiment.yaml`. If prose and YAML differ, the
YAML is controlling and the discrepancy must be corrected before any affected phase
begins. Thresholds may not move after the corresponding test output is visible.

This program may train only the declared OpenAlpha reconstruction head. It may not
train or modify the Kronos encoder, implicit codebook, tokenizer decoder trunk,
forecast-model embeddings, forecasting transformer, or original six-output decoder
head. It may not change a generated coarse or fine ID.

## Research question

Can the same ordered Tokenizer-2k hierarchical identifiers already produced for
Kronos-mini be reconstructed through a learned hard financial parameterization with
zero invalid candles, materially better range fidelity than deterministic terminal
projection, and no material loss in OHLC or close/return reconstruction?

Forecast quality is a later gated question. The first decision concerns real-token
reconstruction only.

## Compatibility lock

Bridge-2K uses:

- official source revision
  `67b630e67f6a18c9e9be918d9b4337c960db1e9a`;
- `model/kronos.py` SHA-256
  `638a56e035856c600c9848b368be087cb706a61603a0790124968c95b8c69f3a`;
- `model/module.py` SHA-256
  `a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f`;
- Kronos-mini revision
  `f4e68697d9d5aed55cef5c96aabc3376bcad9f81`;
- Tokenizer-2k revision
  `26966d0035065a0cae0ebad7af8ece35bc1fb51c`;
- tokenizer config and weight hashes recorded in `experiment.yaml`; and
- implicit 10+10-bit BSQ codebook fingerprint
  `ea1825698624a0835a2238d9bfe8c5ebbd39e074fb4daa9613423f0e4df23af2`.

The runtime fails closed on any incompatibility. Tokenizer-base and larger Kronos
models are not Bridge v0 fallbacks.

## Phase 1 mathematical contract

Before learned work, implement the representation and inverse specified in
`docs/BRIDGE_MATHEMATICAL_REPRESENTATION.md`.

The implementation must:

- accept typed batches of supported valid OHLC with optional volume;
- represent each candle as gap return, body return, nonnegative upper wick,
  nonnegative lower wick, and optional nonnegative log-volume;
- use the last observed close as the first anchor and each reconstructed close as
  the next anchor;
- map finite neural logits through bounded `tanh` and bounded stable-softplus
  functions;
- produce no candle on nonfinite or unsupported numerical input;
- guarantee the complete OHLC(V) contract by construction;
- retain missing volume as missing and derive amount only when requested;
- serialize deterministically with no NaN or infinity; and
- pass the independent Sentinel structural validator.

At least 10,000 property-generated supported examples must pass. Supported
source-representation round trips use `rtol=1e-12` and `atol=1e-12` in float64.
Two identical serialization replays must produce identical canonical hashes.
Perturbing a future parameter must not change an earlier candle.

Any need for post-output projection fails Phase 1. A numerical implementation bug
may be corrected before Phase 1 is sealed, but no learned work begins until fresh
full verification passes.

## Selected decoder architecture

The official tokenizer decoder is sequence-level and causal. Bridge therefore uses
the full ordered token window rather than independent pair lookup.

The frozen path is:

1. official coarse/fine IDs to official 20-dimensional bipolar latent;
2. official `post_quant_embed` from 20 to 256 dimensions; and
3. the three official causal tokenizer decoder blocks.

The trainable path concatenates the 256-dimensional state with 13 causal scale
features and applies `Linear(269,64) -> SiLU -> Linear(64,5)`. It has 17,605
trainable parameters. The five logits enter the Phase 1 hard representation.

No architecture search is authorized after test results. Failure of this fixed
small head within the data and compute cap stops Bridge v0 rather than authorizing a
larger decoder.

## Causal reconstruction examples

Each example is a 512-candle valid real sequence divided into a 448-candle prefix
and 64-candle target suffix. The six-feature population mean and standard deviation
are computed from the prefix only. The prefix and real suffix are normalized with
that frozen state, clipped exactly as the official predictor, and encoded by the
frozen official tokenizer.

Bridge sees the ordered identifiers, their frozen official decoder state, causal
scale features, and the prefix's final close. It is supervised only on the 64 target
candles. The real target suffix is used to create reconstruction target tokens and
loss targets during training; it is not supplied as a decoder feature.

Training windows have stride 64. Validation and test target suffixes do not overlap,
and no example crosses a chronological split.

## Data and storage

Primary feasibility data uses unadjusted daily observations through the repository's
pinned yfinance adapter with exact request options in `experiment.yaml`.

- Train symbols: SPY, QQQ, XLF, XLK, XLE, XLI, XLV, XLP, XLY, XLU, TLT, HYG,
  EFA, and VNQ.
- Unseen-symbol set: IWM, DIA, GLD, EEM, SLV, and USO.
- Train period: 2010-01-01 through 2021-12-31.
- Validation period: calendar year 2022.
- Reconstruction test: 2023-01-01 through 2024-06-28.
- Later external period: 2024-07-01 through 2025-06-30.

The first complete Bridge-2K program may retrieve no more than 250,000 complete
candles. Raw data is streamed in bounded batches, validated and hashed in memory,
and not committed. Temporary cache is outside Git, capped at 10 GiB, and cleanable
after the compact manifest is sealed. Checkpoints stay outside Git.

Phase 4 may add the declared Binance Vision BTCUSDT and ETHUSDT one-day and one-hour
archives only after a data-policy review and only for the locked later period. No
replacement symbols, periods, frequencies, or providers may be selected after a
generalization result is visible.

Low, middle, and high volatility regimes are assigned by the causal 20-bar
population standard deviation of log-close returns. The one-third and two-third
thresholds are calculated from the pooled training split only, serialized, hashed,
and frozen. Each external regime requires at least 500 candles.

No observation on or after 2025-07-01 may enter any Bridge development activity.

## Training lock

Train only after Phase 1 succeeds. Use AdamW at learning rate `1e-3`, betas
`[0.9,0.999]`, epsilon `1e-8`, weight decay `1e-4`, gradient norm cap `1.0`, batch
size 64, seed 1729, and at most 50 epochs. Stop after five validation epochs without
an improvement of at least `1e-4`. Select the lowest validation-total-loss
checkpoint. Primary feasibility uses float32 without mixed precision.

Loss coordinate scales are training-only median absolute deviations with floor
`1e-4`; they are serialized and sealed before validation/test reporting. Huber delta
is 1.0 standardized unit.

The loss contains only:

- weight 1.0 for the mean gap/body loss;
- weight 1.0 for the mean upper/lower-wick loss;
- weight 1.0 for derived log high-low range loss; and
- weight 0.25 for masked log-volume loss.

When volume is absent, its group is omitted and active weights are renormalized.
There is no validity penalty and no direct OHLC level loss.

The Phase 2 cap is one accelerator, at most 24 GiB device memory, and at most 24
GPU-hours. The Bridge may have no more than one million trainable parameters or a
25 MiB checkpoint.

## Baselines

All baselines receive identical held-out tokens and causal state.

1. Official Tokenizer-2k sequence decoder.
2. Terminal projection that preserves raw open and close, sets high to
   `max(raw high, raw open, raw close)`, and sets low to
   `min(raw low, raw open, raw close)`.
3. Ridge residual baseline with alpha `1e-3`, fit intercept, inputs consisting of
   the official normalized six-vector and 20 latent bits, six normalized residual
   outputs, and the same explicit terminal projection.
4. OpenAlpha Bridge-2K with no post-output projection.

Projection outputs and Bridge output are always separately labeled from the raw
official output.

## Phase 2 reconstruction decision

Evaluation requires at least 5,000 unique held-out target candles and at least 500
unique unseen-symbol target candles. Reconstruction price errors are divided by
each target row's close. Statistical units are nonoverlapping target suffix
sequences.

Paired confidence intervals use 10,000 sequence bootstrap resamples with seed
20260731 at 95% confidence.

Bridge-2K passes only if every gate below passes:

1. Invalid reconstructed-candle fraction is exactly 0.
2. Token-ID parity and frozen-weight hash parity are exactly 1.0.
3. No post-output projection exists.
4. Range MAE is at most 0.90 times terminal-projection range MAE and the upper 95%
   bound for the paired Bridge-minus-projection difference is below zero.
5. Full OHLC MAE and every individual OHLC MAE are no greater than
   `1.05 * official + 0.0005`.
6. Close MAE is no greater than `1.05 * official + 0.0005`.
7. Close-return MAE is no greater than `1.05 * official + 0.00025`.
8. Two same-runtime replays produce the same canonical artifact hash.
9. Median latency is no greater than two times official decoding, p95 incremental
   latency is at most 50 ms, and incremental peak memory is at most 512 MiB.
10. The parameter and checkpoint caps pass.

No forecast run is authorized when any gate fails.

## Phase 3 forecast decision

Only after Phase 2 success, use the exact 12 preserved Sentinel v1 development
origins for SPY and QQQ, five daily steps, context 512, seeds 1729/2027/7919,
temperature 1.0, top-p 0.9, top-k 0, and one sample per token path. Preserve each
generated pair sequence before either decoder runs.

Compare the official decoder, terminal projection, and Bridge on paired tokens.
Forecast price errors, path distances, and diversity reuse the preserved
`openalpha_sentinel.decoding_metrics` definitions and origin cutoff-close
denominator. Barrier accuracy is the equal mean of positive-touch, negative-touch,
and either-touch correctness at 0.5%, 1.0%, and 2.0% barriers.

The forecast gates repeat zero invalid candles/paths, exact token and weight parity,
no projection, 10% range improvement, OHLC and close/return non-inferiority,
same-runtime deterministic replay, and resource limits. In addition:

- five-session return MAE must be no greater than
  `1.05 * official + 0.0005`;
- mean barrier-event accuracy degradation may not exceed 0.05 absolute;
- pairwise path diversity must be at least 75% of raw diversity;
- final return variance must be positive; and
- repeated-path rate must not exceed 10%.

The untouched holdout is not run even when Phase 3 succeeds.

## Phase 4 external generalization

Only after Phase 3 success, evaluate the six declared slices in `experiment.yaml`:
unseen equities in the reconstruction and later periods, plus BTCUSDT and ETHUSDT
at daily and hourly frequency in the later period.

Every slice must have zero invalid candles and range MAE no greater than 0.95 times
terminal projection. Pooled close and close-return errors must meet the Phase 2
non-inferiority margins. Failure of any required slice fails external
generalization; slices are not silently pooled away. The same zero-invalid and 0.95
range-ratio gates must pass separately in each low, middle, and high volatility
regime.

## Stop rule and preserved failure product

Stop Bridge v0 when a locked phase fails at its data or compute cap, the method
needs future information or a repair pass, improvement is confined to training
symbols/periods, compatibility requires Kronos retraining, or any resource cap is
exceeded.

Do not respond by changing the threshold, adding an undeclared corpus, expanding
the head, accessing the untouched holdout, or beginning a new conceptual pivot.

A negative result remains a complete OpenAlpha result. The product then consists of
Sentinel structural validation, immutable raw-output audit, deterministic projection
gateway, tokenizer compatibility profiling, and the reproducible Bridge evidence.

## Claims

Permitted claims are limited to measured structural validity, reconstruction
quality, paired forecast metrics, compatibility, and operational performance under
the exact audited configuration.

Prohibited claims include improved profitability, universal Kronos failure,
provider-independent market truth, untouched-holdout performance, compatibility
with untested revisions, or a drop-in checkpoint before runtime rejection and all
gates are demonstrated.
