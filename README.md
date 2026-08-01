# OpenAlpha

Constraint-preserving assurance and compatibility for financial foundation models.

OpenAlpha makes financial foundation-model forecasts structurally safe, auditable,
and deployment-ready. Its first integration adds a constraint-preserving
compatibility decoder and assurance layer to Kronos without retraining the
pretrained forecasting backbone.

The project remains **OpenAlpha**. Its primary public integration is **OpenAlpha
for Kronos**.

## Current status

OpenAlpha has completed the Sentinel v0, v1, and v1.1 development investigations.
Phase 1 proved the Bridge mathematical/runtime contract. Phase 2 then stopped at
its preregistered pre-data gate with `OPERATIONALLY_BLOCKED`: the locked 512-candle
window cannot fit inside the locked one-year validation partition or the daily
reconstruction-test partition without prohibited split crossing. No market data or
Kronos checkpoint was accessed, no Bridge checkpoint was trained, no benchmark was
evaluated, and the untouched Sentinel holdout remains unaccessed.

## The problem

Financial-model output can look plausible while violating elementary candle
structure. A downstream system must not silently treat a row as valid when, for
example, its high is below its open or close, its low is above them, or a price is
nonfinite or nonpositive.

OpenAlpha addresses structural correctness and safe downstream use. It does not
claim to make Kronos more profitable, improve every forecasting error, or create
alpha.

## What OpenAlpha found

All findings below are limited to the pinned revisions, assets, periods,
frequencies, horizons, and configurations recorded in this repository.

- Sentinel v0 found 473 of 936 generated paths invalid (50.53%), 961 of 4,680
  generated candles invalid (20.53%), and 57 of 104 canonical forecasts invalid
  (54.81%).
- The direct official Kronos output was invalid under identical inputs and seeds;
  OpenAlpha's mapping introduced no transformation error.
- Kronos-Tokenizer-2k reconstructed 4,357 of 15,360 valid observed candles as
  invalid (28.3659%).
- Kronos-Tokenizer-base reconstructed 3,555 of 15,360 valid observed candles as
  invalid (23.1445%).
- In the Sentinel v1.1 generated-token audit, 68 of 180 forecast candles were
  invalid (37.7778%). Supported token pairs were not safer than unsupported pairs,
  and median bounded valid probability mass was approximately 60%.
- The preregistered v1.1 conclusion was `TOKENIZER_CONSTRAINT_DEFECT`. The
  inference-only constrained-decoding track stopped under its locked rules.

The complete evidence remains under `research/sentinel-v0/`,
`research/sentinel-v1/`, and `research/sentinel-v1_1/`.

## What failed

Negative results are part of the product evidence, not discarded prototypes.

- Forecast-time diagnostics did not usefully rank later Kronos return error. The
  pooled chronological out-of-fold risk/error Spearman correlation was 0.08148.
- Abstention did not reduce accepted-forecast error, and the zero-return baseline
  beat Kronos at every declared coverage.
- Stepwise projection and re-encoding returned valid paths when successful but
  hard-failed 21 of 36 paths.
- Valid-candidate resampling returned 36 of 36 valid paths but failed the locked
  high-low range-error gate.

No untouched holdout result was accessed or used in any of these decisions.

## The root cause

At the pinned source revision, the official tokenizer converts each normalized
six-feature candle into hierarchical 10-bit coarse and 10-bit fine identifiers.
The official decoder reconstructs the combined 20-bit latent through a causal
sequence model and an unrestricted linear head for open, high, low, close, volume,
and amount. That head does not encode the financial ordering constraints.

The round-trip experiment demonstrated that valid observed candles can become
invalid before autoregressive forecasting begins. This does not establish that
Kronos is universally broken; it identifies a measured compatibility defect at the
tested continuous reconstruction boundary.

## The solution

OpenAlpha is a connected three-layer stack:

- **OpenAlpha Sentinel** validates timestamps, shapes, values, financial structure,
  and provenance; preserves raw output; creates immutable audits; and prevents
  invalid raw paths from being silently consumed.
- **OpenAlpha Bridge** consumes the unchanged Kronos hierarchical token sequence,
  reuses the frozen official tokenizer latent and causal decoder trunk, and learns
  only a small constraint-preserving reconstruction head.
- **OpenAlpha Evidence** preserves the reproducible investigation, failed methods,
  comparison artifacts, limitations, and exact claim boundaries.

Bridge output is separately labeled. It never overwrites or disguises the official
Kronos result.

## How OpenAlpha and Kronos work together

```text
Historical OHLCV
    |
    v
Official Kronos tokenizer encoder
    |
    v
Official coarse/fine token identifiers
    |
    v
Pretrained Kronos forecasting transformer (frozen)
    |
    v
Generated official token sequence (preserved)
    |---------------------------------------------|
    v                                             v
Official Kronos sequence decoder             OpenAlpha Bridge
    |                                         constrained head
    v                                             |
Raw official forecast                             v
    |                                     Separately labeled safe forecast
    |                                             |
    |---------------------> OpenAlpha Sentinel <--|
                              validation + audit
```

Kronos remains the forecasting model. OpenAlpha makes the generated token sequence
safe and auditable at the decoding boundary.

## Planned interface

The paired predictor is a Phase 6 target, not a currently released API:

```python
from openalpha.kronos import OpenAlphaKronosPredictor

predictor = OpenAlphaKronosPredictor.from_pretrained(
    model="NeoQuasar/Kronos-mini",
    tokenizer="NeoQuasar/Kronos-Tokenizer-2k",
    bridge="openalpha/bridge-2k-v0",
)

result = predictor.predict(df=history, pred_len=5)

result.raw_forecast
result.safe_forecast
result.raw_validation
result.safe_validation
result.token_audit
result.provenance
```

The runtime will reject model, tokenizer, source, feature, normalization, or Bridge
checkpoint combinations that do not match the checkpoint compatibility manifest.

## Benchmarks

Bridge benchmarks have not been run. Phase 2 stopped before data access because the
locked chronological partitions cannot form the locked 512-candle validation and
test sequences. The preserved experiment was intended to compare the same official
token sequences under four reconstruction paths:

| Decoder | Validity mechanism | Current evidence |
|---|---|---|
| Official Tokenizer-2k decoder | None | Measured round-trip invalidity: 28.3659% |
| Terminal projection | Deterministic post-decoding projection | Valid by construction; forecast range MAE 0.00959934 in Sentinel v1 |
| Linear residual + projection | Learned affine residual, then explicit projection | Baseline only; not yet evaluated |
| OpenAlpha Bridge-2K | Hard constrained financial parameterization | Design locked; not yet trained |

Success requires zero invalid Bridge candles, materially better range reconstruction
than terminal projection, non-inferior close/return and OHLC reconstruction, exact
token compatibility, deterministic replay, bounded resource use, and external
generalization. The numerical gates are frozen in
`research/bridge-v0/experiment.yaml` before training.

## Reproducibility

- Kronos source revision:
  `67b630e67f6a18c9e9be918d9b4337c960db1e9a`
- Kronos-mini revision:
  `f4e68697d9d5aed55cef5c96aabc3376bcad9f81`
- Kronos-Tokenizer-2k revision:
  `26966d0035065a0cae0ebad7af8ece35bc1fb51c`
- Kronos-Tokenizer-base revision:
  `0e0117387f39004a9016484a186a908917e22426`

See `docs/KRONOS_COMPATIBILITY_BOUNDARY.md` for source and checkpoint hashes,
`docs/OPENALPHA_KRONOS_BRIDGE.md` for the selected architecture, and the research
directories for immutable experiment artifacts.

Raw provider data, caches, and model checkpoints remain outside Git. Only compact
derived manifests, hashes, reports, and synthetic test fixtures may be committed.

## Limitations

- Bridge-2K feasibility is not yet established and no checkpoint exists.
- Structural validity does not imply forecast accuracy, profitability, calibration,
  or economic usefulness.
- Existing empirical results are development evidence, not untouched-holdout
  evidence.
- The tokenizer findings do not automatically generalize to other Kronos revisions,
  assets, frequencies, horizons, providers, or preprocessing choices.
- A constrained decoder cannot recreate information absent from the frozen token
  sequence.
- The initial data and compute budget is deliberately small; a negative result will
  be preserved rather than rescued with an unbounded corpus or model.

OpenAlpha is designed as an interoperable Kronos extension and a potential factual
upstream contribution, not as a replacement for Kronos.
