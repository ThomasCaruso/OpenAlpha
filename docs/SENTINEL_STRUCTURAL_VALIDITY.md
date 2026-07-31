# Sentinel Structural Validity

## Decision

Phase 2.5 concludes:

OFFICIAL_RAW_PATH_STRUCTURAL_INVALIDITY_CONFIRMED

This is a development integration finding, not evidence that structural invalidity predicts forecast error and not a general claim about all Kronos forecasts.

The pinned official predictor produced reproducible OHLC-ordering violations before OpenAlpha canonicalization. OpenAlpha's typed provider boundary and the sealed Phase 2 artifact preserved the direct official output exactly. No OpenAlpha transformation error was found.

## Pinned integration

- Official source: shiyu-coder/Kronos at 67b630e67f6a18c9e9be918d9b4337c960db1e9a
- Model: NeoQuasar/Kronos-mini at f4e68697d9d5aed55cef5c96aabc3376bcad9f81
- Tokenizer: NeoQuasar/Kronos-Tokenizer-2k at 26966d0035065a0cae0ebad7af8ece35bc1fb51c
- Executed source file: C:\Users\Tommy\.cache\openalpha-sentinel\phase2\kronos-src\model\kronos.py
- Traced functions: KronosPredictor.predict, KronosPredictor.generate, auto_regressive_inference, KronosTokenizer.encode, KronosTokenizer.decode, and calc_time_stamps

The official predictor accepts named OHLCV input, derives amount as volume times the mean of open, high, low, and close, normalizes and tokenizes the six features, decodes in normalized space, averages generated samples when sample_count is greater than one, applies its inverse transformation, and returns a DataFrame. It performs no OHLC validity projection or repair.

OpenAlpha removed yfinance MultiIndex structure, selected named open, high, low, close, and volume columns, and passed them in that order. It selected returned fields by name rather than position. The first prediction mapped to 2024-07-08 and the last to 2024-07-12. Baseline and realized returns used the same 2024-07-05 raw close.

## First discrepant boundary

There was no boundary discrepancy.

- Normalized input SHA-256, current versus sealed: f09446b7f7d541907401ca133580eac205c99acb5e27d84bf922943cc4d57bbe
- Direct official, typed provider, and sealed path SHA-256: 2155d9da024b0d0b878a029cca4a180520f59b67acab879eea98e6499cc7fe2c
- Direct equals provider: true
- Direct equals sealed: true
- Provider equals sealed: true

The direct official Phase 2 path at context 512/seed 1729 was already invalid: all five candles were invalid, with eight constraints violated and maximum normalized severity 0.0068618330. The independent official golden fixture also returned one small HIGH_BELOW_OPEN violation at step 3 with normalized severity 0.0003010225.

## Phase 2 frequency and severity

Across the sealed nine individual paths:

- invalid paths: 7/9 (0.7777777778)
- invalid candles: 24/45 (0.5333333333)
- total violations: 52
- high-low inversions: 11
- high-below-body violations: 36
- low-above-body violations: 5
- maximum normalized severity: 0.0140865932
- mean normalized severity: 0.0034681328
- nonfinite outputs: 0
- nonpositive prices: 0

Severity divides the absolute price constraint gap by the final observed raw close at the cutoff. Path validity, candle validity, counts, and severity remain separate diagnostics.

## Averaging and repeatability

The offline average of the three canonical 512-context paths remained invalid in 5/5 candles with seven violations. The diagnostic-only average of all nine paths remained invalid in 4/5 candles with four violations.

Official internal averaging did not remove the issue:

- sample_count=3: 5/5 candles invalid, 12 violations
- sample_count=5: 5/5 candles invalid, 11 violations

Internal averaging is not claimed to equal offline averaging because the internal sample sequence was not proven equivalent to the three separately seeded calls.

Repeating each selected 512-context path with the same input and seed reproduced its exact numerical output hash and the same violation steps, codes, gaps, and severities. Different seeds produced different output and violation hashes.

## Projection experiment

CONSTRAINT_PROJECTION_V0 preserved raw forecasts and changed only high and low:

    repaired_high = max(high, open, close, low)
    repaired_low  = min(low, open, close, repaired_high)

The projection restored the checked OHLC ordering deterministically. Open and close were unchanged for every candle, so every implied close return was exactly unchanged. This demonstrates a validity projection, not improved predictive accuracy; projected paths remain separate from raw model evidence.

## Canary

The canary used only three 512-context seeds at three additional development cutoffs for SPY and QQQ: 18 paths total. Forecast artifacts were sealed before the separate outcome operation.

- invalid paths: 12/18 (0.6666666667)
- invalid candles: 31/90 (0.3444444444)
- total violations: 53
- maximum normalized severity: 0.0137362017
- mean normalized severity: 0.0037325163
- high-low inversions: 8
- high-below-body violations: 38
- low-above-body violations: 7
- nonfinite outputs: 0
- nonpositive prices: 0

The phenomenon recurred in both assets, but one QQQ origin had 0/3 invalid paths. Six origins cannot estimate population frequency or establish correlation with error.

## Consequence

Structural validity is now a locked candidate Sentinel v0 diagnostic and a mandatory runtime gate. Raw output remains immutable. Fatal finite/alignment conditions block use; finite OHLC-ordering violations are labeled and retained for development evaluation. Optional projected output is separate.

## Phase 3A development result

The complete 2024-07-01 through 2025-06-30 development sample contained 104
eligible and completed origins, 52 each for SPY and QQQ, and 936 official
sample-count-one paths. No holdout origin was accessed.

- invalid paths: 473/936 (0.5053418803)
- invalid candles: 961/4680 (0.2053418803)
- invalid canonical 512-context averages: 57/104 (0.5480769231)
- invalid-path prevalence by context: 128 = 0.3429487179, 256 =
  0.4358974359, 512 = 0.7371794872
- invalid-path prevalence by seed: 1729 = 0.3621794872, 2027 =
  0.5641025641, 7919 = 0.5897435897
- violations by forecast step 1-5: 376, 262, 274, 328, 219

Structural invalidity was widespread and strongly context- and seed-dependent,
but it did not become monotonically more frequent later in the five-step
horizon. Canonical-invalid forecasts had lower mean absolute return error
than canonical-valid forecasts in this development sample
(0.0336116547 versus 0.0367770848). Individual invalid paths had higher
descriptive mean absolute return error than individual valid paths
(0.0311702181 versus 0.0290077646), but the chronological OOF structural-only
risk model had weak pooled risk/error rank association. Structural validity is
therefore justified as a downstream domain contract and safety gate, not
established as an incremental return-error warning signal.

CONSTRAINT_PROJECTION_V0 passed every checked structural constraint while
preserving every open, close, and implied close return. It remains a separate
path artifact and is not an accuracy intervention. Development-only valid-path
aggregation was applied at 22 origins and worsened mean absolute error by
0.0013143109 on average; it is not retained.

The verified compact evidence is under `research/sentinel-v0/development/`.
Complete per-origin chains and provider-derived rows remain private outside Git.

Compact evidence is under research/sentinel-v0/phase2_5/. Full comparison receipts and model/data caches remain outside Git.
