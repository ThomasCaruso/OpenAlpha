# Sentinel v0 Postmortem

**Status:** closed with a negative development result on 2026-07-31.

Sentinel v0 asked whether forecast-time instability, regime, analogue, recent-error,
and structural diagnostics could predict later five-session Kronos return error well
enough to support selective use, blending, or abstention. The answer from the locked
development study is no.

## Findings

1. Structural invalidity was widespread: 473 of 936 individual paths (50.53%), 961
   of 4,680 candles (20.53%), and 57 of 104 canonical forecasts (54.81%) violated at
   least one locked structural constraint.
2. Structural diagnostics did not effectively predict five-session return error.
   Canonical-invalid forecasts had mean absolute return error 0.03361, compared with
   0.03678 for canonical-valid forecasts; the chronological out-of-fold risk/error
   Spearman correlation was only 0.08148 pooled and had inconsistent signs by asset.
3. Selective abstention did not improve accepted-forecast MAE. Out-of-fold Kronos
   MAE was 0.03496 at 100% coverage and 0.03457 at 50% coverage, with worse results
   at 90%, 80%, and 70% coverage. Risk-bucket error was not monotonic.
4. The zero-return baseline outperformed Kronos throughout development: full-sample
   MAE was 0.02235 versus 0.03504, and the baseline also won at every declared
   out-of-fold coverage level.
5. The v0 reliability-risk model is retired. It will not be rescued through new
   features, classifiers, threshold tuning, or repeated analysis of these outcomes.
   The rejected valid-path aggregation candidate, which worsened mean error by
   0.00131 where applied, remains preserved.
6. The untouched holdout was not accessed. No holdout origin was created, forecast,
   fitted, inspected, scored, or summarized. The frozen but unused v0 policy remains
   preserved under SHA-256
   `c073d0e8d6cc4760211fc07f20c896444cd31d72a02d31a048e8c8d1ae6038a9`.

## What remains useful

The negative result does not erase the structural finding. The official pinned
Kronos-mini output frequently violated the OHLC domain contract, while OpenAlpha's
integration preserved that output faithfully. The validator, immutable artifacts,
projection audit, provider boundary, and untouched holdout remain useful. Sentinel
v1 therefore studies prevention of impossible autoregressive states, not prediction
of return error.

All v0 artifacts and Git history are retained unchanged.
