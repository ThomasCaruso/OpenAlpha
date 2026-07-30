# OpenAlpha Methodology

## Governing question

Can information available at forecast time predict the future error of a financial forecasting model, and can selective use, blending, or abstention outperform blindly accepting every forecast?

The complete Sentinel v0 method is locked in `docs/SENTINEL_METHODOLOGY.md` and `research/sentinel-v0/experiment.yaml`.

## Study boundary

Sentinel v0 is development-only signal discovery:

- SPY and QQQ;
- Alpaca SIP daily adjusted bars;
- five-session horizon;
- last XNYS session of each ISO week;
- development: 2024-07-01 through 2025-06-30;
- untouched holdout: 2025-07-01 through 2026-06-30.

Protocol v1 remains unfrozen. The result is not a trading claim or scientific publication.

## Forecast ensemble

Real pinned Kronos-mini inference runs at context lengths 128, 256, and 512 under seeds 1729, 2027, and 7919. Every request uses temperature 1.0, top-p 0.9, and one generated path. The canonical forecast is the timestamp-wise arithmetic mean of only the three 512-context close paths; shorter contexts are stress tests. The raw last-value baseline predicts a flat path and zero five-session return.

All nine forecasts and failures are persisted before outcome access.

## Diagnostics

Fourteen causal diagnostics cover sampling agreement/dispersion, path divergence, context sensitivity, baseline disagreement, recent volatility and change, trend, gaps/outliers, historical analogue support/outcome dispersion, and recent resolved Kronos error.

Analogue outcomes and recent model errors must have resolved before the current cutoff. Exogenous events, news, filings, and macro data are excluded.

## Targets

- Continuous primary: absolute error of the canonical 512-context predicted raw five-session close-to-close log return.
- Binary failure: error at or above the pooled development 75th percentile, with that threshold frozen for holdout.
- Deployability: Kronos absolute error is strictly worse than the same-cutoff baseline.
- Secondary: direction correctness, realized return magnitude, and raw-close path error.

Profitability does not define v0 failure.

## Sentinel model

Development-only median imputation, missingness indicators, and standardization feed L2 logistic regression for failure probability. Ridge regression estimates future absolute error secondarily. Fixed expanding time-series folds select from declared regularization grids. Asset identity is not a feature.

Features missing above 20% or redundant above absolute Spearman 0.95 may be removed under the fixed interpretation rule. The retained set, preprocessing, models, thresholds, and reasons freeze before holdout.

## Actions

- USE: risk at or below the development median.
- BLEND: risk above the median through the development 80th percentile; combine Kronos and baseline 50/50.
- ABSTAIN: risk above the development 80th percentile.

Holdout outcomes cannot tune thresholds or weights.

## Evaluation

Report pooled and per-asset risk/error Spearman, error by risk quintile, risk-coverage curve, error and direction at 100/90/80/70/50% coverage, baseline comparison, probability calibration, confusion matrix, quarterly stability, diagnostic contribution, failures, latency, cache size, and cost.

The locked numeric criteria choose PROCEED, PROCEED WITH A NARROWER DIAGNOSTIC SET, CHANGE THE FAILURE-DETECTION APPROACH, or STOP THE PROJECT.

## Integrity

Forecast-time operations cannot read outcomes. Forecasts, diagnostics, decisions, outcomes, and postmortems are distinct immutable artifacts. Failed dates remain visible. All results resolve to the experiment bytes, data/model pins, code and dependency lock, and verified manifest.
