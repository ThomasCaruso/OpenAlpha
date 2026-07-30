# Research Question and Evidentiary Standard

## Central question

> Does Kronos generate statistically meaningful and economically useful out-of-sample forecasts relative to simple forecasting methods, and does any apparent advantage survive realistic costs, different forecast horizons, and forward testing?

Kronos is the subject of the study, not the assumed winner. Negative, mixed, and inconclusive findings are valid outcomes.

## Evidence classes

Every forecast, outcome, metric, comparison, and report claim carries exactly one evidence class:

1. `historical_replay`: generated retrospectively under a locked rolling-origin process;
2. `sealed_historical_test`: generated for a period whose protocol was frozen before its outcomes were evaluated;
3. `live_precommitted_forecast`: stored immutably before the outcome was knowable.

Results may be displayed side by side, but they are never pooled or described as equivalent evidence.

## Initial estimands

For SPY, QQQ, IWM, TLT, and GLD at 1-, 5-, and 20-session horizons, the preregistered study estimates:

- the paired difference between Kronos and each baseline in log-return MAE and RMSE;
- directional accuracy, balanced directional accuracy where defined, and calibrated-probability quality where supported;
- correlation between predicted and realized returns, including cross-asset rank correlation where defined;
- the net results of one locked long/cash rule after declared costs and slippage;
- the stability of those results by asset and through time.

Price-level error, return prediction, direction prediction, volatility prediction, and plausible candle generation remain separate claims.

## Hypotheses

- **H1 — forecast accuracy:** Kronos reduces out-of-sample return loss relative to the declared baselines.
- **H2 — direction:** Kronos direction predictions outperform chance and the declared baselines.
- **H3 — economic value:** the locked Kronos signal improves net risk-adjusted results relative to cash, buy-and-hold, and equivalent baseline signals.
- **H4 — stability:** any improvement is not confined to one asset, horizon, or short time segment.

The null for a comparison is no improvement. Failure to reject is reported as insufficient evidence, not proof of equivalence.

## Comparison set

Protocol v1 must declare exactly these model families before sealed evaluation:

- Kronos with pinned code, checkpoint, tokenizer, and file hashes;
- last-value forecast;
- random walk or drift;
- causal moving average;
- exponential smoothing or ARIMA;
- one regularized linear model;
- one gradient-boosted tree model.

All models implement the same typed forecast-adapter contract. A deterministic fake is allowed only in automated tests and its outputs are always labeled synthetic.

## Temporal boundary

Development is 2017-01-01 through 2022-12-31 and validation is 2023-01-01 through 2023-12-31. Because Kronos reports pretraining through June 2024, January through June 2024 is historical-replay quarantine. The candidate sealed test is 2024-07-01 through 2025-12-31 and may be frozen only after provider-availability and model-feasibility checks that do not inspect performance. Live evaluation begins with the first successfully persisted production forecast.

## Interpretation rules

- Every Kronos metric appears beside declared baselines with sample counts and uncertainty.
- All forecast origins are rolling, causal, and retained, including model failures.
- A forecast is persisted before scoring; a later outcome is a new immutable record.
- No sealed-test threshold tuning, favorable-date selection, or post-result protocol edit is permitted.
- Signals formed at a close execute no earlier than the next permitted bar.
- Numerical report claims are rendered from verified artifacts, never typed by hand.

## Sources

- [Kronos paper](https://arxiv.org/html/2508.02739v1)
- [Kronos official implementation](https://github.com/shiyu-coder/Kronos)
