# OpenAlpha Methodology

## Research question

Does Kronos generate statistically meaningful and economically useful out-of-sample forecasts relative to simple forecasting methods, and does any apparent advantage survive realistic costs, different forecast horizons, and forward testing?

Plausible candles and low price-level error are not sufficient. Primary evidence concerns future return, return direction, and defensible magnitude/volatility targets, compared directly with baselines.

## Preregistration and evidence classes

Protocol v1 locks hypotheses, assets, periods, data representations, models, horizons, metrics, tests, trading rule, costs, failure/success criteria, exclusions, missing/corporate-action policies, and seeds before sealed evaluation. Changes require v2; unattractive outcomes never justify editing v1.

Every result is one of:

- **Historical replay:** retrospective rolling-origin forecasts generated under a locked procedure.
- **Sealed historical test:** a preregistered untouched evaluation segment.
- **Live precommitted forecast:** persisted before the outcome was knowable.

These classes remain separate in metrics and presentation.

## Universe and periods

The candidate v1 universe is SPY, QQQ, IWM, TLT, and GLD. Assets cannot be added or removed after sealed results are viewed.

- development: 2017-01-01 through 2022-12-31;
- validation: 2023-01-01 through 2023-12-31;
- replay-only contamination quarantine: 2024-01-01 through 2024-06-30;
- candidate sealed historical test: 2024-07-01 through 2025-12-31;
- live: first successful precommitted production forecast onward.

Kronos reports pretraining through June 2024, so the first half of 2024 cannot be represented as clean post-cutoff evidence. Final v1 dates freeze only after Alpaca availability, asset history, calendar coverage, and checkpoint feasibility are validated without inspecting candidate sealed performance.

## Data

Alpaca SIP daily bars are requested with explicit start, end, feed, timeframe, adjustment, and as-of behavior using environment credentials. There is no fallback. Forecast inputs and targets use `adjustment=all`; hypothetical fills use separately hashed `adjustment=raw` bars. Both preserve provider timestamps and normalized XNYS sessions.

Modern adjusted historical responses may contain later corrections and symbol mappings. Replay and sealed evidence therefore carry a point-in-time warning. Live forecasts preserve the snapshot actually retrieved before prediction. See `docs/DATA_POLICY.md`.

## Rolling-origin procedure

For every asset, horizon, model, and origin:

1. Construct only data available at the declared cutoff.
2. Normalize and fit transformations within the allowed window.
3. Fit models that require fitting.
4. Run the pinned Kronos checkpoint with the declared causal context and seed policy.
5. Persist all forecast or failure records before scoring.
6. Advance by the preregistered origin schedule without deleting dates.
7. Append outcomes only after the target horizon and calculate scores from forecast/outcome artifacts.

Random splits, future normalization, sealed-period tuning, favorable-date selection, and deleted failures are prohibited.

## Targets and horizons

Fixed horizons are 1, 5, and 20 XNYS sessions. All are reported.

Primary targets:

- future log return from the adjusted close at cutoff to the declared horizon close;
- sign of that return, with zero handling preregistered;
- absolute return or volatility magnitude only when the estimator and target window are explicit.

Secondary targets are close-price MAE/RMSE, OHLC path error, interval coverage, and supported volume error. Reports distinguish price level, return, direction, volatility, and visual candle realism.

## Models

The full benchmark contains Kronos, last value, random walk/drift, moving average, exponential smoothing or ARIMA, ridge, and one gradient-boosted tree. Every adapter receives identical allowed information and records model version, checkpoint/configuration, fit window, cutoff, horizon, runtime, hardware, seed, and artifact hashes. Real findings require real Kronos. Fakes exist only in tests.

## Forecast metrics

Primary metrics are return MAE/RMSE, directional accuracy, balanced directional accuracy where classes justify it, Brier score only for calibrated probabilities, Pearson correlation, and cross-sectional Spearman rank correlation when multiple assets resolve together. Secondary metrics cover price, volatility, interval coverage, horizon, asset, and time.

No isolated Kronos metric is admissible; every table includes the declared baselines, sample count, period, horizon, asset scope, evidence class, and uncertainty.

## Statistical evidence

Dependence-aware block bootstrap intervals are primary. Diebold–Mariano comparisons are used only with documented loss and overlap correction. Directional accuracy receives a dependence-aware interval. Holm correction covers the preregistered model/horizon hypothesis family. Stability is reported by asset and through time. Undefined or underpowered tests produce diagnostics, never favorable defaults.

## Economic evaluation

One preregistered long/cash rule holds the asset when predicted return exceeds the fixed threshold and otherwise holds cash. It executes no earlier than the next eligible raw bar, applies declared commission and slippage, and is never tuned on sealed results. Compare Kronos and baseline signals with buy-and-hold and cash.

Report total/annualized return, volatility, Sharpe, Sortino, maximum drawdown, turnover, cost drag, trade count, and hit rate. Forecast gains that disappear after costs are reported as economically unsuccessful.

## Live forecast protocol

A live forecast record includes protocol version/hash, creation time, asset/session/cutoff, provider/feed/snapshot, model/checkpoint/configuration/context, horizon, prediction/return/direction/uncertainty, strategy decision, expected resolution, runtime/hardware/seed, record hash, and previous hash. It is immutable. Resolution appends the realized outcome, errors, costs, and hypothetical P&L.

## Reporting

The scoreboard never collapses performance into one score. The audit view exposes original forecast, cutoff data, realized path, decision, costs, P&L, failures, neighboring records, protocol, and verification. The research paper obtains every numerical value from verified artifacts and treats negative or inconclusive evidence as a valid conclusion.

