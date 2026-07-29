# Research Question and Evidentiary Standard

## Central question

> Do pretrained financial foundation models generate statistically robust and economically meaningful improvements over conventional forecasting methods after accounting for data leakage, transaction costs, market regimes, model-selection bias, and realistic execution constraints?

Kronos is the flagship case study, not the assumed winner. OpenAlpha is model-agnostic and must be capable of producing a credible negative result.

## Phase 1 estimands

The first study uses one liquid US ETF and daily bars after the Kronos pretraining cutoff. It estimates four distinct quantities:

1. **Forecast value:** the difference in out-of-sample horizon-return loss between Kronos and each prespecified baseline.
2. **Directional value:** the difference in correct horizon-return sign classification.
3. **Trading value:** the net performance difference between a prespecified forecast-derived strategy and its benchmarks after costs.
4. **Stability:** how the first three quantities vary through time and across deterministic market regimes.

Price-level resemblance, return predictability, and implementable trading value are not interchangeable.

## Prespecified hypotheses

For model `m`, forecast origin `t`, horizon `h`, realized log return `r[t,h]`, and forecast `r_hat[m,t,h]`:

- **H1 — Forecast loss:** Kronos has lower mean absolute horizon-return error than the random-walk, last-value, moving-average, statistical, and tree baselines.
- **H2 — Direction:** Kronos directional accuracy is greater than 0.5 and greater than the corresponding baseline accuracy.
- **H3 — Economic value:** the locked Kronos-derived rule improves net risk-adjusted performance over cash and buy-and-hold without unacceptable drawdown or turnover.
- **H4 — Stability:** any measured improvement is not concentrated entirely in one short time window or one deterministic regime.

The null for each comparison is no improvement. Failure to reject is reported as insufficient evidence, not proof of equivalence.

## Outcomes

Primary forecast outcome:

- Mean absolute error of the `h`-bar log-return forecast.

Secondary forecast outcomes:

- RMSE of horizon log returns.
- MAE and RMSE of predicted close prices, labeled as price-level metrics.
- MASE using only the training window for scaling.
- Directional accuracy, balanced accuracy, precision, and recall when both classes occur.
- Pearson and Spearman association, with undefined cases surfaced.
- Error by horizon step, forecast origin, and deterministic regime.

Primary economic outcome:

- Net excess return of the locked strategy over its declared benchmark, accompanied by turnover, costs, volatility, Sharpe, Sortino, Calmar, maximum drawdown, and drawdown duration.

No single metric may determine the conclusion.

## Comparison set

The first comparison set is fixed before evaluation:

- Random walk with innovations estimated from the causal context.
- Last observed value.
- Causal moving average.
- Exponential smoothing or ARIMA, selected by a training-only rule.
- Regularized or linear regression.
- Gradient-boosted trees.
- Kronos using a pinned official model and tokenizer revision.

The official Kronos paper benchmarks a broader task set and reports strong RankIC results, but those claims are not imported as OpenAlpha findings. OpenAlpha performs an independent, post-cutoff, cost-aware evaluation.

## Evaluation discipline

- Evaluation data begins after June 2024 because the paper states that Kronos pretraining extends through that month.
- Model and strategy hyperparameters are chosen on training/validation windows, never the final test observations.
- Each forecast origin has an explicit information cutoff and an ex-ante trading calendar.
- Overlapping horizon labels require a purge or dependence-aware inference.
- Signals generated from a close cannot execute at that same close.
- Costs are specified before viewing strategy results.
- The one-asset vertical slice is an engineering and methodology demonstration, not evidence of universal alpha.
- Additional models, assets, metrics, and parameter sweeps increase the hypothesis family and must be registered in the run manifest and corrected for multiplicity.

## Interpretation categories

Every conclusion must distinguish:

- **Computed fact:** a value read from a verified run artifact.
- **Statistical interpretation:** uncertainty and test result under declared assumptions.
- **Economic interpretation:** magnitude after costs, turnover, and risk.
- **Assumption:** a choice not established by the data.
- **Limitation:** a reason the estimate may not generalize or identify causal skill.

## Falsification and negative controls

Required controls as the platform matures:

- Shuffled or random-label signals.
- Deliberately lagged and placebo features.
- A zero-signal cash strategy.
- Transaction-cost monotonicity checks.
- Stability across windows and assets.
- Parameter perturbations around the registered choice.
- An untouched final segment after model/strategy selection.

## Sources

- [Kronos paper](https://arxiv.org/html/2508.02739v1)
- [Kronos official implementation](https://github.com/shiyu-coder/Kronos)

