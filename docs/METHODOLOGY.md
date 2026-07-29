# OpenAlpha Methodology

## Scientific posture

OpenAlpha separates hypothesis definition, model selection, final evaluation, and interpretation. It reports negative and unstable results. A passed software test does not establish alpha; a profitable backtest does not establish a valid forecast; statistical significance does not establish economic significance.

## Phase 1 protocol

### Universe and period

The vertical slice uses one prespecified liquid US ETF and daily bars. Evaluation begins after 2024-06-30 because the Kronos paper states that pretraining data extends through June 2024. The exact symbol, provider, retrieval time, calendar, adjustment policy, and period are stored in the experiment specification.

This cutoff reduces direct temporal overlap but cannot prove absence of all pretraining contamination because the released checkpoint lacks source-data hashes and per-instrument membership.

### Data handling

1. Fetch through a provider adapter with explicit identity and version.
2. Save the raw response before transformation when provider terms permit.
3. Normalize column names, types, timezone, session calendar, and sort order.
4. Reject duplicate timestamps, non-finite prices, nonpositive prices, invalid OHLC relationships, and non-monotonic order.
5. Detect and report missing sessions, suspicious jumps, zero/stale volume, and outliers.
6. Record adjustment and corporate-action policy; never infer it from column names alone.
7. Store normalized Parquet and a provenance manifest under content hashes.
8. Never forward-fill prices or features across missing trading sessions without a specification rule and warning.

### Forecast origins

Each origin contains:

- information cutoff;
- causal context or training interval;
- optional validation interval;
- purge/embargo intervals;
- target timestamps;
- signal timestamp;
- earliest eligible execution timestamp.

The splitter generates timestamps from the declared exchange calendar, not by looking at future realized rows. A model adapter receives only its allowed slice. Preprocessors fit inside each training/context window.

### Kronos

The official implementation expects lowercase `open`, `high`, `low`, and `close`; `volume` and `amount` are optional. It z-scores the six channels using the supplied historical window, uses calendar embeddings, and samples coarse/fine tokens autoregressively.

OpenAlpha additionally:

- pins official code, model, and tokenizer revisions and verifies hashes;
- calls `eval()` explicitly on tokenizer and model;
- validates timestamp lengths, ordering, uniqueness, regularity, and cutoff;
- fixes and records RNG seeds and deterministic settings;
- records temperature, top-k, top-p, sample count, device, dtype, and runtime;
- constrains base/small context to no more than 512 bars;
- rejects non-finite outputs and records any causal OHLC/nonnegativity repair policy;
- never replaces a failed Kronos run with another model.

Stochastic paths are preserved when probabilistic evaluation is requested. Averaging paths is a declared estimator, not an undocumented implementation detail.

### Baselines

All baselines share the same origins and target timestamps:

- Random walk.
- Last value.
- Causal moving average.
- Exponential smoothing or ARIMA with training-only selection.
- Regularized linear model.
- Gradient-boosted trees using strictly causal features.

Baseline failure is reported; it does not remove that comparison from an inconvenient result table.

### Feature generation

Features declare their maximum source timestamp. Rolling statistics use trailing windows and do not center. Scaling, imputation, clipping, winsorization, and feature selection fit on the current training window. Fundamental, macroeconomic, and revised series are excluded until an as-of-aware adapter exists.

### Strategy and execution

The first strategy is intentionally simple and prespecified:

- derive expected horizon return from the forecast close and last observed close;
- enter long when it exceeds a declared threshold;
- otherwise remain in cash;
- generate the signal after the origin close;
- execute at the next eligible bar under the configured price convention;
- apply commission, spread/slippage, and execution delay;
- apply cash and position limits;
- record rejected orders and their reasons.

An idealized zero-cost view may be computed for cost attribution, but only the realistic view is admissible as the primary strategy result.

Buy-and-hold uses the same initial capital, date interval, execution-price policy, and applicable costs.

## Metrics

### Forecast

- Horizon-return MAE and RMSE.
- Price-level MAE and RMSE, labeled separately.
- MASE with a training-window scale.
- Directional accuracy, balanced accuracy, precision, recall, and class counts.
- Pearson and Spearman association when defined.
- Error by horizon, time window, and regime.

### Trading and risk

- Gross and net total/annualized return.
- Cost drag and turnover.
- Volatility, downside deviation, Sharpe, Sortino, Calmar.
- Maximum drawdown and duration.
- Exposure, hit rate, trade count, and average holding period.
- Benchmark-relative return, beta, tracking error, and information ratio when sample size permits.

Annualization conventions, risk-free rate, return frequency, and treatment of active drawdowns are stored with the metric artifact.

## Statistical inference

Dependence is assumed, not ignored.

- Confidence intervals use a stationary or moving-block bootstrap with block length registered before final evaluation.
- Forecast loss comparisons use a Diebold-Mariano-style statistic with a dependence correction appropriate to horizon overlap.
- Directional uncertainty uses a dependence-aware bootstrap; naive binomial intervals are labeled if shown.
- Multiple model/metric comparisons define a hypothesis family and apply Holm correction initially.
- Strategy Sharpe and drawdown uncertainty use resampled return paths with the limits of that procedure stated.
- Advanced phases add deflated Sharpe, probability of backtest overfitting, SPA/Reality Check where assumptions and sample size justify them.

Tests that are underpowered or undefined return an explicit diagnostic rather than a favorable default.

## Model selection

The final evaluation segment is untouched by:

- baseline order/hyperparameter selection;
- Kronos sampling-parameter selection;
- feature selection;
- strategy thresholds;
- cost-sensitivity choices;
- report narrative selection.

If the available history cannot support train, validation, purge, and final evaluation windows, the run fails validation. It does not shrink safeguards silently.

## Leakage-audit policy

### Failed checks

- Future timestamp in model or feature input.
- Preprocessing fitted outside the allowed training/context window.
- Target-derived feature.
- Same-bar signal and impossible fill.
- Overlapping labels without declared purge/dependence treatment.
- Post-selection evaluation on the selection segment.
- Benchmark used as a model input without declaration.
- Revised/as-known data mismatch.

### Warnings

- Unknown foundation-model pretraining membership.
- Provider lacks point-in-time or survivorship guarantees.
- Adjustment policy is inferred rather than provider-certified.
- Sample size below a registered inferential threshold.
- Unstable metrics or sparse direction classes.

Runs with a failed check are labeled `Failed` and cannot be presented as valid evidence. Warnings produce `Passed with warnings`.

## Reproducibility manifest

Every completed run records:

- experiment and run identifiers;
- canonical specification and schema version;
- Git commit and dirty-tree status;
- raw and normalized data hashes, provider, retrieval timestamp, snapshot ID;
- model/checkpoint/tokenizer revisions and hashes;
- all preprocessing, sampling, split, execution, cost, and reporting parameters;
- RNG seeds and deterministic flags;
- OS, architecture, Python, Node, dependency-lock, container/image, and hardware details;
- artifact inventory and hashes;
- methodology-audit findings;
- test/build evidence associated with the software revision.

## Reporting policy

Reports are generated from verified artifacts. They state:

- what was computed;
- what assumptions were imposed;
- what statistical evidence exists;
- whether magnitudes are economically meaningful after costs;
- what limitations prevent generalization.

No language model supplies or changes numerical report values.

## Primary sources

- [Kronos paper](https://arxiv.org/html/2508.02739v1)
- [Kronos predictor implementation](https://github.com/shiyu-coder/Kronos/blob/master/model/kronos.py)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [VectorBT](https://vectorbt.dev/)

