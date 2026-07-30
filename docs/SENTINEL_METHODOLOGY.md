# Sentinel v0 Methodology

## Status and claim boundary

Sentinel v0 is a locked development and feasibility experiment. It is not a sealed scientific publication, production reliability system, trading strategy, or public claim that Sentinel or Kronos works.

The governing question is:

> Can information available at forecast time predict the future error of a financial forecasting model, and can selective use, blending, or abstention outperform blindly accepting every forecast?

Every forecast and diagnostic vector is persisted before its five-session outcome is supplied to the resolver. Decisions and reasons begin only after the development risk configuration is frozen; for each holdout origin they are then persisted before that origin's outcome is supplied to the resolver.

## Fixed experiment

| Element | Decision |
|---|---|
| Assets | SPY, QQQ |
| Frequency | Yahoo Finance raw `1d` bars through `yfinance==1.5.2` |
| Forecast horizon | 5 XNYS sessions |
| Development period | 2024-07-01 through 2025-06-30 |
| Untouched holdout | 2025-07-01 through 2026-06-30 |
| Cutoff rule | Last valid XNYS session of each ISO week |
| Expected sample | About 52 cutoffs per segment and 104 per asset |
| Baseline | Raw last-value path; predicted five-session return is zero |
| Kronos checkpoint | `NeoQuasar/Kronos-mini` |
| Context lengths | 128, 256, 512 sessions |
| Sampling seeds | 1729, 2027, 7919 |
| Sampling configuration | `temperature=1.0`, `top_p=0.9`, one generated path per request |

Before the first model fit, a calendar preflight materializes the exact ordered cutoff list and records every excluded or unusable cutoff. It first computes the final XNYS session of each complete ISO week and includes that origin only when the computed session is inside the applicable segment. It does not create partial-week origins at the development/holdout boundary. Provider or model failures remain records and do not cause a replacement date to be selected.

Development and holdout boundaries do not change after this commit. The preflight may reduce usable counts only for declared insufficient-history, missing-outcome, provider-quality, or inference-failure reasons.

## Data boundary

Sentinel v0 requests SPY and QQQ from Yahoo Finance through the pinned open-source `yfinance==1.5.2` client. yfinance is not an official Yahoo integration or institutional point-in-time source. Each request explicitly sets `interval="1d"`, `auto_adjust=False`, `back_adjust=False`, `repair=False`, `actions=True`, `progress=False`, `threads=False`, `timeout=30.0`, `prepost=False`, `rounding=False`, `keepna=False`, and `multi_level_index=False`. There is no provider fallback.

The Phase 2 creation request starts inclusively on 2022-06-01 and ends exclusively on 2024-07-06. Its constructor rejects post-cutoff rows and requires 2024-07-05 as the exact final XNYS session. Only after the immutable forecast is sealed may the separate resolver request 2024-07-06 through the exclusive end 2024-07-13 and accept exactly the declared five outcome sessions. Later development/holdout request windows will preserve the same per-origin causal rule.

Raw provider responses are ephemeral and never committed or redistributed. Complete reusable histories, yfinance caches, and CSV exports remain outside Git. Normalized raw OHLCV input may be retained only in a private user-local artifact when permitted and required; Git stores request parameters, client version, retrieval timestamp, schema, row count/range, normalized-input hash, quality results, corporate-action warnings, predictions, outcomes, and derived metrics.

Only Open, High, Low, Close, and Volume enter Kronos. Adj Close is excluded, and dividend/split columns are audit metadata only. The primary outcome is the raw five-session close-to-close log return. This avoids treating a currently revised adjusted history as point-in-time truth, but it omits dividend return and can contain corporate-action discontinuities. Sentinel v0 does not add a corporate-action engine or make trading-return claims.

Phase 2 performs no cross-provider check. Before serious publication, a representative sample must be rerun through an independently sourced provider such as Alpaca, with row, return, forecast, diagnostic, and conclusion differences reported.

## Real Kronos inference boundary

The official Hugging Face page currently reports that no Inference Provider deploys Kronos-mini. Sentinel therefore selects the third-priority fallback: an on-demand local process using a temporary Hugging Face cache outside the repository. No permanent endpoint is created.

Phase 2 begins with these immutable feasibility pins:

- official source: `shiyu-coder/Kronos`;
- source revision: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`;
- checkpoint: `NeoQuasar/Kronos-mini@f4e68697d9d5aed55cef5c96aabc3376bcad9f81`;
- tokenizer: `NeoQuasar/Kronos-Tokenizer-2k@26966d0035065a0cae0ebad7af8ece35bc1fb51c`;
- reported model parameters: 4,108,192;
- reported Hub storage: 16,440,776 bytes for the model and 15,842,376 bytes for the tokenizer.

Downloaded file hashes, runtime environment, device, and cache path class are recorded before the first forecast. A deterministic fake provider is admissible only in automated tests and is always marked synthetic.

## Minimal contracts

The contracts below define the fields Phase 2 must implement. They are Sentinel-v0 contracts, not a universal public schema.

### Forecast provider request

| Field | Meaning |
|---|---|
| `model_id`, `checkpoint_id`, `source_revision` | Exact forecast implementation identity |
| `timestamps` | Strictly ordered causal observation timestamps |
| `observations` | Finite OHLCV rows aligned one-to-one with timestamps |
| `context_length` | One of 128, 256, 512 |
| `forecast_horizon` | Exactly 5 |
| `sampling_seed` | One of the three declared seeds |
| `temperature`, `top_p` | Exactly 1.0 and 0.9 in v0 |
| `generated_path_count` | Exactly 1 per request |
| `request_metadata` | Symbol, cutoff, calendar, data hash, experiment hash, creation time |

### Forecast provider response

| Field | Meaning |
|---|---|
| `generated_paths` | Ordered five-step OHLCV paths; empty on failure |
| `forecast_summary` | Implied cumulative log return and direction |
| `inference_duration_ms` | Provider-measured wall time |
| `provider_id`, `checkpoint_id`, `request_id` | Exact provenance |
| `failure` | Typed code/message without credentials, or null |

### Diagnostic vector

| Field | Meaning |
|---|---|
| `asset`, `cutoff`, `forecast_horizon` | SPY/QQQ identity, causal origin, and fixed five-session horizon |
| `forecast_ids` | Exact nine response artifact identities |
| `baseline_forecast_id` | Causal last-value artifact identity |
| `values` | Fixed-order entries containing diagnostic name, finite value or null, units, and missingness code |
| `available_at` | Timestamp before outcome access |
| `causal_through` | Latest timestamp any diagnostic input may contain |
| `input_sha256` | Context, forecast, baseline, eligible-analogue, and resolved-prior-error identities |
| `configuration_sha256` | Hash of the locked diagnostic definitions |

### Sentinel decision record

| Field | Meaning |
|---|---|
| `forecast_ids` | The nine ensemble request/response artifact identities |
| `baseline_forecast_id` | Causal last-value artifact identity |
| `diagnostic_vector_id` | Immutable pre-outcome diagnostic-vector identity |
| `failure_probability` | Logistic model probability, or null before model fitting |
| `predicted_absolute_error` | Secondary ridge estimate, or null before model fitting |
| `reliability_score` | Rounded `100 × (1 - failure_probability)` |
| `action` | USE, BLEND, or ABSTAIN; never another value |
| `failure_reasons` | Up to three deterministic taxonomy records |
| `decision_created_at` | Time before outcome access |
| `provenance` | Experiment, data, model, diagnostic, and risk-config hashes |

A Sentinel decision is created only after Phase 3 has fitted and frozen the development risk configuration. Phase 2 persists the forecast and diagnostic vector but does not manufacture a failure probability, reliability score, action, or reason from an unfitted model.

## Controlled ensemble

Each cutoff creates nine independent requests: three context lengths crossed with three seeds. Each request uses `sample_count=1` so the system receives one actual path instead of an average that hides sampling disagreement. No temperature/top-p sensitivity branch is included in v0.

Only the three 512-context paths define the canonical Kronos forecast. Their predicted raw close values are averaged timestamp by timestamp. The canonical predicted return is `log(final averaged predicted close / final observed raw close at cutoff)`. Each individual 512 path remains an immutable artifact.

The six 128- and 256-context paths are stress-test evidence only. They contribute to sampling and context diagnostics but never to the canonical point forecast. For context comparisons, each context-level summary is the return implied by the timestamp-wise arithmetic mean of its three seeded close paths.

Before the official nine paths, Phase 2 runs 512/1729 twice and 512/2027 once. Python, NumPy, PyTorch CPU, and applicable accelerator RNGs reset before every call. Canonical output hashes determine whether exact seeded replay is supported; a mismatch is preserved and reported, never hidden by modifying Kronos.

## Pre-outcome diagnostics

Let `r[i]` be path `i`'s five-session cumulative log return and `r[i,s]` its cumulative return through step `s`. All price-derived windows end at the cutoff.

1. **DIRECTIONAL_AGREEMENT:** modal share of the nine signs of `r[i]`.
2. **RETURN_DISPERSION:** sample standard deviation of `r[i]`, reported in basis points.
3. **PATH_DISPERSION:** mean across steps 1–5 of the cross-path standard deviation of `r[i,s]`, divided by trailing 20-session daily volatility times `sqrt(s)`; a zero volatility denominator is an explicit missing value.
4. **CONTEXT_DIRECTION_AGREEMENT:** modal share of the three context-level median-return signs.
5. **CONTEXT_RETURN_SPREAD:** absolute difference in basis points between the 128- and 512-context summary returns.
6. **BASELINE_DISAGREEMENT:** absolute difference between the canonical three-path 512-context return and the baseline's zero return.
7. **RECENT_VOLATILITY:** annualized sample standard deviation of the last 20 daily log returns using `sqrt(252)`.
8. **VOLATILITY_CHANGE:** absolute log-ratio of 20-session to 60-session realized volatility; zero denominators are missing.
9. **TREND_STRENGTH:** absolute 20-session cumulative log return divided by the sum of absolute daily log returns over the same window, bounded to `[0, 1]`.
10. **GAP_OR_OUTLIER_SCORE:** maximum absolute robust z-score among the last 20 close-to-close returns and open-versus-prior-close gaps, using the preceding 252 eligible sessions' median and MAD.
11. **HISTORICAL_ANALOGUE_DISTANCE:** mean Euclidean distance, divided by `sqrt(20)`, to ten eligible z-normalized 20-return windows whose five-session outcomes resolved before the cutoff.
12. **ANALOGUE_OUTCOME_DISPERSION:** sample standard deviation of the resolved five-session returns following those ten nearest contexts.
13. **RECENT_MODEL_ERROR:** mean absolute return error from the eight most recent resolved scheduled Kronos forecasts for the same asset; missing until eight exist.
14. **HORIZON_PATH_DIVERGENCE:** ordinary least-squares slope of cross-path cumulative-return standard deviation against steps 1–5.

Analogue search uses only data and outcomes available before the current cutoff. Candidate windows are normalized from their own causal values, ordered by distance, and greedily selected while excluding candidates whose input or outcome sessions overlap an already selected candidate. Both analogue diagnostics are `not_computable` unless ten candidates remain. The search never uses unresolved outcomes or holdout future.

At the first origin, `RECENT_MODEL_ERROR` is `not_computable` with reason `NO_PRIOR_RESOLVED_FORECASTS`. `UNCERTAINTY_MISCALIBRATION` is separately reported as unavailable with reason `NO_PRIOR_CALIBRATION_SAMPLE`; neither is replaced with zero.

Expected univariate associations with future absolute error are locked as follows: directional and context-direction agreement are negative; return/path dispersion, context spread, baseline disagreement, recent volatility, volatility change, gap/outlier score, analogue distance, analogue outcome dispersion, recent model error, and horizon path divergence are positive. Trend strength has no preregistered sign and is ineligible to satisfy S5 by itself. These signs are research expectations, not assumptions used to calculate the diagnostics.

## Targets and labels

The primary continuous target is:

```text
absolute_return_error =
    abs(canonical_512_predicted_raw_log_return - realized_raw_log_return)
```

The development failure threshold is the pooled 75th percentile of development absolute return errors. The binary label is `absolute_return_error >= frozen_development_threshold`. That numeric threshold is applied unchanged to holdout observations.

The secondary deployability label is true when Kronos absolute return error is strictly greater than the last-value baseline absolute return error at the same cutoff. Ties are false.

Each resolved outcome also records realized return magnitude, direction correctness, and mean absolute error of the canonical averaged raw-close path. V0 never defines failure from trading profitability.

## Development discipline

The pooled model uses SPY and QQQ rows but does not receive symbol identity as a predictor. Both rows for a cutoff date stay in the same fold.

Development processing is fixed:

1. Remove a diagnostic only if it is invalid by definition, missing in more than 20% of development rows, or has an absolute development Spearman correlation above 0.95 with a more directly interpretable diagnostic.
2. Fit median imputation and a missingness indicator for each retained diagnostic using development data only.
3. Standardize retained values with development-only means and standard deviations.
4. Select L2-logistic `C` from `[0.01, 0.1, 1.0, 10.0]` by mean log loss under a three-split expanding `TimeSeriesSplit` over unique cutoff dates with a one-week gap.
5. Fit secondary ridge absolute-error regression with `alpha` from `[0.1, 1.0, 10.0, 100.0]` by mean validation MAE under the same folds.
6. Freeze retained diagnostics, preprocessing statistics, coefficients, failure threshold, reason thresholds, action thresholds, and blend weight before any holdout outcome is scored.

Gradient-boosted trees are excluded from the primary v0 proof. They may be reported later as a secondary diagnostic only after the linear proof is complete.

## Risk score and actions

The failure probability is the logistic output. Reliability is `round(100 × (1 - failure_probability))`, clamped to 0–100.

Development risk-score quantiles freeze the policy:

- **USE:** predicted risk at or below the development 50th percentile; return the Kronos median forecast.
- **BLEND:** predicted risk above the 50th and at or below the 80th percentile; return a deterministic 50/50 average of the Kronos and last-value paths.
- **ABSTAIN:** predicted risk above the development 80th percentile; return no deployable forecast.

No action threshold or blend weight is selected from holdout outcomes.

## Holdout evaluation

The holdout is run once with the frozen development pipeline. Required outputs are:

- pooled and per-asset Spearman correlation between risk and future absolute Kronos error;
- mean error by holdout risk quintile;
- risk-coverage curve;
- accepted-forecast MAE at 100%, 90%, 80%, 70%, and 50% coverage;
- directional accuracy at the same coverage levels;
- Kronos and last-value baseline results on identical rows;
- Brier score, calibration intercept/slope, and five-bin calibration table for the failure label;
- confusion matrix at the development-frozen 50% failure-probability decision threshold;
- USE, BLEND, and ABSTAIN performance;
- stability by asset and calendar quarter;
- univariate diagnostic/error correlations with their expected direction;
- request success rate, model/provider failures, median and 95th-percentile latency, cache size, and monetary cost when applicable.

Coverage accepts the lowest-risk holdout rows. Holdout quintiles use risk ranks only, never outcomes, and all five declared coverage levels are reported.

## Continuation criteria

These criteria are locked before holdout evaluation.

### Signal criteria

- **S1 — rank signal:** pooled holdout Spearman correlation is at least 0.20.
- **S2 — selective error reduction:** Kronos MAE at 70% coverage is at most 90% of Kronos MAE at 100% coverage.
- **S3 — risk monotonicity:** highest-risk-quintile error exceeds lowest-risk-quintile error and at least three of four adjacent quintile changes are nondecreasing.
- **S4 — cross-asset support:** SPY and QQQ each have positive risk/error Spearman correlation and each has 70%-coverage MAE no greater than its own 100%-coverage MAE.
- **S5 — diagnostic support:** at least one diagnostic with a non-null preregistered sign has that expected holdout correlation sign with absolute error and absolute Spearman magnitude of at least 0.15.

### Feasibility criterion

- **S6 — operational feasibility:** at least 95% of declared Kronos requests succeed; median nine-request ensemble latency is at most 600 seconds per cutoff; model/tokenizer cache remains at most 2 GiB outside Git; and hosted-equivalent cost, if charged, is at most USD 0.50 per cutoff.

### Decision rule

Apply rules in this order:

1. **STOP THE PROJECT** if S6 fails after the selected local-cache path and one documented ephemeral alternative are each found unreasonable, or if pooled Spearman is nonpositive and 70%-coverage MAE is not lower than 100%-coverage MAE.
2. **PROCEED** if S1 through S6 all pass.
3. **PROCEED WITH A NARROWER DIAGNOSTIC SET** if S1, S2, and S6 pass but one or more of S3–S5 fail.
4. **CHANGE THE FAILURE-DETECTION APPROACH** if S6 passes and either pooled Spearman is positive or 70%-coverage MAE improves, but the requirements for the two proceed decisions are not met.
5. **STOP THE PROJECT** for every remaining outcome.

The report includes the criterion table regardless of outcome. It may discuss effect size and uncertainty but cannot change this decision mapping.

## Artifact and lifecycle mapping

Existing infrastructure is reused directly:

- forecast requests and responses, diagnostic vectors, decisions, resolved outcomes, and reports are content-addressed artifacts;
- run journals record created, running, failed, and completed experiment stages;
- development lifecycle records append forecast creation, diagnostic completion, and outcome resolution without mutating prior payloads;
- holdout lifecycle records additionally append the frozen-configuration Sentinel decision before outcome resolution and a postmortem afterward;
- completed manifests bind data provenance, model pins, diagnostic configuration, risk configuration, code/lock hashes, and evaluation artifacts;
- path confinement controls every local artifact read.

The existing manifest artifact-kind vocabulary may carry Sentinel payloads where meanings align. It is modified only after a failing compatibility test proves a concrete mismatch.

## Failure and stop behavior

- Unavailable or invalid Yahoo/yfinance history and unavailable real Kronos resources are explicit blockers.
- A real Kronos failure is persisted; no fake or baseline output silently replaces it.
- If Python 3.13/Windows compatibility, deterministic seed control, raw-path capture, or latency makes real Kronos unreasonable, Phase 2 stops and records the evidence.
- Provider responses, model weights, and caches are never committed.
- No empirical Sentinel claim exists until the untouched holdout is complete.

## Primary sources

- [Official Kronos repository](https://github.com/shiyu-coder/Kronos)
- [Kronos-mini model card](https://huggingface.co/NeoQuasar/Kronos-mini)
- [Kronos-Tokenizer-2k model card](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k)
- [yfinance package and legal notice](https://pypi.org/project/yfinance/)
- [yfinance download parameters](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars)
