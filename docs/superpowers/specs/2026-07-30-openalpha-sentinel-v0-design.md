# OpenAlpha Sentinel v0 Design

## Outcome

Determine whether forecast-time information predicts later Kronos error and whether USE, 50/50 BLEND, or ABSTAIN improves error/coverage trade-offs relative to accepting every forecast.

This is a development-only feasibility result. It does not create a public API, universal protocol, trading system, or production claim.

## Selected approach

Use a weekly rolling-origin ensemble for SPY and QQQ with nine real Kronos-mini paths per cutoff. Compute fourteen interpretable pre-outcome diagnostics, fit a pooled regularized logistic risk model on the earlier year, freeze the model and action policy, and evaluate once on the later year.

The selected weekly schedule targets roughly 104 cutoffs per asset while bounding inference to about 1,872 requests across development and holdout. Daily origins would multiply correlated samples and cost without providing equivalent independent evidence.

The selected pooled linear meta-model shares information across two related ETFs without exposing symbol identity as a shortcut. Per-asset models would have about 52 development rows each; a deep or tree-first meta-model would be difficult to distinguish from overfit.

## Rejected alternatives

- **Passive Kronos benchmark:** necessary for labels and comparison, but it does not make a forecast-time reliability decision.
- **Daily cutoffs and parameter sweeps:** larger cost and dependence family with little v0 interpretability benefit.
- **Hosted deployment first:** the official Kronos-mini page reports no deployed Inference Provider; a permanent custom endpoint is premature.
- **News/event/context ingestion:** may matter for exogenous failures but adds another provider and causal-timestamp problem before market/model-output diagnostics are tested.
- **Complex repair:** a fixed baseline blend is sufficient to test whether intervention has value.
- **Universal schemas and services:** the experiment needs only small internal contracts and the preserved evidence infrastructure.

## Fixed sample

- Development: 2024-07-01 through 2025-06-30.
- Holdout: 2025-07-01 through 2026-06-30.
- Origin rule: last valid XNYS session of each ISO week.
- Assets: SPY and QQQ.
- Horizon: five XNYS sessions.

A preflight artifact freezes the exact ordered origin list before any diagnostic model is fit. Failed dates remain failed records and are not replaced after outcomes are known.

## Components

### Market data boundary

The existing provider-independent request contract remains. Alpaca is the first adapter, with explicit SIP, 1Day, start/end, adjustment=all, cutoff-date as-of mapping, pagination, and environment credentials. Raw responses are hashed then discarded for v0.

### Forecast provider

A narrow internal port accepts exact model/checkpoint identity, ordered OHLCV context, context length, horizon, seed, temperature, top-p, path count, and request metadata. It returns generated paths, a summary, duration, provider/checkpoint/request identity, and typed failure.

Kronos-mini is pinned to official source/model/tokenizer revisions. With no documented hosted provider currently available, Phase 2 uses an on-demand local process and temporary Hugging Face cache outside Git. Automated tests use a marked deterministic fake only.

### Ensemble and baseline

Context lengths 128, 256, and 512 are crossed with seeds 1729, 2027, and 7919 at temperature 1.0, top-p 0.9, and one path per request. The primary forecast is the pointwise median. The baseline repeats the cutoff adjusted close and predicts zero return.

### Diagnostics

The fourteen diagnostics measure sampling agreement, path dispersion, context sensitivity, baseline disagreement, volatility/trend/outlier regime, historical analogue support, recent resolved model error, and horizon divergence. Every formula and missing-value rule is fixed in `docs/SENTINEL_METHODOLOGY.md`.

### Failure targets

The continuous target is absolute five-session return error. The binary failure threshold is the pooled development 75th percentile and is frozen for holdout. The deployability label records whether Kronos is strictly worse than the last-value baseline.

### Risk and action

The primary model is L2 logistic regression; ridge regression estimates continuous error secondarily. Development-only expanding splits choose among fixed regularization grids. Risk percentiles freeze USE at 0–50%, BLEND at 50–80%, and ABSTAIN at 80–100%; blend weights are 50/50.

### Evaluation and decision

The holdout reports risk/error rank correlation, risk quintiles, coverage curves, error and direction at five coverage levels, calibration, confusion matrix, baseline comparison, asset/time stability, diagnostic contribution, and runtime/cost. Numeric criteria deterministically map the result to one of four project decisions.

## Evidence flow

```text
locked experiment bytes
  -> exact origin-list artifact
  -> development forecasts + pre-outcome diagnostics
  -> later development outcomes
  -> frozen development risk configuration and action policy
  -> holdout forecasts + pre-outcome diagnostics
  -> Sentinel risk/action/reason artifacts
  -> later holdout outcomes and immutable postmortems
  -> one untouched holdout evaluation
  -> verified manifests and project decision
```

Outcome access is a separate resolver operation. A test spy must prove that no outcome loader runs before forecast and diagnostic artifacts are published. After the development freeze, it must additionally prove that each holdout decision is published before its outcome loader runs.

## Failure handling

- Missing credentials or SIP entitlement stops real data acquisition.
- Missing hosted Kronos support selects the declared local-cache fallback, not fake output.
- Incompatible Python/runtime, unsafe loading, uncontrolled seed behavior, or unreasonable resource use stops Phase 2 and produces a blocker report.
- Missing diagnostics use development-fitted missingness handling; they are never replaced with favorable values.
- Inference and data failures remain in sample accounting.
- Holdout output cannot mutate development configuration.

## Testing

TDD covers contract strictness, secret redaction, causal context bounds, ensemble completeness, exact diagnostic formulas, analogue eligibility, recent-error resolution timing, failure labels, development-only preprocessing, action thresholds, coverage calculations, reason derivation, artifact hashes, and outcome-loader ordering.

Network and real-Kronos tests are separately marked. Every implementation commit runs focused tests, the full existing suite where practical, Ruff, Pyright, and `git diff --check`, with exact evidence appended to `docs/STATUS.md`.

## Stop line

Phase 1 ends with documentation, the locked experiment configuration, minimal field contracts, an archived benchmark plan, and a Phase 2 one-origin implementation plan. It creates no Sentinel package and performs no real inference.
