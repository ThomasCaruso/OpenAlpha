# OpenAlpha Sentinel Direction

- Decision date: 2026-07-30; OpenAlpha integration update: 2026-07-31
- Status: v0 reliability retired; Sentinel assurance retained within OpenAlpha for Kronos
- Preserved commits: `c037bd5`, `27f0c69`, `daf3477`

## Current product direction

Sentinel is OpenAlpha's assurance layer. It validates the official raw forecast and
every separately labeled compatible reconstruction, preserves exact violations and
provenance, creates immutable audits, and prevents an invalid raw path from being
silently accepted downstream. Deterministic terminal projection remains an explicit
fallback gateway, never a disguised model output or an accuracy claim.

Sentinel v1.1 established why that assurance is necessary. Both pinned released
tokenizers materially violated OHLC constraints during encode-decode reconstruction
of valid development-only ETF sequences. This places the measured incompatibility
in continuous tokenizer reconstruction before autoregressive token selection. The
locked result remains `TOKENIZER_CONSTRAINT_DEFECT`.

That classification did not authorize support-conditioned sampling or another
inference-only token-selection method. The bounded v1 experiment remains
**VALIDITY SUCCEEDS, QUALITY DEGRADES**: candidate resampling returned 36 of 36
valid paths but failed the high-low range gate, while stepwise project/re-encode
hard-failed 21 of 36 paths. The model-size canary, larger v1 study, and untouched
holdout remain unrun.

OpenAlpha Bridge is a new, separately preregistered training track at the continuous
reconstruction boundary, not another Sentinel phase. It retains all official tokens
and weights, reuses the frozen causal tokenizer decoder trunk, and produces a
separately labeled constrained path. Sentinel validates and audits both the official
and Bridge paths. See `OPENALPHA_PIVOT.md` and
`OPENALPHA_KRONOS_BRIDGE.md`.

Structural assurance remains; v0 forecast-error risk prediction remains retired.

The v0 product definition below is preserved as historical context.

## Sentinel v0 product

OpenAlpha Sentinel is a model-agnostic reliability, failure-detection, and forecast-intervention layer for financial forecasting models.

Its core proposition is:

> Predict the predictor. Detect when a financial forecasting model is likely to fail before the outcome occurs, explain why the forecast is unreliable, and choose whether to use, blend, repair, or reject it.

Kronos is the first supported forecast provider and case study. The immediate proof implements only USE, BLEND, and ABSTAIN; a 50/50 blend with the causal baseline is the only repair.

## Sentinel v0 governing research question

> Can information available at forecast time predict the future error of a financial forecasting model, and can selective use, blending, or abstention outperform blindly accepting every forecast?

A feature is in scope only if it helps predict future forecast failure, explain that predicted failure, reduce damage from an unreliable forecast, repair it, or prove whether an intervention works.

## Evolution from Kronos Reality Check

Kronos Reality Check correctly identified the credibility problem and established that every model result needs causal data, baselines, immutable provenance, and honest negative findings. Benchmarking remains necessary because Sentinel needs forecast errors, baseline comparisons, and holdout evidence.

Benchmarking is no longer the product. The product is a forecast-time judgment about whether the forecast should be trusted and what intervention is justified.

The unimplemented Reality Check design and proof-slice plan were never built. Commit `27f0c69` remains intact as the record of that reasoning.

## Sentinel v0

Sentinel v0 is a development-only signal-discovery experiment:

- assets: SPY and QQQ;
- data: Yahoo Finance daily raw bars through pinned yfinance for the development proof;
- horizon: five XNYS sessions;
- sample: weekly chronological cutoffs from 2024-07-01 through 2026-06-30;
- forecast models: real Kronos-mini and a zero-return last-value baseline;
- ensemble: context lengths 128, 256, and 512 crossed with three declared sampling seeds;
- meta-model: pooled regularized logistic regression, with ridge regression as a secondary continuous-error model;
- actions: USE, 50/50 BLEND, and ABSTAIN.

The earlier half is development data. The later half is untouched holdout data. Sentinel v0 is not a publication, production service, SDK, trading strategy, or public model-performance claim.

Phase 2.5 confirmed that the pinned official Kronos predictor can emit reproducible finite candles that violate OHLC ordering. OpenAlpha did not introduce the violations. Eleven structural-validity measures are therefore locked as candidate development diagnostics, and every raw path is subject to a mandatory validity gate. This is an integration finding only; whether the measures predict future error remains untested.

## Definition of value

The completed v1 feasibility sample is valuable as a bounded engineering result:
hard validity can be enforced without retraining, but the tested methods did not
simultaneously satisfy fidelity, failure-rate, diversity, and operational gates.
The defensible product surface is therefore structural validation and explicit
repair, not a claim of improved forecasting.

Plausible explanations, repository size, interface polish, and development-only
accuracy differences are not evidence of general forecasting improvement.

## V0 closure and v1 question

The complete v0 development study found widespread structural invalidity but no
useful chronological out-of-fold forecast-error ranking. Abstention did not improve
accepted MAE, and the zero-return baseline outperformed Kronos. The reliability-risk
model is retired without holdout access.

Sentinel v1 asks whether hard financial constraints can be enforced inside Kronos's
autoregressive token loop while preserving forecast fidelity, diversity, and
practical runtime. It is not a reformulation or refit of v0.

## Preserved infrastructure

No functioning infrastructure is redesigned:

- content-addressed artifacts identify forecast requests, model outputs, diagnostics, decisions, outcomes, and reports;
- immutable journals record forecast creation, Sentinel decision, outcome resolution, and postmortem classification;
- verified manifests bind data, model, diagnostic, risk-model, and evaluation provenance;
- path confinement protects artifact access;
- existing tests, Ruff, and Pyright remain quality gates;
- the provider-independent market-data boundary remains; Phase 2 uses an explicitly unofficial yfinance adapter rather than a hand-written endpoint, with Alpaca deferred to independent verification.

## Explicit non-goals

Sentinel v0 does not include a frontend, accounts, database, live trading, portfolio optimization, public SDK/API, model marketplace, enterprise MLOps, news or filing interpretation, macro-event ingestion, indicator library, fine-tuning, complex repair, universal protocol schema, many assets/horizons, large local datasets, or committed model weights.

## Completed decision boundary

Sentinel v0 closed without holdout access. Sentinel v1 completed only its 12-origin
development feasibility experiment. A future decoding study requires a new version
and preregistration; the existing untouched holdout is not available for iteration.
Protocol v1 remains unfrozen.
