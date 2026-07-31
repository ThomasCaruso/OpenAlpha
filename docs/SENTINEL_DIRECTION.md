# OpenAlpha Sentinel Direction

- Decision date: 2026-07-30
- Status: v0 reliability retired; v1.1 compatibility study is the final decoder track
- Preserved commits: `c037bd5`, `27f0c69`

## Current product direction

Sentinel v1.1 establishes the durable product boundary. Both pinned released
tokenizers materially violated OHLC constraints during encode-decode reconstruction
of valid development-only ETF sequences. This places the incompatibility in the
continuous tokenizer reconstruction before any autoregressive token selection.
The locked root-cause result is TOKENIZER_CONSTRAINT_DEFECT.

No new support-conditioned decoder is authorized by that classification. The
product becomes structural validation, immutable raw-output audit, deterministic
terminal projection, token-manifold compatibility profiling, and model-selection
safety checks. A constraint-preserving tokenizer is a distinct future training
research direction, not another Sentinel inference phase.

OpenAlpha Sentinel v1 is a financial grammar-constrained decoding layer. It seeks to
prevent invalid generated K-lines from becoming autoregressive context, without
retraining or modifying model weights. Structural assurance and auditable repair
remain; v0 forecast-error risk prediction is retired.

The bounded v1 experiment reached **VALIDITY SUCCEEDS, QUALITY DEGRADES**. Bounded
valid-candidate resampling produced 36 of 36 structurally valid paths without
fallback, but failed the locked high-low range-error gate. Stepwise project and
re-encode returned valid paths when it succeeded but hard-failed 21 of 36 paths.
No in-loop method passed every continuation criterion, so the model-size canary,
larger study, and holdout remain unrun.

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

The unimplemented Reality Check design and proof-slice plan are archived under `docs/superpowers/{specs,plans}/archive/`. Commit `27f0c69` remains intact as the record of that reasoning.

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
