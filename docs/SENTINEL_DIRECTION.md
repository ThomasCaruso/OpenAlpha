# OpenAlpha Sentinel Direction

- Decision date: 2026-07-30
- Status: final product direction; Phase 1 design only
- Preserved commits: `c037bd5`, `27f0c69`

## Product

OpenAlpha Sentinel is a model-agnostic reliability, failure-detection, and forecast-intervention layer for financial forecasting models.

Its core proposition is:

> Predict the predictor. Detect when a financial forecasting model is likely to fail before the outcome occurs, explain why the forecast is unreliable, and choose whether to use, blend, repair, or reject it.

Kronos is the first supported forecast provider and case study. The immediate proof implements only USE, BLEND, and ABSTAIN; a 50/50 blend with the causal baseline is the only repair.

## Governing research question

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

## Definition of value

The next measure of value is whether pre-outcome diagnostics predict later forecast failure. Sentinel v0 is useful only if the locked holdout shows that higher risk corresponds to higher Kronos error and that selective acceptance or blending reduces error at declared coverage levels.

Plausible explanations, repository size, interface polish, and in-sample fit are not evidence.

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

## Delivery gates

1. Lock this direction, experiment configuration, targets, continuation criteria, and minimal contracts.
2. Prove one real SPY context-to-diagnostic-to-outcome chain.
3. Run the chronological development sample and freeze the risk model/action policy.
4. Run the untouched holdout once and generate the go/no-go decision.
5. Decide whether any broader product surface is justified.

Protocol v1 remains unfrozen until after the Sentinel v0 decision.
