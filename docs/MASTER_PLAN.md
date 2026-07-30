# OpenAlpha Sentinel Master Plan

## Mission

OpenAlpha Sentinel predicts when a financial forecasting model is likely to fail before its outcome, explains the forecast-time evidence, and selects USE, BLEND, or ABSTAIN.

Kronos is the first supported model. Benchmarking remains required to produce labels and evaluate interventions, but benchmarking is not the product.

## Definition of success

Sentinel v0 succeeds as a research decision process when it can truthfully state:

> We used only forecast-time information to estimate later Kronos error, froze the detector before an untouched holdout, measured selective use and blending at every declared coverage level, and preserved negative or positive evidence without changing the rules.

The detector itself may fail. A verified STOP decision is a valid outcome.

## Stage 0 — Preserved evidence infrastructure

Status: complete in `c037bd5` and `27f0c69`.

- content-addressed, path-confined artifacts;
- immutable run journals;
- verified completed-run manifests;
- experiment identity prototype;
- provider-independent documented market-data boundary;
- existing tests, Ruff, and Pyright;
- no undocumented Yahoo dependency.

## Stage 1 — Direction and experiment lock

Scope:

- Sentinel direction, failure taxonomy, and methodology;
- SPY/QQQ daily, five-session experiment;
- development 2024-07-01–2025-06-30;
- untouched holdout 2025-07-01–2026-06-30;
- weekly cutoff rule;
- fixed Kronos ensemble, baseline, diagnostics, targets, risk/action policies, evaluation outputs, and continuation criteria;
- minimal internal market-data, forecast-provider, diagnostic-vector, and later decision field contracts;
- exact pinned source/model/tokenizer revisions;
- archived unimplemented Reality Check plan/spec.

Acceptance:

- protocol v1 remains unfrozen;
- no experiment code or real forecast runs;
- no empirical Sentinel claim is made;
- the full preserved quality gates pass.

## Stage 2 — Smallest real diagnostic

For one historical SPY cutoff:

1. fetch one causal context through Alpaca;
2. run the seed-reproducibility probe and nine declared real Kronos requests;
3. generate the last-value baseline;
4. average only the three 512-context close paths and publish every path before outcome access;
5. compute and publish the diagnostic vector without inventing an unfitted Sentinel action;
6. resolve the five-session outcome;
7. append errors and a human-readable audit record;
8. verify artifacts, lifecycle, and manifest.

Stop if real Kronos cannot operate within the locked feasibility boundary.

## Stage 3 — Development sample

- Materialize and hash the exact weekly origin list.
- Run chronological SPY/QQQ development cutoffs.
- Inspect only development diagnostic validity, missingness, and redundancy under fixed rules.
- Fit logistic failure probability and secondary ridge error regression.
- Freeze preprocessing, coefficients, failure/reason thresholds, and USE/BLEND/ABSTAIN policy.

## Stage 4 — Untouched holdout

- Run the frozen pipeline once on the later year.
- Report all five coverage levels, risk quintiles, calibration, baseline comparison, asset/time stability, and runtime/cost.
- Do not alter diagnostics, labels, models, thresholds, or actions.
- Apply the locked project-decision mapping.

## Stage 5 — Decision

Record exactly one:

- PROCEED;
- PROCEED WITH A NARROWER DIAGNOSTIC SET;
- CHANGE THE FAILURE-DETECTION APPROACH;
- STOP THE PROJECT.

Only then decide whether public interfaces, other models, exogenous-event data, complex repair, or protocol v1 are justified.

## Explicit non-goals

No frontend, accounts, database, trading, portfolio optimization, public SDK/API, generalized copilot, marketplace, MLOps platform, event ingestion, technical-indicator library, fine-tuning, complex repair, universal protocol schema, broad universe/horizons, large dataset, or committed checkpoint is part of Sentinel v0.

## Dependency principles

- Preserve and reuse `openalpha-research-core`; change it only for a proven incompatibility.
- Add at most one narrow `packages/sentinel` package when Phase 2 begins.
- Provider, diagnostics, labels, risk, action, evaluation, and report modules remain small internal boundaries.
- Real claims require real Kronos; fakes are test-only and explicitly synthetic.
- Raw provider data and model caches remain outside Git.
