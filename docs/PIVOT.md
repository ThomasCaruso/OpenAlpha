# OpenAlpha Product Pivots

- Final decision date: 2026-07-30
- Current product: OpenAlpha Sentinel
- Preserved commits: `c037bd5`, `27f0c69`

## Final decision

OpenAlpha Sentinel is a model-agnostic reliability, failure-detection, and forecast-intervention layer for financial forecasting models.

The governing question is:

> Can information available at forecast time predict the future error of a financial forecasting model, and can selective use, blending, or abstention outperform blindly accepting every forecast?

The product predicts the predictor. It returns a reliability assessment, failure probability, deterministic reasons, and an action: USE, BLEND, or ABSTAIN in v0.

## How the direction evolved

The original generalized quantitative platform was rejected because infrastructure breadth preceded empirical value. Commit `c037bd5` nevertheless produced valuable content-addressed artifacts, path confinement, immutable run journals, verified manifests, tests, Ruff, and Pyright configuration.

Kronos Reality Check then focused the project on credible evaluation. Commit `27f0c69` preserved the provider-independent data boundary, removed dependence on undocumented Yahoo endpoints, and established causal forecasts, baselines, immutable outcomes, and honest evidence classes.

That benchmark remains necessary, but it is passive. It explains whether a forecast failed after resolution; it does not tell a developer whether to trust the forecast before the outcome. Sentinel makes that forecast-time reliability decision the product.

The unimplemented Reality Check design and plan are archived without rewriting history.

## Preserved infrastructure

- Content-addressed artifacts store request, forecast, diagnostic, decision, outcome, and report identities.
- Immutable journals record experiment and forecast lifecycles.
- Verified manifests bind data, model, diagnostic, risk-model, and evaluation provenance.
- Path confinement protects artifact reads and publication.
- Existing tests and static-analysis configuration remain mandatory.
- Alpaca remains the first documented market-data provider through a provider-independent port.
- There is no Yahoo adapter and no silent provider fallback.

## Sentinel v0 value test

The immediate experiment asks whether pre-outcome Kronos instability and regime diagnostics predict later five-session Kronos error for SPY and QQQ.

It uses an earlier development year, a later untouched holdout year, one zero-return baseline, a nine-path Kronos ensemble, an interpretable risk model, and fixed USE/BLEND/ABSTAIN rules. Numeric continuation criteria are locked before holdout.

Benchmarking supplies errors, labels, and comparison. It is no longer the product.

## Scope removed or deferred

Sentinel v0 excludes frontend work, accounts, databases, live trading, portfolio optimization, public SDK/API, broad model support, universal protocol schemas, marketplaces, MLOps platforms, news/filing/macro ingestion, indicator libraries, fine-tuning, complex repair, many assets/horizons, large datasets, and committed weights.

Protocol v1 remains unfrozen. No empirical Sentinel claim exists.

## Delivery order

1. Lock Sentinel direction, periods, targets, diagnostics, contracts, actions, and continuation criteria.
2. Prove one real SPY context-to-diagnostic-to-outcome chain.
3. Run development cutoffs and freeze the risk model and action policy.
4. Run the untouched holdout once.
5. Apply the locked go/no-go rule before deciding whether to build a product interface or freeze protocol v1.
