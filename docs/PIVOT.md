# OpenAlpha Product Pivot: Kronos Reality Check

- Decision date: 2026-07-30
- Status: Approved
- Preserved checkpoint: `c037bd5efafd728cb4e0df961dd6a93889b0ce56`

## Decision

OpenAlpha is no longer a generalized quantitative-research operating system. Its flagship product and research study is **OpenAlpha: Kronos Reality Check**: an independent, auditable evaluation of whether Kronos produces statistically meaningful and economically useful out-of-sample forecasts beyond simple baselines.

The governing question is:

> Does Kronos generate statistically meaningful and economically useful out-of-sample forecasts relative to simple forecasting methods, and does any apparent advantage survive realistic costs, different forecast horizons, and forward testing?

The project creates credible evidence. Repository size, architectural complexity, attractive charts, and rising historical equity curves are not success criteria.

## Why the original scope was rejected

The original plan front-loaded a generalized platform: API and worker services, MLflow, a broad web workspace, deployment profiles, portfolio tooling, and extensibility for unrelated research workflows. Those components would consume effort before the central empirical claim had been tested. They also encouraged product polish to become a substitute for evidence.

The rejected plan was archived at `docs/superpowers/plans/archive/2026-07-28-phase-1-vertical-slice.md`. It remains historical context and is not an active backlog.

## Preserved infrastructure

Commit `c037bd5` remains intact. The following completed infrastructure directly strengthens the new study:

- canonical, content-addressed artifacts for snapshots, configurations, forecasts, outcomes, metrics, and reports;
- path confinement and symlink/junction defenses for artifact access;
- immutable run-state journals for experiment lifecycle events;
- completed-run manifests for reproducibility and report provenance;
- artifact hash, size, schema, media-type, lineage, and environment verification;
- strict tests plus Ruff and Pyright configuration.

The existing `experiment-spec` package remains preserved. A new research-protocol contract may consume or supersede parts of it through an explicit versioned migration; it will not be silently rewritten.

## Three evidence classes

Every forecast and aggregate result has exactly one evidence class:

1. **Historical replay** — generated retrospectively by a locked rolling-origin procedure. It tests implementation and historical behavior but is vulnerable to dataset revision and researcher knowledge.
2. **Sealed historical test** — generated on a preregistered test period after models, rules, metrics, baselines, and exclusions are frozen. It is stronger than development evidence but is not equivalent to a prediction made before market outcomes existed.
3. **Live precommitted forecast** — content-addressed and timestamped before its outcome window begins. Resolution appends a separate outcome record and never mutates the forecast.

Interfaces, tables, reports, and exported data must retain this classification. They may not merge the classes into an unexplained score.

## New definition of value

OpenAlpha is valuable when a researcher can determine:

- where Kronos beats last-value, random-walk/drift, moving-average, statistical, linear, and tree baselines;
- where it fails or produces only plausible-looking price paths;
- whether differences are stable across assets, horizons, and time;
- whether uncertainty supports a meaningful conclusion;
- whether a fixed signal survives costs and next-bar execution;
- whether each claim resolves to a preregistered protocol and verified artifacts.

Negative, mixed, and inconclusive findings are successful research outcomes when produced honestly.

## Removed and deferred scope

The initial release explicitly excludes generalized portfolio optimization, Black-Litterman, a model or strategy marketplace, brokerage execution, an AI research copilot, generalized MLOps, multi-tenant enterprise accounts, arbitrary trading agents, and unrelated financial tools.

FastAPI, a durable worker, MLflow, PostgreSQL, object storage, and a dashboard are deferred until a CLI can complete and verify the real forecast-to-outcome vertical slice. The eventual application surface is limited to Study Overview, Live Forecasts, Model Scoreboard, Forecast Audit, Historical Results, Methodology, Research Report, and Reproducibility.

## Delivery order

1. Lock a versioned research protocol.
2. Validate an authenticated, documented Alpaca data boundary.
3. Complete one real SPY forecast date with one real Kronos checkpoint and one baseline.
4. Persist the forecast before scoring, append its outcome, compute one metric and one costed decision, and publish a verified manifest.
5. Demonstrate and reproduce the flow through a CLI.
6. Expand only then to the preregistered assets, models, horizons, sealed test, live ledger, report, and evidence interfaces.
