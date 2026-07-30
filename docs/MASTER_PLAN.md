# OpenAlpha Master Plan

## Mission

OpenAlpha: Kronos Reality Check creates credible, reproducible evidence about whether Kronos forecasts future market behavior better than simple methods and whether any advantage survives costs, horizon changes, asset variation, and genuine forward testing.

The repository is not a generalized quantitative-finance platform. Every planned feature must help a researcher inspect the preregistered question, the original forecast, its evidence class, its later outcome, the comparison with baselines, or the provenance of a numerical claim.

## Definition of success

The project succeeds when it can truthfully state:

> We preregistered an evaluation protocol, generated reproducible Kronos and baseline forecasts, permanently recorded live predictions before outcomes were known, measured their later performance, and published exactly where the model did and did not add value.

Test count, repository size, interface polish, plausible candles, or an attractive backtest are not substitutes.

## Delivery stages

### Stage 0 — Preserved research infrastructure

Status: complete at `c037bd5`.

Preserved deliverables:

- canonical experiment identity;
- content-addressed artifacts and path confinement;
- immutable run-state journals;
- verified completed-run manifests;
- reproducible Python/npm workspace, tests, Ruff, and Pyright.

### Stage 1 — One real forecast-to-outcome proof slice

Scope:

- versioned research-protocol schema and canonical hash;
- documented Alpaca provider port with environment credentials and no fallback;
- one SPY daily forecast date using SIP data;
- paired adjusted forecast data and raw execution data;
- one real pinned Kronos checkpoint and the last-value baseline;
- immutable forecast records persisted before scoring;
- immutable outcome records, one primary return metric, one directional metric, and one fixed costed decision;
- hash-chain and completed-manifest verification;
- CLI demonstration and reproduction command.

Acceptance:

- synthetic fixtures are unmistakably labeled and never appear as research evidence;
- missing credentials fail before network access and never leak secret values;
- a real local run uses the user’s Alpaca credentials and real Kronos inference;
- forecast artifacts are written before the scorer can read future outcomes;
- outcome resolution appends instead of mutating;
- CLI output resolves every value to verified artifacts;
- the full existing and new quality gates pass.

### Stage 2 — Preregistered historical benchmark

Scope:

- freeze `research/protocol/v1.md`, `v1.yaml`, and `v1.sha256` after provider and checkpoint feasibility checks;
- SPY, QQQ, IWM, TLT, and GLD;
- 1-, 5-, and 20-session horizons;
- Kronos, last value, random walk/drift, moving average, exponential smoothing or ARIMA, ridge, and gradient-boosted trees;
- strict rolling origins, failure retention, return/direction/volatility targets, baseline-adjacent metrics, fixed costed strategy, and buy-and-hold/cash comparisons;
- development 2017–2022, validation 2023, replay-only contamination quarantine through June 2024, and candidate sealed evaluation July 2024–December 2025.

Acceptance:

- protocol v1 is immutable and changes require v2;
- sealed results do not tune v1;
- all declared assets, horizons, models, dates, failures, and exclusions appear;
- block-bootstrap intervals, defensible Diebold–Mariano comparisons, directional intervals, Holm correction, and stability analyses disclose assumptions and undefined cases;
- forecasting improvements that fail after costs are reported as failures of economic usefulness.

### Stage 3 — Live precommitted forecast ledger and paper

Scope:

- schedule forecasts using the same verified application service as the CLI;
- append forecast and later outcome records with complete chain verification;
- model scoreboard that preserves evidence class, target, horizon, asset, period, sample count, and uncertainty;
- forecast audit view with adjacent, failed, and unresolved records;
- artifact-grounded paper, “Do Financial Foundation Models Predict Markets? An Independent Historical and Live Evaluation of Kronos.”

Acceptance:

- every live forecast predates its outcome window;
- unresolved and failed forecasts remain discoverable;
- historical replay, sealed test, and live performance are never presented as equivalent;
- all report numbers are generated from verified artifacts;
- negative, mixed, or inconclusive results are published without protocol changes.

### Stage 4 — Narrow evidence application

Only after the CLI and live ledger are proven, build read-only surfaces for Study Overview, Live Forecasts, Model Scoreboard, Forecast Audit, Historical Results, Methodology, Research Report, and Reproducibility. Transport, scheduling, and presentation adapt to the existing evidence kernel.

## Explicit non-goals

The initial release excludes generalized portfolio optimization, Black-Litterman, live brokerage execution, a model or strategy marketplace, an AI research copilot, generalized MLOps, multi-tenant enterprise accounts, arbitrary user-authored agents, and unrelated financial tools.

MLflow, FastAPI, a durable worker, PostgreSQL, object storage, and Next.js are deferred implementation choices, not acceptance requirements.

## Dependency principles

- Python and typed package contracts own research semantics.
- `uv` owns Python resolution and locking; npm remains only for a later evidence UI.
- Provider, model, storage, and presentation code are adapters.
- Real data and real Kronos are required for empirical claims; deterministic substitutes exist only in tests.
- Completed manifests and ledger records, not mutable trackers, are authoritative.

