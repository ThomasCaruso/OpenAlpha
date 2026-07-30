# ADR 0009: Prove the Evidence Slice before Services or UI

- Status: Superseded by ADR 0010 for the current milestone
- Date: 2026-07-30
- Superseded ADR 0006 for the initial release
- Amended ADR 0001 and ADR 0004

## Context

API, worker, MLflow, deployment, and dashboard work do not establish whether the forecast-to-outcome evidence chain is correct. The repository has strong artifact/lifecycle contracts but no real forecast.

## Original decision

The original first product surface was a typed CLI for a one-origin Kronos benchmark. That work was never implemented.

## Sentinel amendment

ADR 0010 narrows the first end-to-end work to an internal development invocation for one real Sentinel diagnostic. It retrieves authenticated data, runs the declared Kronos ensemble and baseline, persists forecasts and a pre-outcome diagnostic vector, appends an outcome, computes error, and verifies artifacts and manifest. It does not invent a Sentinel action before the development risk model is frozen.

No public CLI, SDK, API, FastAPI service, durable worker, MLflow, PostgreSQL, object storage, or web application is required before Sentinel v0 signal validation.

## Consequences

- Research semantics are proven before transport and presentation.
- The existing modular boundary remains useful without a service topology.
- A product interface or live scheduler is considered only after the holdout decision.
