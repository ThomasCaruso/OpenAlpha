# ADR 0009: Prove the Evidence Slice through a CLI before Services or UI

- Status: Accepted
- Date: 2026-07-30
- Supersedes for the initial release: ADR 0006
- Amends: ADR 0001 and ADR 0004

## Context

API, worker, MLflow, deployment, and dashboard work do not establish whether the forecast-to-outcome evidence chain is correct. The current repository has strong artifact and lifecycle contracts but no real completed forecast.

## Decision

The first end-to-end product surface is a typed CLI. It must lock a protocol, retrieve authenticated data, run one real Kronos forecast and one baseline, persist both before scoring, append an outcome, calculate forecast and costed economic results, verify the ledger and manifest, and reproduce the run.

FastAPI, a durable worker, MLflow, PostgreSQL, object storage, and the web application are deferred. Ports remain permissible, but no initial-release acceptance criterion depends on those systems.

## Consequences

- Research semantics are proven before transport and presentation.
- The existing modular package boundary remains useful without requiring process topology.
- Live scheduling will later need a durable execution mechanism, but it must reuse the verified CLI application service.
