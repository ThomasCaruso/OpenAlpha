# ADR 0004: Explicit Local-First Storage with MLflow Behind an Adapter

- Status: Accepted
- Date: 2026-07-28

## Context

The development machine has no Docker installation and must still run the end-to-end slice. MLflow supports local databases and artifacts, while full team deployment benefits from PostgreSQL and object storage. MLflow is valuable lineage infrastructure but does not guarantee full reproducibility.

## Decision

Use explicit SQLite URIs, DuckDB, and local content-addressed artifacts in laptop mode. Use PostgreSQL and S3-compatible artifacts in the full profile. Isolate MLflow 3 calls behind a tracking adapter and pin client/server versions. Do not use implicit `mlruns` defaults. The Model Registry uses a database-backed store.

OpenAlpha's canonical manifest remains authoritative. MLflow model stages are not used; version tags and aliases represent governance state, while audits persist exact immutable versions.

## Consequences

- The demo runs without containers.
- Storage promotion does not change research semantics.
- SQLite concurrency is intentionally limited.
- Operators must back up and migrate MLflow's database separately from application metadata.

## Sources

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)

