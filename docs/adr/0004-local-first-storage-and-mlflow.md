# ADR 0004: Explicit Local-First Storage with MLflow Behind an Adapter

- Status: Amended by ADR 0009
- Date: 2026-07-28

## Context

The development machine has no Docker installation and must still run the end-to-end slice. MLflow supports local databases and artifacts, while full team deployment benefits from PostgreSQL and object storage. MLflow is valuable lineage infrastructure but does not guarantee full reproducibility.

## Decision

Use local content-addressed artifacts as the initial authoritative store. SQLite, DuckDB, PostgreSQL, S3-compatible storage, and an MLflow tracking adapter remain possible later implementations, but ADR 0009 defers them until the CLI proof slice establishes an evidence need. Do not use implicit `mlruns` defaults.

OpenAlpha's canonical manifest remains authoritative. MLflow model stages are not used; version tags and aliases represent governance state, while audits persist exact immutable versions.

## Consequences

- The proof slice runs without containers or an MLflow service.
- Storage promotion does not change research semantics.
- SQLite concurrency is intentionally limited.
- If MLflow is introduced later, operators must back up and migrate its database separately from canonical evidence.

## Sources

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)

