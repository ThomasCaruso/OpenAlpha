# OpenAlpha Phase 0 Design

## Outcome

Build the Phase 1 vertical slice as a modular monorepo whose central artifact is an auditable research run, not a dashboard page or model prediction.

The user-provided project specification is the approved product design. This document records the implementation-level interpretation established after repository inspection and required primary-source research.

## Selected approach

Use a typed Python research kernel with ports for data, models, tracking, metadata, and artifacts; separate FastAPI and worker processes; and a Next.js web workspace. Start with local SQLite/DuckDB/filesystem services, preserving configuration-compatible PostgreSQL/object-storage paths.

Rejected alternatives:

- early microservices, because operational surface would dominate the first research slice;
- a notebook/Streamlit-style product, because it cannot meet the durable job, UI, audit, and architecture standard;
- framework-owned research logic, because hidden defaults would weaken financial reproducibility.

## Phase 1 architecture

The first dependency chain is:

```text
experiment spec
  -> canonical identity
  -> validated real-data snapshot
  -> causal forecast origins
  -> model forecasts
  -> forecast metrics
  -> declared signals
  -> next-bar orders and fills
  -> accounting and risk
  -> methodology audit
  -> immutable manifest
  -> MLflow records and report
  -> API/web presentation
```

Every stage consumes typed inputs and emits schema-versioned artifacts. Completion is a manifest-verification event.

## Error and validity model

Errors are categorized as specification, data quality, methodology, provider, model capability, resource, execution, accounting, artifact integrity, or infrastructure failures. A failure preserves diagnostics but cannot publish valid results.

Methodology state is `passed`, `passed_with_warnings`, or `failed`. UI and reports read this state from the audit artifact and cannot infer it from job success.

## Testing design

Implementation begins with stable experiment hashing and immutability, then causal origins, data contracts, model adapters, accounting, artifacts, and transports. Each financial invariant has an independent oracle or property test. Real Kronos inference is a marked slow integration test; test-only deterministic adapters never appear in real run manifests.

## Local constraints

The inspected machine is CPU-only with 16 GB RAM and no Docker. The slice must therefore offer a real official mini-checkpoint path, checkpoint caching, bounded origins for a smoke run, and explicit estimates/progress. The base checkpoint remains selectable and is verified when compute permits.

## Source-driven corrections to the initial concept

- VectorBT is optional because its Commons Clause and execution limitations conflict with an authoritative open-source accounting core.
- MLflow is an adapter, not the sole lineage store.
- Kronos uses pinned revisions, explicit evaluation mode, external input validation, and post-June-2024 evaluation.
- Training-data opacity is a permanent disclosed limitation, not a problem the UI can hide.

## Phase 1 acceptance boundary

Phase 1 is accepted only when one real-data, real-Kronos run completes from spec through downloadable report; all baselines and buy-and-hold are present; costs and next-bar execution apply; methodology checks pass; hashes verify; and the same run can be reproduced through the documented command.
