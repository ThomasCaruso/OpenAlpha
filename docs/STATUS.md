# Project Status

Last updated: 2026-07-29

## Completed

- Created and initialized the `open-alpha` Git repository.
- Inspected the development environment.
- Read the required official Kronos repository, paper, and Hugging Face model card.
- Read the required official MLflow Tracking and Model Registry documentation.
- Read the required official VectorBT documentation and license.
- Defined the Phase 0 architecture, methodology, research question, threat model, data limitations, master plan, and ADR set.
- Wrote and self-reviewed the complete Phase 1 vertical-slice implementation plan.
- Established the reproducible Python/npm workspace, lock-enforced setup, environment verification, CI smoke workflow, and isolated feature worktree.
- Implemented the v1 immutable experiment specification with strict typed model parameters, semantic finance validation, safe YAML/JSON loading, version gating, formal JSON Schema, canonical serialization, and stable SHA-256 experiment identity.
- Added a valid post-cutoff SPY/Kronos example specification with no generated results.
- Implemented atomic content-addressed artifact publication with portable path confinement, symlink/junction escape rejection, idempotent writes, conflict refusal, hash/size verification, and stable integrity errors.
- Implemented immutable append-only run-state journals with validated lifecycle transitions, terminal-state enforcement, monotonic event time, and retry attempts that preserve prior history.
- Implemented canonical completed-run manifests that require the full pre-report Phase 1 artifact inventory, verified JSON schema/media metadata, acyclic input lineage, matching experiment/Git/environment identity, and a non-failed methodology audit.

## Current work

- Beginning the validated daily-equity snapshot pipeline: provider contracts, normalization, quality findings, provenance, and deterministic Parquet artifacts.

## Blocked work

No architectural blocker exists.

Environment constraints that affect execution:

- Docker is not installed, so the local-first profile must work without containers.
- No NVIDIA runtime is present; real Kronos inference must be verified on CPU and expose resource diagnostics.
- The machine has approximately 16 GB RAM, 12 logical processors, and 261 GB free disk.

## Verification performed

- Confirmed the repository began empty and was not previously a Git worktree.
- Confirmed Git 2.49, Python 3.10/3.13, Node 22, npm 11, and uv 0.11 are available.
- Confirmed the official Kronos implementation requires lowercase OHLC, accepts optional volume/amount, normalizes per causal input window, and produces stochastic autoregressive OHLCVA paths.
- Confirmed the public base checkpoint is 102.3M parameters with a 512-bar context and MIT metadata.
- Confirmed MLflow's current local/server storage behavior and database-backed Registry requirement.
- Confirmed current VectorBT is Apache 2.0 with Commons Clause and therefore unsuitable as OpenAlpha's mandatory authoritative engine.
- Verified lock-enforced `uv sync --locked --group dev` and `npm ci`.
- Verified 163 experiment-spec tests, 15 workspace smoke tests, and 42 research-core tests: 220 total passing.
- Verified Ruff reports no findings and Pyright reports zero errors or warnings.
- Verified duplicate JSON/YAML keys, unsafe updates, non-finite values, invalid cross-field combinations, and externally invalid JSON Schema instances are rejected.
- Verified the checked-in YAML and JSON round-trip to the same identity: `exp_8e72de5fcc485d3d512227f89f3d8e74e6f73bdfe6c90ddced158614d7ec6efb`.
- Completed independent spec-compliance and code-quality reviews for Phase 1 Tasks 1 and 2 with no open Critical or Important findings.
- Completed a production-readiness review for Phase 1 Task 3 and closed all Critical/Important findings before the final quality gate.

## Known limitations

- No data provider has yet been executed or licensed for redistribution.
- No checkpoint has yet been downloaded or timed on this CPU-only machine.
- No application, API, worker, report, or user interface exists yet.
- No financial result has been computed.
- The research-core contracts are not yet wired into a real data/model run.
- Kronos training-data provenance is incomplete: the released checkpoint does not include reconstructable source-data hashes or a training run manifest.
- Any one-ETF result will remain exploratory and cannot establish broad alpha.
- Pydantic's deprecated v1 `copy(update=...)` compatibility method remains a minor defense-in-depth follow-up; the supported `model_copy(update=...)` path is blocked.

## Exact next task

Write failing property and contract tests for daily adjusted OHLCV normalization, duplicate/missing-session findings, provider provenance, deterministic Parquet snapshots, and content-addressed snapshot manifests.
