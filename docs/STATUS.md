# Project Status

Last updated: 2026-07-28

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

## Current work

- Beginning content-addressed artifact storage, append-only run state, and completed-manifest integrity checks.

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
- Verified lock-enforced `uv sync --group dev` and `npm ci`.
- Verified 163 experiment-spec tests and 15 workspace smoke tests: 178 total passing.
- Verified Ruff reports no findings and Pyright reports zero errors or warnings.
- Verified duplicate JSON/YAML keys, unsafe updates, non-finite values, invalid cross-field combinations, and externally invalid JSON Schema instances are rejected.
- Verified the checked-in YAML and JSON round-trip to the same identity: `exp_8e72de5fcc485d3d512227f89f3d8e74e6f73bdfe6c90ddced158614d7ec6efb`.
- Completed independent spec-compliance and code-quality reviews for Phase 1 Tasks 1 and 2 with no open Critical or Important findings.

## Known limitations

- No data provider has yet been executed or licensed for redistribution.
- No checkpoint has yet been downloaded or timed on this CPU-only machine.
- No application, API, worker, report, or user interface exists yet.
- No financial result has been computed.
- Kronos training-data provenance is incomplete: the released checkpoint does not include reconstructable source-data hashes or a training run manifest.
- Any one-ETF result will remain exploratory and cannot establish broad alpha.
- Pydantic's deprecated v1 `copy(update=...)` compatibility method remains a minor defense-in-depth follow-up; the supported `model_copy(update=...)` path is blocked.

## Exact next task

Write failing tests for atomic content-addressed artifact publication, path confinement, hash verification, append-only run-state transitions, and refusal to publish a completed manifest with missing or mismatched artifacts.
