# Project Status

Last updated: 2026-07-28

## Completed

- Created and initialized the `open-alpha` Git repository.
- Inspected the development environment.
- Read the required official Kronos repository, paper, and Hugging Face model card.
- Read the required official MLflow Tracking and Model Registry documentation.
- Read the required official VectorBT documentation and license.
- Defined the Phase 0 architecture, methodology, research question, threat model, data limitations, master plan, and ADR set.

## Current work

- Writing and self-reviewing the Phase 1 vertical-slice implementation plan.

## Blocked work

No architectural blocker exists.

Environment constraints that affect execution:

- Docker is not installed, so the local-first profile must work without containers.
- No NVIDIA runtime is present; real Kronos inference must be verified on CPU and expose resource diagnostics.
- `pnpm` is not installed; the repository will pin a package manager through Corepack rather than rely on a global installation.
- The machine has approximately 16 GB RAM, 12 logical processors, and 261 GB free disk.

## Verification performed

- Confirmed the repository began empty and was not previously a Git worktree.
- Confirmed Git 2.49, Python 3.10/3.13, Node 22, npm 11, and uv 0.11 are available.
- Confirmed the official Kronos implementation requires lowercase OHLC, accepts optional volume/amount, normalizes per causal input window, and produces stochastic autoregressive OHLCVA paths.
- Confirmed the public base checkpoint is 102.3M parameters with a 512-bar context and MIT metadata.
- Confirmed MLflow's current local/server storage behavior and database-backed Registry requirement.
- Confirmed current VectorBT is Apache 2.0 with Commons Clause and therefore unsuitable as OpenAlpha's mandatory authoritative engine.

No application tests or builds have run because Phase 1 source code does not yet exist.

## Known limitations

- No data provider has yet been executed or licensed for redistribution.
- No checkpoint has yet been downloaded or timed on this CPU-only machine.
- No application, API, worker, report, or user interface exists yet.
- No financial result has been computed.
- Kronos training-data provenance is incomplete: the released checkpoint does not include reconstructable source-data hashes or a training run manifest.
- Any one-ETF result will remain exploratory and cannot establish broad alpha.

## Exact next task

Create the Phase 1 plan, then write failing tests for canonical experiment-spec serialization, semantic validation, stable hashing, and run-start immutability before implementing those contracts.
