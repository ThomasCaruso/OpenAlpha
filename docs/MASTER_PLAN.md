# OpenAlpha Master Plan

## Mission

OpenAlpha is an auditable quantitative research operating system for testing whether pretrained financial foundation models deliver robust predictive and economic value after leakage controls, realistic costs, regime variation, and model-selection bias are considered.

The project optimizes for credible evidence, not attractive backtests. Every visible result must resolve to an immutable experiment specification, data snapshot, model checkpoint, code revision, execution assumptions, environment, and generated artifact.

## Delivery strategy

Work proceeds as complete vertical slices. A phase is complete only after its real user flow has executed, its tests and static checks pass, and the evidence and limitations are recorded in `docs/STATUS.md`.

### Phase 0 — Research and architecture

Deliverables:

- Primary-source assessment of Kronos, MLflow, and VectorBT.
- Architecture, research question, methodology, threat model, and data-limitations documents.
- Architecture decision records for consequential choices.
- Explicit phase acceptance criteria and local compute modes.
- A test-driven implementation plan for the first vertical slice.

Acceptance criteria:

- Every mandated source has been read and linked from the documentation.
- Kronos input/output contracts, released checkpoints, stochastic behavior, licensing, pretraining cutoff, and runtime limitations are documented.
- The authoritative boundaries for data, execution, lineage, and experiment immutability are unambiguous.
- Local development constraints are documented without weakening scientific requirements.
- `docs/STATUS.md` names the exact next executable task.

### Phase 1 — Complete vertical slice

Scope:

- One liquid US ETF, daily adjusted OHLCV, and a configurable post-cutoff period.
- Real Kronos inference plus random-walk, last-value, moving-average, exponential-smoothing or ARIMA, and gradient-boosted-tree baselines.
- Leakage-safe rolling-origin evaluation.
- A declared forecast-to-position rule, next-bar execution, commissions, spread/slippage, and buy-and-hold.
- Forecast, directional, portfolio, drawdown, and risk metrics.
- Immutable specification and run manifest, content-addressed data snapshot, MLflow tracking, HTML/PDF report, API, worker, and polished web workflow.

Acceptance criteria:

- A clean machine can execute the documented setup and demo commands without a paid data key.
- The demonstration downloads real data, validates it, snapshots it, and records its provenance; no production result is mocked.
- At least one official Kronos checkpoint performs real inference. Resource failure is explicit and never replaced by a synthetic forecast.
- Every model sees only information available at its forecast origin; automated leakage checks pass.
- Signals execute no earlier than the next eligible bar and all reported strategy results include nonzero configured costs.
- Cash, holdings, fills, fees, and equity reconcile within a documented numerical tolerance.
- A buy-and-hold benchmark and all required simple forecasting baselines appear beside Kronos.
- An experiment can be reproduced from its run ID and artifact hashes are verified.
- Unit, property, data-contract, integration, frontend, report, and reproducibility tests pass.
- The API, worker, web application, and tracking service start through one documented command.
- The generated report derives every number from saved artifacts and identifies the one-asset study as exploratory.

### Phase 2 — Research platform

Scope:

- Versioned YAML/JSON experiment language with migrations.
- Durable asynchronous jobs, data catalog, model registry, artifact browser, and append-only Forecast Ledger.
- Reproduction command, scheduled artifact retention, and platform metadata migrations.

Acceptance criteria:

- Invalid cross-field combinations fail with actionable errors.
- A run stores the canonical specification before work begins and refuses mutation.
- Worker restart, timeout, cancellation, and retry semantics are tested.
- Ledger corrections append a reference to an earlier record; they never overwrite it.
- MLflow registry aliases are convenience pointers only; audits resolve immutable model versions and checkpoint hashes.

### Phase 3 — Quantitative depth

Scope:

- Statistical validation, advanced leakage auditor, execution simulator, multi-asset strategies, portfolio optimization, risk, factor attribution, and deterministic regimes.

Acceptance criteria:

- Multiple-testing correction and parameter-sensitivity reports accompany model comparisons.
- Execution assumptions are inspectable per order and idealized results are visually separated from realistic results.
- Portfolio optimizers return explicit infeasibility and instability diagnostics.
- Factor and regime claims disclose data availability and methodology.
- Golden accounting, statistical, and constraint tests pass.

### Phase 4 — Controlled research copilot

Scope:

- Provider-neutral natural-language hypothesis translation, assumption discovery, specification validation, methodological explanation, artifact-grounded interpretation, and an explicit launch approval.

Acceptance criteria:

- The copilot cannot mutate results or launch expensive work before approval.
- Every numerical statement resolves to a stored artifact field.
- Prompt-injection and untrusted-document boundaries are tested.
- Missing assumptions remain visible rather than being silently invented.

### Phase 5 — Production and presentation

Scope:

- Paper forecasting, governance, observability, deployment, security hardening, accessibility, public sample experiments, and recruiter artifacts.

Acceptance criteria:

- Scheduled runs never cross a live-brokerage boundary.
- Health, queue, duration, inference, memory, fetch, and categorized failure signals are visible.
- CI passes tests, lint, type checks, builds, dependency audit, secret scan, and container scan.
- The hosted demo labels precomputed artifacts and degrades honestly when inference compute is unavailable.
- Documentation, sample report, diagrams, demo scripts, interview guide, case study, and resume bullets are complete and evidence-based.

## Local compute modes

| Mode | Intended use | Kronos | Storage | Services |
|---|---|---|---|---|
| `test` | Fast deterministic CI | Test-only adapter, unmistakably labeled | Temporary SQLite/files | In-process test harness |
| `lite` | CPU laptop demo | Real official mini checkpoint by default; base selectable | SQLite, DuckDB, local artifacts | API, worker, web, MLflow on loopback |
| `full` | Research workstation/server | Base by default, CUDA when available | PostgreSQL, DuckDB, S3-compatible artifacts | Containerized services |
| `precomputed` | Public read-only inspection | No live inference required | Verified real-pipeline artifacts | Web/API read path |

`test` output is never admissible as research evidence. `precomputed` artifacts must carry their originating run manifest and hashes.

## Dependency principles

- Python 3.11 is the baseline runtime because Kronos supports 3.10+, while current scientific packages and VectorBT support it well.
- `uv` owns Python resolution and locking; the JavaScript package manager will be pinned through Corepack.
- The research kernel owns financial semantics. Frameworks are adapters, not domain authorities.
- MLflow tracks experiments and model lifecycle but does not replace the immutable OpenAlpha manifest or data snapshots.
- VectorBT is optional for parity checks and research acceleration. Its Commons Clause license and simplified execution model exclude it from the authoritative accounting path.
- Docker Compose is a supported full profile, not a hidden prerequisite for laptop development.

## Sources reviewed

- [Kronos repository](https://github.com/shiyu-coder/Kronos)
- [Kronos paper](https://arxiv.org/abs/2508.02739)
- [Kronos-base model card](https://huggingface.co/NeoQuasar/Kronos-base)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [VectorBT documentation](https://vectorbt.dev/)

