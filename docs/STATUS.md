# Project Status

Last updated: 2026-07-30

## Product direction

OpenAlpha has pivoted to **Kronos Reality Check**, an evidence-first historical, sealed, and live evaluation of Kronos against simple baselines. The generalized quantitative-platform Phase 1 plan is stopped before any Yahoo adapter work and archived for history.

## Preserved completed work

- Commit `c037bd5efafd728cb4e0df961dd6a93889b0ce56` is preserved.
- Reproducible Python/npm workspace and lock enforcement.
- Immutable experiment-spec prototype with canonical identity.
- Content-addressed artifact publication, path confinement, symlink/junction defenses, hash/size/media/schema verification, and acyclic lineage.
- Immutable run-state journals and retry history.
- Completed-run manifests with methodology, Git, environment, and artifact verification.
- 220 passing tests at the preserved checkpoint, Ruff clean, and Pyright at zero errors/warnings.

## Pivot decisions completed in this worktree

- Added `docs/PIVOT.md` with the rejected scope, new question, preserved infrastructure, evidence classes, non-goals, and value definition.
- Replaced the governing master plan, architecture, and methodology with CLI-first evidence-product documents.
- Added `docs/DATA_POLICY.md`.
- Removed Yahoo’s undocumented endpoint from all active plans.
- Selected an authenticated provider-independent boundary with Alpaca as the first adapter and no fallback.
- Declared SIP daily requests, explicit adjustment/as-of fields, environment credentials, local-only raw data, and no redistribution.
- Archived the superseded generalized Phase 1 plan.
- Added ADRs for the evidence product, Alpaca boundary, and CLI-first proof slice.

## Current work

The replacement test-first implementation plan is written for one real SPY forecast/outcome proof slice before any dashboard or service work. No proof-slice implementation has begun.

## Pivot-planning verification

Verified on 2026-07-30 from this worktree:

- `uv sync --locked --group dev`: exit 0, 26 packages resolved and 25 checked;
- `uv run pytest -q`: 220 passed;
- `uv run ruff check .`: all checks passed;
- `uv run pyright packages/research-core packages/experiment-spec`: 0 errors, 0 warnings, 0 informations;
- `git diff --check`: no whitespace errors;
- active-plan scans: 13 sequential tasks, no placeholder phrases, and no active Yahoo/no-key provider configuration.

## Verified provider facts

- Alpaca documents `https://data.alpaca.markets/v2/stocks/bars` with explicit symbols, timeframe, start, end, adjustment, as-of, feed, pagination, and sort fields.
- Authentication uses `APCA-API-KEY-ID` and `APCA-API-SECRET-KEY` headers sourced from `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` environment variables.
- SIP and IEX have materially different coverage; the protocol must pin one and cannot accept a fallback.
- Alpaca states API market data cannot be redistributed.

## Known limitations and unresolved evidence

- No Alpaca credentials are present or required for automated tests; no real provider request has run.
- No raw market data is committed or publicly redistributable.
- Protocol v1 is not yet frozen; candidate dates remain subject to availability and feasibility checks that do not inspect sealed performance.
- January–June 2024 overlaps Kronos’s reported pretraining range and is replay-only, not clean sealed evidence.
- No real Kronos checkpoint has been downloaded or timed on this CPU-only host.
- No real forecast, outcome, metric, ledger chain, CLI proof slice, scoreboard, or report exists yet.
- Modern adjusted bars do not provide a complete historical-vintage guarantee.
- The existing experiment-spec contract is preserved but does not itself represent the new full research protocol.

## Exact next task

Implement the versioned research-protocol schema test-first, including canonical bytes/hash, immutable v1 locking, evidence classes, five-asset universe, paired Alpaca data representations, candidate periods, three horizons, declared models/metrics/statistics/strategy, and explicit validation that no sealed period overlaps the reported Kronos pretraining cutoff.
