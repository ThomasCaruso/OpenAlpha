# Kronos Reality Check Proof Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task with review checkpoints.

**Goal:** Prove one auditable historical-replay slice for SPY in which authenticated Alpaca data feeds real pinned Kronos and a last-value baseline, forecasts are persisted before outcomes are loaded, outcomes and metrics are appended, and the existing completed-run manifest verifies every artifact.

**Architecture:** Add one focused `openalpha-reality-check` Python package rather than rebuilding the retired platform. Its typed ports isolate protocol, provider, model, ledger, evaluation, orchestration, and CLI concerns. It reuses `openalpha-research-core` for content-addressed storage, path confinement, run journals, and completed manifests. Real provider and Kronos checks are opt-in; automated tests use explicitly synthetic fixtures.

**Tech Stack:** Python 3.13, uv, Pydantic 2, PyYAML, httpx, pandas/PyArrow, NumPy, PyTorch, the pinned official Kronos source and Hugging Face artifacts, pytest/Hypothesis, Ruff, and Pyright.

---

## Fixed proof-slice decisions

- Evidence class: `historical_replay`. This engineering slice was not preregistered before the outcome and must never be labeled sealed or live.
- Asset: SPY.
- Forecast cutoff/session: 2024-07-01 XNYS close.
- Horizon: 1 trading session.
- Provider: Alpaca historical stock bars at the fixed documented host.
- Feed/timeframe: `sip` / `1Day`; missing SIP entitlement is a failure, not an IEX fallback.
- Representations: `adjustment=all` for model input/target and `adjustment=raw` for hypothetical fills.
- As-of behavior: symbol mapping pinned to the forecast cutoff date and recorded in each request.
- Model: real `NeoQuasar/Kronos-mini` with its matching tokenizer, immutable revisions, reviewed upstream commit, and verified file hashes.
- Baseline: last observed adjusted close carried forward.
- Primary proof metric: one-session log-return absolute error, accompanied by squared error and direction correctness.
- Strategy: long at the next permitted raw open only when predicted return exceeds the candidate protocol threshold; otherwise cash; fixed candidate commission and slippage apply without test-period tuning.
- Local evidence root: user-configured and Git-ignored. Raw Alpaca bytes and downloaded checkpoints are never committed.
- Coverage status: `incomplete_engineering_slice`; the run declares its one asset, date, horizon, and two models and is inadmissible as aggregate protocol-v1 performance.

The dates, checkpoint family, threshold, and costs prove plumbing only. Protocol v1 is frozen after availability and model-feasibility checks, before any sealed-result evaluation.

## Task 1: Scaffold the narrow evidence package

**Files:**

- Create: `packages/reality-check/pyproject.toml`
- Create: `packages/reality-check/src/openalpha_reality_check/{__init__,errors,evidence}.py`
- Create: `packages/reality-check/src/openalpha_reality_check/py.typed`
- Create: `packages/reality-check/tests/test_evidence.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

1. Write a failing test that accepts only `historical_replay`, `sealed_historical_test`, and `live_precommitted_forecast`, and proves frozen Pydantic records reject update-based copies.
2. Run `uv run pytest packages/reality-check/tests/test_evidence.py -q`; confirm collection/import fails.
3. Add `EvidenceClass`, the shared strict frozen-model base, typed domain errors, package metadata, and root workspace dependency.
4. Run the focused test, `uv run ruff check packages/reality-check`, and `uv run pyright packages/reality-check`.
5. Commit: `feat(reality-check): add evidence package boundary`.

## Task 2: Create the versioned research-protocol contract

**Files:**

- Create: `packages/reality-check/src/openalpha_reality_check/protocol.py`
- Create: `packages/reality-check/tests/test_protocol.py`
- Create: `packages/reality-check/tests/fixtures/protocol/candidate-v1.yaml`
- Create: `research/protocol/candidate-v1.md`
- Create: `research/protocol/candidate-v1.yaml`

The contract contains typed sections for hypotheses; the five-symbol universe; provider/feed/timeframe and paired adjustment requests; development, validation, replay-quarantine, candidate sealed, and live-start periods; 1/5/20-session horizons; every required model and exact parameter map; targets; metrics; statistical tests; fixed strategy/costs; success/failure rules; exclusions; missing-data and corporate-action policies; random seeds; checkpoint provenance; and protocol status.

1. Write tests for unknown-field rejection, duplicate assets/models/horizons, missing required model families, period overlap, sealed dates on or before 2024-06-30, absent provider fields, empty criteria, mutable structures, non-finite values, and semantic equality under YAML key reordering.
2. Write tests for two hashes:
   - `source_sha256`: exact normalized LF UTF-8 source bytes;
   - `protocol_id`: `protocol_<sha256>` over deterministic canonical JSON from the validated semantic model.
3. Write a test that `status: locked` rejects any unresolved checkpoint revision/hash and rejects a missing `locked_at`; `status: candidate` remains valid for feasibility work.
4. Run `uv run pytest packages/reality-check/tests/test_protocol.py -q`; confirm failure.
5. Implement safe YAML loading, strict models, canonical JSON, source normalization, validation diagnostics, and candidate-file loading. Do not create `v1.*` yet.
6. Run the focused tests, Ruff, and Pyright.
7. Commit: `feat(protocol): add preregistration schema and candidate v1`.

## Task 3: Define the lawful market-data port and Alpaca request adapter

**Files:**

- Create: `packages/reality-check/src/openalpha_reality_check/data.py`
- Create: `packages/reality-check/src/openalpha_reality_check/alpaca.py`
- Create: `packages/reality-check/tests/test_data_contracts.py`
- Create: `packages/reality-check/tests/test_alpaca.py`
- Create: `packages/reality-check/tests/fixtures/alpaca/synthetic_page_{1,2}.json`
- Modify: `.gitignore`

Define `HistoricalBarsRequest`, `ProviderResponsePage`, `ProviderReceipt`, and a `MarketDataProvider` protocol. The request requires symbols, feed, timeframe, start, end, adjustment, as-of date, sort order, page limit, and maximum pages. The receipt records provider, endpoint, retrieval timestamp, request hash, response hashes, pagination count, feed, adjustment, limitations, and redistribution class.

1. Write tests proving every request field is serialized explicitly; the host is fixed; credentials come only from `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`; and missing credentials fail before the transport is called.
2. Test header and exception redaction with distinctive fake secrets. Assert neither secret occurs in `repr`, messages, receipts, or captured logs.
3. Test 401/403 entitlement failures, 429 retry bounds, timeouts, non-JSON/malformed bodies, oversize responses, token cycles, page-count bounds, duplicate pages, and no fallback.
4. Label each fixture with a sibling metadata record containing `synthetic: true` and `inadmissible_as_research_evidence: true`; test that fixtures cannot create a research-admissible receipt.
5. Run focused tests and confirm failure.
6. Implement the fixed `https://data.alpaca.markets/v2/stocks/bars` adapter with injected clock/transport for tests, bounded retries, and raw response hashes. Raw bytes may be written only below the configured local evidence root.
7. Run focused tests, Ruff, and Pyright.
8. Commit: `feat(data): add explicit Alpaca historical-bars adapter`.

## Task 4: Normalize paired snapshots without losing provenance

**Files:**

- Create: `packages/reality-check/src/openalpha_reality_check/snapshots.py`
- Create: `packages/reality-check/tests/test_snapshots.py`
- Create: `packages/reality-check/tests/fixtures/snapshots/synthetic_spy_bars.json`

1. Write tests for deterministic schema/order/timezone normalization, XNYS session mapping, duplicate/missing/out-of-bounds bars, non-finite/negative fields, OHLC consistency, stale and zero-volume warnings, and explicit adjusted/raw pairing.
2. Prove a context snapshot rejects any session after the forecast cutoff and an outcome snapshot cannot be passed to the model-context type.
3. Prove equivalent normalized inputs produce identical Parquet bytes or canonical JSON bytes on the supported platform; record the chosen artifact schema and media type.
4. Implement `ContextSnapshot`, `ExecutionSnapshot`, `OutcomeSnapshot`, `DataQualityReport`, and `PairedSnapshotRef`. Publish through the existing `LocalArtifactStore`; receipts contain hashes, not credentials or public raw bytes.
5. Run focused tests, then the existing artifact/path-confinement tests.
6. Commit: `feat(data): add causal paired market snapshots`.

## Task 5: Add one forecast adapter contract and the last-value baseline

**Files:**

- Create: `packages/reality-check/src/openalpha_reality_check/models.py`
- Create: `packages/reality-check/src/openalpha_reality_check/baselines.py`
- Create: `packages/reality-check/tests/test_model_contract.py`
- Create: `packages/reality-check/tests/test_last_value.py`

`ForecastRequest` supplies asset, session/cutoff, horizon, causal context reference, target definition, seed, and protocol identity. `ForecastResult` records model name/version/checkpoint/config hash, training window, input cutoff/context hash, predicted OHLC path, predicted log return/direction, optional uncertainty, runtime, hardware, seed, artifact hashes, and explicit success/failure status.

1. Write adapter conformance tests shared by all models, including failure retention and rejection of synthetic adapters when `research_admissible=True`.
2. Test the last-value close/path, zero predicted return, deterministic direction policy, metadata, and insufficient context.
3. Implement the typed `ForecastModel` protocol and `LastValueForecastAdapter`.
4. Run focused tests, Ruff, and Pyright.
5. Commit: `feat(models): add forecast contract and last-value baseline`.

## Task 6: Integrate real, pinned Kronos inference

**Files:**

- Create: `packages/reality-check/src/openalpha_reality_check/kronos.py`
- Create: `packages/reality-check/tests/test_kronos_adapter.py`
- Create: `packages/reality-check/tests/test_kronos_real.py`
- Modify: `packages/reality-check/pyproject.toml`
- Modify: `uv.lock`
- Modify: `docs/DATA_POLICY.md`

1. Verify the current official Kronos repository, checkpoint/tokenizer pairing, license, input columns, context limit, sampling arguments, and safe-loading behavior. Record primary-source URLs and the reviewed commit in adapter metadata and the candidate protocol.
2. Pin the official source commit and all model/tokenizer revisions; resolve and store expected downloaded file hashes during feasibility. Never use a moving branch or `trust_remote_code`.
3. Write unit tests around an injected reviewed inference callable for column mapping, evaluation mode, seed controls, context/horizon bounds, return derivation, runtime/hardware capture, and typed resource/checkpoint failures. This callable is a seam, not a fake research forecast.
4. Add a `@pytest.mark.kronos` real test that loads the exact mini checkpoint, runs one bounded synthetic input, asserts finite/schema-valid output, and records its hashes. Skip only with an explicit resource message.
5. Implement `KronosForecastAdapter`; a failed real forecast remains a visible failed result and is never replaced by the baseline or test adapter.
6. Run unit tests. Run `uv run pytest -m kronos packages/reality-check/tests/test_kronos_real.py -q` on the target host and record time/memory.
7. Commit: `feat(kronos): add pinned real inference adapter`.

## Task 7: Implement the append-only forecast ledger

**Files:**

- Create: `packages/reality-check/src/openalpha_reality_check/ledger.py`
- Create: `packages/reality-check/tests/test_ledger.py`

Define immutable `ForecastRecord`, `OutcomeRecord`, and `LedgerEnvelope` schemas. The forecast includes every field required by the protocol plus strategy decision and expected evaluation timestamp. The outcome references the forecast hash and carries realized path/return, errors, costed hypothetical P&L, and resolution time. Each envelope hashes canonical record bytes plus its previous-record hash.

1. Write tests for genesis and multi-record chains, deterministic hashes, complete-field validation, forecast immutability, append-only outcome resolution, early outcome rejection, duplicate outcome rejection, mismatched asset/horizon/snapshot rejection, broken/reordered/truncated chain detection, and unresolved/failed forecast visibility.
2. Write a test proving ledger publication completes and fsyncs before a supplied outcome loader is invoked.
3. Implement `ForecastLedger.create`, `append_forecast`, `append_outcome`, `records`, and `verify` on confined local paths; use content-addressed references for record payloads.
4. Run focused tests and the preserved artifact/run-journal tests.
5. Commit: `feat(ledger): add immutable forecast and outcome chain`.

## Task 8: Compute primary metrics and the fixed economic decision

**Files:**

- Create: `packages/reality-check/src/openalpha_reality_check/evaluation.py`
- Create: `packages/reality-check/src/openalpha_reality_check/strategy.py`
- Create: `packages/reality-check/tests/test_evaluation.py`
- Create: `packages/reality-check/tests/test_strategy.py`

1. Write table-driven tests for future log return, MAE, RMSE components, direction, balanced direction undefined cases, price error, cost application, next-open timing, cash decisions, long decisions, turnover, and P&L.
2. Prove adjusted bars determine forecast/target values while raw bars determine fills, and reject session/corporate-action mismatches.
3. Require model, baseline, horizon, asset, evidence class, sample count, and artifact hashes on metric rows; prevent isolated Kronos metric serialization without its declared comparator.
4. Implement single-origin metrics and the locked long/cash rule. Aggregate statistics, bootstrap, Diebold-Mariano, Holm correction, and multi-period ratios remain out of this proof slice but stay locked in the protocol.
5. Run focused tests, Ruff, and Pyright.
6. Commit: `feat(evaluation): score forecasts and costed decisions`.

## Task 9: Orchestrate the proof slice and reuse completed manifests

**Files:**

- Create: `packages/reality-check/src/openalpha_reality_check/service.py`
- Create: `packages/reality-check/tests/test_service.py`
- Modify only if required by a demonstrated incompatibility: `packages/research-core/src/openalpha_research/manifest.py`

Implement two application operations:

- `create_forecasts`: validates candidate protocol, obtains context-only adjusted/raw snapshots, runs Kronos and last value, publishes both forecasts, appends forecast records, and returns without reading outcome data;
- `resolve_and_complete`: verifies persisted forecasts, obtains the later outcome snapshot, appends outcomes, computes paired metrics and fixed strategy artifacts, completes the existing run journal, and publishes the existing `RunManifest`.

1. Write an in-memory/synthetic integration test whose spies assert the outcome provider is untouched until both forecast envelopes have been durably published.
2. Test all failure stages: provider, data quality, Kronos, baseline, ledger, evaluation, and manifest. Failures remain in the journal/ledger and cannot publish a misleading completed manifest.
3. Map the existing required manifest kinds to proof artifacts: canonical protocol projection, paired snapshot index, quality report, forecast origin, paired forecasts, paired metrics, decisions/signals, hypothetical orders/fills, accounting ledger, single-period risk facts, and methodology audit.
4. Mark the manifest coverage `incomplete_engineering_slice` and list the omitted protocol assets, origins, horizons, and models. Reject its use as aggregate protocol-v1 evidence.
5. Keep the existing manifest verifier authoritative. Change it only if the test demonstrates a concrete incompatibility; add regression tests before any such change.
6. Run the integration test and all `research-core` tests.
7. Commit: `feat(service): complete forecast-to-outcome proof slice`.

## Task 10: Expose the complete flow through a CLI

**Files:**

- Create: `packages/reality-check/src/openalpha_reality_check/cli.py`
- Create: `packages/reality-check/src/openalpha_reality_check/__main__.py`
- Create: `packages/reality-check/tests/test_cli.py`
- Modify: `packages/reality-check/pyproject.toml`

Provide commands:

- `openalpha protocol validate PATH`
- `openalpha protocol feasibility PATH`
- `openalpha protocol lock CANDIDATE --version v1`
- `openalpha snapshot fetch --protocol PATH --asset SPY --cutoff 2024-07-01`
- `openalpha forecast create --protocol PATH --asset SPY --cutoff 2024-07-01 --models kronos-mini,last-value`
- `openalpha outcome resolve --ledger PATH --forecast-id ID`
- `openalpha ledger verify PATH`
- `openalpha run proof-slice --protocol PATH --asset SPY --cutoff 2024-07-01`
- `openalpha reproduce --manifest PATH`

1. Write CLI tests for exit codes, structured errors, missing credentials, secret redaction, synthetic labeling, path confinement, incomplete/unresolved output, manifest verification, and human-readable artifact IDs.
2. Make `forecast create` and `outcome resolve` separate commands so a true live forecast cannot score itself. The historical `run proof-slice` command calls the same two operations sequentially and labels the result `historical_replay`.
3. Print values only from verified records/artifacts and include evidence class in every result heading.
4. Run CLI tests and invoke `uv run openalpha --help`.
5. Commit: `feat(cli): expose auditable reality-check workflow`.

## Task 11: Perform feasibility checks and freeze protocol v1

**Files:**

- Create: `research/protocol/v1.md`
- Create: `research/protocol/v1.yaml`
- Create: `research/protocol/v1.sha256`
- Modify: `research/protocol/candidate-v1.{md,yaml}`
- Modify: `docs/STATUS.md`

1. With the user's local credentials, run provider feasibility for all five assets across development, validation, replay-quarantine, and candidate sealed date bounds. Inspect only availability, schema, adjustment behavior, missingness, entitlements, and timestamp suitability—not model performance.
2. Run real Kronos mini feasibility for context/horizon/schema/resource behavior without evaluating candidate sealed returns.
3. Resolve exact source commit, checkpoint/tokenizer revisions, file hashes, model parameters, request parameters, thresholds, costs, seeds, exclusions, and criteria in the candidate.
4. Run `uv run openalpha protocol lock research/protocol/candidate-v1.yaml --version v1`. It validates `status: locked`, writes v1 files atomically, and refuses to overwrite an existing version.
5. Verify `v1.sha256` against normalized LF UTF-8 `v1.yaml` bytes and record the separate semantic `protocol_id` in `v1.md`.
6. Add a regression test that any semantic or source-byte change fails the lock check and requires v2.
7. Commit: `research: lock Kronos reality check protocol v1`.

Do not execute this task without credentials and verified checkpoint resources. If feasibility changes the candidate dates, the reason must be availability/methodological suitability and must be recorded before any candidate sealed performance is computed.

## Task 12: Run and verify the real one-origin experiment

**Files:**

- Local only: configured evidence root containing provider bytes, checkpoints, ledgers, artifacts, and manifest
- Modify: `docs/STATUS.md`

1. Start from a clean committed tree and record the commit SHA and `uv.lock` hash.
2. Run `uv run openalpha run proof-slice --protocol research/protocol/v1.yaml --asset SPY --cutoff 2024-07-01`.
3. Confirm the CLI labels the run `historical_replay` and `incomplete_engineering_slice`, emits real Alpaca/SIP and Kronos provenance, and stores both Kronos and last-value forecasts before it fetches the realized next session.
4. Run `uv run openalpha ledger verify <ledger-path>` and `uv run openalpha reproduce --manifest <manifest-path>`.
5. Inspect the verified artifact inventory for protocol, paired data snapshots, quality, origin, two forecasts, two outcomes, paired metrics, strategy decision, costs/P&L, methodology audit, run journal, and completed manifest.
6. Record exact commands, exit codes, artifact/manifest hashes, forecast IDs, checkpoint revision, runtime/hardware, evidence class, and any warnings or failures in `docs/STATUS.md`. Do not copy raw bars or secrets into the document.
7. Commit: `research: record verified SPY proof-slice evidence`.

## Task 13: Run final quality and contradiction checks

**Files:**

- Modify as evidence requires: `docs/STATUS.md`

Run:

```powershell
uv sync --locked --group dev
uv run pytest -q
uv run ruff check .
uv run pyright packages/research-core packages/experiment-spec packages/reality-check
git diff --check
rg -n -i 'provider:\\s*yaho[o]|no-paid[-]key|free[-]provider' docs examples --glob '!**/archive/**'
rg -n 'T[B]D|T[O]DO|implement[ ]later|similar[ ]to|appropriate[ ]error[ ]handling' docs/superpowers/plans/2026-07-30-kronos-reality-check-proof-slice.md
```

Expected:

- all automated tests pass using synthetic data only;
- real network/Kronos tests are explicitly marked and separately reported;
- Ruff and Pyright have zero findings;
- no active Yahoo/no-key requirement remains;
- no plan placeholders remain;
- the only empirical claims are tied to verified real-data artifacts and the correct evidence class.

Commit any evidence-only corrections, then stop. Expansion to all assets, horizons, baselines, statistical inference, scoreboard, audit UI, or paper begins only after this proof slice meets every acceptance condition.
