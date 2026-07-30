# Sentinel v0 First Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one real, auditable SPY forecast origin that proves the causal Alpaca-to-Kronos-to-diagnostics-to-outcome chain without starting the development sample.

**Architecture:** Add one internal `openalpha-sentinel` package with typed v0-only boundaries and pure diagnostic functions. Reuse `LocalArtifactStore`, `RunStateJournal`, path confinement, and manifest verification. Forecast creation and outcome resolution are separate commands so the creation path cannot read the future outcome. Real Kronos runs behind a typed subprocess boundary from pinned official source and Hub revisions in an isolated temporary cache/environment outside Git; deterministic synthetic providers exist only in tests.

**Tech Stack:** Python 3.13 for repository code, isolated Python 3.11 for inference if required, Pydantic 2, NumPy, pandas, SciPy, PyTorch, Hugging Face Hub, exchange-calendars, pytest, Ruff, Pyright, uv.

---

## Scope lock

This plan implements Phase 2 only. The single origin is SPY at the weekly cutoff `2024-07-05`, with XNYS forecast sessions 2024-07-08 through 2024-07-12. It uses raw SIP OHLCV, context lengths 128/256/512, seeds 1729/2027/7919, `temperature=1.0`, `top_p=0.9`, and `sample_count=1`. Only the three 512-context close paths form the canonical timestamp-wise arithmetic mean; shorter paths are stress tests. It does not fit a Sentinel risk model, emit an action, inspect the holdout, add a public SDK/API, or begin the chronological development sample.

## Task 1: Create the narrow internal package and v0 contracts

**Files:**

- Create: `packages/sentinel/pyproject.toml`
- Create: `packages/sentinel/src/openalpha_sentinel/__init__.py`
- Create: `packages/sentinel/src/openalpha_sentinel/contracts.py`
- Create: `packages/sentinel/src/openalpha_sentinel/errors.py`
- Create: `packages/sentinel/src/openalpha_sentinel/py.typed`
- Test: `packages/sentinel/tests/test_contracts.py`
- Modify: `pyproject.toml`

- [ ] Write failing strict-model tests for ordered timezone-aware timestamps, one-to-one finite raw OHLCV rows, allowed context lengths, horizon 5, declared seeds, one path, adjustment identity, checkpoint identity, typed failures, and the invariant that a success cannot also carry failure information.
- [ ] Run `uv run pytest packages/sentinel/tests/test_contracts.py -q` and confirm failure because the package does not exist.
- [ ] Implement frozen Pydantic models `OHLCVObservation`, `ForecastRequest`, `ForecastPath`, `ForecastSummary`, `ForecastFailure`, and `ForecastResponse`, plus a `ForecastProvider` protocol with `forecast(request) -> ForecastResponse`.
- [ ] Keep provider identifiers internal and explicit. Do not expose the conceptual public `sentinel.forecast(...)` API.
- [ ] Add `openalpha-sentinel` to the uv workspace/source dependencies and regenerate `uv.lock` only through `uv lock` or `uv sync`.
- [ ] Run the contract test, Ruff on the new package, and Pyright on the package.

## Task 2: Prove and resolve the completed-manifest vocabulary mismatch

**Files:**

- Modify: `packages/research-core/src/openalpha_research/manifest.py`
- Modify: `packages/research-core/src/openalpha_research/__init__.py`
- Test: `packages/research-core/tests/test_manifest.py`

- [ ] Add a failing test that attempts to publish a completed Sentinel origin using only canonical experiment config, causal data provenance, forecasts, diagnostics/decision, resolved metrics, and methodology audit. Confirm the current publisher rejects it because it mandates orders, fills, and accounting artifacts.
- [ ] Add an explicit frozen `ManifestProfile` enum with `LEGACY_RESEARCH_RUN` as the default and `SENTINEL_ORIGIN` as an additive profile. Map each profile to a fixed required-kind set; keep every existing legacy requirement and existing test behavior unchanged.
- [ ] Use existing meanings where they align: `CANONICAL_SPEC`, `DATA_SNAPSHOT`, `DATA_QUALITY`, `FORECAST_ORIGINS`, `FORECASTS`, `DIAGNOSTICS`, `FORECAST_METRICS`, and `METHODOLOGY_AUDIT`. Do not relabel Sentinel decisions as orders, fills, or accounting.
- [ ] Require exactly one of each Sentinel-origin kind and preserve all lineage, schema, canonical-JSON, state, dirty-tree, and artifact-integrity checks.
- [ ] Run `uv run pytest packages/research-core/tests/test_manifest.py -q` and then the complete research-core tests.

## Task 3: Implement the causal Alpaca context boundary

**Files:**

- Create: `packages/sentinel/src/openalpha_sentinel/market_data.py`
- Create: `packages/sentinel/tests/fixtures/alpaca_spy_daily.synthetic.json`
- Create: `packages/sentinel/tests/test_market_data.py`

- [ ] Write failing tests for environment-only credentials, fixed HTTPS host, explicit SIP/1Day/start/end/raw/as-of fields, bounded pagination, ascending XNYS sessions, exact cutoff, duplicate/missing/nonfinite bar rejection, OHLC inequalities, nonnegative volume, response hashing, retrieval timestamps, and secret redaction.
- [ ] Mark the fixture metadata `synthetic: true`; never represent it as empirical market data.
- [ ] Implement a minimal HTTP-client port and `AlpacaHistoricalBarsProvider`. Keep raw bytes in memory only long enough to validate and SHA-256 hash, then return normalized observations plus provenance rather than raw response payloads.
- [ ] Make missing credentials, SIP entitlement, HTTP failure, malformed payload, and page-limit exhaustion typed failures. Do not fall back to IEX, Yahoo, or another provider.
- [ ] Implement a causal context constructor that returns only raw OHLCV sessions at or before `2024-07-05` for creation, with at least 512 observations, no amount field, and a separate outcome request for exactly 2024-07-08/09/10/11/12.
- [ ] Run the market-data tests, Ruff, and Pyright.

## Task 4: Make one pinned real Kronos path operational

**Files:**

- Create: `packages/sentinel/src/openalpha_sentinel/providers/__init__.py`
- Create: `packages/sentinel/src/openalpha_sentinel/providers/kronos.py`
- Create: `scripts/kronos_inference_worker.py`
- Create: `packages/sentinel/tests/test_kronos_provider.py`
- Modify: `packages/sentinel/pyproject.toml`

- [ ] Write unit tests around an injected predictor seam and typed JSON subprocess protocol for shape conversion, OHLCV input without amount, timestamp ordering, `sample_count=1`, seed application, evaluation mode, duration/provenance capture, empty-on-failure semantics, and no fake fallback.
- [ ] Add a `network`/`kronos` marked real smoke test that is skipped unless explicit environment gates and an outside-Git cache path are supplied.
- [ ] Create an isolated inference environment outside Git. Try the repository Python 3.13 runtime only when compatible; otherwise use Python 3.11 behind the same typed subprocess contract. Do not add a service, queue, database, container, or daemon.
- [ ] Resolve and load only the exact source revision `67b630e67f6a18c9e9be918d9b4337c960db1e9a`, model revision `f4e68697d9d5aed55cef5c96aabc3376bcad9f81`, and tokenizer revision `26966d0035065a0cae0ebad7af8ece35bc1fb51c`. Record repository names, downloaded filenames and SHA-256 values, package versions, Python/PyTorch/OS/device identity, cache path, and total cache size.
- [ ] Keep the source checkout and Hugging Face cache outside the repository. Use safe fixed-revision loading without `trust_remote_code`, and reject any unsafe serialized format or unexpected file.
- [ ] Run a synthetic OHLCV shape-validation inference before any Alpaca request and prove the response maps to exactly five declared sessions.
- [ ] Set Python, NumPy, PyTorch CPU, and applicable accelerator seeds before each request; use evaluation/inference mode. Probe A=512/1729, B=512/1729, C=512/2027, canonicalize every output, and record whether A/B hashes match and A/C ordinarily differ. Preserve mismatches without changing Kronos.

## Task 5: Build the nine-request ensemble and baseline

**Files:**

- Create: `packages/sentinel/src/openalpha_sentinel/ensemble.py`
- Create: `packages/sentinel/tests/test_ensemble.py`

- [ ] Write failing tests that require the exact Cartesian product of three contexts and three seeds, reject duplicate/missing/extra combinations, preserve every request failure, and prohibit aggregation of an incomplete real ensemble.
- [ ] Implement the raw last-value baseline and all individual raw-close path returns. Construct the canonical point forecast by timestamp-wise arithmetic mean of only the three 512-context close paths; derive its return as `log(final averaged close / raw cutoff close)`. Use 128/256 paths only for diagnostics.
- [ ] Ensure all output timestamps are the five expected XNYS sessions and every forecast derives from the same cutoff data identity.
- [ ] Run ensemble tests, Ruff, and Pyright.

## Task 6: Compute the fourteen pre-outcome diagnostics

**Files:**

- Create: `packages/sentinel/src/openalpha_sentinel/diagnostics.py`
- Create: `packages/sentinel/tests/test_diagnostics.py`

- [ ] Write table-driven synthetic tests for every formula locked in `docs/SENTINEL_METHODOLOGY.md`, including units, zero-volatility missingness, robust-z behavior, causal analogue eligibility, non-overlapping-candidate selection, minimum support, and horizon-divergence slope.
- [ ] Prove analogue candidates with unresolved or post-cutoff outcomes are excluded.
- [ ] Prove `RECENT_MODEL_ERROR` is `not_computable/NO_PRIOR_RESOLVED_FORECASTS` and `UNCERTAINTY_MISCALIBRATION` is `not_computable/NO_PRIOR_CALIBRATION_SAMPLE`; never replace either with zero.
- [ ] Return a fixed-order diagnostic vector with value, missingness, units, causal cutoff, and input hashes for every diagnostic.
- [ ] Run diagnostic tests, Ruff, and Pyright.

## Task 7: Persist creation separately from outcome resolution

**Files:**

- Create: `packages/sentinel/src/openalpha_sentinel/evidence.py`
- Create: `packages/sentinel/src/openalpha_sentinel/origin_service.py`
- Create: `packages/sentinel/tests/test_evidence.py`
- Create: `packages/sentinel/tests/test_origin_service.py`

- [ ] Write failing tests proving canonical content-addressed publication, immutable forecast/diagnostic/decision payloads, append-only lifecycle hash linkage, path confinement, and refusal to resolve an outcome before the fifth following session exists.
- [ ] Define v0 records for `FORECAST_CREATED`, `DIAGNOSTICS_COMPUTED`, and `OUTCOME_RESOLVED`; each carries its own hash and the previous event hash. Outcome data never appears in the creation-service input type.
- [ ] Do not emit a Sentinel decision, failure probability, reliability score, action, or failure reason before the Phase 3 risk model is fitted and frozen. The only v0 action values remain USE, BLEND, and ABSTAIN.
- [ ] Add a separate resolver that reads a sealed immutable forecast identity, retrieves exactly the five future XNYS sessions, computes raw realized log return, canonical 512-path error, direction, flat-baseline error, and path error, then appends a new outcome without changing the forecast.
- [ ] Publish and verify a `SENTINEL_ORIGIN` completed manifest only after outcome resolution. Include Git, dependency lock, hardware, data/model/config identities, test evidence, and methodology warnings.
- [ ] Run evidence/origin tests and the complete research-core suite.

## Task 8: Produce the single-origin audit through an internal command

**Files:**

- Create: `scripts/run_sentinel_v0_origin.py`
- Create: `packages/sentinel/src/openalpha_sentinel/reporting.py`
- Create: `packages/sentinel/tests/test_reporting.py`
- Modify: `research/sentinel-v0/README.md`

- [ ] Write a failing reporting test that requires `DEVELOPMENT PROOF — NOT EMPIRICAL EVIDENCE`, cutoff/horizon, provider/feed/raw adjustment, model/tokenizer revisions, environment/device, probe result, nine individual returns, three 512 paths, canonical averaged path/return, context summaries, diagnostic availability, realized return, Kronos/baseline errors, direction result, closer model, runtime/latencies/cache, failures/retries, limitations, artifact hashes, and manifest verification.
- [ ] Implement separate `create` and `resolve` subcommands in the internal script. `create` cannot import or call the outcome resolver; `resolve` requires the published creation record hash.
- [ ] Render canonical JSON evidence and a human-readable Markdown audit from verified artifacts only. Store generated output under ignored `research/sentinel-v0/results/generated/` and `reports/generated/` paths.
- [ ] Run `create` for SPY/2024-07-05 using the user's environment credentials and outside-Git cache, then run `resolve` for the five-session outcome ending 2024-07-12.
- [ ] Verify the manifest and artifact hashes independently. Record exact requests, sample count, failures, latency, cache size, and whether seed replay was deterministic.

## Task 9: Stop at the Phase 2 decision gate

**Files:**

- Modify: `docs/STATUS.md`

- [ ] Record the exact real-run result or exact blocker without making a Sentinel efficacy claim from one origin.
- [ ] Confirm no market response, normalized restricted dataset, source checkout, cache, checkpoint, credential, or generated result was added to Git.
- [ ] Run `uv sync --locked --group dev`, `uv run pytest -q`, `uv run ruff check .`, `uv run pyright packages/research-core packages/experiment-spec packages/sentinel`, and `git diff --check`.
- [ ] Inspect `git status --short` and generated-file ignores.
- [ ] Commit only the smallest-real-inference implementation and evidence documentation. Do not start Phase 3.

## Stop conditions

Stop and document a blocker if Alpaca credentials or SIP history are unavailable, 512 raw causal sessions cannot be retrieved, the pinned Kronos model/tokenizer cannot load, five output sessions cannot be mapped, outputs are non-finite, `sample_count=1` paths cannot be preserved, the forecast cannot be sealed before outcome access, memory/runtime is unreasonable, or real output would require a permanent deployment. A deterministic provider may keep automated tests passing, but it is never admissible as completion of this plan.
