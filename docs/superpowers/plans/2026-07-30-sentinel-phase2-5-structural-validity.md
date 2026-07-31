# Sentinel Phase 2.5 Structural-Validity Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate whether Phase 2 OHLC violations originate in pinned official Kronos output or OpenAlpha, quantify recurrence and severity, and publish an auditable Phase 2.5 classification without changing sealed Phase 2 evidence.

**Architecture:** Add one focused model-agnostic structural-validity module and one Phase 2.5 audit path around the existing typed Kronos provider. A trace-enabled worker observes the pinned official numerical pipeline without modifying upstream source; a bounded orchestrator writes private execution material outside Git and compact derived JSON/report outputs inside `research/sentinel-v0/phase2_5`.

**Tech Stack:** Python 3.13 workspace contracts and tests, isolated Python 3.11 PyTorch/Kronos inference, Pydantic, NumPy, pandas, yfinance, exchange-calendars, content-addressed OpenAlpha artifacts, pytest, Ruff, Pyright.

---

### Task 1: Lock read-only evidence and structural-validity behavior

**Files:**
- Create: `packages/sentinel/tests/test_structural_validity.py`
- Create: `packages/sentinel/src/openalpha_sentinel/structural_validity.py`
- Modify: `packages/sentinel/src/openalpha_sentinel/__init__.py`

- [ ] **Step 1: Write failing validity tests**

Define five-session `ForecastPath` fixtures that cover valid candles, every price constraint, nonfinite and nonpositive values through a raw audit-candle input type, negative volume, timestamp shift, duplicate timestamp, and wrong horizon. Assert immutable `StructuralViolation` records and the exact eleven diagnostics. Include a deliberate high/low/open/close label permutation that must produce violations.

```python
result = validate_forecast_path(
    path_id='permuted',
    candles=permuted_candles,
    expected_sessions=expected_sessions,
    cutoff_close=100.0,
    cutoff_volume=1_000_000.0,
)
assert {item.code for item in result.violations} >= {
    'HIGH_BELOW_OPEN',
    'LOW_ABOVE_CLOSE',
}
assert result.diagnostics['INVALID_PATH_FRACTION'] == 1.0
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest packages/sentinel/tests/test_structural_validity.py -q`

Expected: collection fails because `openalpha_sentinel.structural_validity` does not exist.

- [ ] **Step 3: Implement immutable validity contracts and diagnostics**

Create narrow Pydantic records for `AuditCandle`, `StructuralViolation`, `PathValidity`, `StructuralDiagnostics`, and `ProjectionResult`. Implement `validate_forecast_path`, `summarize_structural_validity`, and `constraint_projection_v0`. Use cutoff close for price severity and `max(cutoff_volume, 1.0)` for volume severity. Preserve null severity for nonfinite values. Never mutate input candles.

- [ ] **Step 4: Add projection tests and verify RED/GREEN**

Assert projection changes only high and low, produces valid candles, records every adjustment, and preserves both the complete close path and implied log return exactly. Run the targeted test before and after implementation.

- [ ] **Step 5: Run focused quality checks**

Run:

```text
uv run pytest packages/sentinel/tests/test_structural_validity.py -q
uv run ruff check packages/sentinel/src/openalpha_sentinel/structural_validity.py packages/sentinel/tests/test_structural_validity.py
uv run pyright packages/sentinel/src/openalpha_sentinel/structural_validity.py
```

Expected: all commands exit zero.

### Task 2: Enforce named output columns and exact timestamp alignment

**Files:**
- Modify: `packages/sentinel/tests/test_kronos_provider.py`
- Modify: `scripts/kronos_inference_worker.py`
- Create: `packages/sentinel/tests/test_official_worker_output.py`

- [ ] **Step 1: Write failing named-column and timestamp tests**

Create predictor DataFrames whose physical column order is deliberately permuted but names are correct; extracted OHLCV must remain identical. Create shifted, duplicate, and missing output indices; extraction must fail rather than positionally remap them. Create mislabeled output columns; extraction must fail.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest packages/sentinel/tests/test_official_worker_output.py packages/sentinel/tests/test_kronos_provider.py -q`

Expected: failure because exact named extraction and timestamp validation are not exposed as a tested boundary.

- [ ] **Step 3: Implement exact output extraction**

Implement `_extract_predicted_ohlcv(predicted, expected_sessions, np, pd)` inside the standalone inference worker using named `.loc` selection and exact normalized DatetimeIndex comparison. Keeping the pure helper inside the worker preserves the isolated Python 3.11 process without importing the OpenAlpha package. Call it before serializing a path. Do not validate or repair predicted OHLC ordering in this boundary.

- [ ] **Step 4: Verify GREEN and compatibility**

Run the two focused test files. Confirm the existing sample-count, pin, amount, and output-preservation tests still pass.

### Task 3: Trace the pinned official numerical pipeline

**Files:**
- Create: `scripts/kronos_phase2_5_worker.py`
- Create: `packages/sentinel/src/openalpha_sentinel/integration_trace.py`
- Create: `packages/sentinel/tests/test_integration_trace.py`

- [ ] **Step 1: Write failing safe-summary and trace-schema tests**

Require deterministic canonical hashes and summaries for DataFrames, NumPy arrays, and tensors. Assert columns, order, shape, dtype, timestamp bounds, per-feature minima/maxima, units, and transform metadata. Assert token summaries contain no token payload.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest packages/sentinel/tests/test_integration_trace.py -q`

Expected: collection fails because the trace module does not exist.

- [ ] **Step 3: Implement safe summaries**

Implement canonical finite summaries and source-location metadata in `integration_trace.py`. Refuse nonfinite numeric summaries unless the boundary explicitly records a nonfinite count.

- [ ] **Step 4: Implement trace-enabled direct worker**

Load only the pinned source/model/tokenizer snapshots already used in Phase 2. Wrap the actual tokenizer `encode` and `decode` methods and the predictor `generate` method to capture inputs and outputs, then delegate unchanged. Record pre-normalization DataFrame, derived amount, normalized tensor, time stamps, encoded tokens, decoded tensor, denormalized official DataFrame, structural results, hashes, source file paths, function names, duration, environment, and downloaded-file hashes. Reset every declared RNG immediately before prediction.

- [ ] **Step 5: Test request bounds and source immutability**

Add synthetic request tests proving the worker rejects unpinned identities, undeclared dates, raw-data output requests, and payloads above the Phase 2.5 call bound. Hash pinned source files before and after a synthetic trace and require equality.

### Task 4: Build the bounded Phase 2.5 orchestrator

**Files:**
- Create: `packages/sentinel/src/openalpha_sentinel/phase2_5.py`
- Create: `packages/sentinel/tests/test_phase2_5.py`
- Create: `scripts/run_sentinel_phase2_5.py`

- [ ] **Step 1: Write failing orchestration tests**

Use deterministic synthetic providers to assert operation order: snapshot Phase 2 fingerprints, forecast-only fetch, direct call, provider call, comparison, seal private audit receipt, and only then outcome fetch. Assert no output path resolves inside the Phase 2 root. Assert a normalized-input hash mismatch disables exact A/B/C equality claims.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest packages/sentinel/tests/test_phase2_5.py -q`

Expected: collection fails because the Phase 2.5 orchestrator does not exist.

- [ ] **Step 3: Implement configuration and provenance**

Hard-code only the approved Phase 2 origin, pins, expected sessions, seed set, sample counts `3` and `5`, and canary cutoffs `2024-09-27`, `2025-01-31`, and `2025-05-30`. Compute XNYS outcome sessions rather than using weekday arithmetic. Store private receipts under `~/.cache/openalpha-sentinel/phase2_5`.

- [ ] **Step 4: Implement comparison operations**

Provide separate CLI operations for `trace`, `averaging`, `repeatability`, `canary-forecast`, `canary-resolve`, and `report`. Each operation consumes prior immutable receipts by hash. The report operation cannot fetch data or run inference.

- [ ] **Step 5: Verify GREEN**

Run orchestration tests plus existing market-data, provider, evidence, and reporting tests.

### Task 5: Execute direct golden path and Phase 2 A/B/C isolation

**Private outputs:**
- `~/.cache/openalpha-sentinel/phase2_5/integration_trace.private.json`
- `~/.cache/openalpha-sentinel/phase2_5/official_comparison.private.json`

**Repository outputs:**
- Create: `research/sentinel-v0/phase2_5/integration_trace.json`
- Create: `research/sentinel-v0/phase2_5/official_comparison.json`

- [ ] **Step 1: Fingerprint sealed Phase 2 evidence and pinned source**

Record SHA-256 for Phase 2 `creation.json`, `resolved.json`, every content-addressed artifact, and the pinned official files cited by the design. Do not write into either source root.

- [ ] **Step 2: Run official golden-path trace**

Use the upstream regression fixture, pinned mini/2k snapshots, 512 observations, five fixture timestamps, `sample_count=1`, seed `1729`, temperature `1.0`, and top-p `0.9`. Validate direct official output structurally and retain only the fixture hash and safe summaries in OpenAlpha.

- [ ] **Step 3: Re-fetch and validate Phase 2 SPY context**

Use the pinned yfinance request. Compare the normalized input hash to the sealed Phase 2 hash before any exact reproduction claim. Preserve a mismatch and continue only with current-input A/B comparison if Yahoo history changed.

- [ ] **Step 4: Run A/B/C comparison**

For exact matching input, run direct official and typed-provider context-512 seed `1729` calls and compare both with the immutable Phase 2 path. Compare output indices, named values, row hashes, path hashes, and structural violations. Identify the first unequal numerical boundary.

- [ ] **Step 5: Materialize compact JSON**

Write canonical sorted finite JSON with source paths/functions, pair identities, safe summaries, hashes, equality results, violation details, and limitations. Do not include context rows or raw responses.

### Task 6: Execute repeatability, averaging, and projection experiments

**Repository outputs:**
- Create: `research/sentinel-v0/phase2_5/structural_validity.json`
- Create: `research/sentinel-v0/phase2_5/averaging_comparison.json`
- Create: `research/sentinel-v0/phase2_5/repair_experiment.json`

- [ ] **Step 1: Repeat the three context-512 seeds**

Run `1729`, `2027`, and `7919` with identical Phase 2 input and resets. Compare output and violation hashes, exact steps, codes, gaps, and severities to sealed paths.

- [ ] **Step 2: Analyze sealed individual and offline averages**

Validate all nine sealed paths, the existing offline three-path context-512 canonical average, and an offline all-nine diagnostic average. Do not change the Phase 2 canonical artifact.

- [ ] **Step 3: Run official internal averages**

Execute direct `sample_count=3`. Execute `sample_count=5` only if the preceding call remains inside the existing 1,800-second and reasonable-memory boundary. Record that independent offline seeds are not assumed to reproduce internal sampling order.

- [ ] **Step 4: Run projection**

Apply `CONSTRAINT_PROJECTION_V0` to every invalid sealed path and averaged path. Assert unchanged opens, closes, and implied returns. Record field adjustments, severity, range changes, and path-volatility changes.

- [ ] **Step 5: Write compact derived outputs**

Materialize exact diagnostic names and all path-, candle-, count-, and severity-level summaries without producing a composite score.

### Task 7: Run the bounded six-origin canary

**Repository output:**
- Create: `research/sentinel-v0/phase2_5/canary_results.json`

- [ ] **Step 1: Materialize exact XNYS forecast sessions**

For each approved cutoff and both assets, calculate and record the next five XNYS sessions before inference. Reject missing history and never replace a failed cutoff.

- [ ] **Step 2: Run eighteen official paths**

Run only context 512 with seeds `1729`, `2027`, and `7919`, each with `sample_count=1`. Persist private forecast receipts and structural validity before any outcome access.

- [ ] **Step 3: Resolve outcomes separately**

After all forecast receipts verify, fetch only the required outcome windows, calculate raw close-return error for the canonical three-path mean and the zero-return baseline, and append private outcome receipts.

- [ ] **Step 4: Materialize canary output**

Record per-origin invalid-path fraction, invalid-candle fraction, counts, severities, canonical and realized returns, Kronos and baseline errors, runtimes, failures, input hashes, and aggregate recurrence counts. State that six origins cannot establish correlation or population frequency.

### Task 8: Classify, govern, document, and verify

**Files:**
- Create: `docs/SENTINEL_STRUCTURAL_VALIDITY.md`
- Create: `research/sentinel-v0/phase2_5/report.md`
- Modify: `docs/SENTINEL_DIRECTION.md`
- Modify: `docs/SENTINEL_METHODOLOGY.md`
- Modify: `docs/SENTINEL_FAILURE_TAXONOMY.md`
- Modify: `docs/STATUS.md`
- Conditionally modify: `research/sentinel-v0/experiment.yaml`
- Conditionally modify: `research/sentinel-v0/experiment.sha256`
- Conditionally create: `research/sentinel-v0/amendments/2026-07-30-structural-validity-diagnostics.md`

- [ ] **Step 1: Apply the decision rule**

Choose exactly one of `OPENALPHA_INTEGRATION_BUG`, `OFFICIAL_RAW_PATH_STRUCTURAL_INVALIDITY_CONFIRMED`, `AMBIGUOUS`, or `NONRECURRING_EDGE_CASE` using direct output location, A/B/C equality, repeatability, and canary recurrence. Do not use forecast accuracy to choose the category.

- [ ] **Step 2: Amend Sentinel v0 only if official invalidity is confirmed**

Preserve the previous hash, add the exact eleven diagnostic definitions and expected relationships before any holdout work, regenerate SHA-256, and document fields and reason. Do not change Phase 2 artifacts.

- [ ] **Step 3: Generate artifact-grounded documentation**

The report must source every count and number from the compact JSON outputs. Document official source locations, integration equivalence, structural frequencies, averaging, repeatability, projection magnitude, canary recurrence, limitations, conclusion category, and next justified task.

- [ ] **Step 4: Verify original evidence is unchanged**

Recompute Phase 2 descriptor, artifact inventory, and pinned-source hashes and compare them byte-for-byte with Step 1 fingerprints. Run the existing creation and resolved-chain verifier.

- [ ] **Step 5: Run final gates**

Run:

```text
uv sync --locked --group dev --group sentinel-phase2
uv run pytest -q
uv run ruff check .
uv run pyright packages/research-core packages/experiment-spec packages/sentinel scripts/run_sentinel_v0_origin.py scripts/run_sentinel_phase2_5.py
git diff --check
```

Scan tracked files for raw datasets, Yahoo payloads, caches, credentials, and model/checkpoint files. Record exact outputs in `docs/STATUS.md`.

- [ ] **Step 6: Commit and stop**

Stage only implementation, tests, compact derived evidence, documentation, experiment amendment if required, and dependency lock changes if any. Commit with `feat: complete Sentinel Phase 2.5 integration audit`. Confirm the worktree is clean and do not start the full development sample.
