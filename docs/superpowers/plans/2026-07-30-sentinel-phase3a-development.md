# Sentinel Phase 3A Development Sample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and audit all 104 locked development origins, evaluate structural and reliability assurance chronologically, and freeze a verified holdout configuration without accessing any holdout origin.

**Architecture:** A sealed development manifest drives a resumable origin-at-a-time state machine. Exact inference requests use a private content-addressed cache; forecast evidence is sealed before outcome retrieval; verified compact rows feed deterministic chronological analysis and freeze artifacts. Sentinel v0.4 remains byte-for-byte unchanged.

**Tech Stack:** Python 3.13 repository environment, isolated Python 3.11 Kronos inference worker, Pydantic, NumPy, pandas, scikit-learn 1.7.2, exchange-calendars, yfinance, openalpha-research-core, pytest, Ruff, Pyright.

---

## Execution constraints

- Work only in C:UsersTommyopen-alpha.worktreesphase-1-contracts on branch feature/phase-1-contracts.
- Preserve commits through 41047f1 and every sealed Phase 2/2.5 artifact.
- Do not modify research/sentinel-v0/experiment.yaml or experiment.sha256.
- Reject every origin cutoff on or after 2025-07-01.
- Permit July rows only as the five declared outcome sessions of the 2025-06-27 development origin.
- Keep private state under C:UsersTommy.cacheopenalpha-sentinelphase3a.
- Keep model/source/cache paths under the existing Phase 2 cache.
- Use apply_patch for source and documentation edits.
- Use synthetic data and fake inference in automated tests.
- Do not commit intermediate Phase 3A implementation. The user requires one Phase 3A commit only after all completion gates pass.
- After every task, run the listed tests and git diff --check, then leave changes uncommitted.
- Stop the real run if the operational pilot fails a locked criterion.
- Stop before any holdout-origin action even when the recommendation is A.

## File map

Create focused modules for development_manifest, development_serialization, inference_cache, development_diagnostics, development_origin, development_resolution, development_runner, development_table, risk_model, development_analysis, interventions, development_freeze and development_reporting under packages/sentinel/src/openalpha_sentinel/. Create one matching test module per production module under packages/sentinel/tests/. Create scripts/run_sentinel_phase3a.py.

Modify pyproject.toml and uv.lock for the pinned Phase 3 group. Modify diagnostics.py and ensemble.py only for reusable causal primitives and named-field path averaging. Update the four governed Sentinel documents only after measured results exist.

Materialize the required research/sentinel-v0/development outputs only after the private real run, analysis and freeze verify.

### Task 1: Pin the Phase 3 analysis environment

**Files:**
- Modify: pyproject.toml
- Modify: uv.lock
- Create: packages/sentinel/tests/test_phase3_dependencies.py

- [ ] **Step 1: Write the failing dependency test**

```python
from importlib.metadata import version

def test_phase3_scikit_learn_is_pinned() -> None:
    assert version("scikit-learn") == "1.7.2"
```

- [ ] **Step 2: Run RED**

Run `uv run pytest -q packages/sentinel/tests/test_phase3_dependencies.py`. Expected: failure because version 1.7.2 is absent.

- [ ] **Step 3: Add the exact dependency**

Add `sentinel-phase3 = ["scikit-learn==1.7.2"]` to dependency-groups. Run:

```powershell
uv lock
uv sync --locked --group dev --group sentinel-phase2 --group sentinel-phase3
```

- [ ] **Step 4: Verify GREEN**

Run the test and `uv run --group sentinel-phase3 python -c "import sklearn; print(sklearn.__version__)"`. Expected: 1.7.2.

- [ ] **Step 5: Checkpoint without committing**

Run git diff --check and git status --short.

### Task 2: Seal the exact population and boundary

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/development_manifest.py
- Create: packages/sentinel/tests/test_development_manifest.py

- [ ] **Step 1: Write failing tests**

```python
from datetime import date
import pytest
from openalpha_sentinel.development_manifest import (
    EXPERIMENT_SHA256,
    build_development_manifest,
    require_development_cutoff,
    require_outcome_sessions,
)

def test_manifest_has_exact_locked_population() -> None:
    manifest = build_development_manifest()
    assert EXPERIMENT_SHA256 == "fd50c208f465ce99750b11d4de5d5bc8c391bd73cb9ff450037c596c3e37d8bc"
    assert len(manifest.cutoffs) == 52
    assert len(manifest.origins) == 104
    assert manifest.cutoffs[0] == date(2024, 7, 5)
    assert manifest.cutoffs[-1] == date(2025, 6, 27)
    assert [(item.cutoff, item.asset) for item in manifest.origins[:2]] == [
        (date(2024, 7, 5), "SPY"),
        (date(2024, 7, 5), "QQQ"),
    ]

def test_holdout_cutoff_is_rejected_but_final_outcome_is_allowed() -> None:
    with pytest.raises(ValueError, match="holdout"):
        require_development_cutoff(date(2025, 7, 1))
    assert require_outcome_sessions(date(2025, 6, 27))[-1] == date(2025, 7, 7)
```

- [ ] **Step 2: Run RED**

Run the test. Expected: import failure.

- [ ] **Step 3: Implement immutable manifest models**

Define strict frozen DevelopmentOrigin and DevelopmentManifest models. Use exchange_calendars XNYS, group sessions by ISO year/week, retain the last session only when inside 2024-07-01 through 2025-06-30, then order SPY before QQQ per cutoff. Include the stale target-field override note and a canonical hash.

- [ ] **Step 4: Implement boundary functions**

require_development_cutoff rejects dates outside development. require_outcome_sessions returns the next five XNYS sessions and permits July only when resolving a valid development cutoff.

- [ ] **Step 5: Run GREEN**

Run the manifest tests and print count/hash. Expected: 104 and a 64-character SHA-256.

- [ ] **Step 6: Checkpoint without committing**

Verify git diff shows no change to experiment.yaml or experiment.sha256.

### Task 3: Add deterministic serialization

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/development_serialization.py
- Create: packages/sentinel/tests/test_development_serialization.py
- Modify: packages/sentinel/src/openalpha_sentinel/development_manifest.py

- [ ] **Step 1: Write failing tests**

```python
import math
import pytest
from openalpha_sentinel.development_serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)

def test_canonical_json_is_order_independent_and_finite() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": math.nan})

def test_jsonl_has_manifest_order_and_final_newline() -> None:
    assert canonical_jsonl_bytes(({"id": "a"}, {"id": "b"})) == (
        b'{"id":"a"}
{"id":"b"}
'
    )
```

- [ ] **Step 2: Run RED**

Expected: import failure.

- [ ] **Step 3: Implement exact functions**

```python
def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

def canonical_jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"
" for row in rows)

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
```

Use these functions in development_manifest.

- [ ] **Step 4: Run GREEN and checkpoint**

Run serialization and manifest tests, Ruff, Pyright and git diff --check.

### Task 4: Implement the verified private inference cache

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/inference_cache.py
- Create: packages/sentinel/tests/test_inference_cache.py

- [ ] **Step 1: Write failing tests**

Create synthetic requests and responses. Assert changing created_at or experiment hash does not change the inference key, while changing input hash, sessions, context, seed, temperature, top-p, sample count, horizon or model pins does. Put/get a path, corrupt its private artifact bytes, and assert ArtifactIntegrityError.

- [ ] **Step 2: Run RED**

Expected: import failure.

- [ ] **Step 3: Implement the key**

InferenceCacheKey contains model/tokenizer/source revisions, symbol, cutoff, forecast sessions, observation hash, context, seed, temperature, top-p, sample count and horizon. Its canonical_sha256 excludes created_at and experiment SHA because neither changes model inference.

- [ ] **Step 4: Implement storage and verification**

Store canonical response JSON in LocalArtifactStore outside Git. get_verified verifies the ArtifactRef, revalidates ForecastResponse, checks request identity and timestamps, and recomputes the path hash. Return None only for a true miss; corruption raises.

- [ ] **Step 5: Run GREEN and checkpoint**

Run cache tests, Ruff, Pyright and git diff --check. Confirm no cache file appears in git status.

### Task 5: Generalize path averaging and causal diagnostics

**Files:**
- Modify: packages/sentinel/src/openalpha_sentinel/ensemble.py
- Modify: packages/sentinel/src/openalpha_sentinel/diagnostics.py
- Create: packages/sentinel/src/openalpha_sentinel/development_diagnostics.py
- Modify: packages/sentinel/tests/test_ensemble.py
- Modify: packages/sentinel/tests/test_diagnostics.py
- Create: packages/sentinel/tests/test_development_diagnostics.py

- [ ] **Step 1: Write failing averaging tests**

Use three synthetic ForecastPath objects. Assert average_forecast_paths averages open/high/low/close/volume by name, preserves exact sessions and rejects timestamp mismatch.

- [ ] **Step 2: Write failing recent-error tests**

```python
def test_recent_error_requires_eight_prior_resolved_forecasts() -> None:
    missing = recent_model_error((0.1,) * 7)
    assert missing.status == "not_computable"
    assert missing.reason == "INSUFFICIENT_PRIOR_RESOLVED_FORECASTS"
    available = recent_model_error(tuple(float(i) / 100 for i in range(9)))
    assert available.value == pytest.approx(sum(float(i) / 100 for i in range(1, 9)) / 8)
```

Also test that selected prior outcomes end no later than the cutoff and were sealed before diagnostic creation.

- [ ] **Step 3: Run RED**

Expected: missing functions.

- [ ] **Step 4: Implement reusable functions**

Add named-field average_forecast_paths to ensemble.py. Keep compute_phase_2_diagnostics compatible. Expose causal primitives and implement compute_development_nonstructural_diagnostics returning the locked fourteen diagnostics with structured recent-error missingness.

- [ ] **Step 5: Run GREEN and checkpoint**

Run ensemble, diagnostic and development diagnostic tests. Prove the Phase 2 fixture remains byte-equivalent. Run static checks.
### Task 6: Build forecast creation with separate structural assurance

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/development_origin.py
- Create: packages/sentinel/tests/test_development_origin.py

- [ ] **Step 1: Write the failing synthetic origin test**

Inject a fake market provider and fake forecast function returning the exact nine paths, with one invalid 512 path.

```python
assert result.structural_status == "FAILED"
assert len(result.individual_paths) == 9
assert result.raw_close_return_forecast == result.ensemble.canonical_predicted_log_return
assert result.projected_path_status == "PASSED"
assert result.projected_path.return_unchanged is True
assert result.creation_verified is True
assert outcome_loader.calls == 0
```

Assert the canonical structural OHLCV path is the named-field average of three 512 paths and its closes equal the locked canonical close path.

- [ ] **Step 2: Write boundary and cache tests**

Assert cutoff 2025-07-01 is rejected before provider access. Run the same origin twice and assert the second run performs zero fake inference and records nine cache hits.

- [ ] **Step 3: Run RED**

Expected: development_origin import failure.

- [ ] **Step 4: Implement create_development_forecast**

Use injected MarketDataProvider, forecast_many callable, InferenceCache, LocalArtifactStore, EvidenceContext and prior resolved errors. Validate manifest membership; fetch causal raw data ending cutoff plus one day exclusively; require 512 XNYS sessions; build the exact nine requests; run cache misses only; persist raw paths; assemble canonical forecast; validate individual and averaged 512 OHLCV paths; project all paths; merge fourteen nonstructural and eleven structural diagnostics in locked order; publish and verify creation evidence; atomically write the private descriptor.

Set origin structural_status from canonical full-path validity. Preserve individual status and raw close return separately.

- [ ] **Step 5: Run GREEN and regress Phase 2**

Run development-origin, Phase 2 origin, ensemble and structural tests. Expected: all pass and Phase 2 remains unchanged.

- [ ] **Step 6: Checkpoint without committing**

Run Ruff, Pyright and git diff --check.

### Task 7: Resolve outcomes and compact terminal rows

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/development_resolution.py
- Create: packages/sentinel/tests/test_development_resolution.py

- [ ] **Step 1: Write seal-before-outcome tests**

Tamper with a synthetic creation seal and assert the outcome provider is never called. Test the exact five outcome sessions for 2025-06-27.

- [ ] **Step 2: Write row-separation tests**

```python
required = {
    "structural_status",
    "structural_diagnostics",
    "raw_close_return_forecast",
    "projected_path",
    "projected_path_status",
    "reliability_features",
    "realized_outcome",
    "forecast_error",
}
assert required <= row.keys()
assert "observations" not in json.dumps(row)
assert row["projected_path"]["implied_return_unchanged"] is True
```

Write a terminal-failure test requiring code, stage, attempt hashes and unavailable-analysis status.

- [ ] **Step 3: Run RED**

Expected: missing module.

- [ ] **Step 4: Implement resolution**

Verify CreationEvidence before constructing the outcome loader. Fetch exactly declared sessions, calculate realized raw return, canonical/baseline/individual errors, direction, deployability input and P3 candidate. Append and verify resolved evidence. Write one compact terminal descriptor. Implement record_terminal_failure without deleting failed attempts. Do not compute the pooled failure label yet.

- [ ] **Step 5: Run GREEN and checkpoint**

Run resolution, evidence and Phase 2 outcome tests, then static checks.

### Task 8: Implement resumable orchestration and pilot gates

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/development_runner.py
- Create: packages/sentinel/tests/test_development_runner.py
- Create: scripts/run_sentinel_phase3a.py
- Create: packages/sentinel/tests/test_phase3a_cli.py

- [ ] **Step 1: Write resume tests**

Use twelve synthetic origins and interrupt after five. Resume and assert five verified origins are skipped, no completed inference repeats and all twelve eventually become terminal.

- [ ] **Step 2: Write pilot tests**

```python
gate = evaluate_pilot(records, cache_bytes=32_283_678, hosted_cost_per_cutoff=0.0)
assert gate.request_success_rate >= 0.95
assert gate.median_ensemble_latency_seconds <= 600
assert gate.cache_bytes <= 2_147_483_648
assert gate.continue_automatically is True
```

Add failures for success below 0.95, latency above 600 seconds, cache above 2 GiB and unresumable process instability.

- [ ] **Step 3: Write CLI boundary tests**

Assert the CLI exposes only preflight, pilot, run, analyze, freeze, report and verify. No operation accepts a cutoff or period. Patch network/inference to fail if verify calls them.

- [ ] **Step 4: Run RED**

Expected: missing runner/script.

- [ ] **Step 5: Implement runner**

preflight seals the manifest. pilot selects the first ten Phase 3A-nonterminal origins, runs their full lifecycle, performs one same-input/same-seed replay probe, writes operational metrics, and calls run_remaining automatically only if gates pass. run_remaining follows manifest order and checkpoints every terminal result. verify checks origins and hashes offline.

- [ ] **Step 6: Run GREEN and checkpoint**

Run runner/CLI tests plus existing Phase 2/2.5 CLI tests, Ruff, Pyright and git diff --check.

### Task 9: Build the deterministic development table

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/development_table.py
- Create: packages/sentinel/tests/test_development_table.py

- [ ] **Step 1: Write deterministic-table tests**

Shuffle terminal files and assert table output follows manifest order, includes all terminal failures, ends with a newline and reproduces byte-for-byte.

```python
expected = float(np.quantile(errors, 0.75, method="linear"))
assert table.failure_threshold == expected
assert [row["failure_label"] for row in table.completed_rows] == [
    error >= expected for error in errors
]
```

- [ ] **Step 2: Run RED**

Expected: missing module.

- [ ] **Step 3: Implement table and manifest**

Verify exactly one terminal descriptor per origin, completed chains, declared failures and development cutoffs. Add the pooled linear 75th-percentile failure threshold/labels. Produce canonical JSONL, byte hash, ordered origin hashes and completed/failed counts. Rebuild in memory and require byte equality before publication.

- [ ] **Step 4: Run GREEN and checkpoint**

Run table, serialization, manifest and resolution tests plus static checks.

### Task 10: Implement chronological OOF models and ablations

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/risk_model.py
- Create: packages/sentinel/tests/test_risk_model.py

- [ ] **Step 1: Write fold-isolation tests**

Use 52 paired cutoffs. Assert three expanding folds, one-week gap, paired assets in the same fold, training dates before validation, train-only imputation/scaling and no OOF prediction for early training-only rows.

- [ ] **Step 2: Write family and hyperparameter tests**

Create deterministic predictive data. Assert structural, nonstructural and combined families share folds; all locked C/alpha values are reported; selection is deterministic; one-class folds make a candidate ineligible.

- [ ] **Step 3: Run RED**

Expected: missing module.

- [ ] **Step 4: Implement train-only preprocessing**

Define the eleven structural and fourteen nonstructural features. Per fold, remove more-than-20-percent-missing, zero-variance and redundant features; add missing indicators; median-impute and standardize from train only. Use LogisticRegression penalty l2, solver lbfgs, max_iter 10000 and Ridge with locked grids.

- [ ] **Step 5: Implement OOF selection**

Return fold metrics, row predictions, removals and all candidates. Select C by mean log loss, alpha by mean MAE, and family by the approved one-standard-error rule. Require finite predictions from all three folds.

- [ ] **Step 6: Run GREEN and checkpoint**

Run under sentinel-phase3, then Ruff, Pyright and git diff --check.
### Task 11: Implement prevalence and diagnostic analyses

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/development_analysis.py
- Create: packages/sentinel/tests/test_development_analysis.py

- [ ] **Step 1: Write prevalence tests**

Synthetic rows must yield exact invalid path/candle/canonical fractions and asset, quarter, context, seed and step breakdowns whose counts sum to totals.

- [ ] **Step 2: Write bootstrap/diagnostic tests**

Assert four-week moving-block bootstrap uses paired cutoff clusters, 1000 resamples and seed 314159 reproducibly. Constant diagnostics return explicit unavailable reasons. Every one of twenty-five diagnostics appears even when null or unfavorable.

- [ ] **Step 3: Write structural-control test**

Create confounded data and assert the descriptive control regression reports the structural coefficient without claiming incremental value. OOF ablation remains the predictive evidence.

- [ ] **Step 4: Run RED**

Expected: missing module.

- [ ] **Step 5: Implement analyses**

Implement build_structural_prevalence, moving_block_bootstrap, analyze_diagnostics, analyze_structural_error and analyze_chronological_stability. Use pairwise finite Spearman, explicit sample counts, percentile 95-percent intervals and structured missing reasons.

- [ ] **Step 6: Run GREEN and checkpoint**

Run analysis/risk tests and static checks.

### Task 12: Implement P0-P3 and risk coverage

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/interventions.py
- Create: packages/sentinel/tests/test_interventions.py

- [ ] **Step 1: Write coverage tests**

For ten OOF rows assert sample counts 10, 9, 8, 7 and 5 at 100, 90, 80, 70 and 50 percent. Ties sort by risk, cutoff, asset. Every level reports MAE, direction and count.

- [ ] **Step 2: Write P2 tests**

Risk at/below pooled OOF 50th percentile uses Kronos; above 50th through 80th uses exact 50/50 return blend; above 80th abstains. No asset-specific threshold exists.

- [ ] **Step 3: Write P3 tests**

Fewer than two valid 512 paths retains canonical and flags. Two valid paths average their closes without changing the canonical field. Retention requires at least ten applications, positive paired improvement, bootstrap lower bound above zero and no asset worsening.

- [ ] **Step 4: Run RED**

Expected: missing module.

- [ ] **Step 5: Implement policies**

evaluate_policies and build_risk_coverage produce P0 full acceptance, P1 all coverages, P2 fixed actions and P3 candidate, pooled/per-asset/quarterly results, baseline/blend comparisons and counts.

- [ ] **Step 6: Run GREEN and checkpoint**

Run interventions, analysis and risk tests plus static checks.

### Task 13: Fit and verify the freeze candidate

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/development_freeze.py
- Create: packages/sentinel/tests/test_development_freeze.py

- [ ] **Step 1: Write completeness tests**

Assert freeze includes failure threshold, retained/removed diagnostics, feature order, missing indicators, medians, means, scales, logistic/ridge parameters and coefficients, reliability formula, 50th/80th thresholds, blend 0.50, structural gate, P3 decision, reason thresholds, versions and artifact hashes.

- [ ] **Step 2: Write tamper/boundary tests**

A valid freeze verifies; changing one coefficient fails. No holdout origin or raw observations appear.

- [ ] **Step 3: Run RED**

Expected: missing module.

- [ ] **Step 4: Implement final fit and freeze**

Refit the selected family on all completed development rows. Extract coefficients in feature order. Compute final risk quantiles and locked reason quantiles. STRUCTURALLY_INVALID_MODEL_OUTPUT remains deterministic. Canonically serialize, hash and verify the descriptor.

- [ ] **Step 5: Run GREEN and checkpoint**

Run freeze/risk tests and static checks.

### Task 14: Generate recommendation and report

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/development_reporting.py
- Create: packages/sentinel/tests/test_development_reporting.py

- [ ] **Step 1: Write recommendation tests**

```python
assert recommend(_all_pass()) == "PROCEED_TO_LOCKED_HOLDOUT"
assert recommend(_structural_only()) == "PROCEED_AS_STRUCTURAL_CONTRACT_ONLY"
assert recommend(_weak_positive_signal()) == "CHANGE_THE_RELIABILITY_APPROACH"
assert recommend(_no_assurance()) == "STOP"
```

- [ ] **Step 2: Write provenance tests**

The report begins DEVELOPMENT ANALYSIS - NOT HOLDOUT EVIDENCE, answers all thirteen questions, includes failed diagnostics/origins and exact freeze hash, and sources all numbers from verified artifacts.

- [ ] **Step 3: Run RED**

Expected: missing module.

- [ ] **Step 4: Implement mapping and renderer**

recommend applies S6, R1-R5 and material structural prevalence exactly in approved order. render_development_report accepts verified artifact payloads only and rejects hash mismatch.

- [ ] **Step 5: Run GREEN and checkpoint**

Run reporting/freeze/intervention/analysis tests and static checks.

### Task 15: Complete CLI offline integration and policy scanning

**Files:**
- Modify: packages/sentinel/src/openalpha_sentinel/development_runner.py
- Modify: scripts/run_sentinel_phase3a.py
- Modify: packages/sentinel/tests/test_development_runner.py
- Modify: packages/sentinel/tests/test_phase3a_cli.py

- [ ] **Step 1: Add end-to-end synthetic tests**

Build synthetic private state for 104 origins. Invoke analyze, freeze, report and verify. Assert every required output is produced and verify makes no network/inference call.

- [ ] **Step 2: Add policy scans**

Reject observations arrays, reusable OHLCV histories, provider response payloads, credentials, repository cache paths, NaN/Infinity and any origin outside the manifest.

- [ ] **Step 3: Run RED**

Expected: offline operations incomplete.

- [ ] **Step 4: Connect operations**

analyze verifies terminal origins and builds table/analysis receipts. freeze consumes verified analysis only. report consumes verified analysis/freeze only. verify checks manifest, every chain, deterministic table, analysis/freeze/report hashes, source revision and experiment hash offline.

- [ ] **Step 5: Run GREEN and checkpoint**

Run all Phase 3A tests, repository Ruff, configured Pyright scope and git diff --check.
### Task 16: Pass the pre-real-run verification gate

**Files:**
- No new files

- [ ] **Step 1: Synchronize the locked environments**

Run:

```powershell
uv sync --locked --group dev --group sentinel-phase2 --group sentinel-phase3
```

Record the resolved `scikit-learn` version and retain the Phase 2 inference environment outside Git.

- [ ] **Step 2: Run the Phase 3A targeted tests**

Run every Phase 3A manifest, boundary, cache, runner, structural, diagnostic, analysis, policy, freeze, reporting and CLI test. Stop before real requests if any test fails.

- [ ] **Step 3: Run the complete repository checks**

```powershell
uv run pytest -q
uv run ruff check .
uv run pyright packages/research-core packages/experiment-spec packages/sentinel scripts/run_sentinel_v0_origin.py scripts/kronos_inference_worker.py scripts/kronos_phase2_5_worker.py scripts/run_sentinel_phase2_5.py scripts/run_sentinel_phase3a.py
git diff --check
```

- [ ] **Step 4: Verify preserved evidence and locks**

Verify the sealed Phase 2 and Phase 2.5 chains. Recompute the Sentinel v0.4 experiment hash and assert the configuration is byte-for-byte unchanged. Record source-tree and configuration hashes before any real run.

- [ ] **Step 5: Run the provider-free preflight**

```powershell
uv run --group sentinel-phase3 python scripts\run_sentinel_phase3a.py preflight --state-root C:\Users\Tommy\.cache\openalpha-sentinel\phase3a
```

Expected: 52 unique cutoffs, 104 ordered origins, first cutoff 2024-07-05, last cutoff 2025-06-27, and zero network or inference requests. Stop if any boundary or count differs.

### Task 17: Execute the operational pilot and complete the real development run

**Files:**
- Create privately outside Git: `C:\Users\Tommy\.cache\openalpha-sentinel\phase3a\...`
- Populate compact repository outputs only after verification: `research/sentinel-v0/development/origins/`

- [ ] **Step 1: Record the execution environment**

Record the model/tokenizer/source revisions, dependency versions, device, cache location and size, and runtime fingerprint. Verify private cache/output paths are ignored and outside the repository.

- [ ] **Step 2: Run the ten-origin operational pilot**

```powershell
uv run --group sentinel-phase3 python scripts\run_sentinel_phase3a.py pilot --state-root C:\Users\Tommy\.cache\openalpha-sentinel\phase3a --inference-python C:\Users\Tommy\.cache\openalpha-sentinel\phase2\venv\Scripts\python.exe --source-path C:\Users\Tommy\.cache\openalpha-sentinel\phase2\kronos-src --model-cache C:\Users\Tommy\.cache\openalpha-sentinel\phase2\hf
```

Process exactly the first ten nonterminal chronological origins. Inspect request success, latency, cache growth, artifact size, the declared deterministic replay probe, process stability, provider failures and validation failures only. Do not inspect outcome relationships or alter methodology.

- [ ] **Step 3: Apply the locked pilot decision**

Write and verify a pilot receipt. If every declared feasibility criterion passes, continue automatically with the remaining origins using the same command/state. If a criterion fails, stop with a terminal operational receipt; do not fit models, enlarge infrastructure or commit a partial Phase 3A result.

- [ ] **Step 4: Monitor and resume safely**

Checkpoint after each origin and emit progress at least once per minute while work is active. Exercise one safe resume and one verified cache hit. Never repeat a completed inference unless its cached receipt fails verification.

- [ ] **Step 5: Verify terminal population**

Require exactly 104 terminal origin records, with completed plus failed equal to 104. Verify every completed forecast was sealed before its outcome request, every chain verifies, all failure records are explicit, no cutoff is on or after 2025-07-01, and only the final development origin may contain outcome-session metadata after 2025-06-30.

### Task 18: Analyze, freeze, report and materialize compact outputs

**Files:**
- Create: `research/sentinel-v0/development/development_table.jsonl`
- Create: `research/sentinel-v0/development/development_manifest.json`
- Create: `research/sentinel-v0/development/structural_prevalence.json`
- Create: `research/sentinel-v0/development/diagnostic_analysis.json`
- Create: `research/sentinel-v0/development/out_of_fold_predictions.jsonl`
- Create: `research/sentinel-v0/development/model_comparison.json`
- Create: `research/sentinel-v0/development/intervention_analysis.json`
- Create: `research/sentinel-v0/development/risk_coverage.json`
- Create: `research/sentinel-v0/development/freeze_candidate.json`
- Create: `research/sentinel-v0/development/report.md`

- [ ] **Step 1: Build and verify the development analysis**

```powershell
uv run --group sentinel-phase3 python scripts\run_sentinel_phase3a.py analyze --state-root C:\Users\Tommy\.cache\openalpha-sentinel\phase3a
```

Require a deterministic one-row-per-origin table, explicit diagnostic missingness, structural prevalence, all diagnostic relationships, chronological out-of-fold predictions, three feature-family ablations, P0-P3 intervention results, risk coverage at all five locked levels, confidence intervals, asset splits and chronological stability. Preserve unfavorable and null results.

- [ ] **Step 2: Freeze the development-selected system**

```powershell
uv run --group sentinel-phase3 python scripts\run_sentinel_phase3a.py freeze --state-root C:\Users\Tommy\.cache\openalpha-sentinel\phase3a
```

Verify that the freeze records the selected and removed diagnostics, feature order, missing-data policy, preprocessing values, coefficients/intercepts, reliability conversion, USE/BLEND/ABSTAIN thresholds, blend weight, structural gate, P3 decision, failure-reason thresholds, software versions and all upstream hashes. This command must fail closed if no eligible model family exists.

- [ ] **Step 3: Generate the evidence-backed report**

```powershell
uv run --group sentinel-phase3 python scripts\run_sentinel_phase3a.py report --state-root C:\Users\Tommy\.cache\openalpha-sentinel\phase3a
```

The report must begin `DEVELOPMENT ANALYSIS - NOT HOLDOUT EVIDENCE`, answer the thirteen required questions, cite exact artifact hashes and make the deterministic recommendation without claiming holdout evidence.

- [ ] **Step 4: Materialize only verified compact repository artifacts**

Use `apply_patch` to add the verified private `repository_outputs` payloads to the paths above. Never copy model weights, raw Yahoo responses, reusable bar histories, caches or private exact inference paths. Recompute public hashes and compare them with the verified private receipts.

- [ ] **Step 5: Verify offline reproduction**

Run the CLI `verify` operation with network and inference disabled. Rebuild the development-table serialization and confirm the same content hash. Verify every output provenance edge through the freeze and report.

### Task 19: Update Sentinel documentation from verified artifacts

**Files:**
- Modify: `docs/SENTINEL_STRUCTURAL_VALIDITY.md`
- Modify: `docs/SENTINEL_METHODOLOGY.md`
- Modify: `docs/SENTINEL_FAILURE_TAXONOMY.md`
- Modify: `docs/STATUS.md`

- [ ] **Step 1: Update structural-validity findings**

Record exact development prevalence, severity, categories, context/seed/asset/time/horizon summaries, projection invariants and whether structural diagnostics add incremental return-error signal. Keep structural safety conclusions separate from reliability conclusions.

- [ ] **Step 2: Update methodology and taxonomy**

Document the actual chronological folds, causal feature behavior, failure threshold, selected/removed diagnostics, reason thresholds, structural gate, action policy, P3 decision and freeze hash. State that no holdout origin was accessed.

- [ ] **Step 3: Record exact status evidence**

Record commands, test counts, origin/path counts, completed and failed records, public artifact hashes, OOF results, all coverage results, selected feature set, frozen action policy, recommendation and the exact next task. Call out the stale v0.4 target-count field without changing the locked file.

- [ ] **Step 4: Scan the claim boundary**

Reject wording that implies holdout, provider-independent or production evidence. Ensure all result claims are traceable to verified compact artifacts.

### Task 20: Complete final verification and create the single Phase 3A commit

**Files:**
- Modify only if verification evidence requires it: `docs/STATUS.md`

- [ ] **Step 1: Run targeted and complete tests**

Run every Phase 3A targeted test, then `uv run pytest -q`. Record exact passing, skipped and failed counts in status.

- [ ] **Step 2: Run static and artifact verification**

Run repository Ruff, configured Pyright, artifact verification, ledger verification, experiment-hash verification, offline Phase 3A verification and `git diff --check`. No check may be omitted from the recorded evidence.

- [ ] **Step 3: Run repository-policy and holdout-boundary scans**

Scan tracked changes for model/tokenizer weights, Hugging Face/yfinance caches, raw provider payloads, reusable OHLCV histories, CSV exports, credentials, NaN/Infinity, absolute private cache paths and any cutoff on or after 2025-07-01. Permit post-June dates only where they are the declared five-session outcome metadata for the last development origin.

- [ ] **Step 4: Verify every completion gate**

Require: 104 eligible origins; 104 terminal origins; completed plus failed equals 104; every completed chain verifies; every failure is accounted for; expected official-path/probe counts reconcile; projected-path invariants hold; the table reproduces deterministically; OOF analysis is complete; the freeze verifies; the report verifies; Sentinel v0.4 is unchanged; Phase 2 and Phase 2.5 artifacts are unchanged; and only intended files differ.

- [ ] **Step 5: Stage the exact Phase 3A scope**

Inspect `git status`, `git diff` and `git diff --cached`. Stage only the implementation, tests, compact permitted outputs and required documentation.

- [ ] **Step 6: Create the one implementation commit**

```powershell
git commit -m 'feat: complete Sentinel Phase 3A development analysis'
```

- [ ] **Step 7: Verify the committed state**

Confirm the commit hash, clean worktree, preserved history and unchanged locked artifacts. Do not access holdout data.

- [ ] **Step 8: Report and stop**

Report the commit hash, eligible/completed/failed origins, generated paths, structural prevalence, OOF performance, all coverage results, selected feature set, frozen policy, recommendation and exact next task. Stop before any holdout operation.
