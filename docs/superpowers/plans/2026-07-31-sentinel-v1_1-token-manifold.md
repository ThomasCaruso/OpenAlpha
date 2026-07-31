# Sentinel v1.1 Token-Manifold Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Determine whether Kronos invalidity originates in tokenizer reconstruction, unsupported token pairs, or low valid probability mass, and test support-conditioned decoding only when preregistered rules permit it.

**Architecture:** Add v1.1 beside unchanged v0/v1 trees. Pure modules own experiment identity, support, mass bounds, classification, and gates; a Python 3.11 worker owns pinned inference; a locked runner owns provider access, immutable checkpoints, analysis, and reports. No CLI accepts symbol, cutoff, holdout, model, tokenizer, budget, or method overrides.

**Tech Stack:** Python 3.13 repository environment, isolated Python 3.11 PyTorch/Kronos environment, Pydantic, NumPy, pandas, exchange-calendars, yfinance 1.5.2, pytest, Ruff, Pyright, and SHA-256 artifacts.

---

## File structure

- Create packages/sentinel/src/openalpha_sentinel/sentinel_v1_1.py for immutable scope and request builders.
- Create packages/sentinel/src/openalpha_sentinel/token_manifold.py for round trips and token support.
- Create packages/sentinel/src/openalpha_sentinel/constraint_compatibility.py for mass, classification, sampling, and gates.
- Create scripts/kronos_token_manifold_worker.py for pinned encode/decode and token/logit audits.
- Create scripts/run_sentinel_v1_1.py for fixed orchestration and verification.
- Create five matching test files under packages/sentinel/tests.
- Create research/sentinel-v1_1, two new research docs, and update the four required status/direction docs.

### Task 1: Lock and commit the v1.1 experiment

**Files:**
- Create: research/sentinel-v1_1/experiment.yaml
- Create: research/sentinel-v1_1/experiment.sha256
- Create: research/sentinel-v1_1/preregistration.md
- Create: README.md in each required v1.1 results directory
- Create: packages/sentinel/src/openalpha_sentinel/sentinel_v1_1.py
- Test: packages/sentinel/tests/test_sentinel_v1_1.py

- [ ] **Step 1: Write the failing lock test**

    from datetime import date
    from pathlib import Path
    from openalpha_sentinel.sentinel_v1_1 import (
        SUPPORT_SYMBOLS, locked_v1_1_origins, verify_experiment_hash,
    )

    def test_v1_1_scope_is_fixed_and_pre_holdout() -> None:
        assert SUPPORT_SYMBOLS == (
            'SPY', 'QQQ', 'IWM', 'DIA', 'TLT',
            'HYG', 'GLD', 'EFA', 'EEM', 'XLF',
        )
        origins = locked_v1_1_origins()
        assert len(origins) == 12
        assert max(
            session for item in origins for session in item.forecast_sessions
        ) < date(2025, 7, 1)

    def test_v1_1_experiment_hash_matches() -> None:
        assert verify_experiment_hash(
            Path('research/sentinel-v1_1/experiment.yaml'),
            Path('research/sentinel-v1_1/experiment.sha256'),
        )

- [ ] **Step 2: Run the test and verify collection fails**

    uv run pytest packages/sentinel/tests/test_sentinel_v1_1.py -q

Expected: sentinel_v1_1 cannot be imported.

- [ ] **Step 3: Create the exact YAML and preregistration**

Encode immutable revisions, the ten-symbol corpus, fixed dates and windows, exact twelve v1 origins, seeds, sampling settings, candidate budgets, support rule, Jeffreys smoothing, round-trip thresholds, ordered A-F rules, canary limits, conditional-method gate, and all continuation thresholds. Set holdout_access to prohibited and outcomes_access to only_after_forecast_seal. Hash the exact YAML bytes:

    (Get-FileHash -Algorithm SHA256 research/sentinel-v1_1/experiment.yaml).Hash.ToLowerInvariant()

Label the preregistration DEVELOPMENT COMPATIBILITY RESEARCH - NOT HOLDOUT EVIDENCE.

- [ ] **Step 4: Add the immutable identity module**

    SUPPORT_SYMBOLS = (
        'SPY', 'QQQ', 'IWM', 'DIA', 'TLT',
        'HYG', 'GLD', 'EFA', 'EEM', 'XLF',
    )
    SUPPORT_START = date(2017, 1, 1)
    SUPPORT_END_EXCLUSIVE = date(2024, 6, 29)
    SUPPORT_WINDOW_LENGTH = 512
    SUPPORT_WINDOWS_PER_SYMBOL = 3
    SEEDS = (1729, 2027, 7919)
    MASS_BUDGETS = ((8, 8), (16, 16), (32, 32))
    HOLDOUT_START = date(2025, 7, 1)

Implement frozen V11Origin values with require_outcome_sessions. verify_experiment_hash compares the recorded lowercase SHA-256 to hashlib.sha256(path.read_bytes()).hexdigest().

- [ ] **Step 5: Verify and commit the preregistration before real execution**

    uv run pytest packages/sentinel/tests/test_sentinel_v1_1.py -q
    uv run ruff check packages/sentinel/src/openalpha_sentinel/sentinel_v1_1.py packages/sentinel/tests/test_sentinel_v1_1.py
    uv run pyright packages/sentinel/src/openalpha_sentinel/sentinel_v1_1.py
    git diff --check
    git add research/sentinel-v1_1 packages/sentinel/src/openalpha_sentinel/sentinel_v1_1.py packages/sentinel/tests/test_sentinel_v1_1.py
    git commit -m "research: preregister Sentinel v1.1 compatibility audit"

Expected: tests and static checks pass before the commit.

### Task 2: Implement tokenizer support and round-trip accounting

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/token_manifold.py
- Test: packages/sentinel/tests/test_token_manifold.py

- [ ] **Step 1: Write failing deterministic tests**

    def test_right_aligned_windows_are_nonoverlapping() -> None:
        windows = select_support_windows(tuple(range(1600)), 512, 3)
        assert windows[0] == tuple(range(64, 576))
        assert windows[2] == tuple(range(1088, 1600))

    def test_pair_support_uses_floor_two_and_jeffreys_smoothing() -> None:
        support = build_token_support(((1, 2), (1, 2), (1, 3)), 1024, 1024)
        pair = support.describe(1, 2)
        assert pair.exact_pair_count == 2
        assert pair.empirically_supported is True
        assert pair.smoothed_probability == pytest.approx(
            2.5 / (3 + 0.5 * 1024 * 1024)
        )

    def test_roundtrip_invalidity_has_wilson_interval() -> None:
        summary = summarize_roundtrip_rows(rows_with_one_invalid(), 0.95)
        assert summary.invalid_candle_count == 1
        assert 0.0 <= summary.wilson_lower <= summary.wilson_upper <= 1.0

- [ ] **Step 2: Run tests and verify import failure**

    uv run pytest packages/sentinel/tests/test_token_manifold.py -q

- [ ] **Step 3: Implement immutable support and round-trip types**

    class PairSupport(FrozenModel):
        coarse_count: int = Field(ge=0)
        fine_count: int = Field(ge=0)
        exact_pair_count: int = Field(ge=0)
        exact_pair_frequency: float = Field(ge=0.0, le=1.0)
        smoothed_probability: float = Field(gt=0.0, le=1.0)
        smoothed_surprisal: float = Field(ge=0.0)
        empirically_supported: bool

    class RoundTripSummary(FrozenModel):
        reconstructed_candle_count: int = Field(ge=0)
        invalid_candle_count: int = Field(ge=0)
        invalid_candle_fraction: float = Field(ge=0.0, le=1.0)
        wilson_lower: float = Field(ge=0.0, le=1.0)
        wilson_upper: float = Field(ge=0.0, le=1.0)
        material_defect: bool
        overwhelmingly_valid: bool
        violation_categories: dict[str, int]

Require at least 1,536 rows and return only the final three non-overlapping slices. Count pairs without retaining market rows. Wilson z is 1.959963984540054. Material-defect and overwhelmingly-valid thresholds are exactly those in the design.

- [ ] **Step 4: Implement causal regimes and reconstruction errors**

Use pooled 1/3 and 2/3 quantiles of causal trailing-20 close-return volatility. The first 20 rows are not_computable with INSUFFICIENT_CAUSAL_RETURNS. Normalize OHLC/range errors by positive observed close. Never substitute zero for unavailable or nonfinite values.

- [ ] **Step 5: Verify and commit**

    uv run pytest packages/sentinel/tests/test_token_manifold.py -q
    uv run ruff check packages/sentinel/src/openalpha_sentinel/token_manifold.py packages/sentinel/tests/test_token_manifold.py
    uv run pyright packages/sentinel/src/openalpha_sentinel/token_manifold.py
    git add packages/sentinel/src/openalpha_sentinel/token_manifold.py packages/sentinel/tests/test_token_manifold.py
    git commit -m "feat: add token-manifold support metrics"

### Task 3: Implement mass bounds, classification, and support selection

**Files:**
- Create: packages/sentinel/src/openalpha_sentinel/constraint_compatibility.py
- Test: packages/sentinel/tests/test_constraint_compatibility.py

- [ ] **Step 1: Write failing mass-accounting tests**

    def test_mass_bounds_keep_truncation_honest() -> None:
        result = probability_mass_bounds(
            considered=(
                CandidateMass(probability=0.30, valid=True, supported=True),
                CandidateMass(probability=0.20, valid=False, supported=True),
            ),
            considered_mass=0.50,
        )
        assert result.valid_lower == pytest.approx(0.30)
        assert result.valid_upper == pytest.approx(0.80)
        assert result.valid_supported_lower == pytest.approx(0.30)
        assert result.uncovered_mass == pytest.approx(0.50)
        assert result.exact is False

    def test_zero_mass_tax_is_finite_json() -> None:
        result = probability_mass_bounds(considered=(), considered_mass=0.0)
        assert result.valid_constraint_tax is None
        assert result.valid_tax_status == "ZERO_ESTIMATED_MASS"
        assert "Infinity" not in result.model_dump_json()

- [ ] **Step 2: Write ordered-classification tests**

Create table-driven CompatibilityFacts cases proving E precedes A, A precedes C, C precedes B, B precedes D, and insufficient or broad-bound evidence becomes F. Assert exact enum values and the matched-rule trace.

- [ ] **Step 3: Write deterministic selection tests**

    def test_support_sampler_rejects_invalid_and_rare_pairs() -> None:
        selected = choose_supported_candidate(
            sample_candidates(), auxiliary_fraction=0.40, minimum_pair_count=2
        )
        assert selected.selected.valid is True
        assert selected.selected.exact_pair_count >= 2
        assert selected.renormalized_probability > 0.0

    def test_empty_supported_set_is_hard_failure() -> None:
        with pytest.raises(NoSupportedCandidateError):
            choose_supported_candidate(
                invalid_or_rare_candidates(),
                auxiliary_fraction=0.25,
                minimum_pair_count=2,
            )

- [ ] **Step 4: Run tests and verify import failure**

    uv run pytest packages/sentinel/tests/test_constraint_compatibility.py -q

- [ ] **Step 5: Implement the pure compatibility module**

Implement CandidateMass, ProbabilityMassBounds, CompatibilityFacts, RootCause, SupportedCandidate, SupportedSelection, ContinuationMeasurements, and ContinuationDecision. Enforce probabilities and mass in [0,1], lower <= upper, probability-ranked deterministic order, exact support floor two, and cumulative sampling. Constraint tax is negative log of the lower bound only when positive.

classify_root_cause implements the exact ordered design rules with no outcomes. evaluate_continuation returns all ten frozen gates plus measurements; it never emits only an aggregate boolean.

- [ ] **Step 6: Verify and commit**

    uv run pytest packages/sentinel/tests/test_constraint_compatibility.py -q
    uv run ruff check packages/sentinel/src/openalpha_sentinel/constraint_compatibility.py packages/sentinel/tests/test_constraint_compatibility.py
    uv run pyright packages/sentinel/src/openalpha_sentinel/constraint_compatibility.py
    git add packages/sentinel/src/openalpha_sentinel/constraint_compatibility.py packages/sentinel/tests/test_constraint_compatibility.py
    git commit -m "feat: add constraint compatibility decisions"

### Task 4: Implement the pinned v1.1 inference worker

**Files:**
- Create: scripts/kronos_token_manifold_worker.py
- Test: packages/sentinel/tests/test_v1_1_worker.py

- [ ] **Step 1: Write failing request-boundary tests**

    def test_worker_rejects_unlocked_budget() -> None:
        request = valid_roundtrip_request()
        request["candidate_budgets"] = [2048]
        with pytest.raises(ValueError, match="candidate budgets"):
            worker._validate_request(request)

    def test_worker_requires_model_tokenizer_pairing() -> None:
        request = valid_canary_request("NeoQuasar/Kronos-small")
        request["tokenizer_repository"] = "NeoQuasar/Kronos-Tokenizer-2k"
        with pytest.raises(ValueError, match="pairing"):
            worker._validate_request(request)

    def test_worker_rejects_arbitrary_fields() -> None:
        with pytest.raises(ValueError, match="unknown fields"):
            worker._validate_request({
                **valid_raw_request(), "cutoff_override": "2025-07-01"
            })

- [ ] **Step 2: Run tests and verify worker import failure**

    uv run pytest packages/sentinel/tests/test_v1_1_worker.py -q

- [ ] **Step 3: Implement resource and preprocessing boundaries**

The JSON-stdin/JSON-stdout worker validates exact source/model/tokenizer revisions, pairing, operation, shape, and allowed fields; rejects unsafe snapshot files; and records file hashes, dependencies, cache path/size, OS, Python, PyTorch, device, duration, and peak working set.

    feature_columns = ("open", "high", "low", "close", "volume", "amount")
    amount = volume * mean(open, high, low, close)
    normalized = clip((float32_values - mean) / (std + 1e-5), -5.0, 5.0)
    tokens = tokenizer.encode(normalized, half=True)
    decoded = tokenizer.decode(tokens, half=True)
    reconstructed = decoded * (std + 1e-5) + mean

Never serialize provider observations into public result artifacts.

- [ ] **Step 4: Implement round-trip and raw-token operations**

ROUNDTRIP accepts only the declared 30 windows and returns token IDs plus safe reconstruction summaries. RAW_COMPATIBILITY accepts one fixed origin/seed and records every raw pair, unfiltered temperature-one log-probabilities/ranks, actual top-p probabilities, decoded validity, empirical support, and nested 8x8/16x16/32x32 grids.

Order candidates by descending joint top-p probability with pair ID as tie-break. considered_mass is the sum of evaluated joint probabilities; uncovered mass stays explicit.

- [ ] **Step 5: Implement canary and conditional operations**

CANARY accepts only the two locked canary cutoffs and maximum 16x16 grid. It enforces mini/2k and small-or-base/base-tokenizer pairings. SUPPORT_CONDITIONED accepts only classification B/C/D, budget 32x32, support floor two, and the frozen auxiliary fraction. It appends only an eligible original token pair; an empty set hard-fails with no fallback. DRAFT_CONDITIONED additionally requires the preregistered eligibility record and fixed exp(-0.5*d^2) adjustment.

- [ ] **Step 6: Add synthetic deterministic tests**

Use tiny fake model/tokenizer objects to prove encode/decode shapes, mass accounting, ranks, raw selection preservation, identical replay hashes, rejection divergence, no unsupported selection, no silent fallback, and no model-state mutation.

- [ ] **Step 7: Verify and commit**

    uv run pytest packages/sentinel/tests/test_v1_1_worker.py -q
    uv run ruff check scripts/kronos_token_manifold_worker.py packages/sentinel/tests/test_v1_1_worker.py
    uv run pyright scripts/kronos_token_manifold_worker.py
    git add scripts/kronos_token_manifold_worker.py packages/sentinel/tests/test_v1_1_worker.py
    git commit -m "feat: add pinned token-manifold worker"

### Task 5: Implement the locked resumable runner

**Files:**
- Create: scripts/run_sentinel_v1_1.py
- Test: packages/sentinel/tests/test_v1_1_cli.py

- [ ] **Step 1: Write failing CLI and seal tests**

    def test_cli_exposes_only_locked_operations() -> None:
        parser = runner.build_parser()
        for operation in (
            "roundtrip", "audit", "canary", "classify",
            "conditional", "analyze", "report", "verify",
        ):
            assert parser.parse_args([operation]).operation == operation
        with pytest.raises(SystemExit):
            parser.parse_args(["audit", "--cutoff", "2025-07-01"])

    def test_outcomes_cannot_load_before_forecast_seal(tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="forecast seal"):
            runner.resolve_method_outcomes(tmp_path / "origin")

- [ ] **Step 2: Run tests and verify runner import failure**

    uv run pytest packages/sentinel/tests/test_v1_1_cli.py -q

- [ ] **Step 3: Implement fixed operations and checkpoints**

roundtrip fetches the fixed raw support corpus, validates XNYS sessions, keeps normalized rows outside Git, invokes the worker, and seals compact results. audit reuses verified v1 contexts or fetches causal context, invokes 36 raw audits, and checkpoints each origin/seed. canary attempts small then base under locked stop conditions. classify reads verified compact artifacts and writes the mechanical classification.

conditional refuses unless classification is B/C/D. analyze reads outcomes only after every forecast seal, reuses v1 metrics, evaluates applicable methods, and applies frozen gates. report and verify are offline and never build a provider or inference client.

- [ ] **Step 4: Implement immutable verification**

Every stage record has schema version, experiment hash, input/output hashes, terminal status, holdout_accessed false, and failure details. Immutable writes reject different existing bytes. verify checks experiment hash, every content hash, fixed inventory, no post-2025-06-30 origin, no raw/cache/model files, and unchanged v0/v1 tree IDs.

- [ ] **Step 5: Verify and commit**

    uv run pytest packages/sentinel/tests/test_v1_1_cli.py packages/sentinel/tests/test_sentinel_v1_1.py -q
    uv run ruff check scripts/run_sentinel_v1_1.py packages/sentinel/tests/test_v1_1_cli.py
    uv run pyright scripts/run_sentinel_v1_1.py
    git add scripts/run_sentinel_v1_1.py packages/sentinel/tests/test_v1_1_cli.py
    git commit -m "feat: add resumable Sentinel v1.1 runner"

### Task 6: Execute and classify the compatibility audit

**Files:**
- Create: research/sentinel-v1_1/tokenizer_roundtrip/results.json
- Create: research/sentinel-v1_1/tokenizer_roundtrip/manifest.json
- Create: research/sentinel-v1_1/token_support/generated_steps.jsonl
- Create: research/sentinel-v1_1/token_support/summary.json
- Create: research/sentinel-v1_1/probability_mass/step_bounds.jsonl
- Create: research/sentinel-v1_1/probability_mass/summary.json
- Create: research/sentinel-v1_1/model_size_canary/results.json
- Create: research/sentinel-v1_1/root_cause.json

- [ ] **Step 1: Verify inference resources**

Check the pinned Python 3.11 environment and confirm caches resolve outside Git. Download only missing pinned snapshots. Stop a canary model at the locked 4 GiB cache, 8 GiB memory, ten-minute first-path, or four-hour projected-run boundary.

- [ ] **Step 2: Run tokenizer round trips**

    uv run python scripts/run_sentinel_v1_1.py roundtrip
    uv run python scripts/run_sentinel_v1_1.py verify

Expected: 30 terminal window records per tokenizer, or explicit pinned-resource failures; raw frames remain outside Git.

- [ ] **Step 3: Run generated support and probability audit**

    uv run python scripts/run_sentinel_v1_1.py audit
    uv run python scripts/run_sentinel_v1_1.py verify

Expected: 36 terminal paths and 180 step records, or explicit failures for every locked path. Regenerated raw hashes match v1.

- [ ] **Step 4: Run the bounded model-size canary**

    uv run python scripts/run_sentinel_v1_1.py canary
    uv run python scripts/run_sentinel_v1_1.py verify

Expected: mini and each feasible larger model have terminal results; skips carry stop-condition evidence.

- [ ] **Step 5: Classify mechanically**

    uv run python scripts/run_sentinel_v1_1.py classify

Expected: root_cause.json contains one A-F classification, rule trace, bound ambiguity, model availability, and conditional_method_authorized.

### Task 7: Conditionally evaluate support-aware decoding

**Files:**
- Create when authorized: research/sentinel-v1_1/method_comparison/forecast_records.jsonl
- Create when authorized: research/sentinel-v1_1/method_comparison/outcome_records.jsonl
- Create when authorized: research/sentinel-v1_1/method_comparison/results.json
- Create always: research/sentinel-v1_1/method_comparison/gate_status.json

- [ ] **Step 1: Apply the method gate**

For root cause A, E, or F, write gate_status.json with NOT_AUTHORIZED and proceed to Task 8. For B, C, or D:

    uv run python scripts/run_sentinel_v1_1.py conditional

Expected: 36 sealed support-conditioned forecasts or explicit hard failures, with no outcome access.

- [ ] **Step 2: Apply the draft-variant gate**

Before outcomes, count eligible candidates. Run the fixed draft variant only if at least two exist at 80% or more of intervention steps; otherwise record NOT_AUTHORIZED and the measured fraction.

- [ ] **Step 3: Seal, resolve, and analyze**

    uv run python scripts/run_sentinel_v1_1.py analyze

Expected: outcomes reference immutable forecast hashes; paired raw, terminal, prior resampling, support-conditioned, and applicable draft results are complete.

- [ ] **Step 4: Apply every continuation gate**

Report structural validity, failures, range/close quality, diversity, raw distance, selected rank, replay, latency, cache, memory, and unchanged weights separately. Continue is true only if every frozen gate passes.

### Task 8: Report, document, verify, and commit

**Files:**
- Create: docs/SENTINEL_TOKEN_MANIFOLD.md
- Create: docs/SENTINEL_CONSTRAINT_COMPATIBILITY.md
- Create: research/sentinel-v1_1/report.md
- Modify: docs/MASTER_PLAN.md
- Modify: docs/SENTINEL_DIRECTION.md
- Modify: docs/SENTINEL_METHODOLOGY.md
- Modify: docs/STATUS.md

- [ ] **Step 1: Generate the report from verified artifacts**

    uv run python scripts/run_sentinel_v1_1.py report

Include the development-only boundary, corpus and pins, round-trip results, token support, probability bounds/truncation, canary/skips, root classification, method gate, all applicable quality/diversity/operations results, decision, limitations, hashes, and untouched holdout.

- [ ] **Step 2: Update product documentation**

Document only the measured incompatibility and justified safeguards. If no method passes, end decoder research and define Sentinel as validator, immutable audit layer, terminal-projection gateway, compatibility profiler, and model-selection safety check. Scope every Kronos statement to tested checkpoints, assets, daily frequency, and horizon.

- [ ] **Step 3: Run targeted and complete tests**

    uv run pytest packages/sentinel/tests/test_sentinel_v1_1.py packages/sentinel/tests/test_token_manifold.py packages/sentinel/tests/test_constraint_compatibility.py packages/sentinel/tests/test_v1_1_worker.py packages/sentinel/tests/test_v1_1_cli.py -q
    uv run pytest -q

Expected: all tests pass.

- [ ] **Step 4: Run static and artifact verification**

    uv run ruff check .
    uv run pyright
    uv run python scripts/run_sentinel_v1_1.py verify
    git diff --check
    git status --short

Expected: zero static findings; experiment, artifact, policy, no-holdout, unchanged-v0/v1, and ledger checks pass; only intended compact v1.1 work is modified.

- [ ] **Step 5: Record evidence and commit**

docs/STATUS.md records every command, exit status, test count, result hash, classification, method-gate result, holdout_accessed false, cache, runtime, failures/skips, and conclusion.

    git add packages/sentinel/src/openalpha_sentinel/sentinel_v1_1.py packages/sentinel/src/openalpha_sentinel/token_manifold.py packages/sentinel/src/openalpha_sentinel/constraint_compatibility.py packages/sentinel/tests/test_sentinel_v1_1.py packages/sentinel/tests/test_token_manifold.py packages/sentinel/tests/test_constraint_compatibility.py packages/sentinel/tests/test_v1_1_worker.py packages/sentinel/tests/test_v1_1_cli.py scripts/kronos_token_manifold_worker.py scripts/run_sentinel_v1_1.py research/sentinel-v1_1 docs
    git commit -m "feat: complete Sentinel v1.1 compatibility study"

- [ ] **Step 6: Stop**

Report commit, classification, tokenizer invalidity, support association, mass bounds, canary, method result or gate, decision, verification, and next justified task. Do not access holdout or start a larger experiment.

## Self-review record

- Spec coverage: approved design sections map to Tasks 1-8, including the no-decoder branch.
- Placeholder scan: no TBD, TODO, generic error-handling instruction, or unstated similar step remains.
- Type consistency: V11Origin, PairSupport, RoundTripSummary, CandidateMass, ProbabilityMassBounds, CompatibilityFacts, SupportedSelection, and ContinuationDecision are consistent.
- Preservation: no task edits v0/v1 artifacts or the v1 constrained worker.
- Evidence boundary: provider/model work precedes no outcome access; conditional outcome analysis follows immutable forecast sealing.
