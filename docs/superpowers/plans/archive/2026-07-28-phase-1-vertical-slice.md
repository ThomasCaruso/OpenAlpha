# Phase 1 Auditable Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver one real-data, real-Kronos, leakage-safe ETF experiment from immutable specification through audited web results and a reproducible report.

**Architecture:** A uv-managed Python workspace holds typed domain packages, FastAPI, and a separate worker; an npm-managed Next.js application consumes stable API DTOs. Content-addressed local artifacts, SQLite/DuckDB, and an explicit local MLflow service form the laptop profile while ports preserve PostgreSQL/object-storage promotion.

**Tech Stack:** Python 3.13, uv, Pydantic 2, pandas, PyArrow, DuckDB, scikit-learn, statsmodels, PyTorch, MLflow 3, FastAPI, SQLAlchemy/Alembic, Next.js, TypeScript, Tailwind, Plotly, pytest, Hypothesis, Ruff, Pyright, Vitest, Playwright.

---

## File map

- `pyproject.toml`: Python workspace, shared test/lint/type configuration, dependency groups.
- `package.json`: npm workspace and web commands.
- `packages/experiment-spec/`: v1 schema, semantic validation, migration, canonical bytes, SHA-256.
- `packages/research-core/`: run lifecycle, artifact identities, manifest publication, orchestration.
- `packages/data-connectors/`: provider contract, no-key daily-equity adapter, validation, Parquet snapshots.
- `packages/model-adapters/`: model protocol, simple/statistical/tree baselines, pinned Kronos integration.
- `packages/evaluation/`: forecast origins, leakage findings, metrics, dependence-aware comparisons.
- `packages/backtesting/`: signal rules, orders, fills, cash/holdings/equity ledger.
- `packages/tracking/`: explicit MLflow 3 adapter.
- `packages/reporting/`: artifact-bound HTML/PDF research report.
- `apps/api/`: transport, metadata persistence, job commands, result queries.
- `apps/worker/`: leases, heartbeats, cancellation, run execution.
- `apps/web/`: experiment builder, run status, forecast/backtest/audit/report views.
- `tests/`: cross-package, real-inference, reproducibility, and end-to-end suites.

### Task 1: Establish the reproducible workspace

**Files:**
- Create: `.editorconfig`, `.gitattributes`, `.gitignore`, `.python-version`
- Create: `pyproject.toml`, `package.json`, `Makefile`
- Create: `scripts/bootstrap.ps1`, `scripts/verify_environment.py`
- Create: `.github/workflows/ci.yml`
- Test: `tests/smoke/test_repository_contract.py`

- [ ] **Step 1: Write the repository-contract test**

```python
from pathlib import Path


def test_required_workspace_files_exist() -> None:
    root = Path(__file__).parents[2]
    for name in ("pyproject.toml", "package.json", "Makefile", ".python-version"):
        assert (root / name).is_file(), name
```

- [ ] **Step 2: Run the test and verify failure**

Run: `uv run pytest tests/smoke/test_repository_contract.py -q`

Expected: failure because the workspace files do not exist.

- [ ] **Step 3: Add pinned workspace configuration**

```toml
[project]
name = "openalpha-workspace"
version = "0.1.0"
requires-python = ">=3.13,<3.14"
dependencies = []

[dependency-groups]
dev = ["hypothesis>=6.0", "pyright>=1.1", "pytest>=8.0", "pytest-asyncio>=0.25", "ruff>=0.12"]

[tool.uv.workspace]
members = ["apps/api", "apps/worker", "packages/*"]

[tool.pytest.ini_options]
testpaths = ["tests", "packages", "apps"]
markers = ["slow: resource-intensive integration", "network: external network required", "kronos: official checkpoint inference"]

[tool.ruff]
line-length = 100
target-version = "py313"
```

Set `.python-version` to `3.13`. Configure `.gitattributes` with `* text=auto eol=lf`; ignore environments, Node output, databases, downloaded checkpoints, and generated artifacts. The environment verifier prints versions and fails with actionable messages when Python, uv, Node, or npm is missing.

- [ ] **Step 4: Run workspace checks**

Run: `uv sync --group dev && uv run pytest tests/smoke/test_repository_contract.py -q && uv run ruff check .`

Expected: the smoke test passes and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add .editorconfig .gitattributes .gitignore .python-version pyproject.toml package.json Makefile scripts .github tests/smoke
git commit -m "build: establish reproducible monorepo workspace"
```

### Task 2: Implement the immutable experiment specification

**Files:**
- Create: `packages/experiment-spec/pyproject.toml`
- Create: `packages/experiment-spec/src/openalpha_experiment_spec/{__init__,models,validation,canonical,migrations}.py`
- Test: `packages/experiment-spec/tests/test_canonical.py`
- Test: `packages/experiment-spec/tests/test_validation.py`
- Create: `examples/experiments/spy-kronos-daily-v1.yaml`

- [ ] **Step 1: Write failing canonicalization and semantic tests**

```python
def test_yaml_and_json_have_the_same_identity(valid_mapping: dict[str, object]) -> None:
    yaml_spec = ExperimentSpec.from_yaml(yaml.safe_dump(valid_mapping))
    json_spec = ExperimentSpec.from_json(json.dumps(valid_mapping))
    assert canonical_bytes(yaml_spec) == canonical_bytes(json_spec)
    assert experiment_id(yaml_spec) == experiment_id(json_spec)


def test_evaluation_must_follow_kronos_pretraining_cutoff(valid_mapping: dict[str, object]) -> None:
    valid_mapping["data"]["start"] = "2024-01-02"
    with pytest.raises(ValueError, match="after 2024-06-30"):
        ExperimentSpec.model_validate(valid_mapping)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest packages/experiment-spec/tests -q`

Expected: import failure because the package is absent.

- [ ] **Step 3: Implement strict v1 models and canonical identity**

```python
class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    metadata: MetadataSpec
    hypothesis: str = Field(min_length=20, max_length=2_000)
    data: DataSpec
    forecast: ForecastSpec
    models: tuple[ModelSpec, ...]
    evaluation: EvaluationSpec
    strategy: StrategySpec
    execution: ExecutionSpec
    benchmark: BenchmarkSpec
    statistics: StatisticsSpec
    reporting: ReportingSpec
    random_seed: int = Field(ge=0, le=2**32 - 1)


def canonical_bytes(spec: ExperimentSpec) -> bytes:
    payload = spec.model_dump(mode="json", exclude_none=False)
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def experiment_id(spec: ExperimentSpec) -> str:
    return f"exp_{hashlib.sha256(canonical_bytes(spec)).hexdigest()}"
```

Cross-field validation requires daily adjusted data, a named calendar, causal windows, positive costs, next-bar execution, the complete baseline set, unique model names, adequate history, and post-cutoff Kronos evaluation. Export JSON Schema and return field-located messages.

- [ ] **Step 4: Run package tests**

Run: `uv run pytest packages/experiment-spec/tests -q && uv run pyright packages/experiment-spec && uv run ruff check packages/experiment-spec`

Expected: all checks pass.

- [ ] **Step 5: Commit**

```bash
git add packages/experiment-spec examples/experiments
git commit -m "feat(spec): add immutable versioned experiment contract"
```

### Task 3: Add content-addressed artifacts and completed-run manifests

**Files:**
- Create: `packages/research-core/pyproject.toml`
- Create: `packages/research-core/src/openalpha_research/{artifacts,manifest,run_state,errors}.py`
- Test: `packages/research-core/tests/test_artifacts.py`
- Test: `packages/research-core/tests/test_manifest.py`

- [ ] **Step 1: Write failing immutability tests**

```python
def test_completed_manifest_rejects_missing_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ManifestIntegrityError, match="missing artifact"):
        publish_manifest(store, completed_manifest_with_unknown_hash())


def test_published_artifact_cannot_be_overwritten(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = store.put_bytes(b"first", media_type="application/octet-stream")
    with pytest.raises(ArtifactConflictError):
        store.publish_at(ref.relative_path, b"second")
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest packages/research-core/tests -q`

Expected: import failure for the missing research package.

- [ ] **Step 3: Implement atomic content-addressed storage**

```python
@dataclass(frozen=True)
class ArtifactRef:
    sha256: str
    size_bytes: int
    media_type: str
    relative_path: str


class LocalArtifactStore:
    def put_bytes(self, value: bytes, media_type: str) -> ArtifactRef:
        digest = hashlib.sha256(value).hexdigest()
        relative = Path("sha256") / digest[:2] / digest
        atomic_publish_under_root(self.root, relative, value)
        return ArtifactRef(digest, len(value), media_type, relative.as_posix())
```

Manifest publication verifies every referenced hash, required artifact kind, terminal run state, methodology status, experiment identity, Git metadata, environment metadata, and dirty-tree disclosure before one atomic write.

- [ ] **Step 4: Run tests and static checks**

Run: `uv run pytest packages/research-core/tests -q && uv run pyright packages/research-core && uv run ruff check packages/research-core`

Expected: all checks pass.

- [ ] **Step 5: Commit**

```bash
git add packages/research-core
git commit -m "feat(core): add content-addressed run manifests"
```

### Task 4: Build the market-data snapshot pipeline

**Files:**
- Create: `packages/data-connectors/pyproject.toml`
- Create: `packages/data-connectors/src/openalpha_data/{contracts,equity_provider,quality,snapshot}.py`
- Test: `packages/data-connectors/tests/{test_contract,test_quality,test_snapshot}.py`
- Test: `tests/integration/test_real_daily_equity_snapshot.py`

- [ ] **Step 1: Write data-contract and quality failures**

```python
@given(valid_ohlcv_frames())
def test_normalization_is_order_invariant(frame: pd.DataFrame) -> None:
    assert_frame_equal(normalize_daily_bars(frame), normalize_daily_bars(frame.sample(frac=1)))


def test_duplicate_session_is_fatal(frame_with_duplicate_session: pd.DataFrame) -> None:
    report = inspect_daily_bars(frame_with_duplicate_session, calendar="XNYS")
    assert report.status == QualityStatus.FAILED
    assert "duplicate_session" in {finding.code for finding in report.findings}
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest packages/data-connectors/tests -q`

Expected: import failure for the missing data package.

- [ ] **Step 3: Implement provider, validation, provenance, and Parquet snapshot**

```python
class MarketDataProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity: ...
    async def fetch_daily_adjusted(self, request: DailyBarsRequest) -> ProviderPayload: ...


class SnapshotManifest(BaseModel):
    snapshot_id: str
    provider: ProviderIdentity
    retrieved_at: datetime
    raw_sha256: str
    parquet_sha256: str
    schema_version: Literal["1.0"]
    timezone: str
    calendar: str
    adjustment_policy: AdjustmentPolicy
    quality_report_sha256: str
```

The no-key adapter uses a fixed allow-listed endpoint, bounded response, timeouts, and explicit terms/adjustment limitations. Parquet writing fixes schema, row-group behavior, timezone, compression, and column order before hashing.

- [ ] **Step 4: Run tests including the marked real-provider contract**

Run: `uv run pytest packages/data-connectors/tests -q && uv run pytest tests/integration/test_real_daily_equity_snapshot.py -m network -q`

Expected: local tests pass; the network test either passes with a real snapshot or skips only when the provider is unreachable, recording the reason.

- [ ] **Step 5: Commit**

```bash
git add packages/data-connectors tests/integration/test_real_daily_equity_snapshot.py
git commit -m "feat(data): add validated content-addressed OHLCV snapshots"
```

### Task 5: Implement causal forecast origins and the leakage auditor

**Files:**
- Create: `packages/evaluation/pyproject.toml`
- Create: `packages/evaluation/src/openalpha_evaluation/{origins,causal_frame,audit}.py`
- Test: `packages/evaluation/tests/{test_origins,test_causal_frame,test_audit}.py`

- [ ] **Step 1: Write property tests for timestamp causality**

```python
@given(origin_configs())
def test_no_model_timestamp_exceeds_cutoff(config: OriginConfig) -> None:
    for origin in build_origins(config):
        assert origin.training.end <= origin.information_cutoff
        assert origin.target.start > origin.information_cutoff
        assert origin.execution_time > origin.signal_time


def test_future_access_is_rejected(causal_frame: CausalFrame) -> None:
    with pytest.raises(FutureDataAccessError):
        causal_frame.loc_after(causal_frame.cutoff)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest packages/evaluation/tests/test_origins.py packages/evaluation/tests/test_causal_frame.py -q`

Expected: import failure for missing evaluator classes.

- [ ] **Step 3: Implement expanding/rolling origins, purge, embargo, and findings**

```python
class MethodologyStatus(StrEnum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


@dataclass(frozen=True)
class ForecastOrigin:
    origin_id: str
    information_cutoff: pd.Timestamp
    training: TimeWindow
    validation: TimeWindow | None
    purge: TimeWindow | None
    embargo: TimeWindow | None
    targets: tuple[pd.Timestamp, ...]
    signal_time: pd.Timestamp
    execution_time: pd.Timestamp
```

Audit future inputs, preprocessing fit ranges, label overlap, target-derived columns, calendar construction, same-bar execution, benchmark contamination, and post-selection reuse.

- [ ] **Step 4: Run all evaluation tests**

Run: `uv run pytest packages/evaluation/tests -q && uv run pyright packages/evaluation`

Expected: all checks pass.

- [ ] **Step 5: Commit**

```bash
git add packages/evaluation
git commit -m "feat(evaluation): enforce leakage-safe forecast origins"
```

### Task 6: Add baseline model adapters

**Files:**
- Create: `packages/model-adapters/pyproject.toml`
- Create: `packages/model-adapters/src/openalpha_models/{contracts,naive,statistical,linear,tree,registry}.py`
- Test: `packages/model-adapters/tests/test_baselines.py`
- Test: `packages/model-adapters/tests/test_contract.py`

- [ ] **Step 1: Write shared model-contract tests**

```python
@pytest.mark.parametrize("adapter", baseline_adapters())
def test_adapter_never_changes_causal_input(adapter: ModelAdapter, causal_batch: ForecastBatch) -> None:
    before = causal_batch.frame.copy(deep=True)
    forecast = adapter.forecast(causal_batch)
    assert_frame_equal(before, causal_batch.frame)
    assert forecast.index.equals(causal_batch.target_index)
    assert np.isfinite(forecast.values).all()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest packages/model-adapters/tests/test_contract.py -q`

Expected: import failure for missing model contracts.

- [ ] **Step 3: Implement the protocol and prespecified baselines**

```python
class ModelAdapter(Protocol):
    @property
    def capabilities(self) -> ModelCapabilities: ...
    def fit(self, training: CausalFrame, seed: int) -> FittedModel: ...
    def forecast(self, fitted: FittedModel, context: CausalFrame,
                 target_index: pd.DatetimeIndex, seed: int) -> ForecastArtifact: ...
```

Implement random walk, last value, moving average, exponential smoothing/ARIMA, ridge regression, and histogram gradient boosting. Time features and lags are trailing; each trained artifact records library versions, seed, feature schema, fit window, and selected training-only parameters.

- [ ] **Step 4: Run contract tests**

Run: `uv run pytest packages/model-adapters/tests -q && uv run pyright packages/model-adapters`

Expected: all adapters satisfy the shared contract.

- [ ] **Step 5: Commit**

```bash
git add packages/model-adapters
git commit -m "feat(models): add causal forecasting baselines"
```

### Task 7: Integrate pinned real Kronos inference

**Files:**
- Create: `packages/model-adapters/src/openalpha_models/kronos/{adapter,checkpoint,input_validation,determinism}.py`
- Create: `packages/model-adapters/src/openalpha_models/kronos/LICENSE.kronos`
- Test: `packages/model-adapters/tests/kronos/{test_input_validation,test_checkpoint,test_determinism}.py`
- Test: `tests/integration/test_real_kronos_inference.py`

- [ ] **Step 1: Write failures for unsafe input and checkpoint mismatch**

```python
def test_kronos_rejects_context_past_cutoff(kronos_adapter: KronosAdapter, batch: ForecastBatch) -> None:
    batch.frame.loc[batch.target_index[0], "close"] = 1.0
    with pytest.raises(FutureDataAccessError):
        kronos_adapter.forecast(batch)


def test_checkpoint_hash_must_match(tmp_path: Path) -> None:
    with pytest.raises(CheckpointIntegrityError):
        verify_checkpoint(tmp_path / "model.safetensors", expected_sha256="0" * 64)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest packages/model-adapters/tests/kronos -q`

Expected: import failure for the absent Kronos adapter.

- [ ] **Step 3: Implement the reviewed upstream boundary**

```python
KRONOS_BASE = CheckpointSpec(
    model_repo="NeoQuasar/Kronos-base",
    model_revision="31afc02ed0a3cfaa1a97238c8b948d663e45b597",
    model_sha256="abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83",
    tokenizer_repo="NeoQuasar/Kronos-Tokenizer-base",
    tokenizer_revision="9ef143b98ee3c2488eebd85404e0c215c112b46a",
    tokenizer_sha256="59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
    max_context=512,
)
```

Load safetensors from pinned snapshots without remote code, call `eval()`, configure PyTorch determinism, validate six-channel inputs/timestamps, generate from the official predictor, preserve path diagnostics, validate output constraints, and report resource/device/runtime metadata. Add the official MIT notice.

- [ ] **Step 4: Execute unit tests and one real checkpoint inference**

Run: `uv run pytest packages/model-adapters/tests/kronos -q && uv run pytest tests/integration/test_real_kronos_inference.py -m kronos -s`

Expected: unit tests pass and the marked test downloads/verifies an official checkpoint, produces finite dated OHLCVA output, and reports real runtime; resource exhaustion is a test failure with diagnostics.

- [ ] **Step 5: Commit**

```bash
git add packages/model-adapters tests/integration/test_real_kronos_inference.py
git commit -m "feat(kronos): integrate pinned official checkpoint inference"
```

### Task 8: Implement forecast and statistical evaluation

**Files:**
- Create: `packages/evaluation/src/openalpha_evaluation/{forecast_metrics,comparisons,bootstrap}.py`
- Test: `packages/evaluation/tests/{test_forecast_metrics,test_comparisons,test_bootstrap}.py`

- [ ] **Step 1: Write metric edge-case tests**

```python
def test_price_error_is_distinct_from_return_error() -> None:
    actual = pd.Series([100.0, 101.0, 102.0])
    forecast = pd.Series([110.0, 111.1, 112.2])
    result = evaluate_forecast(actual, forecast, last_context_close=100.0)
    assert result.price_mae > 9.0
    assert result.horizon_return_mae < 1e-12


def test_undefined_correlation_is_explicit() -> None:
    result = safe_spearman(np.ones(8), np.arange(8))
    assert result.value is None
    assert result.reason == "constant_input"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest packages/evaluation/tests/test_forecast_metrics.py -q`

Expected: import failure for missing metrics.

- [ ] **Step 3: Implement registered metrics and dependence-aware inference**

Implement price/return MAE and RMSE, causal MASE, directional classification with class diagnostics, rank/linear association, per-origin/horizon/regime tables, moving-block bootstrap intervals, Diebold-Mariano loss comparisons with overlap-aware lag, and Holm adjustment.

```python
@dataclass(frozen=True)
class Estimate:
    value: float | None
    lower: float | None
    upper: float | None
    n: int
    method: str
    reason: str | None = None
```

- [ ] **Step 4: Run tests against hand-calculated fixtures**

Run: `uv run pytest packages/evaluation/tests -q`

Expected: all metric, bootstrap-seed, and comparison tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/evaluation
git commit -m "feat(statistics): add forecast comparisons and uncertainty"
```

### Task 9: Build the canonical strategy and execution ledger

**Files:**
- Create: `packages/backtesting/pyproject.toml`
- Create: `packages/backtesting/src/openalpha_backtesting/{signals,orders,execution,ledger,metrics}.py`
- Test: `packages/backtesting/tests/{test_signals,test_execution,test_accounting,test_metrics}.py`

- [ ] **Step 1: Write accounting invariants**

```python
@given(executable_order_streams())
def test_cash_holdings_and_equity_reconcile(stream: OrderStream) -> None:
    ledger = simulate(stream)
    for state in ledger.states:
        assert state.equity == pytest.approx(state.cash + state.quantity * state.mark_price)


def test_positive_costs_cannot_improve_terminal_equity(simple_market: MarketPath) -> None:
    free = simulate(simple_market.orders(cost_bps=0))
    costly = simulate(simple_market.orders(cost_bps=10))
    assert costly.terminal_equity <= free.terminal_equity
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest packages/backtesting/tests -q`

Expected: import failure for missing ledger types.

- [ ] **Step 3: Implement transparent long/cash signals and next-bar fills**

```python
@dataclass(frozen=True)
class FillEvent:
    order_id: str
    signal_time: datetime
    intended_execution_time: datetime
    actual_execution_time: datetime
    side: Side
    quantity: Decimal
    reference_price: Decimal
    fill_price: Decimal
    commission: Decimal
    slippage: Decimal
    reason: str
```

Use Decimal at money/accounting boundaries, deterministic rounding, cash checks, next-bar price convention, spread/slippage and commissions, rejected-order events, buy-and-hold parity, and independently calculated equity/drawdown/risk metrics.

- [ ] **Step 4: Run properties and golden examples**

Run: `uv run pytest packages/backtesting/tests -q`

Expected: all reconciliation, cost monotonicity, constraint, and drawdown fixtures pass.

- [ ] **Step 5: Commit**

```bash
git add packages/backtesting
git commit -m "feat(backtest): add auditable execution and accounting ledger"
```

### Task 10: Orchestrate runs and track them in MLflow

**Files:**
- Create: `packages/tracking/pyproject.toml`
- Create: `packages/tracking/src/openalpha_tracking/{contracts,mlflow3}.py`
- Create: `packages/research-core/src/openalpha_research/runner.py`
- Test: `packages/tracking/tests/test_mlflow_adapter.py`
- Test: `tests/integration/test_vertical_runner_baselines.py`

- [ ] **Step 1: Write the orchestration integration test**

```python
def test_runner_publishes_only_after_audit_and_hash_verification(real_snapshot: SnapshotRef) -> None:
    result = run_experiment(baseline_only_spec(), snapshot=real_snapshot)
    assert result.state == RunState.COMPLETED
    assert result.manifest.methodology_status == "passed"
    assert all(result.artifact_store.verify(ref) for ref in result.manifest.artifacts)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/integration/test_vertical_runner_baselines.py -q`

Expected: failure because the runner and tracking port are absent.

- [ ] **Step 3: Implement staged orchestration and explicit MLflow logging**

```python
class ExperimentTracker(Protocol):
    def start(self, manifest_seed: RunManifestSeed) -> TrackingRun: ...
    def log_artifact(self, run: TrackingRun, ref: ArtifactRef) -> None: ...
    def finish(self, run: TrackingRun, status: RunState) -> None: ...
```

The runner freezes the spec, snapshots data, constructs origins, executes models, evaluates, simulates, audits, writes artifacts, verifies hashes, then publishes. MLflow uses an explicit URI/experiment, manual params/metrics/dataset inputs, and exact run/model identifiers. An MLflow failure leaves diagnostics and does not falsify manifest completion.

- [ ] **Step 4: Run baseline vertical integration with local SQLite MLflow**

Run: `uv run pytest packages/tracking/tests tests/integration/test_vertical_runner_baselines.py -q`

Expected: tests pass and the temporary MLflow backend contains the linked run.

- [ ] **Step 5: Commit**

```bash
git add packages/tracking packages/research-core tests/integration/test_vertical_runner_baselines.py
git commit -m "feat(runs): orchestrate and track audited experiments"
```

### Task 11: Generate artifact-grounded HTML and PDF reports

**Files:**
- Create: `packages/reporting/pyproject.toml`
- Create: `packages/reporting/src/openalpha_reporting/{model,render,charts,pdf}.py`
- Create: `packages/reporting/src/openalpha_reporting/templates/research_report.html.j2`
- Test: `packages/reporting/tests/{test_report_model,test_render_snapshot}.py`

- [ ] **Step 1: Write tests that reject unverified values**

```python
def test_report_rejects_value_not_present_in_artifacts(verified_run: VerifiedRun) -> None:
    with pytest.raises(UnverifiedReportValueError):
        ReportModel.from_run(verified_run, overrides={"sharpe": 99.0})
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest packages/reporting/tests -q`

Expected: import failure for the reporting package.

- [ ] **Step 3: Implement the 21-section report model**

```python
class ReportStatement(BaseModel):
    text: str
    category: Literal["computed_fact", "interpretation", "assumption", "limitation"]
    artifact_sha256: str | None
    field_path: str | None
```

Bind tables/charts to verified Parquet/JSON artifacts, render self-contained HTML, create PDF through a pinned renderer, include lineage footers and methodology banners, and label exploratory/negative findings exactly.

- [ ] **Step 4: Run report snapshots and inspect one real artifact**

Run: `uv run pytest packages/reporting/tests -q && uv run openalpha report --latest-verified`

Expected: stable HTML/PDF outputs with matching manifest hashes for the newest verified integration run.

- [ ] **Step 5: Commit**

```bash
git add packages/reporting
git commit -m "feat(reporting): generate verified research reports"
```

### Task 12: Add durable API and worker processes

**Files:**
- Create: `apps/api/pyproject.toml`, `apps/api/src/openalpha_api/`
- Create: `apps/worker/pyproject.toml`, `apps/worker/src/openalpha_worker/`
- Create: `infrastructure/alembic/`
- Test: `apps/api/tests/test_runs_api.py`
- Test: `apps/worker/tests/test_job_leases.py`
- Test: `tests/integration/test_api_worker_flow.py`

- [ ] **Step 1: Write API/lease lifecycle tests**

```python
def test_submit_returns_before_research_executes(client: TestClient, valid_spec: dict) -> None:
    response = client.post("/v1/runs", json=valid_spec)
    assert response.status_code == 202
    assert response.json()["state"] == "queued"


def test_expired_lease_never_marks_run_complete(job_store: JobStore) -> None:
    lease = job_store.acquire(worker_id="worker-a", ttl_seconds=1)
    clock.advance(seconds=2)
    with pytest.raises(LeaseExpiredError):
        job_store.complete(lease)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest apps/api/tests apps/worker/tests -q`

Expected: import failure for absent apps.

- [ ] **Step 3: Implement API DTOs, SQLite job leases, worker stages, cancellation, and health**

Expose specification validation, run submit/get/list, event stream, artifact metadata/download, report download, and health endpoints. Use stable errors and correlation IDs. The worker heartbeats between bounded stages and checks cancellation.

- [ ] **Step 4: Run API, worker, migration, and integration tests**

Run: `uv run pytest apps/api/tests apps/worker/tests tests/integration/test_api_worker_flow.py -q`

Expected: all lifecycle, idempotency, cancellation, timeout, and diagnostic-artifact tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/api apps/worker infrastructure tests/integration/test_api_worker_flow.py
git commit -m "feat(platform): add durable research API and worker"
```

### Task 13: Build the institutional research web workflow

**Files:**
- Create: `apps/web/` via pinned Next.js package configuration
- Create: `apps/web/app/{page.tsx,experiments/new/page.tsx,runs/[runId]/page.tsx}`
- Create: `apps/web/components/{provenance,methodology,forecast-chart,equity-chart,drawdown-chart,metric-table}.tsx`
- Test: `apps/web/**/*.test.tsx`
- Test: `tests/e2e/vertical-slice.spec.ts`

- [ ] **Step 1: Write component tests for validity and provenance**

```tsx
it("blocks evidence styling when methodology failed", () => {
  render(<MethodologyBanner status="failed" findings={[futureLeak]} />);
  expect(screen.getByRole("alert")).toHaveTextContent("Failed");
  expect(screen.getByText(/future timestamp/i)).toBeVisible();
});

it("shows chart lineage", () => {
  render(<ProvenanceBadge experimentId="exp_abcd" artifactSha256="a".repeat(64) />);
  expect(screen.getByText(/exp_abcd/)).toBeVisible();
});
```

- [ ] **Step 2: Run and verify failure**

Run: `npm test --workspace apps/web`

Expected: failure because the web workspace is absent.

- [ ] **Step 3: Implement accessible experiment and audit views**

Use server-rendered initial data, typed API clients, dense responsive tables, Plotly with accessible summaries, skeleton/failure/empty states, keyboard navigation, visible provenance, and no fake market tape. The run page shows forecast versus actual, error tables, strategy rule, fills, equity, drawdown, risk, costs, warnings, report download, and reproduction command.

- [ ] **Step 4: Run component, accessibility, build, and e2e checks**

Run: `npm test --workspace apps/web && npm run typecheck --workspace apps/web && npm run build --workspace apps/web && npm run e2e`

Expected: all checks pass; e2e submits the small real-data spec and reaches a terminal audited result.

- [ ] **Step 5: Commit**

```bash
git add apps/web tests/e2e package.json package-lock.json
git commit -m "feat(web): add auditable research workspace"
```

### Task 14: Add CLI reproduction, one-command startup, and the real demo

**Files:**
- Create: `packages/python-sdk/pyproject.toml`
- Create: `packages/python-sdk/src/openalpha_sdk/{client,cli,reproduce}.py`
- Create: `scripts/dev.ps1`, `scripts/demo.ps1`, `scripts/verify_run.ps1`
- Create: `docker-compose.yml`, `infrastructure/docker/`
- Test: `packages/python-sdk/tests/test_reproduce.py`
- Test: `tests/reproducibility/test_demo_run.py`

- [ ] **Step 1: Write reproduction failures**

```python
def test_reproduce_refuses_dirty_code_without_override(verified_manifest: RunManifest) -> None:
    with pytest.raises(ReproductionMismatch, match="dirty working tree"):
        reproduce(verified_manifest, environment=dirty_environment())


def test_reproduce_verifies_every_artifact(reproducer: Reproducer) -> None:
    result = reproducer.verify_only()
    assert result.mismatches == ()
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest packages/python-sdk/tests tests/reproducibility -q`

Expected: import failure for absent reproduction code.

- [ ] **Step 3: Implement commands and startup profiles**

Provide `openalpha validate`, `run`, `status`, `report`, and `reproduce`; `make setup/dev/test/demo/benchmark/report/lint/typecheck/security/reproduce`; and PowerShell equivalents. The demo runs a bounded post-cutoff real-data experiment with real Kronos and records runtime/hardware. Compose supplies PostgreSQL and MLflow but laptop scripts use explicit local SQLite services.

- [ ] **Step 4: Execute the complete real flow twice**

Run: `make demo` followed by `uv run openalpha reproduce --latest-verified`.

Expected: both commands complete; deterministic artifacts match where declared, stochastic/Kronos artifacts match the recorded seeded policy, and all deviations are explicit.

- [ ] **Step 5: Commit**

```bash
git add packages/python-sdk scripts Makefile docker-compose.yml infrastructure tests/reproducibility
git commit -m "feat(cli): add one-command demo and run reproduction"
```

### Task 15: Verify, document, and freeze Phase 1 evidence

**Files:**
- Create: `README.md`, `QUICKSTART.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`
- Create: `docs/{LEAKAGE_PREVENTION,STATISTICAL_TESTING,BACKTESTING_ASSUMPTIONS,MODEL_GOVERNANCE,REPRODUCIBILITY,DEPLOYMENT}.md`
- Modify: `docs/STATUS.md`
- Create: `research/reports/sample-run/manifest.json`

- [ ] **Step 1: Add documentation link and command tests**

```python
def test_documented_local_commands_exist() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    for target in ("setup", "dev", "test", "demo", "report", "lint", "typecheck", "security", "reproduce"):
        assert re.search(rf"^{target}:", makefile, re.MULTILINE)
```

- [ ] **Step 2: Run the full quality gate**

Run: `make test && make lint && make typecheck && make security && npm run build --workspace apps/web`.

Expected: every command exits zero with no skipped critical invariant tests.

- [ ] **Step 3: Execute and inspect the real user flow**

Run: `make demo` and open the emitted HTML report plus the run page. Verify the manifest, data-quality report, methodology status, forecasts, baselines, costs, ledger, benchmark, equity, drawdown, risk, and reproduction command all resolve to artifacts.

- [ ] **Step 4: Update status with exact evidence**

Record command timestamps, test counts, build outputs, run ID, checkpoint hashes, dataset hash, report hash, runtime, hardware, warnings, limitations, and the exact Phase 2 next task. Include only the real pipeline's sample artifacts when provider terms permit redistribution.

- [ ] **Step 5: Commit the verified phase**

```bash
git add README.md QUICKSTART.md CONTRIBUTING.md SECURITY.md CODE_OF_CONDUCT.md CHANGELOG.md docs research/reports/sample-run
git commit -m "docs: publish verified Phase 1 research evidence"
```

## Plan self-review

- Spec coverage: the tasks cover the complete Phase 1 dependency chain, real Kronos, all required baselines, causal evaluation, costs, benchmark, risk, tracking, report, web, startup, tests, and reproduction.
- Scope boundary: advanced multi-asset optimization, full statistical governance, copilot, paper scheduling, and production deployment remain in later phases.
- Type consistency: experiment IDs, artifact references, forecast origins, methodology states, fill events, and run manifests retain one definition and direction of dependency.
- Evidence rule: no task permits mocked production results or completion before artifact verification.
