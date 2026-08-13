# Retire Bridge Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale Bridge product architecture with a reusable `openalpha_research` core and a subject-specific `openalpha_kronos` package while preserving every scientific record and adding the approved README diagram.

**Architecture:** Extract only demonstrated model-independent primitives into the existing research-core package. Move completed and dormant Kronos studies into a new `kronos-research` package, retain historical identifiers inside study contracts, expose only completed-study research functions through a thin Modal shell, and then delete the abandoned Bridge package and control/deployment surfaces. All verification is offline and produces no scientific evidence.

**Tech Stack:** Python 3.13, Pydantic 2, NumPy, pytest, Ruff, Pyright, uv workspace, GitHub Actions, Modal static packaging, Markdown/PNG documentation.

---

## Target files

Create subject-neutral primitives under `packages/research-core/src/openalpha_research/`: `failures.py`, `identity.py`, `protocols.py`, `calendars.py`, `providers.py`, `runtime.py`, `objectstore.py`, `redaction.py`, `resampling.py`, and `safe_logging.py`.

Create `packages/kronos-research/src/openalpha_kronos/` with:

```text
model/                         assets, contracts, input, normalization, official runtime, source
evaluation/                    shared metrics and OHLC validity
studies/structural_validity/   common, mini, base
studies/zero_shot/
studies/frozen_representation/ preserved inactive source only
```

Create `cloud/modal/kronos_research.py`, `.github/workflows/research-integrity.yml`, `tests/smoke/test_research_architecture.py`, `tests/smoke/test_research_record_integrity.py`, and `docs/assets/openalpha-research-flow.png`.

Preserve `research/**`, `vendor/kronos/**`, `NOTICE`, and every historical run ID, artifact key, schema, experiment ID, and sealed filename. Delete `packages/bridge/`, the old Modal application, Bridge workflows, Docker/client surfaces, and obsolete operational docs after every survivor has moved.

## Task 1: Preserve the dormant probe exactly

**Files:**

- Modify: `cloud/modal/bridge_phase2_app.py`
- Modify: `packages/bridge/src/openalpha_bridge/zero_shot/aggregation.py`
- Create: `packages/bridge/src/openalpha_bridge/resampling.py`
- Create: `packages/bridge/src/openalpha_bridge/representation_probe/*.py`
- Create: `packages/bridge/tests/test_frozen_representation_probe.py`
- Create: `packages/bridge/tests/test_moving_block_resampling.py`
- Exclude: `.superpowers/**`, `representation-probe-review.patch`, and `research/**`

- [ ] **Step 1: Confirm the preservation set**

Run `git status --short`, `git diff --name-only`, and list the probe directory. Expected: implementation/tests/shared resampling/dormant Modal additions only; no scientific output.

- [ ] **Step 2: Run the existing offline probe regression suite**

```powershell
uv run pytest -q packages/bridge/tests/test_frozen_representation_probe.py packages/bridge/tests/test_moving_block_resampling.py packages/bridge/tests/test_kronos_zero_shot_benchmark.py
```

Expected: PASS; no `network` or `kronos` test is selected.

- [ ] **Step 3: Stage only the preservation set**

```powershell
git add -- cloud/modal/bridge_phase2_app.py packages/bridge/src/openalpha_bridge/zero_shot/aggregation.py packages/bridge/src/openalpha_bridge/resampling.py packages/bridge/src/openalpha_bridge/representation_probe packages/bridge/tests/test_frozen_representation_probe.py packages/bridge/tests/test_moving_block_resampling.py
git diff --cached --name-only
```

Expected: the exclusions above are absent.

- [ ] **Step 4: Commit the inactive snapshot**

```powershell
git commit -m "research: preserve inactive frozen-representation probe" -m "Governed by the sealed development-only preregistration, but not authorized or executed. Produces no scientific conclusion. Preserved solely across the namespace migration."
```

## Task 2: Extract a subject-neutral research core

**Files:**

- Create: `packages/research-core/tests/test_research_primitives.py`
- Create: the ten research-core modules listed under Target files.
- Modify: `packages/research-core/src/openalpha_research/__init__.py`
- Modify: `packages/research-core/pyproject.toml`

- [ ] **Step 1: Write failing public-contract tests**

Use these APIs and assertions:

```python
from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from openalpha_research.identity import canonical_json, canonical_sha256, file_sha256
from openalpha_research.objectstore import InMemoryObjectStore, put_json
from openalpha_research.protocols import verify_sha256_files
from openalpha_research.providers import Candle, MarketSeries, ProviderMode, RetrievalRequest, validate_series
from openalpha_research.resampling import moving_block_percentile_interval
from openalpha_research.runtime import gpu_snapshot


def test_core_names_and_schemas_are_subject_neutral() -> None:
    failure = ResearchFailure(
        category=FailureCategory.INVALID_CONFIGURATION,
        code="LOCK_MISMATCH",
        message="sealed input changed",
    )
    assert failure.schema_version == "openalpha.research.failure.v1"
    assert "bridge" not in failure.model_dump_json().lower()
    assert "kronos" not in failure.model_dump_json().lower()


def test_protocol_verification_uses_the_callers_mapping(tmp_path) -> None:
    target = tmp_path / "study.yaml"
    target.write_bytes(b"sealed\n")
    digest = file_sha256(target)
    assert verify_sha256_files(tmp_path, {"study.yaml": digest}) == {"study.yaml": digest}


def test_object_store_accepts_a_caller_owned_evidence_class() -> None:
    metadata = put_json(
        InMemoryObjectStore(),
        "study/runs/run_1/result.json",
        {"result": 1},
        schema_version="study.result.v1",
        run_id="run_1",
        experiment_hash="0" * 64,
        evidence_class="development",
    )
    assert metadata.evidence_class == "development"


def test_runtime_probe_is_safe_offline() -> None:
    assert gpu_snapshot().cuda_available in {True, False}
```

Port the current provider-validation and moving-block arithmetic cases without study constants.

- [ ] **Step 2: Verify RED**

```powershell
uv run pytest -q packages/research-core/tests/test_research_primitives.py
```

Expected: import failure because the modules do not exist.

- [ ] **Step 3: Implement only generic behavior**

Use current implementations as behavior references. Mandatory interfaces:

```python
class ResearchFailure(BaseModel):
    schema_version: Literal["openalpha.research.failure.v1"] = "openalpha.research.failure.v1"
    category: FailureCategory
    code: str
    sequence_index: int | None = None
    candle_index: int | None = None
    field: str | None = None
    observed_value: float | str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    message: str


class ResearchFailureError(ValueError):
    def __init__(self, *failures: ResearchFailure) -> None: ...


def verify_sha256_files(root: Path, expected: Mapping[str, str]) -> dict[str, str]: ...
```

`ObjectMetadata.evidence_class` is a string supplied by the study. Core contains no evidence enum, prefixes, run patterns, experiment hashes, assets, or decisions. Keep `torch`, `yfinance`, and `boto3` lazy; add `yahoo` and `s3` optional extras.

- [ ] **Step 4: Verify core and Sentinel consumers**

```powershell
uv run pytest -q packages/research-core/tests packages/sentinel/tests
uv run ruff check packages/research-core
uv run pyright packages/research-core
```

- [ ] **Step 5: Commit**

```powershell
git add packages/research-core
git commit -m "refactor: complete the reusable research core"
```

## Task 3: Move completed studies into `kronos-research`

**Files:**

- Create: `packages/kronos-research/pyproject.toml`, `README.md`, and package initializers.
- Move: diagnostic/model/evaluation, base-study, zero-shot, and their surviving tests.
- Modify: root `pyproject.toml`, `uv.lock`.

Surviving tests are `diagnostic_fakes.py`, base inventory, diagnostic worker, frozen inference, base study, zero shot, normalization parity, official backend/source/window decode, publication observability, runtime isolation/probe, v3 decisions, and v4 artifact identity.

- [ ] **Step 1: Write a failing lightweight-import test**

```python
import sys


def test_import_is_lightweight_and_names_the_subject() -> None:
    import openalpha_kronos

    assert openalpha_kronos.__name__ == "openalpha_kronos"
    assert "torch" not in sys.modules
    assert "huggingface_hub" not in sys.modules
    assert "yfinance" not in sys.modules
```

Run it and expect an import failure.

- [ ] **Step 2: Create package metadata**

```toml
[project]
name = "openalpha-kronos-research"
version = "0.1.0"
description = "Reproducible empirical studies of the Kronos financial foundation model"
requires-python = ">=3.13,<3.14"
dependencies = ["numpy>=2.5,<3", "openalpha-research-core", "pydantic>=2.11,<3"]

[project.optional-dependencies]
yahoo = ["openalpha-research-core[yahoo]"]
runtime = ["huggingface-hub>=0.34,<1", "safetensors>=0.4,<1", "einops>=0.8,<1", "tqdm>=4.66,<5", "pandas>=2.2,<4"]
gpu = ["torch>=2.5,<3"]
modal = ["modal>=0.64,<2", "openalpha-research-core[s3]"]

[tool.uv.sources]
openalpha-research-core = { workspace = true }

[tool.hatch.build.targets.wheel]
packages = ["src/openalpha_kronos"]
```

No project script and no probe extra.

- [ ] **Step 3: Move model/evaluation code before editing imports**

```text
diagnostic/backends.py            -> model/contracts.py
diagnostic/normalization.py       -> model/normalization.py
diagnostic/official_input.py      -> model/input.py
diagnostic/official_backend.py    -> model/official.py
diagnostic/source_conformance.py  -> model/source.py
diagnostic/metrics.py             -> evaluation/metrics.py
diagnostic/validity.py            -> evaluation/validity.py
diagnostic/conclusion.py          -> studies/structural_validity/common/conclusion.py
diagnostic/methods.py             -> studies/structural_validity/common/methods.py
```

Build `model/assets.py` from only `PinnedAssetSpec`, `PinnedSourceSpec`, official source identity, and mini/base asset specifications. Do not move Bridge dimensions, feature ordering, training backends, or deterministic Bridge fakes.

- [ ] **Step 4: Move studies as owned units**

Move remaining diagnostic files to `mini/`, `base_study/*` to `base/`, and `zero_shot/*` to `zero_shot/`. Preserve:

```python
MINI_RUN_ID_PATTERN = re.compile(r"^canary_[0-9a-f]{8,32}$")
MINI_ARTIFACT_ROOT = "openalpha-compatibility/bridge-phase2"
MINI_EXPERIMENT_SHA256 = "d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2"


class EvidenceClass(StrEnum):
    DEVELOPMENT_COMPATIBILITY_CANARY = "development_compatibility_canary"
```

These values are historical provenance inside Kronos ownership. Update imports to research-core and rename live `BridgeTransformError` references to `ResearchFailureError`; do not change persisted completed-study schema strings.

- [ ] **Step 5: Move tests and keep evidence assertions unchanged**

Update live import paths and path-introspection assertions. Do not edit assertions over historical IDs, keys, schemas, digests, or numeric results.

- [ ] **Step 6: Update workspace and verify**

Temporarily keep both workspace distributions until Task 6 deletes Bridge.

```powershell
uv lock
uv sync --locked --group dev --group sentinel-phase3
uv run pytest -q packages/kronos-research/tests
uv run ruff check packages/kronos-research packages/research-core
uv run pyright packages/kronos-research packages/research-core
```

- [ ] **Step 7: Commit**

```powershell
git add packages/kronos-research packages/bridge packages/research-core pyproject.toml uv.lock
git commit -m "refactor: give Kronos studies explicit ownership"
```

## Task 4: Move the dormant probe and remove every execution surface

**Files:** Move probe source/tests to `openalpha_kronos.studies.frozen_representation`; remove the snapshot's probe block from `cloud/modal/bridge_phase2_app.py`; create `tests/smoke/test_research_architecture.py`.

- [ ] **Step 1: Write the failing dormant-surface guard**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_representation_has_no_execution_surface() -> None:
    forbidden = ("representation_probe", "frozen_representation")
    surfaces = (ROOT / "cloud", ROOT / ".github" / "workflows", ROOT / "scripts")
    offenders = []
    for surface in surfaces:
        for path in surface.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                if any(word in text for word in forbidden):
                    offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
```

Run the single test. Expected: FAIL on probe functions in `bridge_phase2_app.py`.

- [ ] **Step 2: Move mechanically** — update imports and generic failure names only; keep thresholds, grids, windows, seeds, schemas, and authorization fields unchanged.

- [ ] **Step 3: Delete every probe Modal function** — remove runtime construction, deployment/runtime verification, fit, test, and inventory. Add no replacement.

- [ ] **Step 4: Verify**

```powershell
uv run pytest -q packages/kronos-research/tests/test_frozen_representation_probe.py packages/kronos-research/tests/test_moving_block_resampling.py tests/smoke/test_research_architecture.py::test_frozen_representation_has_no_execution_surface
rg -n "representation_probe|frozen_representation" cloud .github/workflows scripts
```

Expected: tests PASS and `rg` has no match.

- [ ] **Step 5: Commit**

```powershell
git add packages/kronos-research packages/bridge cloud/modal/bridge_phase2_app.py tests/smoke/test_research_architecture.py
git commit -m "refactor: preserve the dormant probe without an execution surface"
```

## Task 5: Replace the Modal monolith with a completed-study shell

**Files:** Create `cloud/modal/kronos_research.py`; update Modal/source/deployed-commit smoke tests; preserve `cloud/modal/deployed_commit.py` unless neutrally renamed.

- [ ] **Step 1: Write a failing static shell contract**

```python
APP_PATH = ROOT / "cloud" / "modal" / "kronos_research.py"
ALLOWED_PUBLIC_MODAL_FUNCTIONS = {
    "verify_mini_runtime",
    "run_mini_structural_validity",
    "verify_base_runtime",
    "run_base_structural_validity",
    "inventory_base_artifacts",
    "verify_zero_shot_runtime",
    "run_zero_shot_benchmark",
    "inventory_zero_shot_artifacts",
}
```

Use AST to collect `app.function` functions and require exact equality. Reject every `app.cls`, `app.local_entrypoint`, and ASGI decorator as well as `control_api`, `training`, `test_opening`, `representation_probe`, `frozen_representation`, and `official_protocol`. Expected initial failure: shell absent.

- [ ] **Step 2: Extract only completed-study functions**

```python
APP_NAME = "openalpha-kronos-research"
app = modal.App(APP_NAME)
```

Retain exact pins, package mounting, commit binding, lazy imports, immutable publication, `retries=0`, and isolation. Rename wrappers without changing workers. Copy no API/control code, compute backend, training/synthetic worker, Stage A canary, test gate, generic deployment check, or future-study code.

- [ ] **Step 3: Update offline shell tests** — keep pin, lockfile, flattened-import, mount, worker-thinness, and logging checks; delete product-only assertions.

- [ ] **Step 4: Verify**

```powershell
uv run pytest -q tests/smoke/test_modal_app_packaging.py tests/smoke/test_kronos_source_conformance.py tests/smoke/test_deployed_commit_binding.py packages/kronos-research/tests/test_runtime_isolation_and_logging.py
```

- [ ] **Step 5: Commit**

```powershell
git add cloud/modal/kronos_research.py cloud/modal/deployed_commit.py tests/smoke packages/kronos-research/tests
git commit -m "refactor: narrow Modal to completed Kronos studies"
```

## Task 6: Delete the abandoned Bridge architecture

**Files:** Complete architecture guards; delete `packages/bridge/`, old Modal app, Bridge workflows/Docker/client/product tests; create research CI; update workspace metadata and lock.

- [ ] **Step 1: Add failing final architecture guards**

```python
def test_abandoned_bridge_package_is_absent() -> None:
    assert not (ROOT / "packages" / "bridge").exists()


def test_active_python_has_no_openalpha_bridge_imports() -> None:
    roots = (ROOT / "packages", ROOT / "cloud", ROOT / "scripts", ROOT / "tests")
    offenders = [
        path.relative_to(ROOT).as_posix()
        for root in roots
        for path in root.rglob("*.py")
        if "openalpha_bridge" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_research_core_knows_no_research_subject() -> None:
    source = ROOT / "packages" / "research-core" / "src" / "openalpha_research"
    forbidden = ("openalpha_kronos", "openalpha_sentinel", "kronos", "sentinel", "bridge")
    offenders = {
        path.relative_to(ROOT).as_posix(): token
        for path in source.rglob("*.py")
        for token in forbidden
        if token in path.read_text(encoding="utf-8").lower()
    }
    assert offenders == {}


def test_research_core_distribution_has_no_study_dependency() -> None:
    metadata = (ROOT / "packages" / "research-core" / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "openalpha-kronos" not in metadata
    assert "openalpha-sentinel" not in metadata
    assert "openalpha-bridge" not in metadata


def test_no_official_protocol_implementation_exists() -> None:
    source = ROOT / "packages" / "kronos-research" / "src"
    assert not any("official_protocol" in path.as_posix() for path in source.rglob("*"))


def test_active_ci_has_no_bridge_phase2_identity() -> None:
    workflows = ROOT / ".github" / "workflows"
    paths = [*workflows.glob("*.yml"), *workflows.glob("*.yaml")]
    offenders = [
        path.name
        for path in paths
        if "bridge" in path.name.lower()
        or "bridge phase 2" in path.read_text(encoding="utf-8").lower()
    ]
    assert offenders == []


def test_workspace_metadata_has_no_live_bridge_distribution_or_cli() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "openalpha-bridge" not in metadata
    assert "bridge-phase2" not in metadata
```

Run and confirm failures against current Bridge paths.

- [ ] **Step 2: Confirm every survivor moved, then delete Bridge**

Run `rg -n "openalpha_bridge" packages/kronos-research packages/research-core cloud/modal/kronos_research.py tests/smoke`; expect no match. Delete remaining `packages/bridge/`.

- [ ] **Step 3: Delete product surfaces**

```text
.github/workflows/bridge-phase2.yml
.github/workflows/deploy-bridge-phase2-cloud.yml
.github/workflows/start-bridge-phase2-run.yml
cloud/modal/bridge_phase2_app.py
docker/bridge-phase2.Dockerfile
docker/bridge-phase2.env.template
scripts/bridge_phase2_cloud.py
tests/smoke/test_bridge_phase2_terminal_artifacts.py
tests/smoke/test_verify_deployment_contract.py
```

Retain `research/bridge-v0/phase2/**` as historical evidence.

- [ ] **Step 4: Create `Research Integrity` CI** — Python 3.13/uv; sync dev plus Sentinel phase 3; assert Torch absent; run full pytest, both verification scripts, Ruff, and Pyright. Include no deploy, secret, input, provider/model invocation, or study trigger.

- [ ] **Step 5: Remove Bridge metadata**

```toml
dependencies = ["openalpha-kronos-research", "openalpha-research-core", "openalpha-sentinel"]
```

Remove the Bridge source entry/extras/CLI and run `uv lock`.

- [ ] **Step 6: Verify and commit**

```powershell
uv run pytest -q tests/smoke/test_research_architecture.py tests/smoke/test_workflow_manifests.py
uv run ruff check packages cloud tests scripts
uv run pyright
git add -A packages/bridge .github/workflows cloud/modal docker scripts tests/smoke pyproject.toml uv.lock package.json Makefile
git commit -m "refactor: retire abandoned Bridge product infrastructure"
```

## Task 7: Retire obsolete operational documentation

**Files:** Delete the ten `docs/BRIDGE_*.md`/`OPENALPHA_KRONOS_BRIDGE.md` product documents named in the design; rewrite `docs/ARCHITECTURE.md` and `docs/STATUS.md`; repair live links in `DATA_POLICY.md` and `KRONOS_COMPATIBILITY_BOUNDARY.md`.

- [ ] **Step 1: Add failing current-document guards** — current docs (`README.md`, architecture, status, package READMEs) may not link to deleted files or describe `packages/bridge`, `openalpha_bridge`, `bridge_phase2_app.py`, or an executable Bridge Phase 2 system. Historical research/spec/ADR files are excluded.

- [ ] **Step 2: Delete instead of archiving**

Delete:

```text
docs/BRIDGE_CLOUD_API.md
docs/BRIDGE_CLOUD_ARCHITECTURE.md
docs/BRIDGE_CLOUD_STORAGE.md
docs/BRIDGE_GPU_RUNBOOK.md
docs/BRIDGE_MATHEMATICAL_REPRESENTATION.md
docs/BRIDGE_MODAL_DEPLOYMENT.md
docs/BRIDGE_NUMERICAL_CONTRACT.md
docs/BRIDGE_PHASE2_EXECUTION.md
docs/BRIDGE_TEST_OPENING_POLICY.md
docs/OPENALPHA_KRONOS_BRIDGE.md
```

Git history is the archive. Do not modify `research/**`, completed reports, ADRs, or the committed migration design.

- [ ] **Step 3: Rewrite current status/architecture**

Show the strict dependency direction and state exactly:

```text
structural-validity direction: closed
zero-shot generation direction: stopped
frozen-representation probe: preregistered, preserved, inactive, not authorized or executed
next direction: official-protocol replication; protocol and sealed preregistration precede code
```

Update only live links/commands in data-policy and compatibility-boundary docs.

- [ ] **Step 4: Verify and commit**

```powershell
uv run pytest -q tests/smoke/test_research_architecture.py
git diff --exit-code e6e15b6 -- research
git add -A docs
git commit -m "docs: replace Bridge operations with research architecture"
```

## Task 8: Add the approved README research-flow image

**Files:** Create `docs/assets/openalpha-research-flow.png`; modify root/package READMEs and the architecture guard.

- [ ] **Step 1: Add a failing diagram contract**

```python
import hashlib


def test_readme_uses_the_approved_research_flow_image() -> None:
    image = ROOT / "docs" / "assets" / "openalpha-research-flow.png"
    assert image.is_file()
    assert image.stat().st_size == 1_419_345
    assert hashlib.sha256(image.read_bytes()).hexdigest() == (
        "4782c0251fe9a21bd6984b4b03b39ffe6d2549cb34fac6b5f14a11afece88929"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    marker = "![OpenAlpha research flow"
    assert marker in readme
    assert readme.index(marker) < readme.index("| Study | Question | Conclusion |")
```

Expected initial failure: asset absent.

- [ ] **Step 2: Copy exact bytes**

Copy without recompression:

```text
C:\Users\Tommy\Downloads\ChatGPT Image Aug 12, 2026, 04_58_20 PM.png
→ docs/assets/openalpha-research-flow.png
```

Require SHA-256 `4782c0251fe9a21bd6984b4b03b39ffe6d2549cb34fac6b5f14a11afece88929` and size `1,419,345`.

- [ ] **Step 3: Rewrite the README opening/tour**

After the one-sentence description and before the study table:

```markdown
![OpenAlpha research flow: the Bridge thesis led through Kronos-mini and Kronos-base structural studies, abandonment of the structural-validity direction, a zero-shot benchmark, and the current official-protocol replication direction. Every transition is gated by sealed preregistrations, pinned commits, immutable artifacts, and predefined decision rules.](docs/assets/openalpha-research-flow.png)
```

Tour `research-core`, `kronos-research`, `sentinel`, `experiment-spec`, and immutable `research/`. Remove stale exact line/test counts rather than guessing. State the probe is frozen and official-protocol replication is next; claim no implementation.

- [ ] **Step 4: Verify and commit**

```powershell
uv run pytest -q tests/smoke/test_research_architecture.py
Get-FileHash docs\assets\openalpha-research-flow.png -Algorithm SHA256
git add README.md docs/assets/openalpha-research-flow.png packages/research-core/README.md packages/kronos-research/README.md packages/sentinel/README.md tests/smoke/test_research_architecture.py
git commit -m "docs: present OpenAlpha as an empirical research codebase"
```

## Task 9: Pin research-record integrity

**Files:** Create `tests/smoke/test_research_record_integrity.py`; preserve both verification scripts.

- [ ] **Step 1: Add exact published-summary guards**

```python
SUMMARY_DIGESTS = {
    "research/reports/kronos-structural-validity/results-summary.json": "fabd86207ca37dd7d6ec1e04e6cca916813c5e324d9fa18f03d368b821d1f25a",
    "research/reports/kronos-zero-shot-benchmark/results-summary.json": "1d5eef1b293addb55fbc10dc2e16eddda4c60c68b3be80ebd2041f2369217c20",
}


@pytest.mark.parametrize("relative, expected", SUMMARY_DIGESTS.items())
def test_published_numeric_summary_is_byte_identical(relative: str, expected: str) -> None:
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
```

Retain existing numerical assertions over mini/base results and the zero-shot moving-block interval recomputation.

- [ ] **Step 2: Run all evidence checks**

```powershell
uv run python scripts/verify_specifications.py
uv run python scripts/verify_artifacts.py
uv run pytest -q tests/smoke/test_published_artifacts.py tests/smoke/test_research_record_integrity.py packages/kronos-research/tests/test_kronos_zero_shot_benchmark.py
git diff --exit-code e6e15b6 -- research
```

Expected: hashes, bytes, summaries, numeric regressions pass; research diff empty.

- [ ] **Step 3: Commit**

```powershell
git add tests/smoke/test_research_record_integrity.py packages/kronos-research/tests/test_kronos_zero_shot_benchmark.py
git commit -m "test: guard the published research record"
```

## Task 10: Full offline verification and cleanup

**Files:** Remove only `.superpowers/brainstorm/retire-bridge-20260812/`; preserve `representation-probe-review.patch` unless separately authorized.

- [ ] **Step 1: Run complete offline verification**

```powershell
uv sync --locked --group dev --group sentinel-phase3
uv run pytest -q -m "not network and not kronos"
uv run ruff check .
uv run pyright
uv run python scripts/verify_specifications.py
uv run python scripts/verify_artifacts.py
```

Expected: PASS with no provider, checkpoint, model, Modal, or partition access.

- [ ] **Step 2: Run static acceptance scans**

```powershell
rg -n "openalpha_bridge" packages cloud scripts tests
rg -n -i "Bridge Phase 2" .github packages cloud scripts tests README.md docs/ARCHITECTURE.md docs/STATUS.md
rg -n "representation_probe|frozen_representation" cloud .github/workflows scripts
rg -n "official_protocol" packages/kronos-research/src
git diff --exit-code e6e15b6 -- research
```

Expected: no live matches and empty research diff. Historical evidence is outside live scans.

- [ ] **Step 3: Remove only this session's visual-companion files** — delete `.superpowers/brainstorm/retire-bridge-20260812/`; do not delete `representation-probe-review.patch`.

- [ ] **Step 4: Inspect final status/history**

```powershell
git status --short
git diff --stat e6e15b6..HEAD
git log --oneline --decorate -12
```

Expected: no accidental `.superpowers/`; the review patch may remain untracked and is reported.

- [ ] **Step 5: If verification requires corrections, commit narrowly and rerun affected checks** — do not squash the probe preservation commit into the migration.

## Completion checklist

- [ ] `packages/bridge/` is absent and active code has no `openalpha_bridge`.
- [ ] Research-core is subject-neutral and has no reverse dependency.
- [ ] All surviving Kronos executable research lives under `openalpha_kronos`.
- [ ] Dormant probe source/tests survive with no execution surface.
- [ ] No `official_protocol` implementation exists.
- [ ] Modal exposes only the eight explicit completed-study functions.
- [ ] Active CI verifies research integrity only.
- [ ] Research files, artifacts, summaries, and digests are unchanged.
- [ ] Approved README PNG is byte-identical and near the top.
- [ ] No scientific execution occurred.
