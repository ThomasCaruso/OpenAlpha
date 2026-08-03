"""The Kronos-base replication, and its separation from the mini study.

Every test runs without Torch, without an official weight, and without a
network call. The store, the runtime and the provider are doubles; the base
model and tokenizer appear only as pinned metadata.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from diagnostic_fakes import FakeCodec, FakeForecastModel, frozen_digest
from openalpha_bridge.base_study import artifact as base_artifact_module
from openalpha_bridge.base_study import runner as base_runner_module
from openalpha_bridge.base_study import spec as base_spec_module
from openalpha_bridge.base_study import worker as base_worker_module
from openalpha_bridge.base_study.artifact import (
    BASE_FAILURE_CODE,
    BASE_OPERATIONAL_FAILURE_CODE,
    BASE_OPERATIONAL_FAILURE_MESSAGE,
    BASE_SUCCESS_CODE,
)
from openalpha_bridge.base_study.invocation import BaseStudyInvocation
from openalpha_bridge.base_study.runtime_probe import (
    BASE_PROBE_OUTCOME_PASSED,
    BaseRuntimeProbeResult,
)
from openalpha_bridge.base_study.spec import (
    BASE_ARTIFACT_ROOT,
    BASE_EXPERIMENT_ID,
    BASE_FAILURE_SCHEMA_VERSION,
    BASE_RUN_ID_PATTERN,
    BASE_SPECIFICATION_NAME,
    BASE_SPECIFICATION_SHA256,
    BASE_SUCCESS_SCHEMA_VERSION,
    KRONOS_BASE_SPEC,
    KRONOS_BASE_TOKENIZER_SPEC,
    MAXIMUM_CONTEXT,
    base_artifact_key,
    prove_context_budget,
    verify_base_specification,
)
from openalpha_bridge.base_study.worker import run_base_study_worker
from openalpha_bridge.cloud.objectstore import InMemoryObjectStore
from openalpha_bridge.diagnostic.artifact import diagnostic_artifact_key
from openalpha_bridge.diagnostic.backends import ResolvedDiagnosticAssets
from openalpha_bridge.diagnostic.spec import (
    CONTEXT_CANDLES,
    KRONOS_MINI_SPEC,
    OFFICIAL_SNAPSHOT_ALLOW_PATTERNS,
    OFFICIAL_TOKEN_VOCABULARY,
    TARGET_CANDLES,
    TOTAL_CANDLES,
    V4_SPECIFICATION_SHA256,
    verify_diagnostic_specifications,
)
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.invocation import WorkerInvocation
from openalpha_bridge.phase2.kronos import TOKENIZER_SPEC
from test_diagnostic_worker import COMMIT, NOW, _shift, _SingleWindowProvider
from test_frozen_inference_diagnostic import RESEARCH, TRUE_TARGET

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "cloud" / "modal" / "bridge_phase2_app.py"
BASE_RUN_ID = "base_0f1e2d3c4b5a6978"

# The identities resolved independently from the Hub during implementation.
EXPECTED_BASE_WEIGHTS_SHA256 = "abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83"
EXPECTED_BASE_WEIGHTS_BYTES = 409_264_008
EXPECTED_BASE_TOKENIZER_WEIGHTS_SHA256 = (
    "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
)


# ============================================== doubles


class _BaseRuntime:
    def __init__(self, codec, model, assets, parameter_digest) -> None:
        self.codec = codec
        self.model = model
        self.assets = assets
        self.parameter_digest = parameter_digest


def base_assets(**overrides: Any):
    """Fake resolved assets carrying the base pair's identities.

    Built here rather than by widening the shared mini fake: the mini fake
    describes the mini pair and should keep doing exactly that.
    """
    from openalpha_bridge.diagnostic.spec import OFFICIAL_SOURCE_FILES
    from openalpha_bridge.phase2.kronos import SOURCE_SPEC

    fields = {
        "tokenizer_repository": KRONOS_BASE_TOKENIZER_SPEC.repository,
        "tokenizer_revision": KRONOS_BASE_TOKENIZER_SPEC.revision,
        "tokenizer_config_sha256": KRONOS_BASE_TOKENIZER_SPEC.config_sha256,
        "tokenizer_weights_sha256": KRONOS_BASE_TOKENIZER_SPEC.weights_sha256,
        "model_repository": KRONOS_BASE_SPEC.repository,
        "model_revision": KRONOS_BASE_SPEC.revision,
        "model_config_sha256": KRONOS_BASE_SPEC.config_sha256,
        "model_weights_sha256": KRONOS_BASE_SPEC.weights_sha256,
        "source_revision": SOURCE_SPEC.revision,
        "source_as_committed_sha256": {
            f.relative_path: f.as_committed_sha256 for f in OFFICIAL_SOURCE_FILES
        },
        "source_crlf_normalized_sha256": {
            f.relative_path: f.sealed_sha256_crlf_normalized for f in OFFICIAL_SOURCE_FILES
        },
        "parameter_sha256": "c" * 64,
        "trainable_parameter_count": 0,
        "total_parameter_count": 102_300_000,
    }
    fields.update(overrides)
    return ResolvedDiagnosticAssets(**fields)


class _BaseResolver:
    """Stands in for the shared official runtime context."""

    def __init__(self, *, assets=None, raise_on_enter: Exception | None = None) -> None:
        self._assets = assets if assets is not None else base_assets()
        self._raise_on_enter = raise_on_enter
        self.calls = 0
        self.entered = 0
        self.exited = 0
        self.open = False

    @contextlib.contextmanager
    def _context(self):
        self.entered += 1
        self.open = True
        try:
            if self._raise_on_enter is not None:
                raise self._raise_on_enter
            codec = FakeCodec()
            model = FakeForecastModel(
                codec=codec, path_for=lambda seed, ctx: _shift(TRUE_TARGET, 1.01)
            )
            yield _BaseRuntime(codec, model, self._assets, frozen_digest())
        finally:
            self.open = False
            self.exited += 1

    def __call__(self):
        self.calls += 1
        return self._context()


class _Factory:
    def __init__(self) -> None:
        self.calls = 0
        self.last: _SingleWindowProvider | None = None

    def __call__(self):
        self.calls += 1
        self.last = _SingleWindowProvider()
        return self.last


def _run(*, store=None, resolver=None, factory=None, run_id=BASE_RUN_ID, logger=None):
    store = store or InMemoryObjectStore()
    resolver = resolver or _BaseResolver()
    factory = factory or _Factory()
    result = run_base_study_worker(
        store=store,
        resolve_runtime=resolver,
        provider_factory=factory,
        research_root=RESEARCH,
        source_commit=COMMIT,
        deployed_commit=COMMIT,
        run_id=run_id,
        now=NOW,
        logger=logger,
    )
    return result, store, resolver, factory


# ====================================== 1-4: nothing is downloaded locally


def _tracked_files() -> list[Path]:
    import subprocess

    output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return [REPO / name for name in output.split("\0") if name]


def test_no_model_weights_are_present_in_the_repository() -> None:
    """Neither base nor mini weights, tracked or untracked, anywhere in-tree."""
    weight_suffixes = {".safetensors", ".bin", ".ckpt"}
    offenders = [
        path
        for path in _tracked_files()
        if path.suffix in weight_suffixes or path.name == "model.safetensors"
    ]
    assert offenders == [], f"weight files are tracked: {offenders}"

    # And nothing weight-sized is lying around untracked in the source tree.
    skip = {".git", ".venv", ".worktrees", "__pycache__", "node_modules"}
    for directory in (REPO / "packages", REPO / "cloud", REPO / "research", REPO / "scripts"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if any(part in skip for part in path.parts):
                continue
            if path.is_file() and path.suffix in weight_suffixes:
                pytest.fail(f"a weight file is present in the tree: {path}")


def _download_calls(path: Path) -> set[str]:
    """Names of download verbs actually *called* in a module.

    A string check would be wrong here: several structural tests legitimately
    assert on the text "snapshot_download" without ever calling it. What
    matters is whether the module can perform a download, so only call sites
    and imports count.
    """
    verbs = {"snapshot_download", "hf_hub_download", "from_pretrained"}
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if name in verbs:
                found.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in verbs:
                    found.add(alias.name)
    return found


def test_no_test_or_source_module_downloads_an_asset() -> None:
    """Nothing importable by the local suite can fetch an official asset."""
    allowed = {
        # The one module that loads official weights, and only from an
        # already-resolved local directory inside the deployed container.
        (REPO / "packages/bridge/src/openalpha_bridge/diagnostic/official_backend.py").resolve(),
        (REPO / "packages/bridge/src/openalpha_bridge/phase2/kronos.py").resolve(),
    }
    searched = [
        *(REPO / "packages" / "bridge" / "tests").rglob("test_*.py"),
        *(REPO / "packages" / "bridge" / "src").rglob("*.py"),
    ]
    for path in searched:
        if path.resolve() in allowed:
            continue
        calls = _download_calls(path)
        assert calls == set(), f"{path.relative_to(REPO)} can download: {sorted(calls)}"

    # The two allowed modules load from a local directory, never from the Hub.
    backend = (
        REPO / "packages/bridge/src/openalpha_bridge/diagnostic/official_backend.py"
    ).read_text(encoding="utf-8")
    assert "snapshot_download" not in backend
    assert "from_pretrained(str(Path(tokenizer_directory)))" in backend
    assert "from_pretrained(str(Path(model_directory)))" in backend


def test_every_snapshot_download_lives_in_modal_execution_code() -> None:
    app_source = APP.read_text(encoding="utf-8")
    assert "snapshot_download(" in app_source
    tree = ast.parse(app_source)
    functions = {
        node.name: ast.get_source_segment(app_source, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    downloading = {name for name, body in functions.items() if "snapshot_download(" in body}
    assert downloading == {
        "frozen_inference_diagnostic",
        "verify_frozen_inference_runtime",
        "_download_base_pair",
    }, downloading


def test_every_base_download_is_restricted_to_the_two_allowed_files() -> None:
    app_source = APP.read_text(encoding="utf-8")
    tree = ast.parse(app_source)
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_download_base_pair"
    )
    body = ast.get_source_segment(app_source, node) or ""
    assert body.count("snapshot_download(") == 2
    assert body.count("allow_patterns=list(OFFICIAL_SNAPSHOT_ALLOW_PATTERNS)") == 2
    assert OFFICIAL_SNAPSHOT_ALLOW_PATTERNS == ("config.json", "model.safetensors")
    assert str(EXPECTED_BASE_WEIGHTS_BYTES) not in body, "no weight is materialised here"

    # Both base functions route their downloads through that one helper.
    for name in ("verify_base_frozen_inference_runtime", "kronos_base_frozen_inference_diagnostic"):
        target = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
        )
        text = ast.get_source_segment(app_source, target) or ""
        assert "_download_base_pair(cache_root)" in text
        assert "snapshot_download(" not in text


# ====================== 5-7: base and mini cannot be substituted or collide


def test_base_and_mini_repositories_cannot_be_substituted() -> None:
    assert KRONOS_BASE_SPEC.repository != KRONOS_MINI_SPEC.repository
    assert KRONOS_BASE_SPEC.revision != KRONOS_MINI_SPEC.revision
    assert KRONOS_BASE_SPEC.weights_sha256 != KRONOS_MINI_SPEC.weights_sha256
    assert KRONOS_BASE_SPEC.config_sha256 != KRONOS_MINI_SPEC.config_sha256
    assert KRONOS_BASE_TOKENIZER_SPEC.repository != TOKENIZER_SPEC.repository
    assert KRONOS_BASE_TOKENIZER_SPEC.weights_sha256 != TOKENIZER_SPEC.weights_sha256

    # The base study refuses a crossed pair at runtime, not merely by naming.
    for wrong in (
        {"model_repository": KRONOS_MINI_SPEC.repository},
        {"tokenizer_repository": TOKENIZER_SPEC.repository},
        {"model_revision": KRONOS_MINI_SPEC.revision},
        {"tokenizer_revision": TOKENIZER_SPEC.revision},
    ):
        resolver = _BaseResolver(assets=base_assets(**wrong))
        result, _, _, _ = _run(resolver=resolver)
        assert result.outcome == BASE_FAILURE_CODE, wrong
        assert result.payload["failure_code"].startswith("BASE_STUDY_WRONG_"), wrong


def test_base_and_mini_artifact_namespaces_cannot_collide() -> None:
    base_key = base_artifact_key(BASE_RUN_ID)
    mini_key = diagnostic_artifact_key("canary_0a92fde788bd685c")
    assert base_key.startswith(f"{BASE_ARTIFACT_ROOT}/runs/")
    assert not base_key.startswith("openalpha-compatibility/bridge-phase2/")
    assert not mini_key.startswith(f"{BASE_ARTIFACT_ROOT}/")
    assert base_key != mini_key
    # No shared directory prefix beyond the shared bucket-level root.
    assert base_key.split("/")[1] != mini_key.split("/")[1]
    assert "kronos_base_diagnostic_terminal.json" in base_key
    assert "frozen_inference_diagnostic_terminal.json" not in base_key


def test_base_and_mini_run_identifiers_cannot_collide() -> None:
    mini_ids = ("canary_0a92fde788bd685c", "canary_a91fcd689de78428", "canary_deadbeef")
    base_ids = ("base_deadbeef", "base_0f1e2d3c4b5a6978")

    for mini in mini_ids:
        assert not BASE_RUN_ID_PATTERN.fullmatch(mini)
        with pytest.raises(BridgeTransformError) as excinfo:
            BaseStudyInvocation.validate_all(
                run_id=mini, source_commit=COMMIT, deployed_commit=COMMIT
            )
        assert excinfo.value.failures[0].code == "MINI_RUN_ID_REFUSED_BY_BASE_STUDY"

    for base in base_ids:
        assert BASE_RUN_ID_PATTERN.fullmatch(base)
        with pytest.raises(BridgeTransformError) as excinfo:
            WorkerInvocation.validate_all(run_id=base, source_commit=COMMIT, deployed_commit=COMMIT)
        assert excinfo.value.failures[0].code == "INVALID_RUN_ID"


# ============================= 8-10: specifications and mini compatibility


def test_base_and_mini_specification_hashes_are_independently_verified() -> None:
    base = verify_base_specification(RESEARCH)
    assert base == {BASE_SPECIFICATION_NAME: BASE_SPECIFICATION_SHA256}

    observed = hashlib.sha256((Path(RESEARCH) / BASE_SPECIFICATION_NAME).read_bytes()).hexdigest()
    assert observed == BASE_SPECIFICATION_SHA256

    mini = verify_diagnostic_specifications(RESEARCH)
    assert len(mini) == 4
    assert mini["phase2-frozen-inference-diagnostic-v4.yaml"] == V4_SPECIFICATION_SHA256
    # Neither verifier accepts the other's document as its own.
    assert BASE_SPECIFICATION_SHA256 not in mini.values()
    assert BASE_SPECIFICATION_NAME not in mini


def test_the_base_specification_names_the_mini_study_without_claiming_equivalence() -> None:
    text = (Path(RESEARCH) / BASE_SPECIFICATION_NAME).read_text(encoding="utf-8")
    assert "canary_0a92fde788bd685c" in text
    assert V4_SPECIFICATION_SHA256 in text
    assert "interchangeable" in text
    reference = base_runner_module.PriorMiniStudyReference()
    assert reference.results_are_interchangeable is False
    assert reference.completed_run_id == "canary_0a92fde788bd685c"
    assert reference.operative_specification_sha256 == V4_SPECIFICATION_SHA256


def _mini_success_payload() -> dict[str, Any]:
    """The identity envelope the completed mini run actually recorded.

    Only the fields mini verification checks. The measurement body is not
    reproduced: it is 1.7 MB and none of it participates in verification.
    """
    from openalpha_bridge.diagnostic.spec import (
        V1_SPECIFICATION_NAME,
        V1_SPECIFICATION_SHA256,
        V2_SPECIFICATION_NAME,
        V2_SPECIFICATION_SHA256,
        V3_SPECIFICATION_NAME,
        V3_SPECIFICATION_SHA256,
        V4_SPECIFICATION_NAME,
    )
    from openalpha_bridge.phase2.identity import EXPERIMENT_SHA256

    return {
        "schema_version": "openalpha.bridge.diagnostic.frozen_inference.v4",
        "outcome": "FROZEN_INFERENCE_DIAGNOSTIC_COMPLETED",
        "conclusion": "ROUNDTRIP_MATERIAL_INVALIDITY",
        "claim_boundary": "DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE",
        "evidence_class": "development_compatibility_canary",
        "run_id": "canary_0a92fde788bd685c",
        "source_commit": "8e90533aefa94e552de58d2d2a003489769e7f87",
        "deployed_commit": "8e90533aefa94e552de58d2d2a003489769e7f87",
        "experiment_sha256": EXPERIMENT_SHA256,
        "specification_v1_name": V1_SPECIFICATION_NAME,
        "specification_v1_sha256": V1_SPECIFICATION_SHA256,
        "specification_v2_name": V2_SPECIFICATION_NAME,
        "specification_v2_sha256": V2_SPECIFICATION_SHA256,
        "specification_v3_name": V3_SPECIFICATION_NAME,
        "specification_v3_sha256": V3_SPECIFICATION_SHA256,
        "specification_v4_name": V4_SPECIFICATION_NAME,
        "specification_v4_sha256": V4_SPECIFICATION_SHA256,
        "operative_specification": V4_SPECIFICATION_NAME,
    }


def _mini_operational_failure_payload() -> dict[str, Any]:
    payload = _mini_success_payload()
    payload.update(
        {
            "schema_version": "openalpha.bridge.diagnostic.frozen_inference_failure.v2",
            "outcome": "FROZEN_INFERENCE_DIAGNOSTIC_OPERATIONAL_FAILURE",
            "conclusion": "DIAGNOSTIC_OPERATIONAL_FAILURE",
            "run_id": "canary_a91fcd689de78428",
            "source_commit": "1b2e2e233a284710011f284ae05ad03450ce7977",
            "deployed_commit": "1b2e2e233a284710011f284ae05ad03450ce7977",
            "failure_stage": "frozen_inference_diagnostic",
            "failure_code": "DIAGNOSTIC_OPERATIONAL_FAILURE",
            "exception_class": "SystemError",
        }
    )
    return payload


def _seed_mini(store, payload: dict[str, Any], run_id: str) -> None:
    from openalpha_bridge.cloud.objectstore import put_json
    from openalpha_bridge.phase2.identity import EXPERIMENT_SHA256
    from openalpha_bridge.phase2.states import EvidenceClass

    put_json(
        store,
        diagnostic_artifact_key(run_id),
        payload,
        schema_version=payload["schema_version"],
        run_id=run_id,
        experiment_hash=EXPERIMENT_SHA256,
        evidence_class=EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY,
        immutable=True,
    )


def test_the_completed_mini_success_artifact_still_verifies() -> None:
    """The base work must not change what a mini artifact means."""
    from openalpha_bridge.diagnostic.artifact import verify_existing_artifact

    store = InMemoryObjectStore()
    payload = _mini_success_payload()
    _seed_mini(store, payload, payload["run_id"])

    invocation = WorkerInvocation.validate_all(
        run_id=payload["run_id"],
        source_commit=payload["source_commit"],
        deployed_commit=payload["deployed_commit"],
    )
    verified = verify_existing_artifact(store, invocation=invocation)
    assert verified.outcome == "FROZEN_INFERENCE_DIAGNOSTIC_COMPLETED"
    assert verified.conclusion == "ROUNDTRIP_MATERIAL_INVALIDITY"
    assert verified.already_existed is True
    assert verified.key == diagnostic_artifact_key(payload["run_id"])


def test_the_spent_mini_operational_failure_artifact_still_verifies() -> None:
    from openalpha_bridge.diagnostic.artifact import verify_existing_artifact

    store = InMemoryObjectStore()
    payload = _mini_operational_failure_payload()
    _seed_mini(store, payload, payload["run_id"])

    invocation = WorkerInvocation.validate_all(
        run_id=payload["run_id"],
        source_commit=payload["source_commit"],
        deployed_commit=payload["deployed_commit"],
    )
    verified = verify_existing_artifact(store, invocation=invocation)
    assert verified.outcome == "FROZEN_INFERENCE_DIAGNOSTIC_OPERATIONAL_FAILURE"
    assert verified.conclusion == "DIAGNOSTIC_OPERATIONAL_FAILURE"
    # And the base verifier refuses it outright rather than reading it.
    base_invocation = BaseStudyInvocation.validate_all(
        run_id=BASE_RUN_ID, source_commit=COMMIT, deployed_commit=COMMIT
    )
    assert (
        base_artifact_module.find_existing_base_artifact(store, invocation=base_invocation) is None
    )


# ==================================== 11-12: context budget, no truncation


def test_the_base_runtime_uses_maximum_context_512() -> None:
    assert MAXIMUM_CONTEXT == 512
    assert KRONOS_BASE_SPEC.maximum_context == 512
    from openalpha_bridge.diagnostic.spec import OFFICIAL_INFERENCE_SETTINGS

    assert OFFICIAL_INFERENCE_SETTINGS.max_context == 512

    result, _, _, _ = _run()
    assert result.outcome == BASE_SUCCESS_CODE
    assert result.payload["context_budget"]["maximum_context"] == 512
    assert result.payload["inference_settings"]["max_context"] == 512


def test_the_direct_replication_fills_the_budget_without_truncation() -> None:
    proof = prove_context_budget(context_candles=CONTEXT_CANDLES, target_candles=TARGET_CANDLES)
    assert (CONTEXT_CANDLES, TARGET_CANDLES, TOTAL_CANDLES) == (448, 64, 512)
    assert proof["sum"] == 512 == MAXIMUM_CONTEXT
    assert proof["fills_budget_exactly"] is True
    assert proof["truncation_occurs"] is False

    result, _, _, _ = _run()
    budget = result.payload["context_budget"]
    assert budget["context_candles"] + budget["target_candles"] == budget["maximum_context"]
    assert budget["truncation_occurs"] is False
    assert result.payload["context_candles"] == 448
    assert result.payload["target_candles"] == 64
    assert result.payload["retrieved_sessions"] == 512

    # Anything that overflows the budget is refused, not silently truncated.
    with pytest.raises(BridgeTransformError) as excinfo:
        prove_context_budget(context_candles=480, target_candles=64)
    assert excinfo.value.failures[0].code == "BASE_CONTEXT_BUDGET_EXCEEDED"


# ============================================ 13: model-specific identities


def test_parameter_identities_are_model_specific() -> None:
    assert KRONOS_BASE_SPEC.model_family == "kronos-base"
    assert KRONOS_BASE_SPEC.name == "Kronos-base"
    assert KRONOS_MINI_SPEC.name == "Kronos-mini"
    assert not hasattr(KRONOS_MINI_SPEC, "model_family"), "mini fields must not be renamed"

    # Architecture is read from the base config, not inherited from mini.
    assert (KRONOS_BASE_SPEC.model_dimension, KRONOS_BASE_SPEC.layers) == (832, 12)
    assert (KRONOS_MINI_SPEC.model_dimension, KRONOS_MINI_SPEC.decoder_layers) == (256, 4)
    assert KRONOS_BASE_SPEC.attention_heads == 16
    assert KRONOS_BASE_SPEC.feedforward_dimension == 2048
    assert KRONOS_BASE_SPEC.learned_temporal_embedding is True

    # The independently resolved Hub identities, not copied from the brief.
    assert KRONOS_BASE_SPEC.weights_sha256 == EXPECTED_BASE_WEIGHTS_SHA256
    assert KRONOS_BASE_SPEC.weights_size_bytes == EXPECTED_BASE_WEIGHTS_BYTES
    assert KRONOS_BASE_TOKENIZER_SPEC.weights_sha256 == EXPECTED_BASE_TOKENIZER_WEIGHTS_SHA256
    assert len(KRONOS_BASE_SPEC.revision) == 40
    assert len(KRONOS_BASE_TOKENIZER_SPEC.revision) == 40
    assert KRONOS_BASE_SPEC.revision != "main"

    # The token space is shared, and said so rather than borrowed.
    assert KRONOS_BASE_SPEC.coarse_vocabulary == OFFICIAL_TOKEN_VOCABULARY
    assert KRONOS_MINI_SPEC.coarse_vocabulary == OFFICIAL_TOKEN_VOCABULARY
    assert KRONOS_BASE_SPEC.fine_vocabulary == OFFICIAL_TOKEN_VOCABULARY

    # The runtime is parameterised, and still defaults to mini for mini code.
    signature = inspect.signature(
        __import__(
            "openalpha_bridge.diagnostic.official_backend", fromlist=["official_runtime"]
        ).official_runtime
    )
    assert signature.parameters["model_spec"].default is KRONOS_MINI_SPEC


# ================================ 14-15: duplicate invocation, authorization


def test_a_duplicate_base_invocation_loads_nothing_and_fetches_nothing() -> None:
    _, store, first, first_factory = _run()
    assert (first.calls, first.entered, first_factory.calls) == (1, 1, 1)

    second = _BaseResolver()
    second_factory = _Factory()
    result, _, _, _ = _run(store=store, resolver=second, factory=second_factory)

    assert result.already_existed is True
    assert (second.calls, second.entered) == (0, 0)
    assert second_factory.calls == 0
    assert len(store.list_keys("")) == 1


def test_no_base_result_authorizes_anything() -> None:
    result, _, _, _ = _run()
    assert result.outcome == BASE_SUCCESS_CODE
    for field in (
        "authorizes_training",
        "authorizes_stage_b",
        "authorizes_stage_c",
        "authorizes_test_opening",
        "authorizes_production_inference",
        "authorizes_trading_claims",
        "scientific_result_available",
    ):
        assert getattr(result, field) is False
        assert result.payload[field] is False
    assert result.payload["training_performed"] is False
    assert result.payload["optimizer_constructed"] is False
    assert result.payload["held_out_partition_opened"] is False
    assert result.payload["claim_boundary"] == (
        "DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE"
    )

    probe_fields = BaseRuntimeProbeResult.model_fields
    assert probe_fields["authorizes_the_base_diagnostic"].default is False
    assert BASE_PROBE_OUTCOME_PASSED == "KRONOS_BASE_RUNTIME_PROBE_PASSED"


# ================================================= the study runs end to end


def test_a_successful_base_run_writes_one_artifact_under_its_own_identity() -> None:
    result, store, _, factory = _run()
    assert result.outcome == BASE_SUCCESS_CODE
    assert result.experiment_id == BASE_EXPERIMENT_ID
    assert result.model_family == "kronos-base"
    assert result.artifact_key == base_artifact_key(BASE_RUN_ID)
    assert store.list_keys("") == (base_artifact_key(BASE_RUN_ID),)
    assert result.payload["schema_version"] == BASE_SUCCESS_SCHEMA_VERSION
    assert result.payload["specification_sha256"] == BASE_SPECIFICATION_SHA256
    assert result.payload["forecast_model"]["repository"] == "NeoQuasar/Kronos-base"
    assert result.payload["tokenizer_repository"] == "NeoQuasar/Kronos-Tokenizer-base"
    assert result.payload["provider_request_count"] == 1
    assert factory.last is not None
    assert factory.last.requests == ["SPY:2015-05-07:2017-05-18"]


def test_a_base_operational_failure_is_preserved_without_exception_text() -> None:
    secret = "https://provider.invalid/v1?apikey=hunter2-base"
    logger = logging.getLogger("openalpha.test.base.op")

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    handler = _Capture()
    logger.addHandler(handler)
    try:
        result, store, _, _ = _run(
            resolver=_BaseResolver(raise_on_enter=ConnectionResetError(secret)), logger=logger
        )
    finally:
        logger.removeHandler(handler)

    assert result.outcome == BASE_OPERATIONAL_FAILURE_CODE
    assert result.payload["schema_version"] == BASE_FAILURE_SCHEMA_VERSION
    assert result.payload["exception_class"] == "ConnectionResetError"
    assert result.payload["message"] == BASE_OPERATIONAL_FAILURE_MESSAGE
    body = store.get(result.artifact_key).body.decode("utf-8")
    for leaked in ("hunter2-base", "provider.invalid", "apikey"):
        assert leaked not in body
    logged = "\n".join(handler.messages)
    assert "exception_class=ConnectionResetError" in logged
    for leaked in ("hunter2-base", "provider.invalid", "apikey"):
        assert leaked not in logged


def test_the_base_artifact_round_trips_through_verification() -> None:
    result, store, _, _ = _run()
    invocation = BaseStudyInvocation.validate_all(
        run_id=BASE_RUN_ID, source_commit=COMMIT, deployed_commit=COMMIT
    )
    verified = base_artifact_module.verify_existing_base_artifact(store, invocation=invocation)
    assert verified.content_sha256 == result.content_sha256
    assert verified.already_existed is True
    assert json.loads(store.get(result.artifact_key).body)["experiment_id"] == BASE_EXPERIMENT_ID


# ============================================== structural separation


def _referenced_names(module) -> set[str]:
    """Every identifier the module's *code* mentions, comments excluded.

    Docstrings say things like "no checkpoint code", which a text search would
    read as a checkpoint reference. Only names the parser sees count.
    """
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_the_base_study_never_imports_the_mini_worker_or_namespace() -> None:
    for module in (base_worker_module, base_runner_module, base_artifact_module, base_spec_module):
        names = _referenced_names(module)
        for forbidden in (
            "run_diagnostic_worker",
            "publish_diagnostic_artifact",
            "diagnostic_artifact_key",
            "verify_existing_artifact",
            "DiagnosticArtifact",
            "CloudRunner",
            "TorchTrainingBackend",
            "Phase2Config",
        ):
            assert forbidden not in names, f"{module.__name__} references {forbidden}"
        source = inspect.getsource(module)
        assert "..diagnostic.worker" not in source
        assert "..diagnostic.artifact" not in source


def test_the_base_modal_functions_use_the_base_volume_only() -> None:
    app_source = APP.read_text(encoding="utf-8")
    tree = ast.parse(app_source)
    bodies = {
        node.name: ast.get_source_segment(app_source, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    for name in (
        "verify_base_frozen_inference_runtime",
        "kronos_base_frozen_inference_diagnostic",
        "inventory_base_remote_cache",
    ):
        body = bodies[name]
        assert "BASE_CACHE_ROOT" in body
        assert "cache_volume.commit()" not in body.replace("base_cache_volume.commit()", "")
        assert "CACHE_ROOT)" not in body.replace("BASE_CACHE_ROOT)", "")

    # And the mini functions were not repointed at the base volume.
    for name in ("frozen_inference_diagnostic", "verify_frozen_inference_runtime"):
        assert "BASE_CACHE_ROOT" not in bodies[name]
        assert "KRONOS_BASE" not in bodies[name]

    assert '"openalpha-kronos-base-cache"' in app_source
    assert '"openalpha-bridge-phase2-cache"' in app_source


def test_the_cache_inventory_is_read_only_and_not_on_the_control_api() -> None:
    app_source = APP.read_text(encoding="utf-8")
    tree = ast.parse(app_source)
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "inventory_base_remote_cache"
    )
    body = ast.get_source_segment(app_source, node) or ""
    for destructive in ("rmtree", "unlink(", "os.remove", "shutil.rm", ".delete("):
        assert destructive not in body, f"the inventory can {destructive}"

    # The report is built by an importable module now, so the read-only
    # guarantees are asserted on what it actually returns rather than on the
    # Modal shell's source text. test_base_cache_inventory covers the rest.
    from datetime import UTC, datetime

    from openalpha_bridge.base_study.cache_inventory import build_base_cache_inventory

    payload = build_base_cache_inventory(
        reload=lambda: None,
        mount=str(REPO / "does-not-exist"),
        volume_name="openalpha-kronos-base-cache",
        deployed_commit="f" * 40,
        inspected_at=datetime(2026, 8, 3, tzinfo=UTC),
    )
    assert payload["read_only"] is True
    assert payload["deletion_supported"] is False
    assert payload["reloaded_before_inspection"] is True
    assert "build_base_cache_inventory(" in body
    # Not exposed through the ASGI control plane.
    control = ast.get_source_segment(
        app_source,
        next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "control_api"
        ),
    )
    assert control is not None
    assert "inventory_base_remote_cache" not in control


def test_the_modal_base_pins_agree_with_the_specification() -> None:
    app_source = APP.read_text(encoding="utf-8")
    assert f'KRONOS_BASE_REPOSITORY = "{KRONOS_BASE_SPEC.repository}"' in app_source
    assert f'KRONOS_BASE_REVISION = "{KRONOS_BASE_SPEC.revision}"' in app_source
    assert (
        f'KRONOS_BASE_TOKENIZER_REPOSITORY = "{KRONOS_BASE_TOKENIZER_SPEC.repository}"'
        in app_source
    )
    assert f'KRONOS_BASE_TOKENIZER_REVISION = "{KRONOS_BASE_TOKENIZER_SPEC.revision}"' in app_source
    # The mini pins are untouched.
    assert f'KRONOS_MINI_REPOSITORY = "{KRONOS_MINI_SPEC.repository}"' in app_source
    assert f'KRONOS_MINI_REVISION = "{KRONOS_MINI_SPEC.revision}"' in app_source


def test_the_base_functions_are_named_for_the_family_they_run() -> None:
    app_source = APP.read_text(encoding="utf-8")
    tree = ast.parse(app_source)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {
        "verify_base_deployment",
        "verify_base_frozen_inference_runtime",
        "kronos_base_frozen_inference_diagnostic",
        "inventory_base_remote_cache",
    } <= names
    # No top-level function whose name implies mini touches a base pin.
    # Nested helpers inherit their enclosing function's identity, so only
    # module-level definitions are checked.
    for node in ast.parse(app_source).body:
        if not isinstance(node, ast.FunctionDef) or "base" in node.name:
            continue
        body = ast.get_source_segment(app_source, node) or ""
        assert "KRONOS_BASE_SPEC" not in body, f"{node.name} loads base while not named base"
        assert "KRONOS_BASE_REPOSITORY" not in body, f"{node.name} pins base while not named base"


def test_the_base_specification_is_shipped_into_the_image() -> None:
    """The container verifies the document, so it has to be there."""
    assert (Path(RESEARCH) / BASE_SPECIFICATION_NAME).is_file()
    app_source = APP.read_text(encoding="utf-8")
    assert 'root / "research" / "bridge-v0"' in app_source
    assert 'remote_path="/root/research/bridge-v0"' in app_source


def test_completed_at_is_carried_through_untouched() -> None:
    result, _, _, _ = _run()
    assert result.payload["completed_at"] == datetime(2026, 8, 2, tzinfo=UTC).isoformat().replace(
        "+00:00", "Z"
    )
