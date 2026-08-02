"""The isolated diagnostic worker, executed with doubles.

No network, no official asset, no Torch, no held-out partition.
"""

from __future__ import annotations

import ast
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from diagnostic_fakes import FakeCodec, FakeForecastModel, fake_assets, frozen_digest
from openalpha_bridge.cloud.objectstore import InMemoryObjectStore, run_prefix
from openalpha_bridge.diagnostic import artifact as artifact_module
from openalpha_bridge.diagnostic import worker as worker_module
from openalpha_bridge.diagnostic.artifact import (
    DIAGNOSTIC_FAILURE_CODE,
    DIAGNOSTIC_OPERATIONAL_FAILURE_CODE,
    DIAGNOSTIC_SUCCESS_CODE,
    OPERATIONAL_FAILURE_MESSAGE,
    diagnostic_artifact_key,
)
from openalpha_bridge.diagnostic.conclusion import DiagnosticConclusion
from openalpha_bridge.diagnostic.worker import run_diagnostic_worker
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.canary_artifact import terminal_artifact_key
from openalpha_bridge.phase2.states import EvidenceClass
from test_frozen_inference_diagnostic import (
    RESEARCH,
    TRUE_TARGET,
    _shift,
    _SingleWindowProvider,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)
COMMIT = "a" * 40
RUN_ID = "canary_0badc0de"


class _Resolver:
    """Stands in for asset resolution, and counts whether it was called."""

    def __init__(self, *, policy=None, assets=None, raises: Exception | None = None) -> None:
        self._policy = policy or (lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
        self._assets = assets or fake_assets()
        self._raises = raises
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        codec = FakeCodec()
        model = FakeForecastModel(codec=codec, path_for=self._policy)
        return codec, model, self._assets, frozen_digest()


class _ProviderFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.last: _SingleWindowProvider | None = None

    def __call__(self):
        self.calls += 1
        self.last = _SingleWindowProvider()
        return self.last


def _run(*, store=None, resolver=None, factory=None, run_id=RUN_ID, commit=COMMIT, deployed=None):
    store = store or InMemoryObjectStore()
    resolver = resolver or _Resolver()
    factory = factory or _ProviderFactory()
    result = run_diagnostic_worker(
        store=store,
        resolve=resolver,
        provider_factory=factory,
        research_root=RESEARCH,
        source_commit=commit,
        deployed_commit=deployed or commit,
        run_id=run_id,
        now=NOW,
    )
    return result, store, resolver, factory


# ============================================ validation before anything else


@pytest.mark.parametrize(
    ("run_id", "code"),
    [
        ("canary_../escape", "INVALID_RUN_ID"),
        ("canary_/abs", "INVALID_RUN_ID"),
        ("not_a_canary", "INVALID_RUN_ID"),
        ("", "INVALID_RUN_ID"),
    ],
)
def test_a_malformed_run_id_is_refused_before_any_use(run_id: str, code: str) -> None:
    store = InMemoryObjectStore()
    resolver = _Resolver()
    factory = _ProviderFactory()
    with pytest.raises(BridgeTransformError) as excinfo:
        _run(store=store, resolver=resolver, factory=factory, run_id=run_id)
    assert excinfo.value.failures[0].code == code
    # Nothing was named, resolved, or retrieved.
    assert store.list_keys("") == ()
    assert resolver.calls == 0
    assert factory.calls == 0


@pytest.mark.parametrize("commit", ["", "HEAD", "a" * 39, "A" * 40])
def test_a_malformed_commit_is_refused_before_any_use(commit: str) -> None:
    store = InMemoryObjectStore()
    resolver = _Resolver()
    with pytest.raises(BridgeTransformError):
        _run(store=store, resolver=resolver, commit=commit, deployed=commit)
    assert resolver.calls == 0
    assert store.list_keys("") == ()


def test_a_commit_that_disagrees_with_the_image_is_refused() -> None:
    store = InMemoryObjectStore()
    resolver = _Resolver()
    with pytest.raises(BridgeTransformError) as excinfo:
        _run(store=store, resolver=resolver, commit="b" * 40, deployed=COMMIT)
    assert excinfo.value.failures[0].code == "SOURCE_COMMIT_MISMATCH"
    assert resolver.calls == 0


def test_validation_precedes_everything_in_the_source() -> None:
    source = inspect.getsource(run_diagnostic_worker)
    validation = source.index("WorkerInvocation.validate_all")
    assert validation < source.index("def execute()")
    assert validation < source.index("publish_diagnostic_artifact")


# ================================================ existing artifact comes first


def test_a_duplicate_invocation_loads_no_model_and_fetches_nothing() -> None:
    _, store, first_resolver, first_factory = _run()
    assert first_resolver.calls == 1
    assert first_factory.calls == 1

    second_resolver = _Resolver()
    second_factory = _ProviderFactory()
    result, _, _, _ = _run(store=store, resolver=second_resolver, factory=second_factory)

    assert result.already_existed is True
    assert second_resolver.calls == 0, "a duplicate invocation resolved assets"
    assert second_factory.calls == 0, "a duplicate invocation built a provider"
    assert len(store.list_keys("")) == 1


def test_the_existence_check_precedes_execution_in_the_source() -> None:
    source = inspect.getsource(artifact_module.publish_diagnostic_artifact)
    assert source.index("find_existing_artifact") < source.index("execute()")


def test_a_duplicate_verifies_rather_than_overwrites() -> None:
    first, store, _, _ = _run()
    original = store.get(first.artifact_key).body
    second, _, _, _ = _run(store=store)
    assert second.content_sha256 == first.content_sha256
    assert store.get(first.artifact_key).body == original


def test_an_artifact_from_a_different_commit_is_refused() -> None:
    _, store, _, _ = _run()
    with pytest.raises(BridgeTransformError) as excinfo:
        _run(store=store, commit="b" * 40, deployed="b" * 40)
    assert excinfo.value.failures[0].code == ("DIAGNOSTIC_TERMINAL_ARTIFACT_COMMIT_MISMATCH")


# ============================================================ the window


def test_exactly_one_retrieval_of_the_specified_window() -> None:
    result, _, _, factory = _run()
    assert factory.calls == 1
    assert factory.last is not None
    assert factory.last.requests == ["SPY:2015-05-07:2017-05-18"]
    assert result.payload["provider_request_count"] == 1
    assert result.payload["retrieved_sessions"] == 512


def test_the_session_count_is_required_to_be_512() -> None:
    source = inspect.getsource(
        __import__(
            "openalpha_bridge.diagnostic.runner", fromlist=["run_frozen_inference_diagnostic"]
        ).run_frozen_inference_diagnostic
    )
    assert "DIAGNOSTIC_UNEXPECTED_SESSION_COUNT" in source
    assert "len(series.candles) != TOTAL_CANDLES" in source


# ================================================= the terminal artifact


def test_a_success_is_written_under_the_diagnostic_only_root() -> None:
    result, store, _, _ = _run()
    expected_prefix = run_prefix(RUN_ID, EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY)
    assert result.artifact_key == diagnostic_artifact_key(RUN_ID)
    assert result.artifact_key.startswith(f"{expected_prefix}/frozen-inference-diagnostic/")
    assert store.list_keys("openalpha/") == ()
    assert store.list_keys("openalpha-synthetic/") == ()
    assert result.outcome == DIAGNOSTIC_SUCCESS_CODE
    assert result.conclusion in {c.value for c in DiagnosticConclusion}


def test_the_diagnostic_root_is_distinct_from_the_canary_root() -> None:
    """The two development artifacts can never collide or be confused."""
    assert diagnostic_artifact_key(RUN_ID) != terminal_artifact_key(RUN_ID)
    assert "frozen-inference-diagnostic" in diagnostic_artifact_key(RUN_ID)
    assert "frozen-inference-diagnostic" not in terminal_artifact_key(RUN_ID)


def test_a_typed_failure_is_preserved_distinctly() -> None:
    resolver = _Resolver(assets=fake_assets(trainable=17_605))
    result, store, _, _ = _run(resolver=resolver)
    assert result.outcome == DIAGNOSTIC_FAILURE_CODE
    assert result.payload["failure_code"] == "DIAGNOSTIC_PARAMETERS_NOT_FROZEN"
    assert result.conclusion is None
    assert store.exists(result.artifact_key)


def test_an_operational_failure_is_preserved_distinctly_and_carries_no_text() -> None:
    resolver = _Resolver(
        raises=ConnectionResetError(
            "auth failed at https://provider.invalid/v1?apikey=hunter2 in /home/me/x.py"
        )
    )
    result, store, _, _ = _run(resolver=resolver)
    assert result.outcome == DIAGNOSTIC_OPERATIONAL_FAILURE_CODE
    assert result.outcome != DIAGNOSTIC_FAILURE_CODE
    assert result.conclusion == DiagnosticConclusion.DIAGNOSTIC_OPERATIONAL_FAILURE.value
    assert result.payload["exception_class"] == "ConnectionResetError"
    assert result.payload["message"] == OPERATIONAL_FAILURE_MESSAGE

    body = store.get(result.artifact_key).body.decode("utf-8")
    for leaked in ("hunter2", "provider.invalid", "apikey", "/home/me", "x.py", "Traceback"):
        assert leaked not in body


def test_every_outcome_leaves_exactly_one_artifact() -> None:
    for resolver in (
        _Resolver(),
        _Resolver(assets=fake_assets(trainable=1)),
        _Resolver(raises=RuntimeError("boom")),
    ):
        _, store, _, _ = _run(resolver=resolver)
        assert len(store.list_keys("")) == 1


def test_the_artifact_authorizes_nothing() -> None:
    result, _, _, _ = _run()
    assert result.authorizes_training is False
    assert result.authorizes_stage_b is False
    assert result.authorizes_stage_c is False
    assert result.authorizes_test_opening is False
    assert result.authorizes_production_inference is False
    assert result.authorizes_trading_claims is False
    assert result.scientific_result_available is False
    payload = json.loads(json.dumps(result.payload))
    assert payload["claim_boundary"] == ("DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE")


# ====================================================== nothing else is reachable


def _imports(module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
            if node.module:
                names.update(node.module.split("."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.update(alias.name.split("."))
    return names


def _calls(module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


@pytest.mark.parametrize(
    "forbidden",
    [
        "CloudRunner",
        "run_training",
        "TrainingBackend",
        "optimizer",
        "Adam",
        "AdamW",
        "backward",
        "zero_grad",
        "select_checkpoint",
        "freeze_checkpoint",
        "CheckpointRecord",
        "open_test_partition",
        "open_cloud_test_partition",
        "evaluate_conclusion",
        "GateTable",
        "stage_b",
        "stage_c",
    ],
)
def test_the_worker_reaches_none_of_the_forbidden_surfaces(forbidden: str) -> None:
    for module in (worker_module, artifact_module):
        assert forbidden not in _imports(module)
        assert forbidden not in _calls(module)


def test_the_worker_imports_no_runner_or_training_module() -> None:
    for module in (worker_module, artifact_module):
        imported = _imports(module)
        for banned in ("runner", "training", "gates", "testgate", "pipeline", "torch"):
            if banned == "runner":
                # diagnostic.runner is the diagnostic's own entry point, not
                # the cloud runner. Anything named cloud.runner is forbidden.
                assert "cloud.runner" not in imported
                continue
            assert not any(name == banned or name.endswith(f".{banned}") for name in imported)


def test_the_worker_stops_after_the_artifact() -> None:
    source = inspect.getsource(run_diagnostic_worker)
    assert source.rstrip().endswith(")")
    # The last thing it does is build the result from the stored artifact.
    assert "return DiagnosticWorkerResult(" in source


def test_no_held_out_partition_is_reachable() -> None:
    for module in (worker_module, artifact_module):
        source = inspect.getsource(module)
        for partition in ("reconstruction_test", "external_later", "holdout", "validation"):
            assert partition not in source


# ============================================================ the Modal shell


def test_the_modal_function_only_delegates() -> None:
    app = Path(__file__).resolve().parents[3] / "cloud" / "modal" / "bridge_phase2_app.py"
    source = app.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "frozen_inference_diagnostic"
    )
    body = ast.get_source_segment(source, function) or ""
    assert "run_diagnostic_worker(" in body
    assert "deployed_commit=_require_deployed_commit()" in body
    # The scientific logic is not in the deployment file.
    for moved in ("run_frozen_inference_diagnostic", "run_method_a", "decide("):
        assert moved not in body


def test_the_modal_diagnostic_is_its_own_function() -> None:
    app = Path(__file__).resolve().parents[3] / "cloud" / "modal" / "bridge_phase2_app.py"
    tree = ast.parse(app.read_text(encoding="utf-8"))
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "frozen_inference_diagnostic" in names
    assert "stage_a_official_canary" in names


def test_the_modal_model_pin_agrees_with_the_specification() -> None:
    from openalpha_bridge.diagnostic.spec import KRONOS_MINI_SPEC

    app = Path(__file__).resolve().parents[3] / "cloud" / "modal" / "bridge_phase2_app.py"
    tree = ast.parse(app.read_text(encoding="utf-8"))
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert constants["KRONOS_MINI_REPOSITORY"] == KRONOS_MINI_SPEC.repository
    assert constants["KRONOS_MINI_REVISION"] == KRONOS_MINI_SPEC.revision
