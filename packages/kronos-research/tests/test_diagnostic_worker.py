"""The isolated diagnostic worker, executed with doubles.

No network, no official asset, no Torch, no held-out partition.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest
from diagnostic_fakes import FakeCodec, FakeForecastModel, fake_assets, frozen_digest
from openalpha_kronos.studies.provenance import (
    legacy_canary_artifact_key,
    mini_run_prefix,
)
from openalpha_kronos.studies.structural_validity.common.conclusion import DiagnosticConclusion
from openalpha_kronos.studies.structural_validity.mini import artifact as artifact_module
from openalpha_kronos.studies.structural_validity.mini import worker as worker_module
from openalpha_kronos.studies.structural_validity.mini.artifact import (
    DIAGNOSTIC_FAILURE_CODE,
    DIAGNOSTIC_OPERATIONAL_FAILURE_CODE,
    DIAGNOSTIC_SUCCESS_CODE,
    OPERATIONAL_FAILURE_MESSAGE,
    diagnostic_artifact_key,
)
from openalpha_kronos.studies.structural_validity.mini.worker import run_diagnostic_worker
from openalpha_research.failures import ResearchFailureError
from openalpha_research.objectstore import InMemoryObjectStore
from test_frozen_inference_diagnostic import (
    RESEARCH,
    TRUE_TARGET,
    _shift,
    _SingleWindowProvider,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)
COMMIT = "a" * 40
RUN_ID = "canary_0badc0de"


class _OpennessProxy:
    """Delegates to a double, recording whether the context was open per call.

    Frozen inference has to happen while the import context that produced the
    tokenizer and the model is still entered. This is how a test observes that
    for every call rather than only at construction.
    """

    def __init__(self, inner, resolver, record, watched) -> None:
        self._inner = inner
        self._resolver = resolver
        self._record = record
        self._watched = frozenset(watched)

    def __getattr__(self, name: str):
        value = getattr(self._inner, name)
        if name not in self._watched or not callable(value):
            return value

        def watched(*args, **kwargs):
            self._record.setdefault(name, []).append(self._resolver.open)
            return value(*args, **kwargs)

        return watched


class _FakeRuntime:
    """What the shared official runtime yields, in test form."""

    def __init__(self, codec, model, assets, parameter_digest) -> None:
        self.codec = codec
        self.model = model
        self.assets = assets
        self.parameter_digest = parameter_digest


class _Resolver:
    """Stands in for the official runtime context.

    Counts entries and exits, so a test can prove the context is entered only
    when there is work, stays open for the whole computation, and is exited on
    every path including failure.
    """

    def __init__(
        self,
        *,
        policy=None,
        assets=None,
        raises: Exception | None = None,
        raise_on_enter: Exception | None = None,
        record: dict[str, list[bool]] | None = None,
    ) -> None:
        self._policy = policy or (lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
        self._assets = assets or fake_assets()
        self._raises = raises
        self._raise_on_enter = raise_on_enter
        self._record = record
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
            model = FakeForecastModel(codec=codec, path_for=self._policy)
            if self._record is not None:
                codec = _OpennessProxy(codec, self, self._record, ("encode", "decode"))
                model = _OpennessProxy(model, self, self._record, ("generate",))
            yield _FakeRuntime(codec, model, self._assets, frozen_digest())
        finally:
            self.open = False
            self.exited += 1

    def __call__(self):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._context()


class _WatchedProvider(_SingleWindowProvider):
    """Records whether the runtime context was open when retrieval happened."""

    def __init__(self, resolver: _Resolver) -> None:
        super().__init__()
        self._resolver = resolver
        self.open_at_fetch: list[bool] = []

    def fetch(self, request):
        self.open_at_fetch.append(self._resolver.open)
        return super().fetch(request)


class _ProviderFactory:
    def __init__(self, resolver: _Resolver | None = None) -> None:
        self.calls = 0
        self.last: _SingleWindowProvider | None = None
        self._resolver = resolver
        #: Whether the runtime context was open at construction time.
        self.context_open_at_construction: bool | None = None

    def __call__(self):
        self.calls += 1
        if self._resolver is None:
            self.last = _SingleWindowProvider()
        else:
            self.context_open_at_construction = self._resolver.open
            self.last = _WatchedProvider(self._resolver)
        return self.last


def _run(*, store=None, resolver=None, factory=None, run_id=RUN_ID, commit=COMMIT, deployed=None):
    store = store or InMemoryObjectStore()
    resolver = resolver or _Resolver()
    factory = factory or _ProviderFactory()
    result = run_diagnostic_worker(
        store=store,
        resolve_runtime=resolver,
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
    with pytest.raises(ResearchFailureError) as excinfo:
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
    with pytest.raises(ResearchFailureError):
        _run(store=store, resolver=resolver, commit=commit, deployed=commit)
    assert resolver.calls == 0
    assert store.list_keys("") == ()


def test_a_commit_that_disagrees_with_the_image_is_refused() -> None:
    store = InMemoryObjectStore()
    resolver = _Resolver()
    with pytest.raises(ResearchFailureError) as excinfo:
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
    with pytest.raises(ResearchFailureError) as excinfo:
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
            "openalpha_kronos.studies.structural_validity.mini.runner",
            fromlist=["run_frozen_inference_diagnostic"],
        ).run_frozen_inference_diagnostic
    )
    assert "DIAGNOSTIC_UNEXPECTED_SESSION_COUNT" in source
    assert "len(series.candles) != TOTAL_CANDLES" in source


# ================================================= the terminal artifact


def test_a_success_is_written_under_the_diagnostic_only_root() -> None:
    result, store, _, _ = _run()
    expected_prefix = mini_run_prefix(RUN_ID)
    assert result.artifact_key == diagnostic_artifact_key(RUN_ID)
    assert result.artifact_key.startswith(f"{expected_prefix}/frozen-inference-diagnostic/")
    assert store.list_keys("openalpha/") == ()
    assert store.list_keys("openalpha-synthetic/") == ()
    assert result.outcome == DIAGNOSTIC_SUCCESS_CODE
    assert result.conclusion in {c.value for c in DiagnosticConclusion}


def test_the_diagnostic_root_is_distinct_from_the_canary_root() -> None:
    """The two development artifacts can never collide or be confused."""
    assert diagnostic_artifact_key(RUN_ID) != legacy_canary_artifact_key(RUN_ID)
    assert "frozen-inference-diagnostic" in diagnostic_artifact_key(RUN_ID)
    assert "frozen-inference-diagnostic" not in legacy_canary_artifact_key(RUN_ID)


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


# =========================================== the runtime context lifecycle


def test_the_runtime_context_is_not_entered_for_an_existing_artifact() -> None:
    """A duplicate invocation must not import official source or load weights."""
    _, store, first, _ = _run()
    assert (first.calls, first.entered, first.exited) == (1, 1, 1)

    second = _Resolver()
    result, _, _, factory = _run(store=store, resolver=second)

    assert result.already_existed is True
    assert second.calls == 0, "the runtime was resolved for a duplicate"
    assert second.entered == 0, "the runtime context was entered for a duplicate"
    assert factory.calls == 0


def test_the_provider_is_constructed_only_after_the_context_is_open() -> None:
    resolver = _Resolver()
    factory = _ProviderFactory(resolver)
    _run(resolver=resolver, factory=factory)
    assert factory.calls == 1
    assert factory.context_open_at_construction is True

    source = inspect.getsource(worker_module.run_diagnostic_worker)
    assert source.index("with resolve_runtime() as runtime:") < source.index(
        "provider = provider_factory()"
    )


def test_the_context_stays_open_for_retrieval_and_every_method() -> None:
    record: dict[str, list[bool]] = {}
    resolver = _Resolver(record=record)
    factory = _ProviderFactory(resolver)
    result, _, _, _ = _run(resolver=resolver, factory=factory)
    assert result.outcome == DIAGNOSTIC_SUCCESS_CODE

    provider = factory.last
    assert isinstance(provider, _WatchedProvider)
    assert provider.open_at_fetch == [True], "retrieval ran with the context closed"

    # Methods A-D all encode, generate and decode; every one of those calls
    # must have happened while the importing context was still entered.
    for name in ("encode", "generate", "decode"):
        observed = record.get(name, [])
        assert observed, f"{name} was never called"
        assert all(observed), f"{name} ran {observed.count(False)} times with the context closed"

    assert resolver.open is False, "the context outlived the computation"


def test_the_context_is_exited_after_success_and_after_either_failure() -> None:
    success = _Resolver()
    _run(resolver=success)
    assert (success.entered, success.exited, success.open) == (1, 1, False)

    typed = _Resolver(assets=fake_assets(trainable=17_605))
    typed_result, _, _, _ = _run(resolver=typed)
    assert typed_result.outcome == DIAGNOSTIC_FAILURE_CODE
    assert (typed.entered, typed.exited, typed.open) == (1, 1, False)

    operational = _Resolver(raise_on_enter=RuntimeError("import blew up"))
    op_result, _, _, _ = _run(resolver=operational)
    assert op_result.outcome == DIAGNOSTIC_OPERATIONAL_FAILURE_CODE
    assert (operational.entered, operational.exited, operational.open) == (1, 1, False)


def test_a_context_entry_failure_is_preserved_as_an_operational_artifact() -> None:
    """The SystemError shape: the failure was in entering, not in the science."""
    resolver = _Resolver(
        raise_on_enter=SystemError("initialization of _internal failed without raising")
    )
    factory = _ProviderFactory(resolver)
    result, store, _, _ = _run(resolver=resolver, factory=factory)

    assert result.outcome == DIAGNOSTIC_OPERATIONAL_FAILURE_CODE
    assert result.outcome != DIAGNOSTIC_FAILURE_CODE
    assert result.conclusion == DiagnosticConclusion.DIAGNOSTIC_OPERATIONAL_FAILURE.value
    assert result.payload["exception_class"] == "SystemError"
    assert result.payload["message"] == OPERATIONAL_FAILURE_MESSAGE
    assert factory.calls == 0, "a provider was built after the context failed to open"
    assert len(store.list_keys("")) == 1

    body = store.get(result.artifact_key).body.decode("utf-8")
    assert "initialization of _internal" not in body


def test_the_artifact_stays_sanitised_while_the_log_carries_the_detail() -> None:
    """The durable record says nothing; the operator log says enough to debug."""

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    secret = "https://provider.invalid/v1?apikey=hunter2-secret"
    logger = logging.getLogger("openalpha.test.worker")
    handler = _Capture()
    logger.addHandler(handler)
    store = InMemoryObjectStore()
    try:
        result = run_diagnostic_worker(
            store=store,
            resolve_runtime=_Resolver(raise_on_enter=ConnectionResetError(secret)),
            provider_factory=_ProviderFactory(),
            research_root=RESEARCH,
            source_commit=COMMIT,
            deployed_commit=COMMIT,
            run_id=RUN_ID,
            now=NOW,
            logger=logger,
        )
    finally:
        logger.removeHandler(handler)

    logged = "\n".join(handler.messages)
    assert "exception_class=ConnectionResetError" in logged
    assert "stage=enter_official_runtime" in logged
    assert f"run_id={RUN_ID}" in logged
    assert "frames=[" in logged
    assert "test_diagnostic_worker.py:" in logged
    for leaked in ("hunter2", "provider.invalid", "apikey"):
        assert leaked not in logged, "the log leaked exception text"

    body = store.get(result.artifact_key).body.decode("utf-8")
    for leaked in ("hunter2", "provider.invalid", "apikey", "enter_official_runtime", "frames"):
        assert leaked not in body, "the durable artifact gained unsanitised detail"


def test_a_typed_failure_is_never_logged() -> None:
    """It already carries a message this code wrote, so nothing is emitted."""

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    logger = logging.getLogger("openalpha.test.worker.typed")
    handler = _Capture()
    logger.addHandler(handler)
    try:
        result = run_diagnostic_worker(
            store=InMemoryObjectStore(),
            resolve_runtime=_Resolver(assets=fake_assets(trainable=17_605)),
            provider_factory=_ProviderFactory(),
            research_root=RESEARCH,
            source_commit=COMMIT,
            deployed_commit=COMMIT,
            run_id=RUN_ID,
            now=NOW,
            logger=logger,
        )
    finally:
        logger.removeHandler(handler)

    assert result.outcome == DIAGNOSTIC_FAILURE_CODE
    assert handler.messages == []


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
    app = Path(__file__).resolve().parents[3] / "cloud" / "modal" / "kronos_research.py"
    source = app.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_mini_structural_validity"
    )
    body = ast.get_source_segment(source, function) or ""
    assert "run_diagnostic_worker(" in body
    assert "deployed_commit=_require_deployed_commit()" in body
    # The scientific logic is not in the deployment file.
    for moved in ("run_frozen_inference_diagnostic", "run_method_a", "decide("):
        assert moved not in body


def test_the_modal_diagnostic_is_its_own_function() -> None:
    app = Path(__file__).resolve().parents[3] / "cloud" / "modal" / "kronos_research.py"
    tree = ast.parse(app.read_text(encoding="utf-8"))
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "run_mini_structural_validity" in names
    assert "verify_mini_runtime" in names
    assert "stage_a_official_canary" not in names


def test_the_modal_model_pin_agrees_with_the_specification() -> None:
    from openalpha_kronos.studies.structural_validity.mini.spec import KRONOS_MINI_SPEC

    app = Path(__file__).resolve().parents[3] / "cloud" / "modal" / "kronos_research.py"
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
