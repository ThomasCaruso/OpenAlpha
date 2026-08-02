"""Unexpected failures during publication, not only during the computation.

``execute_and_log`` covered the scientific computation. Everything around it --
the existence lookup, verification of an artifact that is already there, both
serialisations, both writes, and the verification that follows an
``OBJECT_ALREADY_EXISTS`` race -- ran outside any sanitised logging. A run
could therefore finish its measurement and then vanish: no durable artifact,
because publication is what writes one, and no log, because nothing logged.

Every test here drives the real worker. The store, the runtime and the provider
are doubles; no network, no official asset, no Torch, no partition.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from typing import Any

import pytest
from openalpha_bridge.cloud.objectstore import (
    InMemoryObjectStore,
    ObjectMetadata,
    StoredObject,
)
from openalpha_bridge.diagnostic import artifact as artifact_module
from openalpha_bridge.diagnostic import worker as worker_module
from openalpha_bridge.diagnostic.artifact import (
    DIAGNOSTIC_FAILURE_CODE,
    DIAGNOSTIC_OPERATIONAL_FAILURE_CODE,
    DIAGNOSTIC_SUCCESS_CODE,
    diagnostic_artifact_key,
)
from openalpha_bridge.diagnostic.safe_logging import STAGES, StageTracker
from openalpha_bridge.diagnostic.worker import run_diagnostic_worker
from openalpha_bridge.errors import BridgeFailure, BridgeTransformError, FailureCategory
from openalpha_bridge.phase2.states import EvidenceClass
from test_diagnostic_worker import COMMIT, NOW, RUN_ID, _ProviderFactory, _Resolver
from test_frozen_inference_diagnostic import RESEARCH

# Text that must never reach a log or an artifact, in the message of every
# exception these tests raise.
CREDENTIAL = "hunter2-publication-secret"
URL = f"https://store.invalid/bucket/object?token={CREDENTIAL}"
FULL_PATH = "/home/tommy/private/objectstore.py"
SECRET_MESSAGE = f"store call failed for {URL} at {FULL_PATH}"

LEAKS = (CREDENTIAL, URL, "store.invalid", "token=", FULL_PATH, "/home/tommy", "store call failed")


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        return "\n".join(record.getMessage() for record in self.records)


class _BrokenStore:
    """An in-memory store that fails one named operation, unexpectedly.

    ``ConnectionResetError`` rather than a typed failure: the point is an
    exception this code never anticipated, carrying text that must not escape.
    """

    def __init__(self, *, fail_on: str, after: int = 0, error: Exception | None = None) -> None:
        self._inner = InMemoryObjectStore()
        self._fail_on = fail_on
        self._after = after
        self._seen: dict[str, int] = {}
        self._error = error or ConnectionResetError(SECRET_MESSAGE)

    def _maybe_fail(self, operation: str) -> None:
        seen = self._seen.get(operation, 0)
        self._seen[operation] = seen + 1
        if operation == self._fail_on and seen >= self._after:
            raise self._error

    # The ObjectStore protocol, delegated.

    def put_immutable(self, key: str, body: bytes, metadata: ObjectMetadata) -> ObjectMetadata:
        self._maybe_fail("put_immutable")
        return self._inner.put_immutable(key, body, metadata)

    def put_overwrite(self, key: str, body: bytes, metadata: ObjectMetadata) -> ObjectMetadata:
        self._maybe_fail("put_overwrite")
        return self._inner.put_overwrite(key, body, metadata)

    def get(self, key: str) -> StoredObject:
        self._maybe_fail("get")
        return self._inner.get(key)

    def exists(self, key: str) -> bool:
        self._maybe_fail("exists")
        return self._inner.exists(key)

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        return self._inner.list_keys(prefix)

    def delete(self, key: str) -> None:
        self._inner.delete(key)


class _RacingStore(_BrokenStore):
    """Reports the object absent, then refuses the write as already present.

    The window every immutable writer has: another worker wrote the key between
    the existence check and the put. The follow-up verification is what fails.
    """

    def put_immutable(self, key: str, body: bytes, metadata: ObjectMetadata) -> ObjectMetadata:
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="OBJECT_ALREADY_EXISTS",
                message=f"immutable object already exists and cannot be replaced: {key}",
            )
        )


def _run(
    *,
    store: Any,
    resolver: _Resolver | None = None,
    factory: _ProviderFactory | None = None,
    logger: logging.Logger,
) -> Any:
    return run_diagnostic_worker(
        store=store,
        resolve_runtime=resolver or _Resolver(),
        provider_factory=factory or _ProviderFactory(),
        research_root=RESEARCH,
        source_commit=COMMIT,
        deployed_commit=COMMIT,
        run_id=RUN_ID,
        now=NOW,
        logger=logger,
    )


def _capturing(name: str) -> tuple[logging.Logger, _Capture]:
    logger = logging.getLogger(name)
    handler = _Capture()
    logger.addHandler(handler)
    return logger, handler


def _expect_unexpected_failure(
    name: str,
    *,
    store: Any,
    expected_stage: str,
    expected_class: str,
    resolver: _Resolver | None = None,
    factory: _ProviderFactory | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    patch: tuple[str, Any] | None = None,
) -> str:
    """Run, require the failure to propagate, and assert the log is safe.

    Returns the captured log text so a caller can make further assertions.
    """
    logger, handler = _capturing(name)
    try:
        if patch is not None and monkeypatch is not None:
            monkeypatch.setattr(artifact_module, patch[0], patch[1])
        with pytest.raises(Exception) as excinfo:
            _run(store=store, resolver=resolver, factory=factory, logger=logger)
    finally:
        logger.removeHandler(handler)

    assert type(excinfo.value).__name__ == expected_class
    assert not isinstance(excinfo.value, BridgeTransformError)

    text = handler.text
    assert f"stage={expected_stage}" in text, f"expected {expected_stage}, log said: {text}"
    assert f"exception_class={expected_class}" in text
    assert f"run_id={RUN_ID}" in text
    assert f"deployed_commit={COMMIT}" in text

    # Safe frames: basename:function:line, and nothing resembling a path.
    frames = text.split("frames=[", 1)[1].rsplit("]", 1)[0].split(" <- ")
    assert frames and frames != ["no frames"]
    for frame in frames:
        assert frame.count(":") == 2, frame
        assert "/" not in frame and "\\" not in frame, frame
        assert frame.rsplit(":", 1)[1].isdigit(), frame

    for leaked in LEAKS:
        assert leaked not in text, f"the log leaked {leaked!r}"
    return text


# ================================== 1-3: the lookup and the existing artifact


def test_an_unexpected_failure_in_the_initial_exists_is_logged_and_propagates() -> None:
    store = _BrokenStore(fail_on="exists")
    text = _expect_unexpected_failure(
        "openalpha.test.pub.exists",
        store=store,
        expected_stage="verify_existing_artifact",
        expected_class="ConnectionResetError",
    )
    assert "artifact.py:find_existing_artifact:" in text
    # Nothing was written, and the test does not pretend otherwise.
    assert store.list_keys("") == ()


def test_an_unexpected_failure_reading_an_existing_artifact_is_logged() -> None:
    seeded = _BrokenStore(fail_on="never")
    _run(store=seeded, logger=logging.getLogger("openalpha.test.pub.seed"))
    assert seeded.list_keys("") != ()

    broken = _BrokenStore(fail_on="get")
    broken._inner = seeded._inner
    text = _expect_unexpected_failure(
        "openalpha.test.pub.get",
        store=broken,
        expected_stage="verify_existing_artifact",
        expected_class="ConnectionResetError",
    )
    assert "artifact.py:verify_existing_artifact:" in text


def test_an_undecodable_existing_artifact_is_logged_as_a_decode_failure() -> None:
    """A real JSONDecodeError, from a real body that is not JSON."""
    store = InMemoryObjectStore()
    key = diagnostic_artifact_key(RUN_ID)
    body = f"{{not json at all, {CREDENTIAL}".encode()
    store.put_overwrite(
        key,
        body,
        ObjectMetadata(
            schema_version="openalpha.bridge.diagnostic.frozen_inference_failure.v2",
            run_id=RUN_ID,
            experiment_hash="0" * 64,
            evidence_class=EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY,
            content_sha256=hashlib.sha256(body).hexdigest(),
            created_at=NOW,
        ),
    )

    text = _expect_unexpected_failure(
        "openalpha.test.pub.decode",
        store=store,
        expected_stage="verify_existing_artifact",
        expected_class="JSONDecodeError",
    )
    assert "artifact.py:verify_existing_artifact:" in text


# ============================================ 4-5: the successful result


def test_an_unexpected_failure_serializing_a_success_is_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnserializableResult:
        schema_version = "irrelevant"

        def model_dump(self, **_: Any) -> dict[str, Any]:
            raise RecursionError(SECRET_MESSAGE)

    monkeypatch.setattr(
        worker_module, "run_frozen_inference_diagnostic", lambda **_: _UnserializableResult()
    )
    store = _BrokenStore(fail_on="never")
    text = _expect_unexpected_failure(
        "openalpha.test.pub.dump",
        store=store,
        expected_stage="serialize_artifact",
        expected_class="RecursionError",
    )
    assert "artifact.py:publish_diagnostic_artifact:" in text
    assert store.list_keys("") == (), "a partial artifact was written"


def test_an_unexpected_failure_writing_a_success_is_logged() -> None:
    store = _BrokenStore(fail_on="put_immutable")
    text = _expect_unexpected_failure(
        "openalpha.test.pub.write",
        store=store,
        expected_stage="write_artifact",
        expected_class="ConnectionResetError",
    )
    assert "artifact.py:_write:" in text
    assert store.list_keys("") == ()


# ================================ 6-8: the two failure artifacts


class _UnserializableFailure:
    """Stands in for DiagnosticFailure when its construction is the fault."""

    def __init__(self, **_: Any) -> None:
        raise OverflowError(SECRET_MESSAGE)


def test_an_unexpected_failure_serializing_a_typed_failure_is_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The computation failed typed, and building its artifact then failed."""
    resolver = _Resolver(assets=__import__("diagnostic_fakes").fake_assets(trainable=17_605))
    _expect_unexpected_failure(
        "openalpha.test.pub.typedser",
        store=_BrokenStore(fail_on="never"),
        expected_stage="serialize_artifact",
        expected_class="OverflowError",
        resolver=resolver,
        monkeypatch=monkeypatch,
        patch=("DiagnosticFailure", _UnserializableFailure),
    )


def test_an_unexpected_failure_serializing_an_operational_failure_is_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The computation failed unexpectedly, and its artifact would not build."""
    logger, handler = _capturing("openalpha.test.pub.opser")
    monkeypatch.setattr(artifact_module, "DiagnosticFailure", _UnserializableFailure)
    try:
        with pytest.raises(OverflowError):
            _run(
                store=_BrokenStore(fail_on="never"),
                resolver=_Resolver(raise_on_enter=SystemError("inner boom")),
                logger=logger,
            )
    finally:
        logger.removeHandler(handler)

    messages = [record.getMessage() for record in handler.records]
    # Two distinct failures, each logged once: the computation fault that the
    # inner logger caught, then the serialisation fault that escaped.
    assert len(messages) == 2
    assert "exception_class=SystemError" in messages[0]
    assert "stage=enter_official_runtime" in messages[0]
    assert "exception_class=OverflowError" in messages[1]
    assert "stage=serialize_artifact" in messages[1]
    for leaked in LEAKS:
        assert leaked not in "\n".join(messages)


def test_an_unexpected_failure_writing_a_failure_artifact_is_logged() -> None:
    store = _BrokenStore(fail_on="put_immutable")
    text = _expect_unexpected_failure(
        "openalpha.test.pub.failwrite",
        store=store,
        expected_stage="write_artifact",
        expected_class="ConnectionResetError",
        resolver=_Resolver(assets=__import__("diagnostic_fakes").fake_assets(trainable=17_605)),
    )
    assert "artifact.py:_write:" in text
    assert store.list_keys("") == (), "no artifact survived, and none is claimed"


# ================================== 9: the already-exists verification race


def test_an_unexpected_failure_verifying_after_a_write_race_is_logged() -> None:
    """The write lost the race, and reading the winner then failed.

    The stage must say verification rather than write: the write is what
    returned OBJECT_ALREADY_EXISTS, which is expected and handled; the
    verification is what actually broke.
    """
    store = _RacingStore(fail_on="get")
    text = _expect_unexpected_failure(
        "openalpha.test.pub.race",
        store=store,
        expected_stage="verify_existing_artifact",
        expected_class="ConnectionResetError",
    )
    assert "artifact.py:_write:" in text
    assert "artifact.py:verify_existing_artifact:" in text


# ====================================== what must not have changed


def test_a_computation_failure_is_logged_once_and_becomes_an_artifact() -> None:
    logger, handler = _capturing("openalpha.test.pub.once")
    store = InMemoryObjectStore()
    try:
        result = run_diagnostic_worker(
            store=store,
            resolve_runtime=_Resolver(raise_on_enter=ConnectionResetError(SECRET_MESSAGE)),
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

    assert result.outcome == DIAGNOSTIC_OPERATIONAL_FAILURE_CODE
    assert len(handler.records) == 1, "the outer logger logged a converted failure again"
    assert "stage=enter_official_runtime" in handler.text
    for leaked in LEAKS:
        assert leaked not in handler.text
    body = store.get(result.artifact_key).body.decode("utf-8")
    for leaked in LEAKS:
        assert leaked not in body


def test_a_typed_failure_stays_typed_and_is_never_unexpected_logged() -> None:
    logger, handler = _capturing("openalpha.test.pub.typed")
    store = InMemoryObjectStore()
    try:
        result = run_diagnostic_worker(
            store=store,
            resolve_runtime=_Resolver(
                assets=__import__("diagnostic_fakes").fake_assets(trainable=17_605)
            ),
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
    assert handler.records == []


def test_a_rejected_invocation_stays_typed_and_is_never_unexpected_logged() -> None:
    """A commit mismatch escapes publication as a BridgeTransformError."""
    logger, handler = _capturing("openalpha.test.pub.reject")
    store = InMemoryObjectStore()
    _run(store=store, logger=logging.getLogger("openalpha.test.pub.reject.seed"))
    try:
        with pytest.raises(BridgeTransformError) as excinfo:
            run_diagnostic_worker(
                store=store,
                resolve_runtime=_Resolver(),
                provider_factory=_ProviderFactory(),
                research_root=RESEARCH,
                source_commit="b" * 40,
                deployed_commit="b" * 40,
                run_id=RUN_ID,
                now=NOW,
                logger=logger,
            )
    finally:
        logger.removeHandler(handler)

    assert excinfo.value.failures[0].code == "DIAGNOSTIC_TERMINAL_ARTIFACT_COMMIT_MISMATCH"
    assert handler.records == []


def test_successful_publication_is_unchanged() -> None:
    logger, handler = _capturing("openalpha.test.pub.success")
    store = InMemoryObjectStore()
    try:
        result = _run(store=store, logger=logger)
    finally:
        logger.removeHandler(handler)

    assert result.outcome == DIAGNOSTIC_SUCCESS_CODE
    assert result.already_existed is False
    assert handler.records == []
    assert len(store.list_keys("")) == 1


def test_a_duplicate_still_enters_neither_runtime_nor_provider() -> None:
    store = InMemoryObjectStore()
    _run(store=store, logger=logging.getLogger("openalpha.test.pub.dup.seed"))

    resolver = _Resolver()
    factory = _ProviderFactory(resolver)
    logger, handler = _capturing("openalpha.test.pub.dup")
    try:
        result = _run(store=store, resolver=resolver, factory=factory, logger=logger)
    finally:
        logger.removeHandler(handler)

    assert result.already_existed is True
    assert (resolver.calls, resolver.entered) == (0, 0)
    assert factory.calls == 0
    assert handler.records == []
    assert len(store.list_keys("")) == 1


def _without_timings(value: Any) -> Any:
    """Strip elapsed-time fields, the one thing that legitimately varies.

    Two identical runs already differ in ``seconds``, ``total_seconds`` and
    ``mean_seconds_per_rollout``, because those record wall-clock duration.
    Everything else is what stage tracking must not touch.
    """
    if isinstance(value, dict):
        return {k: _without_timings(v) for k, v in value.items() if "seconds" not in k}
    if isinstance(value, list):
        return [_without_timings(v) for v in value]
    return value


def test_stage_tracking_changes_no_serialized_payload() -> None:
    """The same execute, published with a tracker and without one.

    Driving both through ``publish_diagnostic_artifact`` directly makes stage
    tracking the only difference between the two publications.
    """
    key = diagnostic_artifact_key(RUN_ID)
    invocation = worker_module.WorkerInvocation.validate_all(
        run_id=RUN_ID, source_commit=COMMIT, deployed_commit=COMMIT
    )
    fakes = __import__("diagnostic_fakes")

    for label, make_resolver in (
        ("success", lambda: _Resolver()),
        ("typed", lambda: _Resolver(assets=fakes.fake_assets(trainable=17_605))),
        ("operational", lambda: _Resolver(raise_on_enter=RuntimeError("boom"))),
    ):

        def execute(make: Any = make_resolver) -> Any:
            with make()() as runtime:
                return worker_module.run_frozen_inference_diagnostic(
                    provider=_ProviderFactory()(),
                    codec=runtime.codec,
                    model=runtime.model,
                    assets=runtime.assets,
                    invocation=invocation,
                    research_root=RESEARCH,
                    parameter_digest=runtime.parameter_digest,
                    now=NOW,
                )

        tracker = StageTracker()
        tracked = InMemoryObjectStore()
        with_tracking = artifact_module.publish_diagnostic_artifact(
            tracked, invocation=invocation, execute=execute, now=NOW, stage=tracker
        )

        # ``stage`` omitted entirely: the private-tracker path.
        plain = InMemoryObjectStore()
        without_tracking = artifact_module.publish_diagnostic_artifact(
            plain, invocation=invocation, execute=execute, now=NOW
        )

        assert _without_timings(json.loads(tracked.get(key).body)) == _without_timings(
            json.loads(plain.get(key).body)
        ), f"stage tracking altered the {label} payload"
        assert with_tracking.outcome == without_tracking.outcome
        assert with_tracking.conclusion == without_tracking.conclusion
        # Not vacuous: the tracker really did advance through publication.
        assert tracker.stage == "write_artifact"


# ======================================================= structural contracts


def test_every_publication_stage_is_declared() -> None:
    for stage in ("verify_existing_artifact", "serialize_artifact", "write_artifact"):
        assert stage in STAGES


def test_publication_sets_a_stage_before_each_operation() -> None:
    source = inspect.getsource(artifact_module.publish_diagnostic_artifact)
    assert source.index('tracker.enter("verify_existing_artifact")') < source.index(
        "find_existing_artifact("
    )
    # Both failure branches and the success branch serialise under a stage.
    assert source.count('tracker.enter("serialize_artifact")') == 4
    assert source.count("stage=tracker") == 2

    write = inspect.getsource(artifact_module._write)
    assert write.index('tracker.enter("write_artifact")') < write.index("put_json(")
    assert write.index('tracker.enter("verify_existing_artifact")') < write.index(
        "verify_existing_artifact(store"
    )


def test_publication_never_reads_the_stage() -> None:
    """Write-only, so it cannot influence a payload, a digest or a branch."""
    for function in (artifact_module.publish_diagnostic_artifact, artifact_module._write):
        source = inspect.getsource(function)
        assert "tracker.stage" not in source
        assert "stage.stage" not in source
        assert "if tracker" not in source


def test_the_worker_wraps_the_whole_publication() -> None:
    source = inspect.getsource(worker_module.run_diagnostic_worker)
    assert "publish_diagnostic_artifact(" in source
    published = source.index("publish_diagnostic_artifact(")
    outer = source.index("except BridgeTransformError:", published)
    assert outer > published
    assert source.index("log_operational_failure(", outer) > outer
    assert "stage=stage" in source


def test_no_artifact_is_claimed_when_publication_failed() -> None:
    """The store is empty and the exception propagates. Nothing is invented."""
    store = _BrokenStore(fail_on="put_immutable")
    logger, handler = _capturing("openalpha.test.pub.noclaim")
    try:
        with pytest.raises(ConnectionResetError):
            _run(store=store, logger=logger)
    finally:
        logger.removeHandler(handler)
    assert store.list_keys("") == ()
    assert handler.records, "the only surviving evidence was not written"


def test_the_emitted_record_carries_only_the_safe_fields() -> None:
    """The whole message, field by field, as an operator will read it."""
    store = _BrokenStore(fail_on="exists")
    logger, handler = _capturing("openalpha.test.pub.text")
    try:
        with pytest.raises(ConnectionResetError):
            _run(store=store, logger=logger)
    finally:
        logger.removeHandler(handler)

    text = handler.text
    assert text.startswith("diagnostic operational failure")
    head, _, frames = text.partition(" frames=[")
    fields = dict(part.split("=", 1) for part in head.split()[3:])
    assert set(fields) == {"run_id", "deployed_commit", "stage", "exception_class"}
    assert fields["stage"] == "verify_existing_artifact"
    assert frames.rstrip("]")
    # The one record logged, and no second one.
    assert len(handler.records) == 1
