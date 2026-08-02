"""Every canary invocation leaves exactly one immutable terminal artifact."""

from __future__ import annotations

import typing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from openalpha_bridge.cloud.objectstore import InMemoryObjectStore
from openalpha_bridge.cloud.redaction import register_secret
from openalpha_bridge.errors import BridgeFailure, BridgeTransformError, FailureCategory
from openalpha_bridge.phase2.cache import FeatureCache
from openalpha_bridge.phase2.canary import CANARY_SUCCESS_CODE, run_stage_a_canary
from openalpha_bridge.phase2.canary_artifact import (
    CANARY_FAILURE_CODE,
    CANARY_OPERATIONAL_FAILURE_CODE,
    TerminalArtifact,
    publish_terminal_artifact,
    terminal_artifact_key,
)
from openalpha_bridge.phase2.states import EvidenceClass
from pydantic import ValidationError
from test_stage_a_canary import (  # the bridge test directory is on sys.path
    RESEARCH,
    _CountingProvider,
    _PseudoOfficialBackend,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)
COMMIT = "a" * 40
RUN_ID = "canary_0badc0de"


def _store() -> InMemoryObjectStore:
    return InMemoryObjectStore()


def _publish(store: InMemoryObjectStore, execute: Any, **kwargs: Any) -> TerminalArtifact:
    return publish_terminal_artifact(
        store,
        run_id=kwargs.pop("run_id", RUN_ID),
        source_commit=COMMIT,
        deployed_commit=COMMIT,
        execute=execute,
        now=NOW,
        **kwargs,
    )


def _successful(tmp_path: Path) -> Any:
    def execute():
        return run_stage_a_canary(
            provider=_CountingProvider(),
            backend=_PseudoOfficialBackend(),
            cache=FeatureCache(tmp_path / "cache"),
            research_root=RESEARCH,
            source_commit=COMMIT,
            deployed_commit=COMMIT,
            run_id=RUN_ID,
            now=NOW,
        )

    return execute


# ------------------------------------------------------------ the three ends


def test_a_success_is_written_under_the_compatibility_root(tmp_path: Path) -> None:
    store = _store()
    artifact = _publish(store, _successful(tmp_path))

    assert artifact.outcome == CANARY_SUCCESS_CODE
    assert artifact.already_existed is False
    assert artifact.key == terminal_artifact_key(RUN_ID)
    assert artifact.key.startswith("openalpha-compatibility/")
    assert store.exists(artifact.key)
    assert artifact.payload["outcome"] == CANARY_SUCCESS_CODE


def test_a_typed_failure_is_preserved_not_discarded() -> None:
    store = _store()

    def execute():
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="CANARY_TOKENIZER_IDENTITY_MISMATCH",
                message="resolved tokenizer does not equal TOKENIZER_SPEC",
            )
        )

    artifact = _publish(store, execute)
    assert artifact.outcome == CANARY_FAILURE_CODE
    assert artifact.payload["failure_code"] == "CANARY_TOKENIZER_IDENTITY_MISMATCH"
    assert artifact.payload["deployed_commit"] == COMMIT
    assert store.exists(artifact.key)


def test_an_operational_failure_is_distinguishable_from_a_result() -> None:
    """Infrastructure dying is not evidence about the official path."""
    store = _store()

    def execute():
        raise ConnectionResetError("the provider connection dropped")

    artifact = _publish(store, execute)
    assert artifact.outcome == CANARY_OPERATIONAL_FAILURE_CODE
    assert artifact.outcome != CANARY_FAILURE_CODE
    assert artifact.payload["failure_code"] == "CANARY_OPERATIONAL_FAILURE"
    assert "ConnectionResetError" in artifact.payload["message"]


def test_nothing_escapes_without_an_artifact() -> None:
    """Any exception type at all still terminates in a stored object."""
    store = _store()

    for index, error in enumerate(
        [MemoryError("out of memory"), RuntimeError("boom"), OSError("disk"), ValueError("bad")]
    ):

        def execute(err=error):
            raise err

        run_id = f"canary_{index:08x}"
        artifact = _publish(store, execute, run_id=run_id)
        assert store.exists(artifact.key)
        assert artifact.outcome == CANARY_OPERATIONAL_FAILURE_CODE


def test_a_failure_message_is_scrubbed_before_storage() -> None:
    store = _store()
    register_secret("hunter2-secret-token")

    def execute():
        raise RuntimeError("auth failed for token hunter2-secret-token")

    artifact = _publish(store, execute)
    assert "hunter2-secret-token" not in artifact.payload["message"]
    assert "hunter2-secret-token" not in store.get(artifact.key).body.decode("utf-8")


# ------------------------------------------------------- immutability


def test_a_duplicate_invocation_verifies_and_never_overwrites(tmp_path: Path) -> None:
    store = _store()
    first = _publish(store, _successful(tmp_path))
    original = store.get(first.key).body

    def execute():
        raise RuntimeError("a second invocation that would have written a different outcome")

    second = _publish(store, execute)

    assert second.already_existed is True
    assert second.key == first.key
    assert second.content_sha256 == first.content_sha256
    assert second.outcome == CANARY_SUCCESS_CODE, "the stored outcome wins, not the new one"
    assert store.get(first.key).body == original
    assert len(store.list_keys("openalpha-compatibility/")) == 1


def test_a_duplicate_after_a_failure_preserves_the_failure(tmp_path: Path) -> None:
    """A failed canary cannot be re-run into a pass under the same run id."""
    store = _store()

    def failing():
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="CANARY_ASSETS_UNVERIFIED",
                message="official asset revisions did not verify",
            )
        )

    first = _publish(store, failing)
    second = _publish(store, _successful(tmp_path))

    assert first.outcome == CANARY_FAILURE_CODE
    assert second.already_existed is True
    assert second.outcome == CANARY_FAILURE_CODE
    assert second.payload["failure_code"] == "CANARY_ASSETS_UNVERIFIED"


def test_an_existing_object_of_another_evidence_class_is_refused() -> None:
    store = _store()
    key = terminal_artifact_key(RUN_ID)
    from openalpha_bridge.cloud.objectstore import put_json

    put_json(
        store,
        key,
        {"outcome": "SOMETHING_ELSE", "evidence_class": EvidenceClass.REAL_PHASE2.value},
        schema_version="openalpha.bridge.phase2.stage_a_canary.v1",
        run_id=RUN_ID,
        experiment_hash="0" * 64,
        evidence_class=EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY,
        immutable=True,
    )

    def execute():
        raise RuntimeError("unused")

    with pytest.raises(BridgeTransformError) as excinfo:
        _publish(store, execute)
    assert excinfo.value.failures[0].code == "CANARY_TERMINAL_ARTIFACT_EVIDENCE_CLASS_MISMATCH"


# --------------------------------------------------- the classification is typed


def test_the_evidence_class_cannot_be_overridden() -> None:
    hints = typing.get_type_hints(TerminalArtifact, include_extras=True)
    assert hints["evidence_class"] == typing.Literal[EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY]
    assert hints["authorizes_stage_b"] == typing.Literal[False]
    assert hints["authorizes_real_run"] == typing.Literal[False]

    with pytest.raises(ValidationError):
        TerminalArtifact(
            key="k",
            content_sha256="0" * 64,
            outcome=CANARY_SUCCESS_CODE,
            evidence_class=EvidenceClass.REAL_PHASE2,  # pyright: ignore[reportArgumentType]
            already_existed=False,
            payload={},
        )


def test_the_artifact_authorizes_nothing(tmp_path: Path) -> None:
    store = _store()
    artifact = _publish(store, _successful(tmp_path))
    assert artifact.authorizes_stage_b is False
    assert artifact.authorizes_real_run is False
    assert artifact.evidence_class is EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY


def test_the_key_is_derived_from_run_prefix_not_a_formatted_string() -> None:
    from openalpha_bridge.cloud.objectstore import run_prefix

    expected = run_prefix(RUN_ID, EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY)
    assert terminal_artifact_key(RUN_ID).startswith(expected + "/")
    # Disjoint from the real and synthetic roots by construction.
    for other in (EvidenceClass.REAL_PHASE2, EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION):
        assert not terminal_artifact_key(RUN_ID).startswith(run_prefix(RUN_ID, other) + "/")
