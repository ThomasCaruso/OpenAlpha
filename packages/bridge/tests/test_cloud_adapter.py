"""SYNTHETIC CLOUD PIPELINE VALIDATION - NOT EMPIRICAL EVIDENCE.

Exercises the cloud control plane, object store, lease, journal chain, cloud
test gate, and secret redaction with an in-memory store and a fake compute
backend. No provider API is contacted and no Kronos asset is loaded.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from openalpha_bridge.cloud.auth import TokenAuthenticator
from openalpha_bridge.cloud.identity import CloudRunIdentity, assert_idempotent_match
from openalpha_bridge.cloud.journal import CloudJournal
from openalpha_bridge.cloud.lease import (
    acquire_lease,
    current_lease,
    release_lease,
)
from openalpha_bridge.cloud.models import (
    CancelRunRequest,
    CreateRunRequest,
    ExecutionMode,
    ResumeRunRequest,
)
from openalpha_bridge.cloud.objectstore import (
    InMemoryObjectStore,
    ObjectMetadata,
    experiment_prefix,
    run_prefix,
)
from openalpha_bridge.cloud.redaction import REDACTED, SecretRedactor, redact, register_secret
from openalpha_bridge.cloud.runner import CloudRunner, ResourceGuards
from openalpha_bridge.cloud.service import Phase2ControlService, ServiceConfig
from openalpha_bridge.cloud.testgate import (
    cloud_test_opening_key,
    is_cloud_test_partition_opened,
    open_cloud_test_partition,
)
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.gates import TerminalConclusion
from openalpha_bridge.phase2.identity import AMENDMENT_3_SHA256, EXPERIMENT_SHA256
from openalpha_bridge.phase2.kronos import DeterministicFakeKronosBackend, KronosMode
from openalpha_bridge.phase2.pipeline import Phase2Config
from openalpha_bridge.phase2.provider import DeterministicFakeProvider, ProviderMode
from openalpha_bridge.phase2.states import EvidenceClass, Phase2State
from openalpha_bridge.phase2.testgate import TestOpeningPreconditions
from openalpha_bridge.phase2.training import NumpyTrainingBackend

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_ROOT = REPOSITORY_ROOT / "research" / "bridge-v0"
COMMIT = "a" * 40


class _Clock:
    def __init__(self) -> None:
        self._now = datetime(2026, 8, 1, 9, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now


class FakeComputeBackend:
    """Records spawns instead of starting cloud work."""

    name = "fake"

    def __init__(self) -> None:
        self.real_spawns: list[str] = []
        self.synthetic_spawns: list[str] = []
        self.cancelled: list[str] = []
        self.payloads: dict[str, dict[str, Any]] = {}

    @property
    def image_digest(self) -> str:
        return "sha256:fakeimagedigest"

    @property
    def app_version(self) -> str:
        return "1.0.0"

    def spawn_real_run(self, run_id: str, payload: dict[str, Any]) -> str:
        self.real_spawns.append(run_id)
        self.payloads[run_id] = payload
        return f"fc-real-{run_id}"

    def spawn_synthetic_run(self, run_id: str, payload: dict[str, Any]) -> str:
        self.synthetic_spawns.append(run_id)
        self.payloads[run_id] = payload
        return f"fc-syn-{run_id}"

    def execution_status(self, cloud_execution_id: str) -> str:
        return "running"

    def cancel(self, cloud_execution_id: str) -> None:
        self.cancelled.append(cloud_execution_id)


def _service(store: InMemoryObjectStore, backend: FakeComputeBackend) -> Phase2ControlService:
    config = ServiceConfig(
        base_url="https://openalpha.example",
        object_store_namespace="openalpha-artifacts",
        dependency_lock_sha256="b" * 64,
        kronos_revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        provider_identity="yahoo_finance/yfinance==1.5.2",
    )
    return Phase2ControlService(store=store, backend=backend, config=config, clock=_Clock())


def _create(mode: ExecutionMode = ExecutionMode.SYNTHETIC, **overrides: Any) -> CreateRunRequest:
    payload: dict[str, Any] = {
        "experiment_hash": EXPERIMENT_SHA256,
        "source_commit": COMMIT,
        "operator": "tester",
        "execution_mode": mode,
        "confirm_real_evidence": mode is ExecutionMode.REAL,
    }
    payload.update(overrides)
    return CreateRunRequest.model_validate(payload)


def _stage_a_report(sequences: int = 1):
    """A minimal valid Stage A report for orchestration tests.

    Stage A no longer advances without evidence, so pipeline-level tests must
    supply one. Extraction itself is covered by test_phase2_features.py.
    """
    from openalpha_bridge.phase2.stage_a import StageAReport, StageASequenceRecord

    records = tuple(
        StageASequenceRecord(
            sequence_id=f"{index:016d}",
            symbol="SPY",
            interval="1d",
            partition="train",
            target_start="2022-01-03",
            target_end="2022-04-04",
            prefix_start="2020-01-02",
            prefix_end="2021-12-31",
            bridge_input_shape=(512, 269),
            bridge_input_dtype="float32",
            canonical_sha256="a" * 64,
            shard_relative_path=f"train/{index:016d}.npz",
            shard_content_sha256="b" * 64,
            shard_size_bytes=1024,
            deterministic_replay_matched=True,
        )
        for index in range(sequences)
    )
    return StageAReport(
        run_id="syn_orchestration",
        experiment_sha256="d" * 64,
        amendment_sha256=("1" * 64, "2" * 64, "3" * 64),
        source_commit="a" * 40,
        evidence_class="synthetic_pipeline_validation",
        provider="deterministic_fake",
        provider_mode="fake",
        provider_client_version="fake-1",
        kronos_mode="fake",
        kronos_repository="fake/kronos-tokenizer-2k",
        kronos_revision="0" * 40,
        kronos_config_sha256=None,
        kronos_weights_sha256=None,
        frozen_parameter_sha256="c" * 64,
        official_source_repository="https://github.com/shiyu-coder/Kronos",
        official_source_revision="67b630e67f6a18c9e9be918d9b4337c960db1e9a",
        official_source_file_sha256={
            "model/kronos.py": ("638a56e035856c600c9848b368be087cb706a61603a0790124968c95b8c69f3a"),
            "model/module.py": ("a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f"),
        },
        cache_schema_version="openalpha.bridge.phase2.cache.v2",
        representation_version="openalpha.bridge.financial.v1",
        prefix_length=448,
        suffix_length=64,
        score_mask_sha256="2fe5b1b3c66dfd7c7e8af2612d69d3c2337a4a0f6dd3896a4d7c2f69911f3711",
        bridge_input_dimension=269,
        retrieved_candles=512,
        sequences=records,
        completed_at=datetime(2026, 8, 2, tzinfo=UTC),
        passed=True,
    )


def _real_stage_a_for(runner_identity, cache_dir: Path, tmp_path: Path):
    """A genuinely extracted Stage A report bound to the runner's own identity.

    The runner audits Stage A evidence against a plan derived from the locks, so
    the report must cover exactly the sequences those locks imply: SPY over the
    intersection of the locked Stage A request period and the training period.
    A single hand-built window no longer satisfies the audit, which is the point
    of deriving the expectation independently.
    """
    from openalpha_bridge.calendars import sessions_in_half_open_range
    from openalpha_bridge.phase2.cache import FeatureCache
    from openalpha_bridge.phase2.features import FeatureExtractor
    from openalpha_bridge.phase2.kronos import SOURCE_SPEC
    from openalpha_bridge.phase2.pipeline import LOCKED_PERIODS
    from openalpha_bridge.phase2.provider import Candle, MarketSeries
    from openalpha_bridge.phase2.stage_a import run_stage_a
    from openalpha_bridge.phase2.stage_a_plan import (
        STAGE_A_LOCKED_MAXIMUM_CANDLES,
        STAGE_A_LOCKED_REQUEST_PERIOD,
    )
    from openalpha_bridge.windowing import Partition, build_scored_sequences

    requested_start = date.fromisoformat(STAGE_A_LOCKED_REQUEST_PERIOD[0])
    requested_end = date.fromisoformat(STAGE_A_LOCKED_REQUEST_PERIOD[1])
    training_start = date.fromisoformat(LOCKED_PERIODS[Partition.TRAIN][0])
    training_end = date.fromisoformat(LOCKED_PERIODS[Partition.TRAIN][1])
    sessions = sessions_in_half_open_range(
        max(requested_start, training_start), min(requested_end, training_end)
    )[:STAGE_A_LOCKED_MAXIMUM_CANDLES]

    # One deterministic candle per real session, so sequence identities match
    # the calendar the plan derives from.
    candles = []
    level = 100.0
    for session in sessions:
        level = max(5.0, level * 1.0005)
        close = level * 1.0002
        volume = 1.0e6
        candles.append(
            Candle(
                session=session,
                open=level,
                high=max(level, close) * 1.001,
                low=min(level, close) / 1.001,
                close=close,
                volume=volume,
                amount=volume * close,
            )
        )
        level = close

    specs = build_scored_sequences(
        symbol="SPY",
        interval="1d",
        partition=Partition.TRAIN,
        partition_sessions=tuple(sessions),
        history_sessions=(),
    )
    series = MarketSeries(
        symbol="SPY",
        interval="1d",
        provider="deterministic_fake",
        provider_mode=ProviderMode.FAKE,
        client_version="fake-1",
        retrieval_timestamp=None,
        candles=tuple(candles),
    )
    backend = DeterministicFakeKronosBackend()
    return run_stage_a(
        run_id=runner_identity.run_id,
        experiment_sha256=runner_identity.active_experiment_sha256,
        source_commit=runner_identity.source_commit,
        evidence_class=runner_identity.evidence_class.value,
        series=series,
        specs=tuple(specs),
        extractor=FeatureExtractor(backend),
        assets=backend.resolve_assets(),
        cache=FeatureCache(cache_dir, repository_root=REPOSITORY_ROOT),
        completed_at=datetime(2026, 8, 2, tzinfo=UTC),
        amendment_sha256=(
            runner_identity.amendment_1_sha256,
            runner_identity.amendment_2_sha256,
            AMENDMENT_3_SHA256,
        ),
        verified_source_file_sha256=dict(SOURCE_SPEC.files),
    )


# ------------------------------------------------------ dependency boundary


def test_importing_cloud_package_loads_no_cloud_sdk() -> None:
    code = (
        "import sys; import openalpha_bridge.cloud; "
        "print('boto3' in sys.modules, 'modal' in sys.modules, 'fastapi' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False False"


def test_importing_cloud_package_does_not_load_torch() -> None:
    code = "import sys; import openalpha_bridge.cloud; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


# --------------------------------------------------------------- redaction


def test_registered_secret_values_are_scrubbed() -> None:
    redactor = SecretRedactor()
    redactor.register("super-secret-token-value")
    scrubbed = redactor.scrub_text("using super-secret-token-value to authenticate")
    assert "super-secret-token-value" not in scrubbed
    assert REDACTED in scrubbed


def test_credential_shaped_patterns_are_scrubbed_without_registration() -> None:
    redactor = SecretRedactor()
    samples = (
        "Authorization: Bearer abcdef1234567890",
        "api_key=AKIAIOSFODNN7EXAMPLE",
        "token: hf_abcdefghijklmnopqrstuvwxyz",
        "https://data.example.com/bars?apikey=zzzzzzzzzzzz&symbol=SPY",
    )
    for sample in samples:
        assert REDACTED in redactor.scrub_text(sample)


def test_redaction_recurses_into_structures() -> None:
    register_secret("nested-secret-abcdef")
    payload = {"a": ["nested-secret-abcdef"], "b": {"c": "nested-secret-abcdef"}}
    scrubbed = redact(payload)
    assert scrubbed["a"][0] == REDACTED
    assert scrubbed["b"]["c"] == REDACTED


def test_environment_secrets_are_registered_by_name() -> None:
    redactor = SecretRedactor()
    redactor.register_environment({"ALPACA_API_SECRET_KEY": "alpaca-secret-value-123"})
    assert REDACTED in redactor.scrub_text("key=alpaca-secret-value-123")


# ------------------------------------------------------------ object store


def _metadata(body: bytes = b"{}") -> ObjectMetadata:
    import hashlib

    return ObjectMetadata(
        schema_version="test.v1",
        run_id="syn_test",
        experiment_hash=EXPERIMENT_SHA256,
        content_sha256=hashlib.sha256(body).hexdigest(),
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        evidence_class=EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
    )


def test_immutable_write_rejects_a_second_write() -> None:
    store = InMemoryObjectStore()
    store.put_immutable("k", b"{}", _metadata())
    with pytest.raises(BridgeTransformError) as excinfo:
        store.put_immutable("k", b"{}", _metadata())
    assert excinfo.value.failures[0].code == "OBJECT_ALREADY_EXISTS"


def test_object_hash_mismatch_is_detected() -> None:
    store = InMemoryObjectStore()
    store.put_immutable("k", b"{}", _metadata(b"different"))
    with pytest.raises(BridgeTransformError) as excinfo:
        store.get("k")
    assert excinfo.value.failures[0].code == "OBJECT_HASH_MISMATCH"


def test_real_and_synthetic_namespaces_are_disjoint() -> None:
    real = run_prefix("r1", EvidenceClass.REAL_PHASE2)
    synthetic = run_prefix("r1", EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    assert real != synthetic
    assert not synthetic.startswith(real)
    assert experiment_prefix(EXPERIMENT_SHA256, EvidenceClass.REAL_PHASE2).startswith(
        "openalpha/bridge-phase2/experiments/"
    )


# ------------------------------------------------------------------- lease


def test_lease_blocks_a_second_holder() -> None:
    store = InMemoryObjectStore()
    now = datetime(2026, 8, 1, tzinfo=UTC)
    kwargs = {
        "run_id": "syn_1",
        "experiment_hash": EXPERIMENT_SHA256,
        "evidence_class": EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
    }
    acquire_lease(store, holder="worker-a", now=now, **kwargs)
    with pytest.raises(BridgeTransformError) as excinfo:
        acquire_lease(store, holder="worker-b", now=now, **kwargs)
    assert excinfo.value.failures[0].code == "RUN_LEASE_HELD"


def test_expired_lease_can_be_taken_over() -> None:
    store = InMemoryObjectStore()
    now = datetime(2026, 8, 1, tzinfo=UTC)
    kwargs = {
        "run_id": "syn_2",
        "experiment_hash": EXPERIMENT_SHA256,
        "evidence_class": EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
    }
    acquire_lease(store, holder="worker-a", now=now, duration_seconds=60, **kwargs)
    later = now + timedelta(seconds=120)
    lease = acquire_lease(store, holder="worker-b", now=later, **kwargs)
    assert lease.holder == "worker-b"


def test_release_requires_ownership() -> None:
    store = InMemoryObjectStore()
    now = datetime(2026, 8, 1, tzinfo=UTC)
    kwargs = {
        "run_id": "syn_3",
        "evidence_class": EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
    }
    acquire_lease(store, holder="worker-a", now=now, experiment_hash=EXPERIMENT_SHA256, **kwargs)
    with pytest.raises(BridgeTransformError) as excinfo:
        release_lease(store, holder="worker-b", **kwargs)
    assert excinfo.value.failures[0].code == "RUN_LEASE_NOT_HELD"
    release_lease(store, holder="worker-a", **kwargs)
    assert current_lease(store, **kwargs) is None


# ----------------------------------------------------------------- journal


def _journal(store: InMemoryObjectStore) -> CloudJournal:
    return CloudJournal(
        store,
        run_id="syn_j",
        experiment_hash=EXPERIMENT_SHA256,
        evidence_class=EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
    )


def test_journal_chains_by_prior_hash() -> None:
    store = InMemoryObjectStore()
    journal = _journal(store)
    clock = _Clock()
    first = journal.append(Phase2State.CREATED, occurred_at=clock())
    second = journal.append(Phase2State.PREFLIGHT_PASSED, occurred_at=clock())
    assert first.prior_entry_sha256 is None
    assert second.prior_entry_sha256 == first.entry_sha256
    assert journal.verify_chain()
    assert journal.current_state() is Phase2State.PREFLIGHT_PASSED


def test_concurrent_journal_write_is_detected() -> None:
    store = InMemoryObjectStore()
    clock = _Clock()
    first = _journal(store)
    first.append(Phase2State.CREATED, occurred_at=clock())
    stale = first.latest()

    # Worker A advances the run to sequence 2.
    _journal(store).append(Phase2State.PREFLIGHT_PASSED, occurred_at=clock())

    # Worker B still holds the pre-advance view, so it also targets sequence 2.
    b = _journal(store)
    b.latest = lambda: stale  # type: ignore[method-assign]
    with pytest.raises(BridgeTransformError) as excinfo:
        b.append(Phase2State.PREFLIGHT_PASSED, occurred_at=clock())
    assert excinfo.value.failures[0].code == "CONCURRENT_JOURNAL_WRITE"


def test_journal_redacts_secrets_in_reasons() -> None:
    store = InMemoryObjectStore()
    register_secret("journal-secret-value-xyz")
    entry = _journal(store).append(
        Phase2State.FAILED,
        occurred_at=_Clock()(),
        reason="failed using journal-secret-value-xyz",
    )
    assert "journal-secret-value-xyz" not in (entry.reason or "")


# --------------------------------------------------------------- cloud gate


def _preconditions(**overrides: Any) -> TestOpeningPreconditions:
    payload: dict[str, Any] = {
        "stage_a_passed": True,
        "stage_b_passed": True,
        "stage_c_completed": True,
        "selected_checkpoint_fixed": True,
        "checkpoint_sha256": "c" * 64,
        "frozen_kronos_weights_verified": True,
        "validation_selection_report_sealed": True,
        "experiment_hash_verified": True,
        "preprocessing_state_sha256": "d" * 64,
        "feature_manifest_sha256": "e" * 64,
    }
    payload.update(overrides)
    return TestOpeningPreconditions.model_validate(payload)


def _identity(evidence_class: EvidenceClass) -> CloudRunIdentity:
    return CloudRunIdentity.derive(
        source_commit=COMMIT,
        provider_identity="yahoo_finance/yfinance==1.5.2",
        kronos_revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        cloud_image_digest="sha256:fakeimagedigest",
        modal_app_version="1.0.0",
        dependency_lock_sha256="b" * 64,
        object_store_namespace="openalpha-artifacts",
        operator="tester",
        evidence_class=evidence_class,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


def _frozen_journal(store: InMemoryObjectStore, identity: CloudRunIdentity) -> CloudJournal:
    journal = CloudJournal(
        store,
        run_id=identity.run_id,
        experiment_hash=identity.active_experiment_sha256,
        evidence_class=identity.evidence_class,
    )
    clock = _Clock()
    for state in (
        Phase2State.CREATED,
        Phase2State.PREFLIGHT_PASSED,
        Phase2State.DATA_RETRIEVED,
        Phase2State.DATA_VALIDATED,
        Phase2State.WINDOWS_BUILT,
        Phase2State.COVERAGE_PASSED,
        Phase2State.ASSETS_RESOLVED,
        Phase2State.STAGE_A_PASSED,
        Phase2State.STAGE_B_PASSED,
        Phase2State.STAGE_C_TRAINED,
        Phase2State.CHECKPOINT_FROZEN,
    ):
        journal.append(state, occurred_at=clock())
    return journal


def test_cloud_test_partition_opens_exactly_once() -> None:
    store = InMemoryObjectStore()
    identity = _identity(EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    journal = _frozen_journal(store, identity)

    record = open_cloud_test_partition(
        store,
        identity=identity,
        journal=journal,
        preconditions=_preconditions(),
        opened_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    assert record.run_id == identity.run_id
    assert is_cloud_test_partition_opened(
        store, run_id=identity.run_id, evidence_class=identity.evidence_class
    )

    with pytest.raises(BridgeTransformError) as excinfo:
        open_cloud_test_partition(
            store,
            identity=identity,
            journal=journal,
            preconditions=_preconditions(),
            opened_at=datetime(2026, 8, 3, tzinfo=UTC),
        )
    assert excinfo.value.failures[0].code in {
        "TEST_PARTITION_ALREADY_OPENED",
        "TEST_OPENED_BEFORE_CHECKPOINT_FROZEN",
    }


def test_cloud_gate_requires_checkpoint_frozen() -> None:
    store = InMemoryObjectStore()
    identity = _identity(EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    journal = CloudJournal(
        store,
        run_id=identity.run_id,
        experiment_hash=identity.active_experiment_sha256,
        evidence_class=identity.evidence_class,
    )
    journal.append(Phase2State.STAGE_C_TRAINED, occurred_at=datetime(2026, 8, 1, tzinfo=UTC))
    with pytest.raises(BridgeTransformError) as excinfo:
        open_cloud_test_partition(
            store,
            identity=identity,
            journal=journal,
            preconditions=_preconditions(),
            opened_at=datetime(2026, 8, 2, tzinfo=UTC),
        )
    assert excinfo.value.failures[0].code == "TEST_OPENED_BEFORE_CHECKPOINT_FROZEN"


def test_cloud_gate_rejects_unmet_preconditions() -> None:
    store = InMemoryObjectStore()
    identity = _identity(EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    journal = _frozen_journal(store, identity)
    with pytest.raises(BridgeTransformError) as excinfo:
        open_cloud_test_partition(
            store,
            identity=identity,
            journal=journal,
            preconditions=_preconditions(stage_b_passed=False),
            opened_at=datetime(2026, 8, 2, tzinfo=UTC),
        )
    assert excinfo.value.failures[0].code == "TEST_OPENING_PRECONDITION_UNMET"


def test_synthetic_gate_cannot_touch_the_real_namespace() -> None:
    store = InMemoryObjectStore()
    identity = _identity(EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    journal = _frozen_journal(store, identity)
    open_cloud_test_partition(
        store,
        identity=identity,
        journal=journal,
        preconditions=_preconditions(),
        opened_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    assert not is_cloud_test_partition_opened(
        store, run_id=identity.run_id, evidence_class=EvidenceClass.REAL_PHASE2
    )
    assert cloud_test_opening_key(identity.run_id, EvidenceClass.REAL_PHASE2).startswith(
        "openalpha/bridge-phase2"
    )


# -------------------------------------------------------------- identity


def test_cloud_image_change_changes_run_identity() -> None:
    base = _identity(EvidenceClass.REAL_PHASE2)
    changed = CloudRunIdentity.derive(
        source_commit=COMMIT,
        provider_identity="yahoo_finance/yfinance==1.5.2",
        kronos_revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        cloud_image_digest="sha256:DIFFERENT",
        modal_app_version="1.0.0",
        dependency_lock_sha256="b" * 64,
        object_store_namespace="openalpha-artifacts",
        operator="tester",
        evidence_class=EvidenceClass.REAL_PHASE2,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    assert base.run_id != changed.run_id


def test_idempotency_conflict_is_rejected() -> None:
    existing = _identity(EvidenceClass.REAL_PHASE2)
    conflicting = CloudRunIdentity.derive(
        source_commit="f" * 40,
        provider_identity="yahoo_finance/yfinance==1.5.2",
        kronos_revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        cloud_image_digest="sha256:fakeimagedigest",
        modal_app_version="1.0.0",
        dependency_lock_sha256="b" * 64,
        object_store_namespace="openalpha-artifacts",
        operator="tester",
        evidence_class=EvidenceClass.REAL_PHASE2,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        assert_idempotent_match(existing, conflicting)
    assert excinfo.value.failures[0].code == "IDEMPOTENCY_KEY_CONFLICT"


# ------------------------------------------------------------------- auth


def test_bearer_token_authentication() -> None:
    auth = TokenAuthenticator("correct-token-value-1234")
    assert auth.authenticate("Bearer correct-token-value-1234").authenticated

    for header in (None, "", "Basic abc", "Bearer wrong-token-value-9999"):
        with pytest.raises(BridgeTransformError) as excinfo:
            auth.authenticate(header)
        assert excinfo.value.failures[0].code == "UNAUTHORIZED"


def test_unconfigured_authenticator_rejects_everything() -> None:
    with pytest.raises(BridgeTransformError):
        TokenAuthenticator(None).authenticate("Bearer anything")


# --------------------------------------------------------- control service


def test_create_run_returns_immediately_with_a_run_id() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    response = _service(store, backend).create_run(_create())

    assert response.run_id.startswith("syn_")
    assert response.state is Phase2State.CREATED
    assert response.cloud_execution_id == f"fc-syn-{response.run_id}"
    assert response.status_url.endswith(f"/runs/{response.run_id}")
    assert backend.synthetic_spawns == [response.run_id]
    assert not backend.real_spawns


def test_real_run_requires_explicit_confirmation() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    request = _create(ExecutionMode.REAL).model_copy(update={"confirm_real_evidence": False})
    with pytest.raises(BridgeTransformError) as excinfo:
        _service(store, backend).create_run(request)
    assert excinfo.value.failures[0].code == "REAL_EVIDENCE_NOT_CONFIRMED"
    assert not backend.real_spawns


def test_unknown_experiment_hash_is_rejected() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    with pytest.raises(BridgeTransformError) as excinfo:
        _service(store, backend).create_run(_create(experiment_hash="0" * 64))
    assert excinfo.value.failures[0].code == "UNKNOWN_EXPERIMENT_HASH"


def test_unapproved_source_commit_is_rejected() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    config = ServiceConfig(
        base_url="https://openalpha.example",
        object_store_namespace="ns",
        dependency_lock_sha256="b" * 64,
        kronos_revision="rev",
        provider_identity="yahoo_finance/yfinance==1.5.2",
        approved_source_commits=frozenset({"b" * 40}),
    )
    service = Phase2ControlService(store=store, backend=backend, config=config, clock=_Clock())
    with pytest.raises(BridgeTransformError) as excinfo:
        service.create_run(_create())
    assert excinfo.value.failures[0].code == "UNAPPROVED_SOURCE_COMMIT"


def test_idempotent_create_returns_the_same_run() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    service = _service(store, backend)
    first = service.create_run(_create(idempotency_key="stable-key-0001"))
    second = service.create_run(_create(idempotency_key="stable-key-0001"))

    assert first.run_id == second.run_id
    assert second.idempotent_replay
    assert len(backend.synthetic_spawns) == 1


def test_idempotent_key_with_conflicting_inputs_fails() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    service = _service(store, backend)
    service.create_run(_create(idempotency_key="stable-key-0002"))
    with pytest.raises(BridgeTransformError) as excinfo:
        service.create_run(_create(idempotency_key="stable-key-0002", operator="someone-else"))
    assert excinfo.value.failures[0].code == "IDEMPOTENCY_KEY_CONFLICT"


def test_run_creation_is_rate_limited() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    config = ServiceConfig(
        base_url="https://openalpha.example",
        object_store_namespace="ns",
        dependency_lock_sha256="b" * 64,
        kronos_revision="rev",
        provider_identity="yahoo_finance/yfinance==1.5.2",
        rate_limit_per_hour=2,
    )
    service = Phase2ControlService(store=store, backend=backend, config=config, clock=_Clock())
    for index in range(2):
        service.create_run(_create(operator=f"op-{index}"))
    with pytest.raises(BridgeTransformError) as excinfo:
        service.create_run(_create(operator="op-3"))
    assert excinfo.value.failures[0].code == "RATE_LIMIT_EXCEEDED"


def test_status_reports_sealed_test_partition() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    service = _service(store, backend)
    created = service.create_run(_create())
    status = service.get_run(created.run_id)

    assert status.run_id == created.run_id
    assert status.test_sealed
    assert not status.test_partition_opened
    assert status.final_conclusion is None
    assert status.cloud_execution_status == "running"


def test_status_for_unknown_run_fails() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    with pytest.raises(BridgeTransformError) as excinfo:
        _service(store, backend).get_run("syn_missing")
    assert excinfo.value.failures[0].code == "RUN_NOT_FOUND"


def test_resume_is_rejected_for_a_running_state() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    service = _service(store, backend)
    created = service.create_run(_create())
    with pytest.raises(BridgeTransformError) as excinfo:
        service.resume_run(created.run_id, ResumeRunRequest(operator="tester"))
    assert excinfo.value.failures[0].code == "RUN_NOT_RESUMABLE"


def test_cancel_records_before_terminating() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    service = _service(store, backend)
    created = service.create_run(_create())
    status = service.cancel_run(
        created.run_id, CancelRunRequest(operator="tester", reason="no longer needed")
    )
    assert status.state is Phase2State.BLOCKED
    assert backend.cancelled == [created.cloud_execution_id]


def test_logs_never_expose_secrets() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    service = _service(store, backend)
    created = service.create_run(_create())
    register_secret("log-leak-secret-abcdef")
    service.cancel_run(
        created.run_id,
        CancelRunRequest(operator="tester", reason="failed with log-leak-secret-abcdef"),
    )
    logs = service.get_logs(created.run_id)
    assert all("log-leak-secret-abcdef" not in entry.message for entry in logs.entries)


def test_artifact_manifest_excludes_private_objects() -> None:
    store, backend = InMemoryObjectStore(), FakeComputeBackend()
    service = _service(store, backend)
    created = service.create_run(_create())

    prefix = run_prefix(created.run_id, EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    for key in (
        f"{prefix}/final/gate_table.json",
        f"{prefix}/checkpoint/weights/bridge.safetensors",
        f"{prefix}/features/shard.npz",
    ):
        store.put_immutable(key, b"{}", _metadata())

    manifest = service.list_artifacts(created.run_id)
    keys = {entry.key for entry in manifest.artifacts}
    assert f"{prefix}/final/gate_table.json" in keys
    assert not any("weights" in key or key.endswith(".npz") for key in keys)


# --------------------------------------------- synthetic cloud end-to-end


def _runner(store: InMemoryObjectStore, tmp_path: Path) -> CloudRunner:
    identity = _identity(EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    config = Phase2Config(
        run_directory=tmp_path / "run",
        cache_directory=tmp_path / "cache",
        research_root=RESEARCH_ROOT,
        repository_root=REPOSITORY_ROOT,
        evidence_class=EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
        provider_mode=ProviderMode.FAKE,
        kronos_mode=KronosMode.FAKE,
        dry_run=True,
        require_accelerator=False,
        training_symbols=("SPY", "QQQ"),
        unseen_symbols=("IWM",),
    )
    report = _real_stage_a_for(identity, tmp_path / "cache", tmp_path)
    return CloudRunner(
        store=store,
        identity=identity,
        pipeline_config=config,
        provider=DeterministicFakeProvider(),
        kronos=DeterministicFakeKronosBackend(),
        training_backend=NumpyTrainingBackend(),
        guards=ResourceGuards(),
        clock=_Clock(),
        stage_a_report=report,
    )


def _measurements() -> dict[str, float | str | None]:
    return {
        "structurally_invalid_candle_fraction": 0.0,
        "token_identifier_parity_fraction": 1.0,
        "frozen_weight_hash_parity_fraction": 1.0,
        "post_output_projection": "false",
        "high_low_range_mae_projection_ratio": 0.80,
        "high_low_range_paired_bootstrap_upper": -0.01,
        "full_ohlc_mae_official_multiplier": 1.0,
        "close_mae_official_multiplier": 1.0,
        "close_return_mae_official_multiplier": 1.0,
        "trainable_parameter_count": 17_605,
        "checkpoint_size_bytes": 100_000,
        "median_latency_official_multiplier": 1.2,
        "p95_incremental_latency_ms": 10.0,
        "incremental_peak_memory_bytes": 1_000_000,
        "canonical_hash_match_fraction": 1.0,
    }


def test_synthetic_cloud_run_reaches_finalized(tmp_path: Path) -> None:
    """SYNTHETIC CLOUD PIPELINE VALIDATION - NOT EMPIRICAL EVIDENCE."""
    store = InMemoryObjectStore()
    runner = _runner(store, tmp_path)
    result = runner.execute(
        holder="worker-1", confirm_open_test_partition=True, measurements=_measurements()
    )

    assert result.final_state is Phase2State.FINALIZED
    assert result.conclusion is TerminalConclusion.BRIDGE_2K_FEASIBLE

    identity = _identity(EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    journal = CloudJournal(
        store,
        run_id=identity.run_id,
        experiment_hash=identity.active_experiment_sha256,
        evidence_class=identity.evidence_class,
    )
    assert journal.verify_chain()
    assert journal.current_state() is Phase2State.FINALIZED

    prefix = run_prefix(identity.run_id, identity.evidence_class)
    assert store.exists(f"{prefix}/final/gate_table.json")
    # The real namespace is never touched by a synthetic run.
    assert not store.list_keys("openalpha/bridge-phase2/")


def test_synthetic_cloud_run_releases_its_lease(tmp_path: Path) -> None:
    store = InMemoryObjectStore()
    runner = _runner(store, tmp_path)
    runner.execute(
        holder="worker-1", confirm_open_test_partition=True, measurements=_measurements()
    )
    identity = _identity(EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    assert (
        current_lease(store, run_id=identity.run_id, evidence_class=identity.evidence_class) is None
    )


def test_unconfirmed_test_opening_blocks_before_the_gate(tmp_path: Path) -> None:
    store = InMemoryObjectStore()
    runner = _runner(store, tmp_path)
    result = runner.execute(holder="worker-1", confirm_open_test_partition=False)

    assert result.final_state is Phase2State.BLOCKED
    assert result.blocker_code == "TEST_OPENING_NOT_CONFIRMED"
    identity = _identity(EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    assert not is_cloud_test_partition_opened(
        store, run_id=identity.run_id, evidence_class=identity.evidence_class
    )


def test_interrupted_cloud_run_resumes_from_verified_state(tmp_path: Path) -> None:
    store = InMemoryObjectStore()
    identity = _identity(EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)

    # A worker dies mid-run, leaving a journal and a live lease.
    journal = _frozen_journal(store, identity)
    acquire_lease(
        store,
        run_id=identity.run_id,
        experiment_hash=identity.active_experiment_sha256,
        evidence_class=identity.evidence_class,
        holder="dead-worker",
        now=datetime(2026, 8, 1, tzinfo=UTC),
        duration_seconds=1,
    )
    assert journal.current_state() is Phase2State.CHECKPOINT_FROZEN

    # A later worker takes over the expired lease and finishes the gate.
    lease = acquire_lease(
        store,
        run_id=identity.run_id,
        experiment_hash=identity.active_experiment_sha256,
        evidence_class=identity.evidence_class,
        holder="resumed-worker",
        now=datetime(2026, 8, 1, 1, 0, tzinfo=UTC),
    )
    assert lease.holder == "resumed-worker"

    record = open_cloud_test_partition(
        store,
        identity=identity,
        journal=journal,
        preconditions=_preconditions(),
        opened_at=datetime(2026, 8, 1, 2, 0, tzinfo=UTC),
    )
    assert record.prior_ledger_sha256 is not None
    assert journal.verify_chain()


def test_resume_never_recreates_the_test_opening_record(tmp_path: Path) -> None:
    store = InMemoryObjectStore()
    runner = _runner(store, tmp_path)
    runner.execute(
        holder="worker-1", confirm_open_test_partition=True, measurements=_measurements()
    )
    identity = _identity(EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    journal = CloudJournal(
        store,
        run_id=identity.run_id,
        experiment_hash=identity.active_experiment_sha256,
        evidence_class=identity.evidence_class,
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        open_cloud_test_partition(
            store,
            identity=identity,
            journal=journal,
            preconditions=_preconditions(),
            opened_at=datetime(2026, 8, 5, tzinfo=UTC),
        )
    assert excinfo.value.failures[0].code in {
        "TEST_PARTITION_ALREADY_OPENED",
        "TEST_OPENED_BEFORE_CHECKPOINT_FROZEN",
    }


def test_real_evidence_rejects_fake_components(tmp_path: Path) -> None:
    store = InMemoryObjectStore()
    identity = _identity(EvidenceClass.REAL_PHASE2)
    config = Phase2Config(
        run_directory=tmp_path / "run",
        cache_directory=tmp_path / "cache",
        research_root=RESEARCH_ROOT,
        repository_root=REPOSITORY_ROOT,
        evidence_class=EvidenceClass.REAL_PHASE2,
        provider_mode=ProviderMode.FAKE,
        kronos_mode=KronosMode.FAKE,
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        CloudRunner(
            store=store,
            identity=identity,
            pipeline_config=config,
            provider=DeterministicFakeProvider(),
            kronos=DeterministicFakeKronosBackend(),
            training_backend=NumpyTrainingBackend(),
        )
    assert excinfo.value.failures[0].code == "FAKE_PROVIDER_IN_REAL_RUN"


# ---------------------------------------------------------------- guards


def test_resource_guards_enforce_the_locked_budgets() -> None:
    guards = ResourceGuards()
    assert guards.maximum_gpu_count == 1
    assert guards.maximum_feature_cache_bytes == 10_737_418_240

    with pytest.raises(BridgeTransformError) as excinfo:
        guards.assert_gpu_hours(24.5)
    assert excinfo.value.failures[0].code == "GPU_HOUR_BUDGET_EXCEEDED"

    with pytest.raises(BridgeTransformError) as excinfo:
        guards.assert_provider_requests(10_000)
    assert excinfo.value.failures[0].code == "PROVIDER_REQUEST_BUDGET_EXCEEDED"


# ------------------------------- evidence-class storage roots (item 3)


def test_the_three_evidence_roots_are_pairwise_disjoint() -> None:
    from openalpha_bridge.cloud.objectstore import EVIDENCE_ROOTS

    assert set(EVIDENCE_ROOTS) == set(EvidenceClass)
    roots = list(EVIDENCE_ROOTS.values())
    assert len(set(roots)) == len(roots)
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            assert not first.startswith(second)
            assert not second.startswith(first)


def test_the_canary_root_is_neither_real_nor_synthetic() -> None:
    from openalpha_bridge.cloud.objectstore import run_prefix

    canary = run_prefix("canary_0badc0de", EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY)
    real = run_prefix("canary_0badc0de", EvidenceClass.REAL_PHASE2)
    synthetic = run_prefix("canary_0badc0de", EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    assert canary != real and canary != synthetic
    assert canary.startswith("openalpha-compatibility/")


def test_every_evidence_class_has_an_explicit_root() -> None:
    """No else-branch may silently map an unmapped class to synthetic."""
    from openalpha_bridge.cloud.objectstore import EVIDENCE_ROOTS

    for member in EvidenceClass:
        assert member in EVIDENCE_ROOTS, f"{member.value} has no explicit storage root"
