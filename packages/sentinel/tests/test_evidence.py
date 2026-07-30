import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from openalpha_research import (
    EnvironmentMetadata,
    LocalArtifactStore,
    MethodologyStatus,
)
from openalpha_sentinel.evidence import (
    EvidenceContext,
    ResolvedEvidence,
    append_outcome_and_complete,
    publish_creation,
    verify_creation,
    verify_resolved_evidence,
)

NOW = datetime(2026, 7, 30, 20, 0, tzinfo=UTC)


def _payload(name: str) -> dict[str, object]:
    return {"schema_version": "1.0", "name": name}


def _context() -> EvidenceContext:
    return EvidenceContext(
        run_id="sentinel_spy_20240705",
        attempt_id="attempt_01",
        code_commit="f" * 40,
        code_dirty=True,
        diff_sha256="d" * 64,
        dependency_lock_sha256="e" * 64,
        environment=EnvironmentMetadata(
            os_name="Windows",
            os_version="11",
            architecture="AMD64",
            python_version="3.13.2",
            node_version=None,
            dependency_lock_sha256="e" * 64,
            container_image=None,
            hardware="CPU",
        ),
    )


def test_creation_seals_forecast_and_diagnostics_before_outcome_access(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    creation = publish_creation(
        store=store,
        context=_context(),
        canonical_spec={
            "schema_version": "1.0",
            "experiment_yaml_sha256": "a" * 64,
            "experiment_yaml_utf8": "schema_version: sentinel-v0.3\n",
        },
        data_snapshot=_payload("data"),
        data_quality=_payload("quality"),
        forecast=_payload("forecast"),
        diagnostics=_payload("diagnostics"),
        created_at=NOW,
    )
    forecast_before = store.read_bytes(creation.forecast_ref)

    assert verify_creation(store, creation)
    assert store.verify(creation.seal_ref)
    assert [event.event_type for event in creation.ledger_events] == [
        "FORECAST_CREATED",
        "DIAGNOSTICS_COMPUTED",
    ]
    assert creation.ledger_events[1].previous_record_sha256 == (
        creation.ledger_events[0].record_sha256
    )

    loader_calls = 0

    def outcome_loader() -> dict[str, object]:
        nonlocal loader_calls
        loader_calls += 1
        assert verify_creation(store, creation)
        return _payload("outcome")

    resolved = append_outcome_and_complete(
        store=store,
        context=_context(),
        creation=creation,
        outcome_loader=outcome_loader,
        methodology_audit={
            "schema_version": "1.0",
            "methodology_status": MethodologyStatus.PASSED_WITH_WARNINGS.value,
        },
        methodology_status=MethodologyStatus.PASSED_WITH_WARNINGS,
        test_evidence=("focused tests passed",),
        completed_at=NOW + timedelta(seconds=10),
    )

    assert loader_calls == 1
    assert store.read_bytes(creation.forecast_ref) == forecast_before
    assert resolved.outcome_event.previous_record_sha256 == (
        creation.ledger_events[-1].record_sha256
    )
    assert verify_resolved_evidence(store, resolved)
    assert store.verify(resolved.manifest_ref)

    outcome_record = json.loads(store.read_bytes(resolved.outcome_ref))
    for field, replacement in (
        ('forecast_seal_sha256', '0' * 64),
        ('ledger_event', {'sequence': 999}),
    ):
        altered = {**outcome_record, field: replacement}
        altered_ref = store.put_bytes(
            json.dumps(
                altered,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            ).encode('utf-8'),
            media_type='application/json',
        )
        altered_evidence = ResolvedEvidence(
            schema_version='sentinel-resolved-evidence-v0',
            creation=resolved.creation,
            outcome_ref=altered_ref,
            audit_ref=resolved.audit_ref,
            outcome_event=resolved.outcome_event,
            manifest_ref=resolved.manifest_ref,
        )

        assert not verify_resolved_evidence(store, altered_evidence)
