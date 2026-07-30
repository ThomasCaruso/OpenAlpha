from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

from openalpha_research import LocalArtifactStore, MethodologyStatus

from .evidence import (
    CreationEvidence,
    EvidenceContext,
    append_outcome_and_complete,
    verify_creation,
    verify_resolved_evidence,
)
from .market_data import MarketDataRequest, default_yfinance_provider
from .reporting import render_phase_2_audit


def resolve_phase_2_origin(
    *,
    repository_root: Path,
    state_root: Path,
) -> dict[str, object]:
    control = json.loads((state_root / "creation.json").read_text(encoding="utf-8"))
    context = EvidenceContext.model_validate_json(json.dumps(control["evidence_context"]))
    creation = CreationEvidence.model_validate_json(json.dumps(control["creation"]))
    store = LocalArtifactStore(state_root / "artifacts")
    if not verify_creation(store, creation):
        raise ValueError("forecast creation seal failed before outcome access")
    forecast = json.loads(store.read_bytes(creation.forecast_ref))

    def outcome_loader() -> dict[str, object]:
        snapshot = default_yfinance_provider().fetch(
            MarketDataRequest(
                purpose="outcome",
                symbol="SPY",
                start_inclusive=date(2024, 7, 6),
                end_exclusive=date(2024, 7, 13),
                cutoff=date(2024, 7, 12),
                minimum_sessions=5,
            )
        )
        cutoff_close = float(forecast["cutoff_close"])
        predicted_return = float(forecast["canonical_predicted_log_return"])
        realized_return = math.log(snapshot.observations[-1].close / cutoff_close)
        kronos_error = abs(predicted_return - realized_return)
        baseline_error = abs(realized_return)
        if kronos_error < baseline_error:
            closer = "KRONOS"
        elif baseline_error < kronos_error:
            closer = "BASELINE"
        else:
            closer = "TIE"
        canonical_path = [float(value) for value in forecast["canonical_close_path"]]
        realized_path = [row.close for row in snapshot.observations]
        return {
            "schema_version": "1.0",
            "forecast_id": creation.seal_ref.sha256,
            "resolved_at": datetime.now(UTC).isoformat(),
            "provider": snapshot.model_dump(mode="json", exclude={"observations"}),
            "outcome_sessions": [row.session.isoformat() for row in snapshot.observations],
            "outcome_ohlcv": [row.model_dump(mode="json") for row in snapshot.observations],
            "realized_log_return": realized_return,
            "canonical_predicted_log_return": predicted_return,
            "kronos_absolute_return_error": kronos_error,
            "direction_correct": _direction(predicted_return) == _direction(realized_return),
            "baseline_predicted_log_return": 0.0,
            "baseline_absolute_return_error": baseline_error,
            "canonical_close_path_mae": sum(
                abs(predicted - realized)
                for predicted, realized in zip(canonical_path, realized_path)
            )
            / 5.0,
            "closer_model": closer,
            "corporate_action_warnings": [
                warning.model_dump(mode="json")
                for warning in snapshot.corporate_action_warnings
            ],
        }

    completed_at = datetime.now(UTC)
    resolved = append_outcome_and_complete(
        store=store,
        context=context,
        creation=creation,
        outcome_loader=outcome_loader,
        methodology_audit={
            "schema_version": "1.0",
            "methodology_status": MethodologyStatus.PASSED_WITH_WARNINGS.value,
            "claim_boundary": "DEVELOPMENT PROOF — NOT EMPIRICAL EVIDENCE",
            "warnings": [
                "ONE_ORIGIN_ONLY",
                "UNOFFICIAL_YAHOO_INTERFACE",
                "NOT_POINT_IN_TIME_DATA",
                "RAW_RETURN_OMITS_DIVIDENDS",
                "NO_CROSS_PROVIDER_VERIFICATION",
                "OFFICIAL_KRONOS_PREDICTOR_DERIVES_INTERNAL_AMOUNT",
                "PRESEAL_IMPLEMENTATION_FAILURE_RETRIED",
            ],
        },
        methodology_status=MethodologyStatus.PASSED_WITH_WARNINGS,
        test_evidence=(
            "uv run pytest packages/sentinel/tests packages/research-core/tests/test_manifest.py",
            "real synthetic Kronos shape validation passed",
        ),
        completed_at=completed_at,
    )
    verified = verify_resolved_evidence(store, resolved)
    if not verified:
        raise ValueError("complete outcome artifact chain failed verification")
    outcome_record = json.loads(store.read_bytes(resolved.outcome_ref))
    outcome = outcome_record["outcome"]
    artifact_hashes = {
        "canonical_spec": creation.canonical_spec_ref.sha256,
        "data_snapshot": creation.data_snapshot_ref.sha256,
        "data_quality": creation.data_quality_ref.sha256,
        "forecast": creation.forecast_ref.sha256,
        "diagnostics": creation.diagnostics_ref.sha256,
        "forecast_seal": creation.seal_ref.sha256,
        "outcome": resolved.outcome_ref.sha256,
        "methodology_audit": resolved.audit_ref.sha256,
        "completed_manifest": resolved.manifest_ref.sha256,
    }
    report = render_phase_2_audit(
        forecast=forecast,
        outcome=outcome,
        forecast_id=creation.seal_ref.sha256,
        manifest_sha256=resolved.manifest_ref.sha256,
        artifact_chain_verified=verified,
        artifact_hashes=artifact_hashes,
    )
    report_path = state_root / "phase2-spy-2024-07-05.md"
    report_path.write_text(report, encoding="utf-8")
    generated_report = (
        repository_root
        / "research"
        / "sentinel-v0"
        / "reports"
        / "generated"
        / "phase2-spy-2024-07-05.md"
    )
    generated_report.parent.mkdir(parents=True, exist_ok=True)
    generated_report.write_text(report, encoding="utf-8")
    resolved_path = state_root / "resolved.json"
    resolved_path.write_text(resolved.model_dump_json(indent=2), encoding="utf-8")
    return {
        "forecast_id": creation.seal_ref.sha256,
        "outcome_ref_sha256": resolved.outcome_ref.sha256,
        "manifest_sha256": resolved.manifest_ref.sha256,
        "artifact_chain_verified": verified,
        "outcome": outcome,
        "artifact_hashes": artifact_hashes,
        "report_path": str(report_path.resolve()),
        "generated_report_path": str(generated_report.resolve()),
        "resolved_receipt_path": str(resolved_path.resolve()),
    }


def _direction(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
