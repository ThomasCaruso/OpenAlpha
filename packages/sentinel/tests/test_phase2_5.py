import json
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TypedDict, cast

import pytest
from openalpha_sentinel.contracts import OHLCVObservation
from openalpha_sentinel.phase2_5 import (
    Phase25Paths,
    average_path_payloads,
    build_path_comparison,
    classify_phase2_5,
    load_private_receipt,
    make_direct_request,
    next_xnys_sessions,
    path_validity_from_payload,
    phase2_fingerprint,
    publish_private_receipt,
    raw_log_return,
    run_direct_worker,
)


class _ClassificationFacts(TypedDict):
    direct_official_invalid: bool
    openalpha_output_invalid: bool
    direct_equals_openalpha: bool
    repeatable: bool
    canary_invalid_path_count: int


def _path(final_close: float = 105.0) -> list[dict[str, object]]:
    sessions = next_xnys_sessions(date(2024, 7, 5))
    return [
        {
            "session": session.isoformat(),
            "timestamp": f"{session.isoformat()}T00:00:00+00:00",
            "open": 100.0 + index,
            "high": 102.0 + index,
            "low": 99.0 + index,
            "close": final_close if index == 4 else 101.0 + index,
            "volume": 1_000.0 + index,
        }
        for index, session in enumerate(sessions)
    ]


def test_phase2_and_phase2_5_roots_must_be_disjoint(tmp_path: Path) -> None:
    phase2 = tmp_path / "phase2"
    phase2.mkdir()

    paths = Phase25Paths(phase2_root=phase2, phase2_5_root=tmp_path / "phase2_5")

    assert paths.phase2_root == phase2.resolve()
    with pytest.raises(ValueError, match="disjoint"):
        Phase25Paths(phase2_root=phase2, phase2_5_root=phase2 / "audit")


def test_phase2_fingerprint_covers_descriptors_and_artifact_inventory(tmp_path: Path) -> None:
    (tmp_path / "artifacts" / "sha256" / "aa").mkdir(parents=True)
    (tmp_path / "creation.json").write_text("creation", encoding="utf-8")
    (tmp_path / "resolved.json").write_text("resolved", encoding="utf-8")
    artifact = tmp_path / "artifacts" / "sha256" / "aa" / ("a" * 64)
    artifact.write_text("artifact", encoding="utf-8")

    first = phase2_fingerprint(tmp_path)
    second = phase2_fingerprint(tmp_path)

    assert first == second
    assert set(first["descriptors"]) == {"creation.json", "resolved.json"}
    assert len(first["artifacts"]) == 1
    assert first["artifacts"][0]["relative_path"].endswith("a" * 64)


def test_path_comparison_disables_exact_claim_when_input_hash_changed() -> None:
    result = build_path_comparison(
        direct_path=_path(),
        provider_path=_path(),
        sealed_path=_path(),
        observed_input_sha256="a" * 64,
        sealed_input_sha256="b" * 64,
    )

    assert result["exact_comparison_permitted"] is False
    assert result["direct_equals_provider"] is True
    assert result["direct_equals_sealed"] is None
    assert result["limitation"] == "NORMALIZED_INPUT_HASH_MISMATCH"


def test_path_comparison_is_named_and_value_exact() -> None:
    changed = _path(final_close=106.0)

    result = build_path_comparison(
        direct_path=_path(),
        provider_path=_path(),
        sealed_path=changed,
        observed_input_sha256="a" * 64,
        sealed_input_sha256="a" * 64,
    )

    assert result["exact_comparison_permitted"] is True
    assert result["direct_equals_provider"] is True
    assert result["direct_equals_sealed"] is False
    assert result["first_direct_sealed_difference"] == {
        "field": "close",
        "step": 5,
        "direct": 105.0,
        "other": 106.0,
    }


def test_private_receipts_are_content_addressed_outside_phase2(tmp_path: Path) -> None:
    paths = Phase25Paths(
        phase2_root=tmp_path / "phase2",
        phase2_5_root=tmp_path / "phase2_5",
    )
    paths.phase2_root.mkdir()

    first = publish_private_receipt(paths, {"operation": "trace", "value": 1})
    second = publish_private_receipt(paths, {"value": 1, "operation": "trace"})

    assert first == second
    assert first.sha256 == "0eb13074cfd199b4b8ab2e0d952615ad2d05e5fc265e19c576b96d7198491e63"
    assert paths.phase2_5_root in first.path.parents
    assert paths.phase2_root not in first.path.parents

    loaded = load_private_receipt(paths, first.sha256)
    assert loaded == {"operation": "trace", "value": 1}


@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (
            {
                "direct_official_invalid": False,
                "openalpha_output_invalid": True,
                "direct_equals_openalpha": False,
                "repeatable": False,
                "canary_invalid_path_count": 0,
            },
            "OPENALPHA_INTEGRATION_BUG",
        ),
        (
            {
                "direct_official_invalid": True,
                "openalpha_output_invalid": True,
                "direct_equals_openalpha": True,
                "repeatable": True,
                "canary_invalid_path_count": 4,
            },
            "OFFICIAL_RAW_PATH_STRUCTURAL_INVALIDITY_CONFIRMED",
        ),
        (
            {
                "direct_official_invalid": True,
                "openalpha_output_invalid": True,
                "direct_equals_openalpha": True,
                "repeatable": True,
                "canary_invalid_path_count": 0,
            },
            "NONRECURRING_EDGE_CASE",
        ),
    ),
)
def test_phase2_5_classification_is_structural_not_accuracy(
    facts: _ClassificationFacts, expected: str
) -> None:
    assert classify_phase2_5(**facts) == expected


def test_phase2_5_classification_is_ambiguous_without_boundary_isolation() -> None:
    assert (
        classify_phase2_5(
            direct_official_invalid=True,
            openalpha_output_invalid=True,
            direct_equals_openalpha=False,
            repeatable=True,
            canary_invalid_path_count=2,
        )
        == "AMBIGUOUS"
    )


def test_next_sessions_use_xnys_calendar() -> None:
    assert next_xnys_sessions(date(2024, 7, 5)) == (
        date(2024, 7, 8),
        date(2024, 7, 9),
        date(2024, 7, 10),
        date(2024, 7, 11),
        date(2024, 7, 12),
    )


def test_direct_request_is_named_causal_and_bounded() -> None:
    observations = tuple(
        OHLCVObservation(
            session=date(2024, 7, 5) - timedelta(days=127 - index),
            timestamp=datetime.combine(
                date(2024, 7, 5) - timedelta(days=127 - index),
                datetime.min.time(),
                tzinfo=UTC,
            ),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1_000.0,
        )
        for index in range(128)
    )

    request = make_direct_request(
        audit_case="phase2_spy_20240705",
        observations=observations,
        forecast_timestamps=next_xnys_sessions(date(2024, 7, 5)),
        context_length=128,
        seed=1729,
        sample_count=1,
        trace=True,
    )

    request_observations = cast(list[dict[str, object]], request["observations"])
    forecast_sessions = cast(list[str], request["forecast_sessions"])
    assert request_observations[0].keys() == {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    assert cast(str, request_observations[-1]["timestamp"]).startswith("2024-07-05")
    assert forecast_sessions[0] == "2024-07-08"


def test_path_validity_and_averaging_use_named_values() -> None:
    first = _path(final_close=105.0)
    second = _path(final_close=107.0)

    averaged = average_path_payloads((first, second))
    validity = path_validity_from_payload(
        path_id="average",
        path=averaged,
        expected_timestamps=next_xnys_sessions(date(2024, 7, 5)),
        cutoff_close=100.0,
        cutoff_volume=1_000.0,
    )

    assert averaged[-1]["close"] == 106.0
    assert validity.valid is True
    assert raw_log_return(averaged, cutoff_close=100.0) == pytest.approx(0.058268908123975824)


def test_direct_worker_runner_preserves_typed_failure_and_success(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def success_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        payload = json.loads(str(kwargs["input"]))
        output = {
            "status": "success",
            "environment": {"device": "cpu"},
            "responses": [{"status": "success", "path": _path()}],
        }
        assert payload["schema_version"] == "sentinel-kronos-phase2_5-worker-v0"
        return subprocess.CompletedProcess(command, 0, json.dumps(output), "")

    result = run_direct_worker(
        requests=[{"audit_case": "phase2_spy_20240705"}],
        inference_python=tmp_path / "python.exe",
        worker_script=tmp_path / "worker.py",
        source_path=tmp_path / "source",
        cache_path=tmp_path / "cache",
        trace_helper_path=tmp_path / "trace.py",
        runner=success_runner,
    )

    assert result["responses"][0]["status"] == "success"
    assert calls[0][1]["timeout"] == 1_800.0
    assert calls[0][1]["capture_output"] is True

    def failure_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output = {
            "status": "failure",
            "failure": {"code": "WORKER_FAILED", "message": "model failed"},
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(output), "")

    with pytest.raises(RuntimeError, match="model failed"):
        run_direct_worker(
            requests=[{"audit_case": "phase2_spy_20240705"}],
            inference_python=tmp_path / "python.exe",
            worker_script=tmp_path / "worker.py",
            source_path=tmp_path / "source",
            cache_path=tmp_path / "cache",
            trace_helper_path=tmp_path / "trace.py",
            runner=failure_runner,
        )
