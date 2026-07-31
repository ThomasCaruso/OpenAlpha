import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType


def _load_cli() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "run_sentinel_phase2_5.py"
    spec = importlib.util.spec_from_file_location("phase2_5_cli_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_intraday_expected_timestamps_are_normalized_to_utc() -> None:
    module = _load_cli()

    result = module._audit_timestamps(("2024-07-03T09:55:00", "2024-07-03T10:00:00+00:00"))

    assert result == (
        datetime(2024, 7, 3, 9, 55, tzinfo=UTC),
        datetime(2024, 7, 3, 10, 0, tzinfo=UTC),
    )


def test_repeatability_violation_signature_ignores_only_path_identity() -> None:
    module = _load_cli()
    first = {
        "path_id": "repeat-a",
        "valid": False,
        "violations": [
            {
                "path_id": "repeat-a",
                "step": 2,
                "timestamp": "2024-07-09",
                "code": "HIGH_BELOW_CLOSE",
                "observed_gap": 1.25,
                "normalized_severity": 0.002,
                "existed_before_outcome": True,
            }
        ],
    }
    second = {
        **first,
        "path_id": "repeat-b",
        "violations": [{**first["violations"][0], "path_id": "repeat-b"}],
    }

    assert module._violation_signature(first) == module._violation_signature(second)


def test_report_reconstructs_compact_trace_without_private_observations() -> None:
    module = _load_cli()
    path = [
        {
            "timestamp": "2024-07-08T00:00:00+00:00",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 10.0,
        },
        {
            "timestamp": "2024-07-12T00:00:00+00:00",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 10.0,
        },
    ]
    trace = {
        "snapshot": {"provider": "yahoo_finance", "observations": [{"private": True}]},
        "golden_metadata": {"fixture_sha256": "a" * 64},
        "direct_batch": {
            "environment": {"device": "cpu"},
            "responses": [
                {"trace": {"boundaries": []}},
                {"trace": {"boundaries": [], "source_functions": []}, "path": path},
            ],
        },
        "phase2_fingerprint_before": {"inventory_sha256": "b" * 64},
        "comparison": {"direct_equals_provider": True},
        "validity": {"official_direct": {"valid": False}},
        "provider_environment": {"device": "cpu"},
        "sealed_forecast": {
            "cutoff": "2024-07-05",
            "forecast_sessions": [
                "2024-07-08",
                "2024-07-09",
                "2024-07-10",
                "2024-07-11",
                "2024-07-12",
            ],
        },
        "execution_attempts": [],
    }

    compact_trace, compact_comparison = module._compact_trace_from_receipt(trace, "c" * 64)

    assert "observations" not in compact_trace["provider"]
    assert compact_trace["private_receipt_sha256"] == "c" * 64
    assert compact_comparison["comparison"]["direct_equals_provider"] is True
