from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _runner() -> ModuleType:
    path = Path("scripts/run_sentinel_v1_1.py")
    spec = importlib.util.spec_from_file_location("run_sentinel_v1_1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Sentinel v1.1 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_exposes_only_locked_operations_and_no_scope_override() -> None:
    runner = _runner()
    parser = runner.build_parser()

    for operation in (
        "roundtrip",
        "audit",
        "canary",
        "classify",
        "conditional",
        "analyze",
        "report",
        "verify",
    ):
        assert parser.parse_args([operation]).operation == operation
    with pytest.raises(SystemExit):
        parser.parse_args(["audit", "--cutoff", "2025-07-01"])


def test_immutable_write_is_idempotent_but_rejects_replacement(
    tmp_path: Path,
) -> None:
    runner = _runner()
    path = tmp_path / "record.json"

    runner._write_immutable(path, b'{"value":1}')
    runner._write_immutable(path, b'{"value":1}')

    with pytest.raises(FileExistsError, match="immutable"):
        runner._write_immutable(path, b'{"value":2}')


def test_outcome_resolution_requires_a_verified_forecast_seal(
    tmp_path: Path,
) -> None:
    runner = _runner()
    origin_root = tmp_path / "origin"
    origin_root.mkdir()

    with pytest.raises(ValueError, match="forecast seal"):
        runner._require_forecast_seal(origin_root)

    forecast = runner._canonical_json_bytes({"outcome_accessed": False})
    runner._write_immutable(origin_root / "forecast.json", forecast)
    bad_seal = runner._canonical_json_bytes(
        {"forecast_sha256": "0" * 64, "sealed_before_outcome_access": True}
    )
    runner._write_immutable(origin_root / "forecast-seal.json", bad_seal)
    with pytest.raises(ValueError, match="hash"):
        runner._require_forecast_seal(origin_root)


def test_public_payload_scan_rejects_raw_market_rows_and_weights() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="forbidden"):
        runner._policy_scan(
            Path("research/sentinel-v1_1/results.json"),
            {"observations": [{"open": 1.0}]},
        )
    with pytest.raises(ValueError, match="forbidden"):
        runner._policy_scan(
            Path("research/sentinel-v1_1/model.safetensors"),
            {"status": "bad"},
        )


def test_report_and_verify_operations_are_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()

    def fail_provider() -> None:
        raise AssertionError("offline operation constructed provider")

    monkeypatch.setattr(runner, "default_yfinance_provider", fail_provider)
    analysis = {
        "claim_boundary": "DEVELOPMENT COMPATIBILITY RESEARCH - NOT HOLDOUT EVIDENCE",
        "root_cause": "MIXED_OR_UNRESOLVED",
        "holdout_accessed": False,
    }
    (tmp_path / "analysis.json").write_text(
        json.dumps(analysis),
        encoding="utf-8",
    )
    report = runner.report_state(tmp_path)

    assert "NOT HOLDOUT EVIDENCE" in report
    assert "MIXED_OR_UNRESOLVED" in report


def test_roundtrip_request_is_fixed_and_public_result_drops_token_pairs() -> None:
    runner = _runner()
    windows = [
        {
            "window_id": f"{symbol}-{index + 1}",
            "symbol": symbol,
            "observations": [
                {
                    "timestamp": "2024-06-28T00:00:00+00:00",
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.0,
                    "volume": 1.0,
                }
                for _ in range(512)
            ],
        }
        for symbol in runner.SUPPORT_SYMBOLS
        for index in range(3)
    ]
    request = runner._build_roundtrip_worker_request(
        tokenizer_repository="NeoQuasar/Kronos-Tokenizer-2k",
        tokenizer_revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        windows=windows,
    )

    assert request["operation"] == "roundtrip"
    assert len(request["windows"]) == 30
    response = {
        "status": "success",
        "token_pairs": [{"coarse": 1, "fine": 2}],
        "token_pairs_sha256": "a" * 64,
        "windows": [],
    }
    compact = runner._compact_roundtrip_response(response)
    assert "token_pairs" not in compact
    assert compact["token_pairs_sha256"] == "a" * 64


def test_main_dispatches_roundtrip_without_scope_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()
    monkeypatch.setattr(
        runner,
        "run_roundtrip",
        lambda: {"operation": "roundtrip", "holdout_accessed": False},
    )

    assert runner.main(["roundtrip"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "holdout_accessed": False,
        "operation": "roundtrip",
    }


def test_compatibility_request_uses_only_locked_scope() -> None:
    runner = _runner()
    origin = runner.locked_v1_1_origins()[0]
    observations = [
        {
            "timestamp": "2024-07-05T00:00:00+00:00",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1.0,
        }
        for _ in range(512)
    ]
    request = runner._build_compatibility_request(
        origin=origin,
        seed=1729,
        observations=observations,
        support={
            "coarse_counts": {"1": 3},
            "fine_counts": {"2": 3},
            "pair_counts": {"1:2": 3},
        },
    )

    assert request["operation"] == "compatibility"
    assert request["model_repository"] == "NeoQuasar/Kronos-mini"
    assert request["tokenizer_repository"] == "NeoQuasar/Kronos-Tokenizer-2k"
    assert request["candidate_budgets"] == [64, 256, 1024]
    assert request["context_length"] == 512
    assert request["forecast_horizon"] == 5
    assert request["sampling_seed"] == 1729
    assert max(request["forecast_sessions"]) < "2025-07-01"


def test_compatibility_summary_separates_support_and_truncation() -> None:
    runner = _runner()
    rows = [
        {
            "asset": "SPY",
            "step": 1,
            "raw_valid": False,
            "support": {"exact_pair_count": 0},
            "probability_mass": [
                {
                    "candidate_budget": 1024,
                    "considered_probability_mass": 0.8,
                    "valid_probability_mass_lower": 0.2,
                    "valid_probability_mass_upper": 0.4,
                    "supported_probability_mass_lower": 0.1,
                    "supported_probability_mass_upper": 0.3,
                    "valid_supported_probability_mass_lower": 0.05,
                    "valid_supported_probability_mass_upper": 0.25,
                }
            ],
        },
        {
            "asset": "QQQ",
            "step": 2,
            "raw_valid": True,
            "support": {"exact_pair_count": 3},
            "probability_mass": [
                {
                    "candidate_budget": 1024,
                    "considered_probability_mass": 0.9,
                    "valid_probability_mass_lower": 0.6,
                    "valid_probability_mass_upper": 0.7,
                    "supported_probability_mass_lower": 0.5,
                    "supported_probability_mass_upper": 0.6,
                    "valid_supported_probability_mass_lower": 0.4,
                    "valid_supported_probability_mass_upper": 0.5,
                }
            ],
        },
    ]

    summary = runner._summarize_compatibility_steps(rows)

    assert summary["raw_invalid_candle_fraction"] == 0.5
    assert summary["invalid_raw_unsupported_fraction"] == 1.0
    assert summary["budget_1024"]["median_valid_mass_lower"] == 0.4
    assert summary["budget_1024"]["median_valid_mass_upper"] == 0.55


def test_tokenizer_defect_gate_stops_canary_and_new_decoder() -> None:
    runner = _runner()
    roundtrip = {
        "tokenizers": {
            "tokenizer_2k": {
                "summary": {"pooled": {"material_defect": True}}
            },
            "tokenizer_base": {
                "summary": {"pooled": {"material_defect": True}}
            },
        }
    }

    canary = runner._canary_disposition(roundtrip)
    conditional = runner._conditional_disposition(
        {
            "root_cause": "TOKENIZER_CONSTRAINT_DEFECT",
            "conditional_method_authorized": False,
        }
    )

    assert canary["status"] == "not_run_tokenizer_defect_gate"
    assert conditional["status"] == "not_authorized"
    assert conditional["method_executed"] is False
