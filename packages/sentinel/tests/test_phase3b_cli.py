import json
from pathlib import Path

import pytest

from scripts import run_sentinel_phase3b


def test_phase3b_cli_has_no_holdout_or_date_override() -> None:
    parser = run_sentinel_phase3b.build_parser()
    operation = next(action for action in parser._actions if action.dest == "operation")

    assert tuple(operation.choices or ()) == (
        "preflight",
        "probe",
        "run",
        "analyze",
        "report",
        "verify",
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--cutoff", "2025-07-01"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--period", "holdout"])


def test_offline_operations_do_not_construct_network_or_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_dependencies(*args, **kwargs):
        raise AssertionError("offline operation constructed real dependencies")

    monkeypatch.setattr(run_sentinel_phase3b, "build_real_dependencies", fail_dependencies)
    monkeypatch.setattr(
        run_sentinel_phase3b,
        "analyze_state",
        lambda root: calls.append("analyze") or {"operation": "analyze"},
    )
    monkeypatch.setattr(
        run_sentinel_phase3b,
        "report_state",
        lambda root: calls.append("report") or {"operation": "report"},
    )
    monkeypatch.setattr(
        run_sentinel_phase3b,
        "verify_state",
        lambda root: calls.append("verify") or {"operation": "verify"},
    )

    for operation in ("analyze", "report", "verify"):
        result = run_sentinel_phase3b.dispatch(
            run_sentinel_phase3b.build_parser().parse_args(
                [operation, "--state-root", str(tmp_path)]
            )
        )
        assert result["operation"] == operation
    assert calls == ["analyze", "report", "verify"]


def test_gate_accepts_zero_repeated_paths_as_noncollapsed() -> None:
    base_quality = {
        "close_mae": 0.02,
        "normalized_ohlc_mae": 0.02,
        "high_low_range_mae": 0.01,
    }
    base = {
        "quality": base_quality,
        "diversity": {
            "mean_pairwise_path_distance": 0.02,
            "mean_final_return_variance": 0.001,
            "mean_repeated_path_rate": 0.0,
        },
        "median_duration_ms": 100.0,
        "maximum_peak_working_set_bytes": 100,
        "invalid_returned_path_count": 0,
        "returned_structural_validity_rate": 1.0,
        "hard_failure_count": 0,
        "hard_failure_rate": 0.0,
    }
    methods = {
        "RAW_AUTOREGRESSIVE": base,
        "TERMINAL_PROJECTION": base,
        "STEPWISE_PROJECT_REENCODE": base,
    }

    result = run_sentinel_phase3b._evaluate_continuation_gates(
        method="STEPWISE_PROJECT_REENCODE",
        methods=methods,
        determinism={
            "execution_count": 2,
            "comparisons": {
                "STEPWISE_PROJECT_REENCODE": {"exact_match": True},
            },
        },
    )

    assert result["gates"]["diversity"] is True
    assert result["all_passed"] is True


def test_determinism_payload_excludes_operational_timing_noise() -> None:
    first = {
        "method": "VALID_CANDIDATE_RESAMPLING",
        "status": "success",
        "path_sha256": "a" * 64,
        "steps": [
            {
                "step": 1,
                "selected_token": {"coarse": 2, "fine": 3},
                "candidate_validation_duration_ms": 10.0,
            }
        ],
    }
    second = {
        **first,
        "steps": [
            {
                "step": 1,
                "selected_token": {"coarse": 2, "fine": 3},
                "candidate_validation_duration_ms": 12.0,
            }
        ],
    }

    assert run_sentinel_phase3b._deterministic_method_payload(
        first
    ) == run_sentinel_phase3b._deterministic_method_payload(second)


def test_generation_quality_summarizes_preserved_audit_fields() -> None:
    records = [
        {
            "generation": {
                "steps": [
                    {
                        "step": 2,
                        "intervened": True,
                        "projection_changes": [
                            {"normalized_adjustment": 0.02}
                        ],
                        "raw_token": {"coarse": 1, "fine": 2},
                        "selected_token": {"coarse": 3, "fine": 4},
                        "candidates_considered": 16,
                        "rejection_count": 12,
                        "search_expansion_count": 1,
                        "candidate_validation_duration_ms": 5.0,
                    }
                ]
            },
            "projection_adjustments": None,
            "audit": None,
        },
        {
            "generation": None,
            "projection_adjustments": [
                {"step": 1, "normalized_adjustment": 0.01}
            ],
            "audit": {
                "completed_steps": [],
                "failed_step_audit": {
                    "step": 3,
                    "projection_changes": [
                        {"normalized_adjustment": 0.03}
                    ],
                    "raw_token": {"coarse": 5, "fine": 6},
                    "selected_token": {"coarse": 5, "fine": 6},
                },
            },
        },
    ]

    result = run_sentinel_phase3b._generation_quality_summary(records)

    assert result["intervention_step_counts"] == {"1": 1, "2": 1, "3": 1}
    assert result["projection_adjustment_count"] == 3
    assert result["mean_projection_magnitude"] == pytest.approx(0.02)
    assert result["maximum_projection_magnitude"] == pytest.approx(0.03)
    assert result["reencoded_token_comparison_count"] == 2
    assert result["reencoded_token_change_count"] == 1
    assert result["candidate_validation_total_ms"] == 5.0
    assert result["search_expansion_count"] == 1


def test_report_includes_all_preregistered_metric_families(tmp_path: Path) -> None:
    quality = {
        "normalized_ohlc_mae": 0.01,
        "open_mae": 0.01,
        "high_mae": 0.01,
        "low_mae": 0.01,
        "close_mae": 0.01,
        "high_low_range_mae": 0.01,
        "candle_body_mae": 0.01,
        "upper_wick_mae": 0.01,
        "lower_wick_mae": 0.01,
        "range_direction_accuracy": 0.5,
        "path_shape_distance": 0.01,
        "parkinson_volatility_error": 0.01,
        "cumulative_range_error": 0.01,
        "five_session_close_return_error": 0.01,
        "direction_accuracy": 0.5,
        "error_by_horizon_step": [0.01] * 5,
    }
    method = {
        "success_count": 36,
        "hard_failure_count": 0,
        "returned_structural_validity_rate": 1.0,
        "quality": quality,
        "barriers": {
            level: {
                "positive_touch_correct": 0.5,
                "negative_touch_correct": 0.5,
                "either_touch_correct": 0.5,
            }
            for level in ("0.005", "0.01", "0.02")
        },
        "diversity": {
            "mean_pairwise_path_distance": 0.01,
            "mean_final_return_variance": 0.001,
            "mean_repeated_path_rate": 0.0,
        },
        "generation_quality": {
            "intervention_step_counts": {"1": 2},
            "projection_adjustment_count": 2,
            "mean_projection_magnitude": 0.001,
            "maximum_projection_magnitude": 0.002,
            "reencoded_token_comparison_count": 2,
            "reencoded_token_change_count": 1,
            "reencoded_token_change_rate": 0.5,
            "candidate_validation_total_ms": 10.0,
            "candidate_validation_mean_ms": 5.0,
            "search_expansion_count": 1,
            "paired_raw_path_count": 36,
            "mean_raw_path_distance": 0.01,
            "maximum_raw_path_distance": 0.02,
        },
        "intervention_count": 2,
        "candidate_rejection_rate": 0.5,
        "fallback_rate": 0.0,
        "average_selected_token_rank": 2.0,
        "median_duration_ms": 100.0,
        "median_intervention_overhead_ms": 0.25,
        "maximum_peak_working_set_bytes": 1000,
    }
    methods = {
        name: {**method, "method": name}
        for name in (
            "RAW_AUTOREGRESSIVE",
            "TERMINAL_PROJECTION",
            "STEPWISE_PROJECT_REENCODE",
            "VALID_CANDIDATE_RESAMPLING",
        )
    }
    analysis = {
        "eligible_origins": 12,
        "completed_origins": 12,
        "failed_origins": 0,
        "methods": methods,
        "continuation_gates": {
            name: {"gates": {"determinism": True}}
            for name in ("STEPWISE_PROJECT_REENCODE", "VALID_CANDIDATE_RESAMPLING")
        },
        "zero_return_baseline_mae": 0.02,
        "model_size_canary_status": "not_run_prerequisite_not_met",
        "conclusion": "VALIDITY_SUCCEEDS_QUALITY_DEGRADES",
    }
    (tmp_path / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")

    run_sentinel_phase3b.report_state(tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert "## Full path and range metrics" in report
    assert "## Barrier-event accuracy" in report
    assert "## Diversity and operational metrics" in report
    assert "## Intervention audit" in report
    assert "Error by horizon step" in report


def test_terminal_duration_includes_raw_rollout_and_projection_overhead() -> None:
    resolved = [
        {
            "method_records": [
                {
                    "method": "RAW_AUTOREGRESSIVE",
                    "sampling_seed": 1729,
                    "status": "success",
                    "duration_ms": 100.0,
                },
                {
                    "method": "TERMINAL_PROJECTION",
                    "sampling_seed": 1729,
                    "status": "success",
                    "duration_ms": 0.25,
                },
            ]
        }
    ]

    result = run_sentinel_phase3b._method_duration_summary(
        "TERMINAL_PROJECTION",
        resolved[0]["method_records"],
        resolved,
    )

    assert result["median_duration_ms"] == 100.25
    assert result["median_intervention_overhead_ms"] == 0.25
