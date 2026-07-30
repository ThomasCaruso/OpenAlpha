from openalpha_sentinel.reporting import render_phase_2_audit


def test_phase_2_audit_has_required_claim_boundary_and_evidence_fields() -> None:
    forecast = {
        "claim_boundary": "DEVELOPMENT PROOF — NOT EMPIRICAL EVIDENCE",
        "symbol": "SPY",
        "cutoff": "2024-07-05",
        "forecast_sessions": [
            "2024-07-08",
            "2024-07-09",
            "2024-07-10",
            "2024-07-11",
            "2024-07-12",
        ],
        "provider": {"provider": "yahoo_finance", "client_version": "1.5.2"},
        "model_environment": {
            "model_repository": "NeoQuasar/Kronos-mini",
            "model_revision": "model-revision",
            "tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-2k",
            "tokenizer_revision": "tokenizer-revision",
            "device": "cpu",
            "python_version": "3.11.15",
            "torch_version": "2.13.0+cpu",
            "cache_size_bytes": 32_283_678,
        },
        "reproducibility_probe": {
            "a_sha256": "a",
            "b_sha256": "a",
            "c_sha256": "c",
            "exact_seed_replay_supported": True,
        },
        "individual_paths": [
            {
                "context_length": context,
                "sampling_seed": seed,
                "predicted_log_return": 0.01,
                "inference_duration_ms": 100.0,
                "close_path": [1, 2, 3, 4, 5],
            }
            for context in (128, 256, 512)
            for seed in (1729, 2027, 7919)
        ],
        "canonical_close_path": [1, 2, 3, 4, 5],
        "canonical_predicted_log_return": 0.01,
        "context_summaries": [],
        "diagnostics": [
            {
                "name": "RECENT_MODEL_ERROR",
                "status": "not_computable",
                "value": None,
                "reason": "NO_PRIOR_RESOLVED_FORECASTS",
            }
        ],
        "total_runtime_seconds": 4.0,
        "request_failures": [],
        "request_retries": 0,
        "limitations": ["ONE_ORIGIN_ONLY"],
    }
    outcome = {
        "realized_log_return": 0.02,
        "kronos_absolute_return_error": 0.01,
        "baseline_absolute_return_error": 0.02,
        "direction_correct": True,
        "closer_model": "KRONOS",
    }

    report = render_phase_2_audit(
        forecast=forecast,
        outcome=outcome,
        forecast_id="forecast-sha",
        manifest_sha256="manifest-sha",
        artifact_chain_verified=True,
    )

    for expected in (
        "DEVELOPMENT PROOF — NOT EMPIRICAL EVIDENCE",
        "2024-07-05",
        "yahoo_finance",
        "NeoQuasar/Kronos-mini",
        "Reproducibility probe",
        "Nine individual predicted returns",
        "Canonical 512-context forecast",
        "RECENT_MODEL_ERROR",
        "NO_PRIOR_RESOLVED_FORECASTS",
        "Realized raw log return",
        "Baseline absolute error",
        "KRONOS",
        "Artifact chain verified: **yes**",
        "ONE_ORIGIN_ONLY",
    ):
        assert expected in report
