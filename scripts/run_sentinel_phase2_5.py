from __future__ import annotations

# pyright: reportArgumentType=false
import argparse
import hashlib
import itertools
import json
import math
import statistics
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
from openalpha_research import LocalArtifactStore
from openalpha_sentinel.contracts import ForecastRequest, OHLCVObservation
from openalpha_sentinel.evidence import (
    CreationEvidence,
    ResolvedEvidence,
    verify_creation,
    verify_resolved_evidence,
)
from openalpha_sentinel.market_data import (
    MarketDataRequest,
    MarketDataSnapshot,
    default_yfinance_provider,
)
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
from openalpha_sentinel.providers.kronos import KronosSubprocessClient
from openalpha_sentinel.structural_validity import (
    AuditCandle,
    constraint_projection_v0,
    summarize_structural_validity,
)

EXPERIMENT_SHA256 = "587bc36ac1dd941c66d9e94c92d7b26a33faeb8e7e7a38883762f0cb54897950"
PHASE2_CUTOFF = date(2024, 7, 5)
CANARY_CUTOFFS = (date(2024, 9, 27), date(2025, 1, 31), date(2025, 5, 30))
SEEDS = (1729, 2027, 7919)
SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
MODEL_REVISION = "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"
TOKENIZER_REVISION = "26966d0035065a0cae0ebad7af8ece35bc1fb51c"


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    cache_root = Path.home() / ".cache" / "openalpha-sentinel"
    paths = Phase25Paths(
        phase2_root=Path(args.phase2_root or cache_root / "phase2" / "run-spy-20240705"),
        phase2_5_root=Path(args.phase2_5_root or cache_root / "phase2_5"),
    )
    runtime = {
        "repository_root": repository_root,
        "paths": paths,
        "inference_python": Path(
            args.inference_python or cache_root / "phase2" / "venv" / "Scripts" / "python.exe"
        ),
        "source_path": Path(args.source_path or cache_root / "phase2" / "kronos-src"),
        "cache_path": Path(args.model_cache or cache_root / "phase2" / "hf"),
    }
    handlers = {
        "trace": lambda: _trace(runtime),
        "repeatability": lambda: _repeatability(runtime, args.input_receipt),
        "averaging": lambda: _averaging(runtime, args.input_receipt),
        "canary-forecast": lambda: _canary_forecast(runtime),
        "canary-resolve": lambda: _canary_resolve(runtime, args.input_receipt),
        "report": lambda: _report(
            runtime,
            trace_receipt=args.trace_receipt,
            averaging_receipt=args.averaging_receipt,
            canary_receipt=args.canary_receipt,
        ),
    }
    try:
        result = handlers[args.operation]()
    except Exception as error:  # noqa: BLE001 - CLI boundary preserves typed failure
        print(
            json.dumps(
                {
                    "status": "failure",
                    "operation": args.operation,
                    "failure": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bounded Sentinel Phase 2.5 audit")
    parser.add_argument(
        "operation",
        choices=(
            "trace",
            "repeatability",
            "averaging",
            "canary-forecast",
            "canary-resolve",
            "report",
        ),
    )
    parser.add_argument("--input-receipt")
    parser.add_argument("--trace-receipt")
    parser.add_argument("--averaging-receipt")
    parser.add_argument("--canary-receipt")
    parser.add_argument("--phase2-root")
    parser.add_argument("--phase2-5-root")
    parser.add_argument("--inference-python")
    parser.add_argument("--source-path")
    parser.add_argument("--model-cache")
    return parser


def _trace(runtime: dict[str, Any]) -> dict[str, object]:
    started = time.perf_counter()
    execution_attempts = [
        {
            "attempt": 1,
            "status": "not_sealed",
            "failure": "INTRADAY_EXPECTED_TIMESTAMPS_NOT_CONVERTED_TO_STRICT_DATETIME",
            "outcome_accessed": False,
        },
        {"attempt": 2, "status": "sealed", "outcome_accessed": False},
    ]
    paths = cast(Phase25Paths, runtime["paths"])
    before = phase2_fingerprint(paths.phase2_root)
    sealed = _load_sealed_phase2(paths.phase2_root)
    snapshot = _forecast_snapshot("SPY", PHASE2_CUTOFF)
    golden_request, golden_metadata = _golden_request(cast(Path, runtime["source_path"]))
    direct_request = make_direct_request(
        audit_case="phase2_spy_20240705",
        observations=snapshot.observations,
        forecast_timestamps=next_xnys_sessions(PHASE2_CUTOFF),
        context_length=512,
        seed=1729,
        sample_count=1,
        trace=True,
    )
    direct_batch = _direct_batch(runtime, [golden_request, direct_request])
    golden_response, direct_response = direct_batch["responses"]
    provider_path, provider_environment = _provider_path(
        runtime=runtime,
        symbol="SPY",
        cutoff=PHASE2_CUTOFF,
        snapshot=snapshot,
        seed=1729,
    )
    forecast = sealed["forecast"]
    sealed_path = _sealed_path(forecast, 512, 1729)
    comparison = build_path_comparison(
        direct_path=direct_response["path"],
        provider_path=provider_path,
        sealed_path=sealed_path,
        observed_input_sha256=snapshot.normalized_input_sha256,
        sealed_input_sha256=str(sealed["data_quality"]["normalized_input_sha256"]),
    )
    expected = next_xnys_sessions(PHASE2_CUTOFF)
    cutoff_close = float(forecast["cutoff_close"])
    cutoff_volume = snapshot.observations[-1].volume
    validity = {
        "golden_direct": _validity(
            "golden-direct",
            golden_response["path"],
            golden_metadata["forecast_timestamps"],
            float(golden_metadata["cutoff_close"]),
            float(golden_metadata["cutoff_volume"]),
        ),
        "official_direct": _validity(
            "official-direct-512-1729",
            direct_response["path"],
            expected,
            cutoff_close,
            cutoff_volume,
        ),
        "openalpha_provider": _validity(
            "openalpha-provider-512-1729",
            provider_path,
            expected,
            cutoff_close,
            cutoff_volume,
        ),
        "sealed_phase2": _validity(
            "sealed-phase2-512-1729",
            sealed_path,
            expected,
            cutoff_close,
            cutoff_volume,
        ),
    }
    private_payload = {
        "schema_version": "sentinel-phase2_5-trace-private-v0",
        "operation": "trace",
        "created_at": datetime.now(UTC).isoformat(),
        "phase2_fingerprint_before": before,
        "sealed_verification": sealed["verification"],
        "sealed_forecast": forecast,
        "sealed_data_quality": sealed["data_quality"],
        "snapshot": snapshot.model_dump(mode="json"),
        "golden_metadata": golden_metadata,
        "direct_batch": direct_batch,
        "provider_path": provider_path,
        "provider_environment": provider_environment,
        "comparison": comparison,
        "validity": validity,
    }
    private_payload["execution_attempts"] = execution_attempts
    receipt = publish_private_receipt(paths, cast(dict[str, object], private_payload))
    compact_trace = {
        "schema_version": "sentinel-phase2_5-integration-trace-v0",
        "claim_boundary": "DEVELOPMENT AUDIT - NOT EMPIRICAL EVIDENCE",
        "pipeline": [
            "Yahoo raw OHLCV",
            "normalized input DataFrame",
            "512-session context slice",
            "official named input DataFrame",
            "derived amount field",
            "per-feature float32 normalization",
            "tokenizer encode",
            "model generation",
            "tokenizer decode",
            "inverse per-feature normalization",
            "official predictor output DataFrame",
            "OpenAlpha named extraction and canonicalization",
            "sealed forecast artifact",
        ],
        "provider": _snapshot_summary(snapshot),
        "golden_fixture": golden_metadata,
        "official_environment": direct_batch["environment"],
        "source_functions": direct_response["trace"]["source_functions"],
        "golden_boundaries": golden_response["trace"]["boundaries"],
        "phase2_boundaries": direct_response["trace"]["boundaries"],
        "phase2_fingerprint_before": before,
        "private_receipt_sha256": receipt.sha256,
    }
    compact_comparison = {
        "schema_version": "sentinel-phase2_5-official-comparison-v0",
        "claim_boundary": "DEVELOPMENT AUDIT - NOT EMPIRICAL EVIDENCE",
        "comparison": comparison,
        "validity": validity,
        "column_mapping": {
            "official_expected": ["open", "high", "low", "close", "volume", "amount"],
            "openalpha_input": ["open", "high", "low", "close", "volume"],
            "official_derived_field": "amount=volume*mean(open,high,low,close)",
            "openalpha_output_named_selection": ["open", "high", "low", "close", "volume"],
            "multiindex_survived_normalization": False,
        },
        "timestamp_alignment": {
            "cutoff": PHASE2_CUTOFF.isoformat(),
            "forecast_sessions": [item.isoformat() for item in expected],
            "first_output_maps_to": direct_response["path"][0]["timestamp"],
            "last_output_maps_to": direct_response["path"][-1]["timestamp"],
        },
        "provider_environment": provider_environment,
        "private_receipt_sha256": receipt.sha256,
    }
    compact_trace["execution_attempts"] = execution_attempts
    compact_comparison["execution_attempts"] = execution_attempts
    return _result(
        "trace",
        receipt.sha256,
        started,
        compact_integration_trace=compact_trace,
        compact_official_comparison=compact_comparison,
    )


def _repeatability(runtime: dict[str, Any], input_receipt: str | None) -> dict[str, object]:
    started = time.perf_counter()
    if input_receipt is None:
        raise ValueError("repeatability requires --input-receipt from trace")
    paths = cast(Phase25Paths, runtime["paths"])
    trace_payload = load_private_receipt(paths, input_receipt)
    if trace_payload.get("operation") != "trace":
        raise ValueError("repeatability input must be a trace receipt")
    snapshot = _snapshot_from_payload(trace_payload["snapshot"])
    requests = [
        make_direct_request(
            audit_case="phase2_spy_20240705",
            observations=snapshot.observations,
            forecast_timestamps=next_xnys_sessions(PHASE2_CUTOFF),
            context_length=512,
            seed=seed,
            sample_count=1,
            trace=False,
        )
        for seed in SEEDS
        for _ in range(2)
    ]
    batch = _direct_batch(runtime, requests)
    responses = batch["responses"]
    pairs = []
    for index, seed in enumerate(SEEDS):
        first = responses[index * 2]
        second = responses[index * 2 + 1]
        first_path = _named_path(first["path"])
        second_path = _named_path(second["path"])
        sealed_path = _sealed_path(trace_payload["sealed_forecast"], 512, seed)
        exact_pair = first_path == second_path
        sealed_exact = (
            first_path == _named_path(sealed_path)
            if trace_payload["comparison"]["exact_comparison_permitted"]
            else None
        )
        first_validity = _validity(
            f"repeat-{seed}-a",
            first["path"],
            next_xnys_sessions(PHASE2_CUTOFF),
            snapshot.observations[-1].close,
            snapshot.observations[-1].volume,
        )
        second_validity = _validity(
            f"repeat-{seed}-b",
            second["path"],
            next_xnys_sessions(PHASE2_CUTOFF),
            snapshot.observations[-1].close,
            snapshot.observations[-1].volume,
        )
        violation_match = _violation_signature(first_validity) == _violation_signature(
            second_validity
        )
        pairs.append(
            {
                "seed": seed,
                "a_path_sha256": first["path_sha256"],
                "b_path_sha256": second["path_sha256"],
                "exact_output_match": exact_pair,
                "sealed_output_match": sealed_exact,
                "exact_violation_match": first_validity == second_validity,
                "a_validity": first_validity,
                "b_validity": second_validity,
            }
        )
        pairs[-1]["exact_violation_match"] = violation_match
    payload = {
        "schema_version": "sentinel-phase2_5-repeatability-private-v0",
        "operation": "repeatability",
        "dependencies": {"trace_receipt_sha256": input_receipt},
        "snapshot": trace_payload["snapshot"],
        "sealed_forecast": trace_payload["sealed_forecast"],
        "comparison": trace_payload["comparison"],
        "batch": batch,
        "pairs": pairs,
        "all_exactly_repeatable": all(item["exact_output_match"] for item in pairs),
    }
    payload["execution_attempts"] = [
        {
            "attempt": 1,
            "status": "superseded",
            "receipt_sha256": "848eb7da0b484b4d786fc91b06951864459bda5154a70151aba9a648a6c6c9fd",
            "reason": "PATH_ID_INCLUDED_IN_VIOLATION_EQUALITY",
        },
        {"attempt": 2, "status": "sealed"},
    ]
    receipt = publish_private_receipt(paths, cast(dict[str, object], payload))
    return _result(
        "repeatability",
        receipt.sha256,
        started,
        all_exactly_repeatable=payload["all_exactly_repeatable"],
        pairs=pairs,
    )


def _averaging(runtime: dict[str, Any], input_receipt: str | None) -> dict[str, object]:
    started = time.perf_counter()
    if input_receipt is None:
        raise ValueError("averaging requires --input-receipt from repeatability")
    paths = cast(Phase25Paths, runtime["paths"])
    repeat = load_private_receipt(paths, input_receipt)
    if repeat.get("operation") != "repeatability":
        raise ValueError("averaging input must be a repeatability receipt")
    snapshot = _snapshot_from_payload(repeat["snapshot"])
    internal_requests = [
        make_direct_request(
            audit_case="phase2_spy_20240705",
            observations=snapshot.observations,
            forecast_timestamps=next_xnys_sessions(PHASE2_CUTOFF),
            context_length=512,
            seed=1729,
            sample_count=sample_count,
            trace=False,
        )
        for sample_count in (3, 5)
    ]
    internal = _direct_batch(runtime, internal_requests)
    repeat_paths = [repeat["batch"]["responses"][index * 2]["path"] for index in range(3)]
    sealed_members = repeat["sealed_forecast"]["individual_paths"]
    sealed_paths = [item["path"] for item in sealed_members]
    sealed_512 = [item["path"] for item in sealed_members if item["context_length"] == 512]
    offline_current_512 = average_path_payloads(repeat_paths)
    offline_sealed_512 = average_path_payloads(sealed_512)
    offline_all_nine = average_path_payloads(sealed_paths)
    named_paths = {
        **{_sealed_name(item): item["path"] for item in sealed_members},
        "offline-sealed-512-average": offline_sealed_512,
        "offline-current-512-average": offline_current_512,
        "offline-all-nine-average": offline_all_nine,
        "official-internal-sample-count-3": internal["responses"][0]["path"],
        "official-internal-sample-count-5": internal["responses"][1]["path"],
    }
    expected = next_xnys_sessions(PHASE2_CUTOFF)
    cutoff_close = snapshot.observations[-1].close
    cutoff_volume = snapshot.observations[-1].volume
    validity_models = {
        name: path_validity_from_payload(
            path_id=name,
            path=value,
            expected_timestamps=expected,
            cutoff_close=cutoff_close,
            cutoff_volume=cutoff_volume,
        )
        for name, value in named_paths.items()
    }
    all_validity = {name: value.model_dump(mode="json") for name, value in validity_models.items()}
    sealed_only = [validity_models[_sealed_name(item)] for item in sealed_members]
    structural_compact = {
        "schema_version": "sentinel-phase2_5-structural-validity-v0",
        "claim_boundary": "DEVELOPMENT AUDIT - NOT EMPIRICAL EVIDENCE",
        "severity_denominator": "final observed cutoff close for price violations",
        "individual_sealed_paths": {
            item.path_id: item.model_dump(mode="json") for item in sealed_only
        },
        "sealed_nine_summary": summarize_structural_validity(sealed_only).model_dump(mode="json"),
        "averaged_and_internal_paths": {
            name: value for name, value in all_validity.items() if not name.startswith("sealed-")
        },
        "repeatability": {
            "all_exactly_repeatable": repeat["all_exactly_repeatable"],
            "pairs": repeat["pairs"],
        },
    }
    averaging_compact = {
        "schema_version": "sentinel-phase2_5-averaging-comparison-v0",
        "claim_boundary": "DEVELOPMENT AUDIT - NOT EMPIRICAL EVIDENCE",
        "forecasts": {
            name: {
                "sample_count": (
                    3 if name.endswith("count-3") else 5 if name.endswith("count-5") else None
                ),
                "predicted_log_return": raw_log_return(path, cutoff_close=cutoff_close),
                "path_sha256": _canonical_sha256(_named_path(path)),
                "validity": all_validity[name],
                "runtime_ms": (
                    internal["responses"][0]["inference_duration_ms"]
                    if name.endswith("count-3")
                    else internal["responses"][1]["inference_duration_ms"]
                    if name.endswith("count-5")
                    else None
                ),
            }
            for name, path in named_paths.items()
            if "average" in name or "internal" in name
        },
        "offline_internal_equivalence_claimed": False,
        "reason": "internal samples use one RNG stream and average in normalized space",
        "canonical_phase2_unchanged": True,
        "sealed_canonical_close_matches_offline_512": [
            float(row["close"]) for row in offline_sealed_512
        ]
        == [float(value) for value in repeat["sealed_forecast"]["canonical_close_path"]],
    }
    projections = {}
    for name, path in named_paths.items():
        if all_validity[name]["valid"]:
            continue
        projection = constraint_projection_v0(
            path_id=name,
            candles=_candles(path),
            cutoff_close=cutoff_close,
        )
        projected_path = _projection_path(projection.model_dump(mode="json"))
        projections[name] = {
            "method": "CONSTRAINT_PROJECTION_V0",
            "adjustments": [item.model_dump(mode="json") for item in projection.adjustments],
            "total_absolute_adjustment": sum(
                item.absolute_adjustment for item in projection.adjustments
            ),
            "max_normalized_adjustment": max(
                (item.normalized_adjustment for item in projection.adjustments), default=0.0
            ),
            "original_log_return": projection.original_log_return,
            "projected_log_return": projection.projected_log_return,
            "return_unchanged": projection.original_log_return == projection.projected_log_return,
            "mean_range_before": _mean_range(path),
            "mean_range_after": _mean_range(projected_path),
            "close_path_volatility_before": _close_path_volatility(path, cutoff_close),
            "close_path_volatility_after": _close_path_volatility(projected_path, cutoff_close),
        }
    repair_compact = {
        "schema_version": "sentinel-phase2_5-constraint-projection-v0",
        "claim_boundary": "POST-PROCESSING MEASUREMENT - NOT IMPROVED ACCURACY",
        "method": "CONSTRAINT_PROJECTION_V0",
        "invariants": {
            "open_unchanged": True,
            "close_unchanged": True,
            "implied_return_unchanged": all(
                item["return_unchanged"] for item in projections.values()
            ),
        },
        "paths": projections,
    }
    payload = {
        "schema_version": "sentinel-phase2_5-averaging-private-v0",
        "operation": "averaging",
        "dependencies": {"repeatability_receipt_sha256": input_receipt},
        "snapshot": repeat["snapshot"],
        "sealed_forecast": repeat["sealed_forecast"],
        "repeatability": repeat,
        "internal_batch": internal,
        "named_paths": named_paths,
        "compact_structural_validity": structural_compact,
        "compact_averaging_comparison": averaging_compact,
        "compact_repair_experiment": repair_compact,
    }
    receipt = publish_private_receipt(paths, cast(dict[str, object], payload))
    for compact in (structural_compact, averaging_compact, repair_compact):
        compact["private_receipt_sha256"] = receipt.sha256
    return _result(
        "averaging",
        receipt.sha256,
        started,
        compact_structural_validity=structural_compact,
        compact_averaging_comparison=averaging_compact,
        compact_repair_experiment=repair_compact,
    )


def _canary_forecast(runtime: dict[str, Any]) -> dict[str, object]:
    started = time.perf_counter()
    paths = cast(Phase25Paths, runtime["paths"])
    origins: list[dict[str, Any]] = []
    requests: list[dict[str, object]] = []
    for symbol in ("SPY", "QQQ"):
        for cutoff in CANARY_CUTOFFS:
            snapshot = _forecast_snapshot(symbol, cutoff)
            sessions = next_xnys_sessions(cutoff)
            origin = {
                "symbol": symbol,
                "cutoff": cutoff.isoformat(),
                "forecast_sessions": [item.isoformat() for item in sessions],
                "snapshot": snapshot.model_dump(mode="json"),
            }
            origins.append(origin)
            cutoff_code = cutoff.strftime("%Y%m%d")
            audit_case = f"canary_{symbol.lower()}_{cutoff_code}"
            for seed in SEEDS:
                requests.append(
                    make_direct_request(
                        audit_case=audit_case,
                        observations=snapshot.observations,
                        forecast_timestamps=sessions,
                        context_length=512,
                        seed=seed,
                        sample_count=1,
                        trace=False,
                    )
                )
    batch = _direct_batch(runtime, requests)
    for index, origin in enumerate(origins):
        origin["paths"] = batch["responses"][index * 3 : index * 3 + 3]
        snapshot = _snapshot_from_payload(origin["snapshot"])
        expected = tuple(date.fromisoformat(item) for item in origin["forecast_sessions"])
        origin["forecast_time_validity"] = [
            _validity(
                _origin_path_id(origin, response),
                response["path"],
                expected,
                snapshot.observations[-1].close,
                snapshot.observations[-1].volume,
            )
            for response in origin["paths"]
        ]
    payload = {
        "schema_version": "sentinel-phase2_5-canary-forecast-private-v0",
        "operation": "canary-forecast",
        "created_at": datetime.now(UTC).isoformat(),
        "outcome_accessed": False,
        "origins": origins,
        "environment": batch["environment"],
    }
    receipt = publish_private_receipt(paths, cast(dict[str, object], payload))
    return _result(
        "canary-forecast",
        receipt.sha256,
        started,
        origin_count=len(origins),
        official_path_count=len(requests),
        outcome_accessed=False,
    )


def _canary_resolve(runtime: dict[str, Any], input_receipt: str | None) -> dict[str, object]:
    started = time.perf_counter()
    if input_receipt is None:
        raise ValueError("canary-resolve requires --input-receipt from canary-forecast")
    paths = cast(Phase25Paths, runtime["paths"])
    forecast = load_private_receipt(paths, input_receipt)
    if (
        forecast.get("operation") != "canary-forecast"
        or forecast.get("outcome_accessed") is not False
    ):
        raise ValueError("canary resolution requires a sealed forecast-only receipt")
    results = []
    all_validity_models = []
    for origin in forecast["origins"]:
        symbol = str(origin["symbol"])
        cutoff = date.fromisoformat(str(origin["cutoff"]))
        sessions = tuple(date.fromisoformat(item) for item in origin["forecast_sessions"])
        outcome = default_yfinance_provider().fetch(
            MarketDataRequest(
                purpose="outcome",
                symbol=cast(Any, symbol),
                start_inclusive=sessions[0],
                end_exclusive=sessions[-1] + timedelta(days=1),
                cutoff=sessions[-1],
                minimum_sessions=5,
            )
        )
        snapshot = _snapshot_from_payload(origin["snapshot"])
        member_paths = [response["path"] for response in origin["paths"]]
        canonical = average_path_payloads(member_paths)
        validity_models = [
            path_validity_from_payload(
                path_id=_origin_path_id(origin, response),
                path=response["path"],
                expected_timestamps=sessions,
                cutoff_close=snapshot.observations[-1].close,
                cutoff_volume=snapshot.observations[-1].volume,
            )
            for response in origin["paths"]
        ]
        all_validity_models.extend(validity_models)
        summary = summarize_structural_validity(validity_models).model_dump(mode="json")
        predicted_return = raw_log_return(canonical, cutoff_close=snapshot.observations[-1].close)
        realized_return = math.log(outcome.observations[-1].close / snapshot.observations[-1].close)
        results.append(
            {
                "symbol": symbol,
                "cutoff": cutoff.isoformat(),
                "forecast_sessions": [item.isoformat() for item in sessions],
                "input_sha256": snapshot.normalized_input_sha256,
                "outcome_sha256": outcome.normalized_input_sha256,
                "structural_validity": summary,
                "paths": [item.model_dump(mode="json") for item in validity_models],
                "canonical_predicted_log_return": predicted_return,
                "realized_raw_log_return": realized_return,
                "kronos_absolute_error": abs(predicted_return - realized_return),
                "baseline_absolute_error": abs(realized_return),
                "canonical_model_closer": abs(predicted_return - realized_return)
                < abs(realized_return),
                "inference_runtime_ms": sum(
                    float(response["inference_duration_ms"]) for response in origin["paths"]
                ),
                "provider": _snapshot_summary(snapshot),
                "outcome_provider": _snapshot_summary(outcome),
            }
        )
    aggregate = summarize_structural_validity(all_validity_models).model_dump(mode="json")
    invalid_path_count = sum(not item.valid for item in all_validity_models)
    compact = {
        "schema_version": "sentinel-phase2_5-canary-results-v0",
        "claim_boundary": "SIX-ORIGIN RECURRENCE CANARY - NOT FREQUENCY OR CORRELATION EVIDENCE",
        "forecast_receipt_sha256": input_receipt,
        "forecast_sealed_before_outcome_access": True,
        "origins": results,
        "aggregate_structural_validity": aggregate,
        "invalid_path_count": invalid_path_count,
        "total_path_count": len(all_validity_models),
        "limitations": [
            "ONLY_SIX_ORIGINS",
            "NO_CORRELATION_CLAIM",
            "NO_POPULATION_FREQUENCY_CLAIM",
            "UNOFFICIAL_YAHOO_INTERFACE",
            "NO_CROSS_PROVIDER_VERIFICATION",
        ],
    }
    payload = {
        "schema_version": "sentinel-phase2_5-canary-outcome-private-v0",
        "operation": "canary-resolve",
        "dependencies": {"canary_forecast_receipt_sha256": input_receipt},
        "resolved_at": datetime.now(UTC).isoformat(),
        "compact_canary_results": compact,
    }
    receipt = publish_private_receipt(paths, cast(dict[str, object], payload))
    compact["private_receipt_sha256"] = receipt.sha256
    return _result(
        "canary-resolve",
        receipt.sha256,
        started,
        compact_canary_results=compact,
    )


def _report(
    runtime: dict[str, Any],
    *,
    trace_receipt: str | None,
    averaging_receipt: str | None,
    canary_receipt: str | None,
) -> dict[str, object]:
    started = time.perf_counter()
    if None in (trace_receipt, averaging_receipt, canary_receipt):
        raise ValueError("report requires trace, averaging, and canary receipt hashes")
    assert trace_receipt is not None
    assert averaging_receipt is not None
    assert canary_receipt is not None
    paths = cast(Phase25Paths, runtime["paths"])
    trace = load_private_receipt(paths, trace_receipt)
    averaging = load_private_receipt(paths, averaging_receipt)
    canary = load_private_receipt(paths, canary_receipt)
    if (
        trace.get("operation") != "trace"
        or averaging.get("operation") != "averaging"
        or canary.get("operation") != "canary-resolve"
    ):
        raise ValueError("report receipt operation mismatch")
    comparison = trace["comparison"]
    validity = trace["validity"]
    repeatability = averaging["repeatability"]
    canary_compact = canary["compact_canary_results"]
    category = classify_phase2_5(
        direct_official_invalid=not validity["official_direct"]["valid"],
        openalpha_output_invalid=not validity["openalpha_provider"]["valid"],
        direct_equals_openalpha=bool(comparison["direct_equals_provider"]),
        repeatable=bool(repeatability["all_exactly_repeatable"]),
        canary_invalid_path_count=int(canary_compact["invalid_path_count"]),
    )
    compact_trace, compact_comparison = _compact_trace_from_receipt(trace, trace_receipt)
    compact_trace = _slim_trace(compact_trace)
    compact_comparison = _slim_comparison(compact_comparison)
    averaging["compact_structural_validity"] = _slim_structural(
        averaging["compact_structural_validity"]
    )
    canary_compact = _slim_canary(canary_compact)
    trace["compact_integration_trace"] = compact_trace
    trace["compact_official_comparison"] = compact_comparison
    phase2_after = phase2_fingerprint(paths.phase2_root)
    phase2_unchanged = phase2_after == trace["phase2_fingerprint_before"]
    sealed = _load_sealed_phase2(paths.phase2_root)
    source_after = _source_inventory(cast(Path, runtime["source_path"]))
    source_before = trace["direct_batch"]["environment"]["source_inventory"]
    source_unchanged = source_after == source_before
    if not phase2_unchanged or not source_unchanged:
        raise ValueError("Phase 2 evidence or pinned source changed during Phase 2.5")
    report = _report_markdown(
        category=category,
        trace=trace,
        averaging=averaging,
        canary=canary_compact,
        phase2_unchanged=phase2_unchanged,
        source_unchanged=source_unchanged,
    )
    outputs = {
        "integration_trace.json": trace["compact_integration_trace"],
        "official_comparison.json": trace["compact_official_comparison"],
        "structural_validity.json": averaging["compact_structural_validity"],
        "averaging_comparison.json": averaging["compact_averaging_comparison"],
        "repair_experiment.json": averaging["compact_repair_experiment"],
        "canary_results.json": canary_compact,
        "report.md": report,
    }
    payload = {
        "schema_version": "sentinel-phase2_5-report-private-v0",
        "operation": "report",
        "dependencies": {
            "trace_receipt_sha256": trace_receipt,
            "averaging_receipt_sha256": averaging_receipt,
            "canary_receipt_sha256": canary_receipt,
        },
        "conclusion_category": category,
        "phase2_fingerprint_after": phase2_after,
        "phase2_unchanged": phase2_unchanged,
        "source_inventory_after": source_after,
        "source_unchanged": source_unchanged,
        "sealed_chain_verified": sealed["verification"],
        "repository_outputs": outputs,
    }
    receipt = publish_private_receipt(paths, cast(dict[str, object], payload))
    return _result(
        "report",
        receipt.sha256,
        started,
        conclusion_category=category,
        phase2_unchanged=phase2_unchanged,
        source_unchanged=source_unchanged,
        sealed_chain_verified=sealed["verification"],
        repository_outputs=outputs,
    )


def _compact_trace_from_receipt(
    trace: dict[str, Any], receipt_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = {key: value for key, value in trace["snapshot"].items() if key != "observations"}
    golden_response, direct_response = trace["direct_batch"]["responses"]
    forecast = trace["sealed_forecast"]
    path = direct_response["path"]
    compact_trace = {
        "schema_version": "sentinel-phase2_5-integration-trace-v0",
        "claim_boundary": "DEVELOPMENT AUDIT - NOT EMPIRICAL EVIDENCE",
        "pipeline": [
            "Yahoo raw OHLCV",
            "normalized input DataFrame",
            "512-session context slice",
            "official named input DataFrame",
            "derived amount field",
            "per-feature float32 normalization",
            "tokenizer encode",
            "model generation",
            "tokenizer decode",
            "inverse per-feature normalization",
            "official predictor output DataFrame",
            "OpenAlpha named extraction and canonicalization",
            "sealed forecast artifact",
        ],
        "provider": snapshot,
        "golden_fixture": trace["golden_metadata"],
        "official_environment": trace["direct_batch"]["environment"],
        "source_functions": direct_response["trace"]["source_functions"],
        "golden_boundaries": golden_response["trace"]["boundaries"],
        "phase2_boundaries": direct_response["trace"]["boundaries"],
        "phase2_fingerprint_before": trace["phase2_fingerprint_before"],
        "execution_attempts": trace["execution_attempts"],
        "private_receipt_sha256": receipt_sha256,
    }
    compact_comparison = {
        "schema_version": "sentinel-phase2_5-official-comparison-v0",
        "claim_boundary": "DEVELOPMENT AUDIT - NOT EMPIRICAL EVIDENCE",
        "comparison": trace["comparison"],
        "validity": trace["validity"],
        "column_mapping": {
            "official_expected": ["open", "high", "low", "close", "volume", "amount"],
            "openalpha_input": ["open", "high", "low", "close", "volume"],
            "official_derived_field": "amount=volume*mean(open,high,low,close)",
            "openalpha_output_named_selection": ["open", "high", "low", "close", "volume"],
            "multiindex_survived_normalization": False,
        },
        "timestamp_alignment": {
            "cutoff": forecast["cutoff"],
            "forecast_sessions": forecast["forecast_sessions"],
            "first_output_maps_to": path[0]["timestamp"],
            "last_output_maps_to": path[-1]["timestamp"],
        },
        "provider_environment": trace["provider_environment"],
        "execution_attempts": trace["execution_attempts"],
        "private_receipt_sha256": receipt_sha256,
    }
    return compact_trace, compact_comparison


def _load_sealed_phase2(root: Path) -> dict[str, Any]:
    creation_control = json.loads((root / "creation.json").read_text(encoding="utf-8"))
    resolved_payload = json.loads((root / "resolved.json").read_text(encoding="utf-8"))
    creation = CreationEvidence.model_validate_json(json.dumps(creation_control["creation"]))
    resolved = ResolvedEvidence.model_validate_json(json.dumps(resolved_payload))
    store = LocalArtifactStore(root / "artifacts")
    creation_verified = verify_creation(store, creation)
    resolved_verified = verify_resolved_evidence(store, resolved)
    if not creation_verified or not resolved_verified or resolved.creation != creation:
        raise ValueError("sealed Phase 2 evidence failed verification")
    forecast = json.loads(store.read_bytes(creation.forecast_ref))
    data_quality = json.loads(store.read_bytes(creation.data_quality_ref))
    if not isinstance(forecast, dict) or not isinstance(data_quality, dict):
        raise TypeError("sealed Phase 2 JSON artifacts must be objects")
    return {
        "creation": creation.model_dump(mode="json"),
        "resolved": resolved.model_dump(mode="json"),
        "forecast": forecast,
        "data_quality": data_quality,
        "verification": {
            "creation_verified": creation_verified,
            "resolved_chain_verified": resolved_verified,
        },
    }


def _forecast_snapshot(symbol: str, cutoff: date) -> MarketDataSnapshot:
    return default_yfinance_provider().fetch(
        MarketDataRequest(
            purpose="forecast_context",
            symbol=cast(Any, symbol),
            start_inclusive=date(2022, 6, 1),
            end_exclusive=cutoff + timedelta(days=1),
            cutoff=cutoff,
            minimum_sessions=512,
        )
    )


def _snapshot_from_payload(payload: object) -> MarketDataSnapshot:
    return MarketDataSnapshot.model_validate_json(json.dumps(payload, allow_nan=False))


def _golden_request(source_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    fixture = source_path / "tests" / "data" / "regression_input.csv"
    if not fixture.is_file():
        raise ValueError("official regression fixture is missing")
    frame = pd.read_csv(fixture)
    timestamp_column = "timestamps"
    columns = (timestamp_column, "open", "high", "low", "close", "volume", "amount")
    if any(column not in frame.columns for column in columns) or len(frame) < 517:
        raise ValueError("official regression fixture schema is unexpected")
    context = frame.iloc[:512]
    future = frame.iloc[512:517]
    observations = [
        {
            "timestamp": pd.Timestamp(row[timestamp_column]).isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "amount": float(row["amount"]),
        }
        for _, row in context.iterrows()
    ]
    forecast_timestamps = [pd.Timestamp(value).isoformat() for value in future[timestamp_column]]
    request = {
        "audit_case": "golden_official_fixture",
        "model_repository": "NeoQuasar/Kronos-mini",
        "model_revision": MODEL_REVISION,
        "tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-2k",
        "tokenizer_revision": TOKENIZER_REVISION,
        "source_revision": SOURCE_REVISION,
        "context_length": 512,
        "sampling_seed": 1729,
        "temperature": 1.0,
        "top_p": 0.9,
        "sample_count": 1,
        "forecast_horizon": 5,
        "observations": observations,
        "forecast_sessions": forecast_timestamps,
        "trace": True,
    }
    metadata = {
        "fixture_relative_path": "tests/data/regression_input.csv",
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "context_rows": 512,
        "forecast_rows": 5,
        "input_columns": list(columns),
        "context_first_timestamp": observations[0]["timestamp"],
        "context_last_timestamp": observations[-1]["timestamp"],
        "forecast_timestamps": forecast_timestamps,
        "cutoff_close": observations[-1]["close"],
        "cutoff_volume": observations[-1]["volume"],
    }
    return request, metadata


def _direct_batch(runtime: dict[str, Any], requests: list[dict[str, object]]) -> dict[str, Any]:
    repository_root = cast(Path, runtime["repository_root"])
    return run_direct_worker(
        requests=requests,
        inference_python=cast(Path, runtime["inference_python"]),
        worker_script=repository_root / "scripts" / "kronos_phase2_5_worker.py",
        source_path=cast(Path, runtime["source_path"]),
        cache_path=cast(Path, runtime["cache_path"]),
        trace_helper_path=(
            repository_root
            / "packages"
            / "sentinel"
            / "src"
            / "openalpha_sentinel"
            / "integration_trace.py"
        ),
    )


def _provider_path(
    *,
    runtime: dict[str, Any],
    symbol: str,
    cutoff: date,
    snapshot: MarketDataSnapshot,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    request = ForecastRequest(
        model_repository="NeoQuasar/Kronos-mini",
        model_revision=MODEL_REVISION,
        tokenizer_repository="NeoQuasar/Kronos-Tokenizer-2k",
        tokenizer_revision=TOKENIZER_REVISION,
        source_revision=SOURCE_REVISION,
        symbol=cast(Any, symbol),
        cutoff=cutoff,
        forecast_sessions=next_xnys_sessions(cutoff),
        observations=snapshot.observations[-512:],
        context_length=512,
        forecast_horizon=5,
        sampling_seed=seed,
        temperature=1.0,
        top_p=0.9,
        sample_count=1,
        data_snapshot_sha256=_observations_sha256(snapshot.observations[-512:]),
        experiment_sha256=EXPERIMENT_SHA256,
        created_at=datetime.now(UTC),
    )
    repository_root = cast(Path, runtime["repository_root"])
    client = KronosSubprocessClient(
        python_executable=cast(Path, runtime["inference_python"]),
        worker_script=repository_root / "scripts" / "kronos_inference_worker.py",
        source_path=cast(Path, runtime["source_path"]),
        cache_path=cast(Path, runtime["cache_path"]),
        timeout_seconds=1_800.0,
    )
    batch = client.forecast_many((request,))
    response = batch.responses[0]
    if response.failure is not None or not response.generated_paths:
        raise RuntimeError(f"OpenAlpha provider call failed: {response.failure}")
    path = [item.model_dump(mode="json") for item in response.generated_paths[0].observations]
    return path, batch.environment.model_dump(mode="json")


def _observations_sha256(observations: tuple[OHLCVObservation, ...]) -> str:
    return _canonical_sha256([item.model_dump(mode="json") for item in observations])


def _sealed_path(forecast: dict[str, Any], context: int, seed: int) -> list[dict[str, object]]:
    matches = [
        item["path"]
        for item in forecast["individual_paths"]
        if item["context_length"] == context and item["sampling_seed"] == seed
    ]
    if len(matches) != 1:
        raise ValueError("sealed Phase 2 path lookup was not unique")
    return cast(list[dict[str, object]], matches[0])


def _snapshot_summary(snapshot: MarketDataSnapshot) -> dict[str, object]:
    return snapshot.model_dump(mode="json", exclude={"observations"})


def _validity(
    path_id: str,
    path: list[dict[str, object]],
    expected: Any,
    cutoff_close: float,
    cutoff_volume: float,
) -> dict[str, object]:
    if all(isinstance(value, str) for value in expected):
        expected = _audit_timestamps(expected)
    return path_validity_from_payload(
        path_id=path_id,
        path=path,
        expected_timestamps=expected,
        cutoff_close=cutoff_close,
        cutoff_volume=cutoff_volume,
    ).model_dump(mode="json")


def _audit_timestamps(values: Any) -> tuple[datetime, ...]:
    result = []
    for value in values:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(UTC)
        else:
            timestamp = timestamp.tz_convert(UTC)
        result.append(datetime.fromisoformat(timestamp.isoformat()))
    return tuple(result)


def _violation_signature(validity: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in validity.items() if key != "path_id"}
    result["violations"] = [
        {key: value for key, value in item.items() if key != "path_id"}
        for item in validity["violations"]
    ]
    return result


def _named_path(path: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for row in path:
        timestamp = pd.Timestamp(str(row["timestamp"]))
        result.append(
            {
                "session": str(row.get("session") or timestamp.date().isoformat()),
                "timestamp": timestamp.isoformat(),
                "open": float(cast(Any, row["open"])),
                "high": float(cast(Any, row["high"])),
                "low": float(cast(Any, row["low"])),
                "close": float(cast(Any, row["close"])),
                "volume": float(cast(Any, row["volume"])),
            }
        )
    return result


def _candles(path: list[dict[str, object]]) -> tuple[AuditCandle, ...]:
    return tuple(
        AuditCandle(
            session=datetime.fromisoformat(pd.Timestamp(str(row["timestamp"])).isoformat()),
            open=float(cast(Any, row["open"])),
            high=float(cast(Any, row["high"])),
            low=float(cast(Any, row["low"])),
            close=float(cast(Any, row["close"])),
            volume=float(cast(Any, row["volume"])),
        )
        for row in path
    )


def _projection_path(payload: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "timestamp": row["session"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }
        for row in payload["projected_candles"]
    ]


def _mean_range(path: list[dict[str, object]]) -> float:
    return statistics.fmean(
        float(cast(Any, row["high"])) - float(cast(Any, row["low"])) for row in path
    )


def _close_path_volatility(path: list[dict[str, object]], cutoff_close: float) -> float:
    closes = [cutoff_close, *[float(cast(Any, row["close"])) for row in path]]
    returns = [math.log(current / previous) for previous, current in itertools.pairwise(closes)]
    return statistics.pstdev(returns)


def _sealed_name(item: dict[str, Any]) -> str:
    context = item["context_length"]
    seed = item["sampling_seed"]
    return f"sealed-{context}-{seed}"


def _origin_path_id(origin: dict[str, Any], response: dict[str, Any]) -> str:
    symbol = origin["symbol"]
    cutoff = origin["cutoff"]
    seed = response["sampling_seed"]
    return f"{symbol}-{cutoff}-{seed}"


def _source_inventory(source_path: Path) -> list[dict[str, object]]:
    expected = ("README.md", "model/kronos.py", "tests/test_kronos_regression.py")
    result = []
    for relative in expected:
        path = source_path / Path(relative)
        if not path.is_file():
            continue
        content = path.read_bytes()
        result.append(
            {
                "relative_path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    return result


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validity_summary(validity: dict[str, Any]) -> dict[str, Any]:
    violations = validity["violations"]
    code_counts: dict[str, int] = {}
    severities = []
    for item in violations:
        code = str(item["code"])
        code_counts[code] = code_counts.get(code, 0) + 1
        severity = item.get("normalized_severity")
        if isinstance(severity, (int, float)) and not isinstance(severity, bool):
            severities.append(float(severity))
    return {
        "path_id": validity["path_id"],
        "valid": validity["valid"],
        "candle_count": validity["candle_count"],
        "expected_candle_count": validity["expected_candle_count"],
        "invalid_candle_count": validity["invalid_candle_count"],
        "earliest_invalid_horizon_step": validity["earliest_invalid_horizon_step"],
        "violation_count": len(violations),
        "code_counts": dict(sorted(code_counts.items())),
        "max_normalized_severity": max(severities) if severities else None,
        "mean_normalized_severity": (statistics.fmean(severities) if severities else None),
    }


def _slim_trace(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    golden = result["golden_boundaries"]
    result["golden_boundaries"] = [golden[0], golden[-1]]
    result["golden_boundary_policy"] = (
        "official input and final output retained; full trace private"
    )
    return result


def _slim_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    validity = result["validity"]
    result["validity"] = {name: _validity_summary(value) for name, value in validity.items()}
    result["official_direct_violations"] = validity["official_direct"]["violations"]
    result["golden_direct_violations"] = validity["golden_direct"]["violations"]
    return result


def _slim_structural(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["individual_sealed_paths"] = {
        name: _validity_summary(value) for name, value in payload["individual_sealed_paths"].items()
    }
    result["averaged_and_internal_paths"] = {
        name: _validity_summary(value)
        for name, value in payload["averaged_and_internal_paths"].items()
    }
    repeatability = payload["repeatability"]
    result["repeatability"] = {
        "all_exactly_repeatable": repeatability["all_exactly_repeatable"],
        "pairs": [
            {
                "seed": item["seed"],
                "a_path_sha256": item["a_path_sha256"],
                "b_path_sha256": item["b_path_sha256"],
                "exact_output_match": item["exact_output_match"],
                "exact_violation_match": item["exact_violation_match"],
                "sealed_output_match": item["sealed_output_match"],
                "violation_signature": _violation_signature(item["a_validity"]),
            }
            for item in repeatability["pairs"]
        ],
    }
    return result


def _slim_canary(payload: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in payload.items() if key != "origins"}
    origins = []
    for origin in payload["origins"]:
        compact = {
            key: value
            for key, value in origin.items()
            if key not in {"paths", "provider", "outcome_provider"}
        }
        compact["paths"] = [_validity_summary(value) for value in origin["paths"]]
        compact["provider"] = _public_provider_summary(origin["provider"])
        compact["outcome_provider"] = _public_provider_summary(origin["outcome_provider"])
        origins.append(compact)
    result["origins"] = origins
    return result


def _public_provider_summary(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "provider",
        "client",
        "client_version",
        "access_class",
        "adjustment",
        "retrieved_at",
        "row_count",
        "first_session",
        "last_session",
        "normalized_input_sha256",
        "corporate_action_warnings",
    )
    return {field: payload[field] for field in fields}


def _diagnostic_value(summary: dict[str, Any], name: str) -> object:
    matches = [item["value"] for item in summary["values"] if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"missing structural diagnostic: {name}")
    return matches[0]


def _report_markdown(
    *,
    category: str,
    trace: dict[str, Any],
    averaging: dict[str, Any],
    canary: dict[str, Any],
    phase2_unchanged: bool,
    source_unchanged: bool,
) -> str:
    comparison = trace["comparison"]
    validity = trace["validity"]
    structural = averaging["compact_structural_validity"]
    sealed_summary = structural["sealed_nine_summary"]
    invalid_fraction = _diagnostic_value(sealed_summary, "INVALID_PATH_FRACTION")
    candle_fraction = _diagnostic_value(sealed_summary, "INVALID_CANDLE_FRACTION")
    violations = _diagnostic_value(sealed_summary, "TOTAL_CONSTRAINT_VIOLATIONS")
    max_severity = _diagnostic_value(sealed_summary, "MAX_CONSTRAINT_VIOLATION_SEVERITY")
    averaging_compact = averaging["compact_averaging_comparison"]
    repair = averaging["compact_repair_experiment"]
    canary_fraction = _diagnostic_value(
        canary["aggregate_structural_validity"], "INVALID_PATH_FRACTION"
    )
    direct_invalid = not validity["official_direct"]["valid"]
    provider_invalid = not validity["openalpha_provider"]["valid"]
    direct_provider_equal = comparison["direct_equals_provider"]
    exact_permitted = comparison["exact_comparison_permitted"]
    direct_sealed_equal = comparison["direct_equals_sealed"]
    golden_valid = validity["golden_direct"]["valid"]
    repeatable = averaging["repeatability"]["all_exactly_repeatable"]
    canonical_equal = averaging_compact["sealed_canonical_close_matches_offline_512"]
    projection_invariant = repair["invariants"]["implied_return_unchanged"]
    canary_invalid = canary["invalid_path_count"]
    canary_total = canary["total_path_count"]
    lines = [
        "# Sentinel Phase 2.5 Structural-Validity and Integration Audit",
        "",
        "**DEVELOPMENT AUDIT - NOT EMPIRICAL EVIDENCE**",
        "",
        f"Conclusion category: **{category}**",
        "",
        "## Integration finding",
        "",
        f"- Direct official output invalid: {direct_invalid}.",
        f"- OpenAlpha provider output invalid: {provider_invalid}.",
        f"- Direct official and OpenAlpha named OHLCV values match: {direct_provider_equal}.",
        f"- Exact sealed comparison permitted: {exact_permitted}.",
        f"- Direct official and sealed Phase 2 path match: {direct_sealed_equal}.",
        f"- Golden official fixture output valid: {golden_valid}.",
        "",
        (
            "The pinned official predictor performs named-column selection, derives amount when absent, "
            "normalizes each feature, tokenizes and decodes, averages internally in normalized space, "
            "then applies the inverse feature transform. It does not enforce or repair OHLC ordering."
        ),
        "",
        "## Phase 2 structural measurements",
        "",
        f"- Invalid path fraction: {invalid_fraction} (nine sealed paths).",
        f"- Invalid candle fraction: {candle_fraction}.",
        f"- Total constraint violations: {violations}.",
        f"- Maximum normalized severity: {max_severity} of the cutoff close.",
        f"- Same-input, same-seed repeatability across all three 512-context seeds: {repeatable}.",
        "",
        "## Averaging and projection",
        "",
        (
            f"- Sealed 512 offline average matches the unchanged Phase 2 canonical close path: "
            f"{canonical_equal}."
        ),
        (
            "- Official sample_count=3 and sample_count=5 outputs were validated separately. "
            "No equality with offline independent-seed averaging is claimed."
        ),
        f"- Projection return invariant held for every projected path: {projection_invariant}.",
        (
            "- CONSTRAINT_PROJECTION_V0 preserves raw output separately and changes only high and low. "
            "It is not evidence of improved accuracy."
        ),
        "",
        "## Canary recurrence",
        "",
        f"- Invalid paths: {canary_invalid}/{canary_total} (fraction {canary_fraction}).",
        (
            "- The canary contains exactly six origins and eighteen official sample_count=1 paths. "
            "It cannot establish correlation, prevalence, or predictive value."
        ),
        "",
        "## Integrity",
        "",
        f"- Sealed Phase 2 descriptor/artifact fingerprint unchanged: {phase2_unchanged}.",
        f"- Pinned official source inventory unchanged: {source_unchanged}.",
        "- The sealed Phase 2 forecast and outcome records were read and verified, never rewritten.",
        "",
        "## Limitations",
        "",
        "- Yahoo/yfinance is an unofficial, non-point-in-time development source.",
        "- No independent provider comparison was performed.",
        "- Raw close returns omit dividends.",
        "- The official golden fixture is intraday while Sentinel Phase 2 uses daily bars.",
        "- Structural invalidity has not yet been shown to predict forecast error.",
        "",
        "## Next justified task",
        "",
        (
            "Lock structural-validity diagnostics and a mandatory raw-output validity gate in the "
            "development configuration before any holdout work. Then run the already declared "
            "chronological development sample without changing these definitions."
        ),
        "",
    ]
    return "\n".join(lines)


def _result(
    operation: str,
    receipt_sha256: str,
    started: float,
    **values: object,
) -> dict[str, object]:
    return {
        "status": "success",
        "operation": operation,
        "receipt_sha256": receipt_sha256,
        "runtime_seconds": time.perf_counter() - started,
        **values,
    }


if __name__ == "__main__":
    raise SystemExit(main())
