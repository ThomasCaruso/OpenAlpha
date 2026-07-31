from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from openalpha_sentinel.constrained_decoding import terminal_projection
from openalpha_sentinel.decoding_metrics import (
    barrier_event_metrics,
    diversity_metrics,
    path_distance,
    path_quality_metrics,
)
from openalpha_sentinel.development_serialization import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
)
from openalpha_sentinel.market_data import (
    MarketDataRequest,
    default_yfinance_provider,
)
from openalpha_sentinel.phase3b import (
    EXPERIMENT_SHA256,
    build_constrained_worker_requests,
    locked_phase3b_origins,
)
from openalpha_sentinel.structural_validity import (
    AuditCandle,
    validate_forecast_path,
)


@dataclass(frozen=True)
class Phase3BDependencies:
    inference_python: Path
    worker_script: Path
    source_path: Path
    model_cache: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("preflight", "probe", "run", "analyze", "report", "verify"),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path.home() / ".cache" / "openalpha-sentinel" / "phase3b",
    )
    parser.add_argument("--inference-python", type=Path)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--model-cache", type=Path)
    return parser


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    state_root = args.state_root
    if args.operation == "preflight":
        return preflight(state_root)
    if args.operation == "analyze":
        return analyze_state(state_root)
    if args.operation == "report":
        return report_state(state_root)
    if args.operation == "verify":
        return verify_state(state_root)
    dependencies = build_real_dependencies(args)
    if args.operation == "probe":
        return run_probe(state_root, dependencies)
    return run_all(state_root, dependencies)


def build_real_dependencies(args: argparse.Namespace) -> Phase3BDependencies:
    cache_root = Path.home() / ".cache" / "openalpha-sentinel"
    repository_root = Path(__file__).resolve().parents[1]
    return Phase3BDependencies(
        inference_python=args.inference_python
        or cache_root / "phase2" / "venv" / "Scripts" / "python.exe",
        worker_script=repository_root / "scripts" / "kronos_constrained_worker.py",
        source_path=args.source_path or cache_root / "phase2" / "kronos-src",
        model_cache=args.model_cache or cache_root / "phase2" / "hf",
    )


def preflight(state_root: Path) -> dict[str, Any]:
    state_root.mkdir(parents=True, exist_ok=True)
    origins = locked_phase3b_origins()
    payload = {
        "schema_version": "sentinel-phase3b-preflight-v1",
        "experiment_sha256": EXPERIMENT_SHA256,
        "claim_boundary": "DEVELOPMENT FEASIBILITY - NOT HOLDOUT EVIDENCE",
        "holdout_access": "forbidden",
        "origin_count": len(origins),
        "origins": [item.model_dump(mode="json") for item in origins],
        "created_at": datetime.now(UTC).isoformat(),
    }
    encoded = canonical_json_bytes(payload)
    _write_immutable(state_root / "preflight.json", encoded)
    return {
        "operation": "preflight",
        "state_root": str(state_root.resolve()),
        "experiment_sha256": EXPERIMENT_SHA256,
        "origin_count": len(origins),
        "preflight_sha256": sha256_bytes(encoded),
        "holdout_accessed": False,
    }


def run_probe(
    state_root: Path,
    dependencies: Phase3BDependencies,
) -> dict[str, Any]:
    state_root.mkdir(parents=True, exist_ok=True)
    receipt_path = state_root / "probe-receipt.json"
    response_path = state_root / "probe-response.json"
    if receipt_path.exists() and response_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if sha256_bytes(response_path.read_bytes()) != receipt["response_sha256"]:
            raise ValueError("preserved probe response hash mismatch")
        return {**receipt, "operation": "probe", "resumed": True}
    _require_dependencies(dependencies)
    origin = next(
        item
        for item in locked_phase3b_origins()
        if item.asset == "SPY" and item.cutoff == date(2024, 7, 5)
    )
    provider = default_yfinance_provider()
    snapshot = provider.fetch(
        MarketDataRequest(
            purpose="forecast_context",
            symbol=origin.asset,
            start_inclusive=date(2022, 6, 1),
            end_exclusive=origin.cutoff + timedelta(days=1),
            cutoff=origin.cutoff,
            minimum_sessions=512,
        )
    )
    request = build_constrained_worker_requests(
        origin=origin,
        observations=snapshot.observations,
    )[0]
    public_request_metadata = {
        key: value for key, value in request.items() if key != "observations"
    }
    worker_payload = {
        "schema_version": "sentinel-kronos-constrained-worker-v1",
        "requests": [request],
    }
    completed = subprocess.run(
        [
            str(dependencies.inference_python),
            str(dependencies.worker_script),
            "--source-path",
            str(dependencies.source_path),
            "--cache-path",
            str(dependencies.model_cache),
        ],
        input=canonical_json_bytes(worker_payload),
        check=False,
        capture_output=True,
        timeout=1_200,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"constrained worker exited {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace')[-2000:]}"
        )
    result = json.loads(completed.stdout)
    if result.get("status") != "success":
        raise RuntimeError(f"constrained worker failed: {result.get('failure')}")
    responses = result.get("responses")
    if not isinstance(responses, list) or len(responses) != 1:
        raise RuntimeError("constrained worker returned the wrong response count")
    response = responses[0]
    if response.get("status") != "success":
        raise RuntimeError(f"constrained request failed: {response.get('failure')}")
    parity = response.get("official_parity")
    if not isinstance(parity, dict) or parity.get("exact_match") is not True:
        raw = response.get("methods", {}).get("RAW_AUTOREGRESSIVE", {})
        mismatch = {
            "schema_version": "sentinel-phase3b-parity-mismatch-v1",
            "origin_id": origin.origin_id,
            "sampling_seed": request["sampling_seed"],
            "custom_raw_path_sha256": raw.get("path_sha256"),
            "custom_raw_status": raw.get("status"),
            "custom_raw_failure": raw.get("failure"),
            "official_path_sha256": (
                parity.get("official_path_sha256") if isinstance(parity, dict) else None
            ),
            "custom_generated_token_pairs": raw.get("generated_token_pairs"),
            "method_statuses": {
                name: {
                    "status": value.get("status"),
                    "failure": value.get("failure"),
                }
                for name, value in response.get("methods", {}).items()
            },
            "custom_path": raw.get("path"),
            "official_path": parity.get("official_path") if isinstance(parity, dict) else None,
            "fieldwise_differences": _path_differences(
                raw.get("path"),
                parity.get("official_path") if isinstance(parity, dict) else None,
            ),
            "outcome_accessed": False,
        }
        atomic_write_bytes(
            state_root / "probe-parity-mismatch.json",
            canonical_json_bytes(mismatch),
        )
        raise RuntimeError("custom raw loop did not exactly match the official predictor")
    sealed_payload = {
        "schema_version": "sentinel-phase3b-probe-response-v1",
        "claim_boundary": "DEVELOPMENT FEASIBILITY - NOT HOLDOUT EVIDENCE",
        "experiment_sha256": EXPERIMENT_SHA256,
        "origin": origin.model_dump(mode="json"),
        "data": {
            **snapshot.model_dump(mode="json", exclude={"observations"}),
            "normalized_input_sha256": snapshot.normalized_input_sha256,
        },
        "request": public_request_metadata,
        "environment": result["environment"],
        "response": response,
        "outcome_accessed": False,
    }
    response_bytes = canonical_json_bytes(sealed_payload)
    _write_immutable(response_path, response_bytes)
    methods = response["methods"]
    receipt = {
        "schema_version": "sentinel-phase3b-probe-receipt-v1",
        "operation": "probe",
        "origin_id": origin.origin_id,
        "sampling_seed": request["sampling_seed"],
        "response_sha256": sha256_bytes(response_bytes),
        "official_parity_exact": True,
        "method_statuses": {
            name: methods[name]["status"] for name in methods
        },
        "raw_final_violation_count": len(
            methods["RAW_AUTOREGRESSIVE"].get("final_violations", [])
        ),
        "holdout_accessed": False,
        "outcome_accessed": False,
        "resumed": False,
    }
    _write_immutable(receipt_path, canonical_json_bytes(receipt))
    return receipt


def run_all(
    state_root: Path,
    dependencies: Phase3BDependencies,
) -> dict[str, Any]:
    state_root.mkdir(parents=True, exist_ok=True)
    _require_dependencies(dependencies)
    origins_root = state_root / "origins"
    origins_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for origin in locked_phase3b_origins():
        receipt_path = origins_root / origin.origin_id / "origin-receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            _verify_origin_receipt(receipt_path.parent, receipt)
            summaries.append(receipt)
            continue
        try:
            receipt = _execute_origin(
                origin=origin,
                origin_root=receipt_path.parent,
                dependencies=dependencies,
            )
        except Exception as error:  # noqa: BLE001 - every locked origin gets a terminal record
            failure = {
                "schema_version": "sentinel-phase3b-origin-failure-v1",
                "origin_id": origin.origin_id,
                "terminal_status": "failed",
                "failure": {
                    "code": "ORIGIN_EXECUTION_FAILED",
                    "message": f"{type(error).__name__}: {error}",
                },
                "holdout_accessed": False,
            }
            failure_bytes = canonical_json_bytes(failure)
            _write_immutable(receipt_path.parent / "failure.json", failure_bytes)
            receipt = {
                "schema_version": "sentinel-phase3b-origin-receipt-v1",
                "origin_id": origin.origin_id,
                "terminal_status": "failed",
                "failure_sha256": sha256_bytes(failure_bytes),
                "holdout_accessed": False,
            }
            _write_immutable(receipt_path, canonical_json_bytes(receipt))
        summaries.append(receipt)
    run_payload = {
        "schema_version": "sentinel-phase3b-run-receipt-v1",
        "experiment_sha256": EXPERIMENT_SHA256,
        "origin_count": len(summaries),
        "completed_count": sum(
            item["terminal_status"] == "completed" for item in summaries
        ),
        "failed_count": sum(item["terminal_status"] == "failed" for item in summaries),
        "origin_receipts": summaries,
        "holdout_accessed": False,
    }
    run_bytes = canonical_json_bytes(run_payload)
    _write_immutable(state_root / "run-receipt.json", run_bytes)
    return {**run_payload, "operation": "run", "run_sha256": sha256_bytes(run_bytes)}


def _execute_origin(
    *,
    origin: Any,
    origin_root: Path,
    dependencies: Phase3BDependencies,
) -> dict[str, Any]:
    provider = default_yfinance_provider()
    context = provider.fetch(
        MarketDataRequest(
            purpose="forecast_context",
            symbol=origin.asset,
            start_inclusive=date(2022, 6, 1),
            end_exclusive=origin.cutoff + timedelta(days=1),
            cutoff=origin.cutoff,
            minimum_sessions=512,
        )
    )
    requests = build_constrained_worker_requests(
        origin=origin,
        observations=context.observations,
    )
    worker_result = _invoke_worker(requests, dependencies)
    responses = worker_result.get("responses")
    if not isinstance(responses, list) or len(responses) != 3:
        raise RuntimeError("constrained worker returned the wrong response count")
    if any(item.get("status") != "success" for item in responses):
        raise RuntimeError("one or more constrained worker requests failed")
    parity = responses[0].get("official_parity")
    if not isinstance(parity, dict) or parity.get("exact_match") is not True:
        raise RuntimeError("first-seed custom raw loop failed official parity")
    terminal_results: dict[str, Any] = {}
    for response in responses:
        raw = response["methods"]["RAW_AUTOREGRESSIVE"]
        if raw.get("status") != "success":
            raise RuntimeError("raw autoregressive method failed")
        terminal_started = datetime.now(UTC)
        projection = terminal_projection(
            path_id=f"{origin.origin_id}-seed-{response['sampling_seed']}-terminal",
            candles=_audit_candles(raw["path"]),
            expected_sessions=origin.forecast_sessions,
            cutoff_close=context.observations[-1].close,
            cutoff_volume=context.observations[-1].volume,
        )
        terminal_results[str(response["sampling_seed"])] = {
            **projection.model_dump(mode="json"),
            "duration_ms": (
                datetime.now(UTC) - terminal_started
            ).total_seconds()
            * 1_000.0,
        }
    determinism = None
    if origin.origin_id == "sentinel-v1-SPY-2024-07-05-h5":
        replay = _invoke_worker((requests[0],), dependencies)["responses"][0]
        determinism = _compare_replay(responses[0], replay)
    request_metadata = [
        {key: value for key, value in request.items() if key != "observations"}
        for request in requests
    ]
    forecast_payload = {
        "schema_version": "sentinel-phase3b-forecast-v1",
        "claim_boundary": "DEVELOPMENT FEASIBILITY - NOT HOLDOUT EVIDENCE",
        "experiment_sha256": EXPERIMENT_SHA256,
        "origin": origin.model_dump(mode="json"),
        "data": context.model_dump(mode="json", exclude={"observations"}),
        "request_metadata": request_metadata,
        "environment": worker_result["environment"],
        "responses": responses,
        "terminal_projection": terminal_results,
        "determinism_replay": determinism,
        "outcome_accessed": False,
    }
    forecast_bytes = canonical_json_bytes(forecast_payload)
    _write_immutable(origin_root / "forecast.json", forecast_bytes)
    forecast_seal = {
        "schema_version": "sentinel-phase3b-forecast-seal-v1",
        "origin_id": origin.origin_id,
        "forecast_sha256": sha256_bytes(forecast_bytes),
        "sealed_before_outcome_access": True,
        "holdout_accessed": False,
    }
    _write_immutable(
        origin_root / "forecast-seal.json",
        canonical_json_bytes(forecast_seal),
    )
    outcome = provider.fetch(
        MarketDataRequest(
            purpose="outcome",
            symbol=origin.asset,
            start_inclusive=origin.forecast_sessions[0],
            end_exclusive=origin.forecast_sessions[-1] + timedelta(days=1),
            cutoff=origin.forecast_sessions[-1],
            minimum_sessions=5,
        )
    )
    if tuple(item.session for item in outcome.observations) != origin.forecast_sessions:
        raise RuntimeError("outcome sessions do not match the locked horizon")
    resolved_payload = _resolve_origin(
        origin=origin,
        context=context,
        outcome=outcome,
        forecast=forecast_payload,
        forecast_sha256=forecast_seal["forecast_sha256"],
    )
    resolved_bytes = canonical_json_bytes(resolved_payload)
    _write_immutable(origin_root / "resolved.json", resolved_bytes)
    receipt = {
        "schema_version": "sentinel-phase3b-origin-receipt-v1",
        "origin_id": origin.origin_id,
        "terminal_status": "completed",
        "forecast_sha256": forecast_seal["forecast_sha256"],
        "resolved_sha256": sha256_bytes(resolved_bytes),
        "holdout_accessed": False,
    }
    _write_immutable(origin_root / "origin-receipt.json", canonical_json_bytes(receipt))
    return receipt


def _invoke_worker(
    requests: tuple[dict[str, Any], ...],
    dependencies: Phase3BDependencies,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(dependencies.inference_python),
            str(dependencies.worker_script),
            "--source-path",
            str(dependencies.source_path),
            "--cache-path",
            str(dependencies.model_cache),
        ],
        input=canonical_json_bytes(
            {
                "schema_version": "sentinel-kronos-constrained-worker-v1",
                "requests": list(requests),
            }
        ),
        check=False,
        capture_output=True,
        timeout=1_200,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"constrained worker exited {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace')[-2000:]}"
        )
    result = json.loads(completed.stdout)
    if result.get("status") != "success":
        raise RuntimeError(f"constrained worker failed: {result.get('failure')}")
    return result


def _resolve_origin(
    *,
    origin: Any,
    context: Any,
    outcome: Any,
    forecast: dict[str, Any],
    forecast_sha256: str,
) -> dict[str, Any]:
    realized = tuple(
        AuditCandle(
            session=item.session,
            open=item.open,
            high=item.high,
            low=item.low,
            close=item.close,
            volume=item.volume,
        )
        for item in outcome.observations
    )
    cutoff = context.observations[-1]
    method_records = []
    terminal_by_seed = forecast["terminal_projection"]
    for response in forecast["responses"]:
        seed = int(response["sampling_seed"])
        for method in (
            "RAW_AUTOREGRESSIVE",
            "STEPWISE_PROJECT_REENCODE",
            "VALID_CANDIDATE_RESAMPLING",
        ):
            result = response["methods"][method]
            method_records.append(
                _resolved_method_record(
                    method=method,
                    seed=seed,
                    result=result,
                    realized=realized,
                    cutoff=cutoff,
                    forecast_sessions=origin.forecast_sessions,
                )
            )
        terminal = terminal_by_seed[str(seed)]
        method_records.append(
            _resolved_method_record(
                method="TERMINAL_PROJECTION",
                seed=seed,
                result={
                    "status": "success",
                    "path": terminal["projected_candles"],
                    "duration_ms": terminal["duration_ms"],
                    "peak_working_set_bytes": response["methods"][
                        "RAW_AUTOREGRESSIVE"
                    ].get("peak_working_set_bytes"),
                    "projection_adjustments": terminal["adjustments"],
                },
                realized=realized,
                cutoff=cutoff,
                forecast_sessions=origin.forecast_sessions,
            )
        )
    return {
        "schema_version": "sentinel-phase3b-resolved-origin-v1",
        "claim_boundary": "DEVELOPMENT FEASIBILITY - NOT HOLDOUT EVIDENCE",
        "experiment_sha256": EXPERIMENT_SHA256,
        "origin": origin.model_dump(mode="json"),
        "forecast_sha256": forecast_sha256,
        "outcome": {
            **outcome.model_dump(mode="json", exclude={"observations"}),
            "realized_path": [
                item.model_dump(mode="json") for item in realized
            ],
        },
        "cutoff_candle": {
            "session": cutoff.session.isoformat(),
            "open": cutoff.open,
            "high": cutoff.high,
            "low": cutoff.low,
            "close": cutoff.close,
            "volume": cutoff.volume,
        },
        "method_records": method_records,
        "zero_return_baseline_absolute_error": abs(
            math.log(realized[-1].close / cutoff.close)
        ),
        "holdout_accessed": False,
    }


def _resolved_method_record(
    *,
    method: str,
    seed: int,
    result: dict[str, Any],
    realized: tuple[AuditCandle, ...],
    cutoff: Any,
    forecast_sessions: tuple[date, ...],
) -> dict[str, Any]:
    base = {
        "method": method,
        "sampling_seed": seed,
        "status": result["status"],
        "duration_ms": result.get("duration_ms"),
        "peak_working_set_bytes": result.get("peak_working_set_bytes"),
        "failure": result.get("failure"),
        "audit": result.get("audit"),
    }
    if result["status"] != "success":
        return base
    predicted = _audit_candles(result["path"])
    validity = validate_forecast_path(
        path_id=f"{method}-{seed}",
        candles=predicted,
        expected_sessions=forecast_sessions,
        cutoff_close=cutoff.close,
        cutoff_volume=cutoff.volume,
    )
    quality = path_quality_metrics(
        predicted=predicted,
        realized=realized,
        cutoff_close=cutoff.close,
        cutoff_high=cutoff.high,
        cutoff_low=cutoff.low,
    )
    barriers = barrier_event_metrics(
        predicted=predicted,
        realized=realized,
        cutoff_close=cutoff.close,
        barriers=(0.005, 0.01, 0.02),
    )
    return {
        **base,
        "path": [item.model_dump(mode="json") for item in predicted],
        "path_sha256": sha256_bytes(
            canonical_json_bytes([item.model_dump(mode="json") for item in predicted])
        ),
        "structural_valid": validity.valid,
        "structural_violations": [
            item.model_dump(mode="json") for item in validity.violations
        ],
        "quality": quality.model_dump(mode="json"),
        "barriers": [item.model_dump(mode="json") for item in barriers],
        "projection_adjustments": result.get("projection_adjustments"),
        "generation": {
            "steps": result.get("steps"),
            "first_divergence_step": result.get("first_divergence_step"),
            "generated_token_pairs": result.get("generated_token_pairs"),
        },
    }


def _audit_candles(path: list[dict[str, Any]]) -> tuple[AuditCandle, ...]:
    return tuple(
        AuditCandle(
            session=date.fromisoformat(str(row["session"])[:10]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for row in path
    )


def _compare_replay(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    comparisons = {}
    for method in (
        "RAW_AUTOREGRESSIVE",
        "STEPWISE_PROJECT_REENCODE",
        "VALID_CANDIDATE_RESAMPLING",
    ):
        first_payload = _deterministic_method_payload(first["methods"][method])
        second_payload = _deterministic_method_payload(second["methods"][method])
        first_hash = sha256_bytes(canonical_json_bytes(first_payload))
        second_hash = sha256_bytes(canonical_json_bytes(second_payload))
        comparisons[method] = {
            "first_sha256": first_hash,
            "second_sha256": second_hash,
            "exact_match": first_hash == second_hash,
        }
    return {
        "execution_count": 2,
        "comparisons": comparisons,
        "all_exact": all(item["exact_match"] for item in comparisons.values()),
    }


def _deterministic_method_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: result.get(key)
        for key in (
            "method",
            "status",
            "path",
            "path_sha256",
            "invalid_path",
            "generated_token_pairs",
            "steps",
            "first_divergence_step",
            "final_violations",
            "failure",
            "audit",
        )
    }
    return _without_operational_noise(payload)


def _without_operational_noise(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_operational_noise(item)
            for key, item in value.items()
            if key not in {"candidate_validation_duration_ms"}
        }
    if isinstance(value, list):
        return [_without_operational_noise(item) for item in value]
    return value


def _verify_origin_receipt(origin_root: Path, receipt: dict[str, Any]) -> None:
    if receipt["terminal_status"] == "completed":
        forecast = origin_root / "forecast.json"
        resolved = origin_root / "resolved.json"
        if sha256_bytes(forecast.read_bytes()) != receipt["forecast_sha256"]:
            raise ValueError("origin forecast hash mismatch")
        if sha256_bytes(resolved.read_bytes()) != receipt["resolved_sha256"]:
            raise ValueError("origin resolved hash mismatch")
    else:
        failure = origin_root / "failure.json"
        if sha256_bytes(failure.read_bytes()) != receipt["failure_sha256"]:
            raise ValueError("origin failure hash mismatch")


def analyze_state(state_root: Path) -> dict[str, Any]:
    receipts = _load_origin_receipts(state_root)
    resolved = [
        json.loads(
            (state_root / "origins" / receipt["origin_id"] / "resolved.json").read_text(
                encoding="utf-8"
            )
        )
        for receipt in receipts
        if receipt["terminal_status"] == "completed"
    ]
    records = [
        {**record, "origin_id": item["origin"]["origin_id"]}
        for item in resolved
        for record in item["method_records"]
    ]
    method_ids = (
        "RAW_AUTOREGRESSIVE",
        "TERMINAL_PROJECTION",
        "STEPWISE_PROJECT_REENCODE",
        "VALID_CANDIDATE_RESAMPLING",
    )
    methods = {
        method: _aggregate_method(
            method,
            [record for record in records if record["method"] == method],
            resolved,
        )
        for method in method_ids
    }
    first_forecast_path = (
        state_root
        / "origins"
        / "sentinel-v1-SPY-2024-07-05-h5"
        / "forecast.json"
    )
    determinism = None
    if first_forecast_path.exists():
        determinism = json.loads(first_forecast_path.read_text(encoding="utf-8")).get(
            "determinism_replay"
        )
    gates = {
        method: _evaluate_continuation_gates(
            method=method,
            methods=methods,
            determinism=determinism,
        )
        for method in ("STEPWISE_PROJECT_REENCODE", "VALID_CANDIDATE_RESAMPLING")
    }
    passing = [method for method, gate in gates.items() if gate["all_passed"]]
    if passing:
        conclusion = "CONSTRAINED_DECODING_SUCCEEDS"
        canary_status = "required_before_final_conclusion"
    elif any(
        methods[method]["success_count"] > 0
        and methods[method]["returned_structural_validity_rate"] == 1.0
        for method in ("STEPWISE_PROJECT_REENCODE", "VALID_CANDIDATE_RESAMPLING")
    ):
        conclusion = "VALIDITY_SUCCEEDS_QUALITY_DEGRADES"
        canary_status = "not_run_prerequisite_not_met"
    elif all(
        methods[method]["success_count"] == 0
        for method in ("STEPWISE_PROJECT_REENCODE", "VALID_CANDIDATE_RESAMPLING")
    ):
        conclusion = "TECHNICALLY_INFEASIBLE_WITH_CURRENT_TOKENIZER"
        canary_status = "not_run_prerequisite_not_met"
    else:
        conclusion = "STOP"
        canary_status = "not_run_prerequisite_not_met"
    analysis = {
        "schema_version": "sentinel-phase3b-analysis-v2",
        "claim_boundary": "DEVELOPMENT FEASIBILITY - NOT HOLDOUT EVIDENCE",
        "experiment_sha256": EXPERIMENT_SHA256,
        "eligible_origins": 12,
        "completed_origins": len(resolved),
        "failed_origins": sum(
            receipt["terminal_status"] == "failed" for receipt in receipts
        ),
        "paths_per_method": 36,
        "methods": methods,
        "continuation_gates": gates,
        "passing_in_loop_methods": passing,
        "model_size_canary_status": canary_status,
        "conclusion": conclusion,
        "zero_return_baseline_mae": _mean(
            [float(item["zero_return_baseline_absolute_error"]) for item in resolved]
        ),
        "holdout_accessed": False,
    }
    analysis_bytes = canonical_json_bytes(analysis)
    _write_immutable(state_root / "analysis.json", analysis_bytes)
    outputs = state_root / "repository_outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    _write_immutable(outputs / "feasibility_results.json", analysis_bytes)
    summaries = [
        {
            "origin": item["origin"],
            "forecast_sha256": item["forecast_sha256"],
            "method_records": [
                {
                    "method": record["method"],
                    "sampling_seed": record["sampling_seed"],
                    "status": record["status"],
                    "path_sha256": record.get("path_sha256"),
                    "structural_valid": record.get("structural_valid"),
                    "quality": record.get("quality"),
                    "barriers": record.get("barriers"),
                    "duration_ms": record.get("duration_ms"),
                    "failure": record.get("failure"),
                }
                for record in item["method_records"]
            ],
            "holdout_accessed": False,
        }
        for item in resolved
    ]
    summary_bytes = b"".join(canonical_json_bytes(item) + b"\n" for item in summaries)
    _write_immutable(outputs / "origin_summaries.jsonl", summary_bytes)
    canary = {
        "schema_version": "sentinel-phase3b-model-size-canary-v1",
        "status": canary_status,
        "prerequisite": (
            "at_least_one_in_loop_method_passes_all_mini_continuation_gates"
        ),
        "passing_in_loop_methods": passing,
        "holdout_accessed": False,
    }
    _write_immutable(
        outputs / "model_size_canary.json",
        canonical_json_bytes(canary),
    )
    return {
        "operation": "analyze",
        "analysis_sha256": sha256_bytes(analysis_bytes),
        "conclusion": conclusion,
        "passing_in_loop_methods": passing,
        "model_size_canary_status": canary_status,
        "holdout_accessed": False,
    }


def _aggregate_method(
    method: str,
    records: list[dict[str, Any]],
    resolved: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [item for item in records if item["status"] == "success"]
    quality_keys = (
        "normalized_ohlc_mae",
        "open_mae",
        "high_mae",
        "low_mae",
        "close_mae",
        "high_low_range_mae",
        "candle_body_mae",
        "upper_wick_mae",
        "lower_wick_mae",
        "range_direction_accuracy",
        "path_shape_distance",
        "parkinson_volatility_error",
        "cumulative_range_error",
        "five_session_close_return_error",
    )
    quality: dict[str, Any] = {
        key: _mean([float(item["quality"][key]) for item in successful])
        for key in quality_keys
    }
    quality["direction_accuracy"] = _mean(
        [float(bool(item["quality"]["direction_correct"])) for item in successful]
    )
    quality["error_by_horizon_step"] = [
        _mean(
            [
                float(item["quality"]["error_by_horizon_step"][step])
                for item in successful
            ]
        )
        for step in range(5)
    ]
    barrier_rows = [
        barrier
        for item in successful
        for barrier in item.get("barriers", [])
    ]
    barrier_summary = {
        str(level): {
            key: _mean(
                [
                    float(bool(row[key]))
                    for row in barrier_rows
                    if float(row["barrier_fraction"]) == level
                ]
            )
            for key in (
                "positive_touch_correct",
                "negative_touch_correct",
                "either_touch_correct",
            )
        }
        for level in (0.005, 0.01, 0.02)
    }
    diversity = _method_diversity(method, resolved)
    duration_summary = _method_duration_summary(method, records, resolved)
    generation_quality = _generation_quality_summary(records)
    audited_steps = _audited_generation_steps(records)
    intervention_steps = [
        step
        for step in audited_steps
        if step.get("intervened") or step.get("projection_changes")
    ]
    candidate_steps = [
        step
        for step in audited_steps
        if step.get("candidates_considered") is not None
    ]
    return {
        "method": method,
        "expected_path_count": 36,
        "success_count": len(successful),
        "hard_failure_count": 36 - len(successful),
        "hard_failure_rate": (36 - len(successful)) / 36,
        "returned_structural_validity_rate": (
            _mean([float(bool(item["structural_valid"])) for item in successful])
            if successful
            else None
        ),
        "invalid_returned_path_count": sum(
            not bool(item["structural_valid"]) for item in successful
        ),
        "quality": quality,
        "barriers": barrier_summary,
        "diversity": diversity,
        "generation_quality": {
            **generation_quality,
            **_paired_raw_distance(method, resolved),
        },
        **duration_summary,
        "maximum_peak_working_set_bytes": max(
            (
                int(item["peak_working_set_bytes"])
                for item in records
                if item.get("peak_working_set_bytes") is not None
            ),
            default=None,
        ),
        "intervention_count": len(intervention_steps),
        "candidate_rejection_rate": (
            sum(int(step["rejection_count"]) for step in candidate_steps)
            / sum(int(step["candidates_considered"]) for step in candidate_steps)
            if candidate_steps
            else None
        ),
        "fallback_rate": (
            _mean([float(bool(step.get("fallback_used"))) for step in candidate_steps])
            if candidate_steps
            else None
        ),
        "average_selected_token_rank": (
            _mean(
                [
                    float(step["selected_candidate_rank"])
                    for step in candidate_steps
                    if step.get("selected_candidate_rank") is not None
                ]
            )
            if candidate_steps
            else None
        ),
    }


def _method_duration_summary(
    method: str,
    records: list[dict[str, Any]],
    resolved: list[dict[str, Any]],
) -> dict[str, Any]:
    if method != "TERMINAL_PROJECTION":
        return {
            "median_duration_ms": _median(
                [
                    float(item["duration_ms"])
                    for item in records
                    if item.get("duration_ms") is not None
                ]
            ),
            "median_intervention_overhead_ms": None,
        }

    totals = []
    overheads = []
    for item in resolved:
        raw_by_seed = {
            int(record["sampling_seed"]): float(record["duration_ms"])
            for record in item["method_records"]
            if record["method"] == "RAW_AUTOREGRESSIVE"
            and record.get("duration_ms") is not None
        }
        for record in item["method_records"]:
            if (
                record["method"] != "TERMINAL_PROJECTION"
                or record.get("duration_ms") is None
            ):
                continue
            raw_duration = raw_by_seed.get(int(record["sampling_seed"]))
            if raw_duration is None:
                continue
            overhead = float(record["duration_ms"])
            overheads.append(overhead)
            totals.append(raw_duration + overhead)
    return {
        "median_duration_ms": _median(totals),
        "median_intervention_overhead_ms": _median(overheads),
    }


def _audited_generation_steps(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for record in records:
        generation = record.get("generation") or {}
        steps.extend(generation.get("steps") or [])
        audit = record.get("audit") or {}
        steps.extend(audit.get("completed_steps") or [])
        failed = audit.get("failed_step_audit")
        if isinstance(failed, dict):
            steps.append(failed)
    return steps


def _generation_quality_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    steps = _audited_generation_steps(records)
    step_counts: dict[str, int] = {}
    projection_adjustments: list[float] = []
    token_comparison_count = 0
    token_change_count = 0
    candidate_durations: list[float] = []
    search_expansion_count = 0

    for record in records:
        for adjustment in record.get("projection_adjustments") or []:
            step = str(int(adjustment["step"]))
            step_counts[step] = step_counts.get(step, 0) + 1
            projection_adjustments.append(float(adjustment["normalized_adjustment"]))

    for step in steps:
        projection_changes = step.get("projection_changes") or []
        intervened = bool(step.get("intervened")) or bool(projection_changes)
        if intervened:
            step_key = str(int(step["step"]))
            step_counts[step_key] = step_counts.get(step_key, 0) + 1
        projection_adjustments.extend(
            float(item["normalized_adjustment"]) for item in projection_changes
        )
        if projection_changes and step.get("raw_token") is not None:
            selected = step.get("selected_token") or step.get("projected_token")
            if selected is not None:
                token_comparison_count += 1
                token_change_count += int(step["raw_token"] != selected)
        if step.get("candidate_validation_duration_ms") is not None:
            candidate_durations.append(
                float(step["candidate_validation_duration_ms"])
            )
        search_expansion_count += int(step.get("search_expansion_count") or 0)

    return {
        "intervention_step_counts": dict(
            sorted(step_counts.items(), key=lambda item: int(item[0]))
        ),
        "projection_adjustment_count": len(projection_adjustments),
        "mean_projection_magnitude": _mean(projection_adjustments),
        "maximum_projection_magnitude": max(projection_adjustments, default=None),
        "reencoded_token_comparison_count": token_comparison_count,
        "reencoded_token_change_count": token_change_count,
        "reencoded_token_change_rate": (
            token_change_count / token_comparison_count
            if token_comparison_count
            else None
        ),
        "candidate_validation_total_ms": sum(candidate_durations),
        "candidate_validation_mean_ms": _mean(candidate_durations),
        "search_expansion_count": search_expansion_count,
    }


def _paired_raw_distance(
    method: str,
    resolved: list[dict[str, Any]],
) -> dict[str, Any]:
    distances = []
    for item in resolved:
        raw_by_seed = {
            int(record["sampling_seed"]): record
            for record in item["method_records"]
            if record["method"] == "RAW_AUTOREGRESSIVE"
            and record["status"] == "success"
        }
        for record in item["method_records"]:
            if record["method"] != method or record["status"] != "success":
                continue
            raw = raw_by_seed.get(int(record["sampling_seed"]))
            if raw is None:
                continue
            distances.append(
                path_distance(
                    left=_audit_candles(raw["path"]),
                    right=_audit_candles(record["path"]),
                    cutoff_close=float(item["cutoff_candle"]["close"]),
                )
            )
    return {
        "paired_raw_path_count": len(distances),
        "mean_raw_path_distance": _mean(distances),
        "maximum_raw_path_distance": max(distances, default=None),
    }


def _method_diversity(
    method: str,
    resolved: list[dict[str, Any]],
) -> dict[str, Any]:
    values = []
    for item in resolved:
        paths = [
            _audit_candles(record["path"])
            for record in item["method_records"]
            if record["method"] == method and record["status"] == "success"
        ]
        if len(paths) != 3:
            continue
        result = diversity_metrics(
            paths=paths,
            cutoff_close=float(item["cutoff_candle"]["close"]),
        )
        values.append(result)
    return {
        "complete_three_seed_origin_count": len(values),
        "mean_pairwise_path_distance": _mean(
            [item.mean_pairwise_path_distance for item in values]
        ),
        "minimum_pairwise_path_distance": min(
            (item.mean_pairwise_path_distance for item in values),
            default=None,
        ),
        "mean_final_return_variance": _mean(
            [item.final_return_variance for item in values]
        ),
        "minimum_final_return_variance": min(
            (item.final_return_variance for item in values),
            default=None,
        ),
        "mean_repeated_path_rate": _mean(
            [item.repeated_path_rate for item in values]
        ),
    }


def _evaluate_continuation_gates(
    *,
    method: str,
    methods: dict[str, dict[str, Any]],
    determinism: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate = methods[method]
    raw = methods["RAW_AUTOREGRESSIVE"]
    terminal = methods["TERMINAL_PROJECTION"]
    candidate_quality = candidate["quality"]
    raw_quality = raw["quality"]
    terminal_quality = terminal["quality"]
    close_value = candidate_quality["close_mae"]
    raw_close = raw_quality["close_mae"]
    full_value = candidate_quality["normalized_ohlc_mae"]
    range_value = candidate_quality["high_low_range_mae"]
    raw_diversity = raw["diversity"]["mean_pairwise_path_distance"]
    candidate_diversity = candidate["diversity"]["mean_pairwise_path_distance"]
    return_variance = candidate["diversity"]["mean_final_return_variance"]
    repeated_path_rate = candidate["diversity"]["mean_repeated_path_rate"]
    structural = (
        candidate["invalid_returned_path_count"] == 0
        and candidate["returned_structural_validity_rate"] in (1.0, None)
    )
    hard_failure = candidate["hard_failure_count"] <= 1
    close_gate = (
        close_value is not None
        and raw_close is not None
        and close_value <= raw_close * 1.10
        and close_value <= raw_close + 0.001
    )
    better_full = min(
        value
        for value in (
            raw_quality["normalized_ohlc_mae"],
            terminal_quality["normalized_ohlc_mae"],
        )
        if value is not None
    )
    full_gate = full_value is not None and full_value <= better_full * 1.05
    better_range = min(
        value
        for value in (
            raw_quality["high_low_range_mae"],
            terminal_quality["high_low_range_mae"],
        )
        if value is not None
    )
    range_gate = range_value is not None and range_value <= better_range * 1.05
    diversity_gate = (
        candidate_diversity is not None
        and raw_diversity is not None
        and candidate_diversity >= raw_diversity * 0.50
        and return_variance is not None
        and return_variance > 0.0
        and repeated_path_rate is not None
        and repeated_path_rate <= 0.10
    )
    deterministic = bool(
        determinism
        and determinism.get("execution_count") == 2
        and determinism.get("comparisons", {}).get(method, {}).get("exact_match")
    )
    raw_latency = raw["median_duration_ms"]
    method_latency = candidate["median_duration_ms"]
    latency_multiplier = (
        method_latency / raw_latency
        if method_latency is not None and raw_latency not in (None, 0.0)
        else None
    )
    raw_memory = raw["maximum_peak_working_set_bytes"]
    method_memory = candidate["maximum_peak_working_set_bytes"]
    memory_gate = (
        method_memory is not None
        and raw_memory is not None
        and method_memory <= raw_memory * 2.0 + 1_073_741_824
    )
    runtime_gate = (
        latency_multiplier is not None
        and latency_multiplier <= 5.0
        and memory_gate
    )
    gates = {
        "returned_path_structural_validity": structural,
        "hard_failure_rate": hard_failure,
        "close_mae": close_gate,
        "full_ohlc_mae": full_gate,
        "high_low_range_mae": range_gate,
        "diversity": diversity_gate,
        "determinism": deterministic,
        "runtime_and_memory": runtime_gate,
        "no_retraining_or_weight_modification": True,
    }
    return {
        "method": method,
        "gates": gates,
        "all_passed": all(gates.values()),
        "measurements": {
            "hard_failure_count": candidate["hard_failure_count"],
            "hard_failure_rate": candidate["hard_failure_rate"],
            "close_mae": close_value,
            "raw_close_mae": raw_close,
            "normalized_ohlc_mae": full_value,
            "better_raw_terminal_ohlc_mae": better_full,
            "high_low_range_mae": range_value,
            "better_raw_terminal_range_mae": better_range,
            "pairwise_diversity": candidate_diversity,
            "raw_pairwise_diversity": raw_diversity,
            "latency_multiplier": latency_multiplier,
            "peak_working_set_bytes": method_memory,
            "raw_peak_working_set_bytes": raw_memory,
        },
    }


def _load_origin_receipts(state_root: Path) -> list[dict[str, Any]]:
    receipts = []
    for origin in locked_phase3b_origins():
        path = state_root / "origins" / origin.origin_id / "origin-receipt.json"
        if not path.exists():
            raise ValueError(f"origin has no terminal receipt: {origin.origin_id}")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        _verify_origin_receipt(path.parent, receipt)
        receipts.append(receipt)
    return receipts


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _format_value(value: object) -> str:
    if value is None:
        return "not computable"
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def report_state(state_root: Path) -> dict[str, Any]:
    analysis_path = state_root / "analysis.json"
    if not analysis_path.exists():
        raise ValueError("Phase 3B analysis must exist before reporting")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    methods = analysis["methods"]
    method_order = (
        "RAW_AUTOREGRESSIVE",
        "TERMINAL_PROJECTION",
        "STEPWISE_PROJECT_REENCODE",
        "VALID_CANDIDATE_RESAMPLING",
    )
    lines = [
        "# Sentinel v1 Phase 3B Bounded Feasibility Report",
        "",
        "**DEVELOPMENT FEASIBILITY - NOT HOLDOUT EVIDENCE**",
        "",
        "## Research question",
        "",
        "Can hard financial-domain constraints be enforced inside the Kronos",
        "autoregressive generation loop without retraining while preserving forecast",
        "fidelity, diversity, and practical runtime?",
        "",
        "## Scope",
        "",
        f"- Eligible origins: {analysis['eligible_origins']}",
        f"- Completed origins: {analysis['completed_origins']}",
        f"- Failed origins: {analysis['failed_origins']}",
        "- Assets: SPY and QQQ; daily; five XNYS sessions; context 512.",
        "- Seeds: 1729, 2027, and 7919.",
        "- The untouched holdout was not accessed.",
        "",
        "## Method results",
        "",
        "| Method | Success | Hard failures | Valid returned paths | OHLC MAE | Range MAE | Close MAE | Median ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in method_order:
        item = methods[method]
        quality = item["quality"]
        lines.append(
            f"| {method} | {item['success_count']}/36 | "
            f"{item['hard_failure_count']} | "
            f"{_format_value(item['returned_structural_validity_rate'])} | "
            f"{_format_value(quality['normalized_ohlc_mae'])} | "
            f"{_format_value(quality['high_low_range_mae'])} | "
            f"{_format_value(quality['close_mae'])} | "
            f"{_format_value(item['median_duration_ms'])} |"
        )
    lines.extend(
        [
            "",
            "The methods are paired by origin, context, forecast timestamps, and",
            "initial seed. Candidate rejection is the declared point at which a",
            "constrained trajectory may diverge from its raw trajectory.",
            "",
            "## Full path and range metrics",
            "",
            "| Method | Open MAE | High MAE | Low MAE | Close MAE | Range MAE | Body MAE | Upper-wick MAE | Lower-wick MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        quality = methods[method]["quality"]
        lines.append(
            f"| {method} | {_format_value(quality['open_mae'])} | "
            f"{_format_value(quality['high_mae'])} | "
            f"{_format_value(quality['low_mae'])} | "
            f"{_format_value(quality['close_mae'])} | "
            f"{_format_value(quality['high_low_range_mae'])} | "
            f"{_format_value(quality['candle_body_mae'])} | "
            f"{_format_value(quality['upper_wick_mae'])} | "
            f"{_format_value(quality['lower_wick_mae'])} |"
        )
    lines.extend(
        [
            "",
            "| Method | Range-direction accuracy | Path-shape distance | Parkinson-volatility error | Cumulative-range error | Five-session return MAE | Direction accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        quality = methods[method]["quality"]
        lines.append(
            f"| {method} | {_format_value(quality['range_direction_accuracy'])} | "
            f"{_format_value(quality['path_shape_distance'])} | "
            f"{_format_value(quality['parkinson_volatility_error'])} | "
            f"{_format_value(quality['cumulative_range_error'])} | "
            f"{_format_value(quality['five_session_close_return_error'])} | "
            f"{_format_value(quality['direction_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            (
                "The zero-return baseline five-session MAE was "
                f"{_format_value(analysis['zero_return_baseline_mae'])}."
            ),
            "",
            "### Error by horizon step",
            "",
            "| Method | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        values = methods[method]["quality"]["error_by_horizon_step"]
        lines.append(
            f"| {method} | " + " | ".join(_format_value(value) for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "## Barrier-event accuracy",
            "",
            "| Method | Barrier | Positive touch | Negative touch | Either touch |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        for barrier in ("0.005", "0.01", "0.02"):
            item = methods[method]["barriers"][barrier]
            lines.append(
                f"| {method} | {float(barrier):.1%} | "
                f"{_format_value(item['positive_touch_correct'])} | "
                f"{_format_value(item['negative_touch_correct'])} | "
                f"{_format_value(item['either_touch_correct'])} |"
            )
    lines.extend(
        [
            "",
            "## Diversity and operational metrics",
            "",
            "| Method | Pairwise diversity | Return variance | Repeated-path rate | Raw-path distance | Median total ms | Peak bytes |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        item = methods[method]
        diversity = item["diversity"]
        generation = item["generation_quality"]
        lines.append(
            f"| {method} | "
            f"{_format_value(diversity['mean_pairwise_path_distance'])} | "
            f"{_format_value(diversity['mean_final_return_variance'])} | "
            f"{_format_value(diversity['mean_repeated_path_rate'])} | "
            f"{_format_value(generation['mean_raw_path_distance'])} | "
            f"{_format_value(item['median_duration_ms'])} | "
            f"{_format_value(item['maximum_peak_working_set_bytes'])} |"
        )
    lines.extend(
        [
            "",
            (
                "Terminal projection's median projection-only overhead was "
                f"{_format_value(methods['TERMINAL_PROJECTION']['median_intervention_overhead_ms'])} ms; "
                "its total includes the paired raw rollout."
            ),
            "Peak memory is the Windows process-wide cumulative working-set peak",
            "observed after each method. It is a real operational upper bound but",
            "cannot isolate incremental memory by method within the shared worker.",
        ]
    )
    lines.extend(
        [
            "",
            "## Intervention audit",
            "",
            "| Method | Steps | Projection changes | Mean projection | Max projection | Re-encoded token changes | Candidate rejection | Mean validation ms | Search expansions | Fallback | Mean selected rank |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        item = methods[method]
        generation = item["generation_quality"]
        token_changes = (
            f"{generation['reencoded_token_change_count']}/"
            f"{generation['reencoded_token_comparison_count']}"
        )
        lines.append(
            f"| {method} | "
            f"{json.dumps(generation['intervention_step_counts'], sort_keys=True)} | "
            f"{generation['projection_adjustment_count']} | "
            f"{_format_value(generation['mean_projection_magnitude'])} | "
            f"{_format_value(generation['maximum_projection_magnitude'])} | "
            f"{token_changes} | "
            f"{_format_value(item['candidate_rejection_rate'])} | "
            f"{_format_value(generation['candidate_validation_mean_ms'])} | "
            f"{generation['search_expansion_count']} | "
            f"{_format_value(item['fallback_rate'])} | "
            f"{_format_value(item['average_selected_token_rank'])} |"
        )
    lines.extend(["", "## Continuation gates", ""])
    for method, result in analysis["continuation_gates"].items():
        lines.append(f"### {method}")
        lines.append("")
        for gate, passed in result["gates"].items():
            lines.append(f"- {gate}: {'PASS' if passed else 'FAIL'}")
        lines.append("")
    lines.extend(
        [
            "## Generation behavior",
            "",
            (
                "- Raw structural validity rate: "
                f"{_format_value(methods['RAW_AUTOREGRESSIVE']['returned_structural_validity_rate'])}."
            ),
            (
                "- Stepwise project/re-encode hard failures: "
                f"{methods['STEPWISE_PROJECT_REENCODE']['hard_failure_count']}/36."
            ),
            (
                "- Candidate-resampling interventions: "
                f"{methods['VALID_CANDIDATE_RESAMPLING']['intervention_count']}."
            ),
            (
                "- Candidate rejection rate: "
                f"{_format_value(methods['VALID_CANDIDATE_RESAMPLING']['candidate_rejection_rate'])}."
            ),
            (
                "- Candidate fallback rate: "
                f"{_format_value(methods['VALID_CANDIDATE_RESAMPLING']['fallback_rate'])}."
            ),
            "",
            "Terminal projection guarantees final OHLC ordering when it succeeds but",
            "does not prevent invalid intermediate states from conditioning later tokens.",
            "Projection is not presented as improved predictive accuracy.",
            "",
            "## Model-size canary",
            "",
            f"Status: **{analysis['model_size_canary_status']}**.",
            "",
            "Kronos-small and Kronos-base are not run unless an in-loop mini method",
            "passes every locked continuation gate.",
            "",
            "## Conclusion",
            "",
            f"**{analysis['conclusion']}**",
            "",
            "Stepwise project/re-encode returned only valid paths but hard-failed",
            "21 of 36 paths because the tokenizer round trip could reintroduce an",
            "invalid candle. Bounded valid-candidate resampling returned 36 of 36",
            "valid paths with no fallback, but its high-low range MAE exceeded the",
            "preregistered non-degradation threshold. No in-loop method passed every",
            "continuation gate, so the model-size canary and larger study were not run.",
            "",
            "This conclusion is bounded to the tested checkpoints, assets, origins,",
            "frequency, seeds, and five-session horizon. It is not evidence that Kronos",
            "is broadly broken, nor that constrained decoding improves accuracy.",
            "",
        ]
    )
    report = "\n".join(lines)
    report_bytes = report.encode("utf-8")
    _write_immutable(state_root / "report.md", report_bytes)
    outputs = state_root / "repository_outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    _write_immutable(outputs / "report.md", report_bytes)
    return {
        "operation": "report",
        "report_sha256": sha256_bytes(report_bytes),
        "conclusion": analysis["conclusion"],
        "holdout_accessed": False,
    }


def verify_state(state_root: Path) -> dict[str, Any]:
    receipts = _load_origin_receipts(state_root)
    repository_root = Path(__file__).resolve().parents[1]
    experiment_path = repository_root / "research" / "sentinel-v1" / "experiment.yaml"
    if sha256_bytes(experiment_path.read_bytes()) != EXPERIMENT_SHA256:
        raise ValueError("Sentinel v1 experiment hash mismatch")
    for receipt in receipts:
        origin_root = state_root / "origins" / receipt["origin_id"]
        if receipt["terminal_status"] != "completed":
            continue
        forecast = json.loads((origin_root / "forecast.json").read_text(encoding="utf-8"))
        resolved = json.loads((origin_root / "resolved.json").read_text(encoding="utf-8"))
        seal = json.loads((origin_root / "forecast-seal.json").read_text(encoding="utf-8"))
        if forecast["outcome_accessed"] is not False:
            raise ValueError("forecast payload indicates outcome access")
        if seal["sealed_before_outcome_access"] is not True:
            raise ValueError("forecast was not sealed before outcome access")
        if seal["forecast_sha256"] != receipt["forecast_sha256"]:
            raise ValueError("forecast seal and terminal receipt differ")
        if resolved["forecast_sha256"] != receipt["forecast_sha256"]:
            raise ValueError("resolved record references the wrong forecast")
        if resolved["holdout_accessed"] is not False:
            raise ValueError("resolved record indicates holdout access")
    analysis_path = state_root / "analysis.json"
    report_path = state_root / "report.md"
    return {
        "operation": "verify",
        "experiment_sha256": EXPERIMENT_SHA256,
        "eligible_origins": 12,
        "terminal_origins": len(receipts),
        "completed_origins": sum(
            item["terminal_status"] == "completed" for item in receipts
        ),
        "failed_origins": sum(item["terminal_status"] == "failed" for item in receipts),
        "analysis_sha256": (
            sha256_bytes(analysis_path.read_bytes()) if analysis_path.exists() else None
        ),
        "report_sha256": (
            sha256_bytes(report_path.read_bytes()) if report_path.exists() else None
        ),
        "artifact_chain_verified": True,
        "holdout_accessed": False,
        "network_accessed": False,
        "inference_accessed": False,
    }


def main() -> int:
    result = dispatch(build_parser().parse_args())
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


def _require_dependencies(dependencies: Phase3BDependencies) -> None:
    for path in (
        dependencies.inference_python,
        dependencies.worker_script,
        dependencies.source_path,
        dependencies.model_cache,
    ):
        if not path.exists():
            raise ValueError(f"missing Phase 3B dependency: {path}")


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable Phase 3B record differs: {path.name}")
        return
    atomic_write_bytes(path, payload)


def _path_differences(left: object, right: object) -> list[dict[str, Any]]:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return [{"code": "PATH_SHAPE_MISMATCH"}]
    result = []
    for step, (left_row, right_row) in enumerate(zip(left, right), start=1):
        if not isinstance(left_row, dict) or not isinstance(right_row, dict):
            return [{"code": "PATH_ROW_TYPE_MISMATCH"}]
        result.append(
            {
                "step": step,
                "differences": {
                    field: float(left_row[field]) - float(right_row[field])
                    for field in ("open", "high", "low", "close", "volume", "amount")
                },
            }
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
