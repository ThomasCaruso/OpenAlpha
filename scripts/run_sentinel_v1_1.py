from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from openalpha_sentinel.constraint_compatibility import (
    CompatibilityFacts,
    classify_root_cause,
)
from openalpha_sentinel.market_data import (
    MarketDataRequest,
    default_yfinance_provider,
)
from openalpha_sentinel.sentinel_v1_1 import (
    MODEL_REVISIONS,
    SOURCE_REVISION,
    SUPPORT_END_EXCLUSIVE,
    SUPPORT_START,
    SUPPORT_SYMBOLS,
    SUPPORT_WINDOW_LENGTH,
    SUPPORT_WINDOWS_PER_SYMBOL,
    TOKENIZER_REVISIONS,
    locked_v1_1_origins,
)
from openalpha_sentinel.token_manifold import (
    select_support_windows,
    summarize_roundtrip_rows,
)

EXPERIMENT_SHA256 = "6dacd9fd0912a77cce9d373d910b7bff9d58b44076d814960a191cdbe183bd76"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = REPOSITORY_ROOT / "research" / "sentinel-v1_1"
PRIVATE_ROOT = Path.home() / ".cache" / "openalpha-sentinel" / "sentinel-v1_1"
V1_STATE_ROOT = Path.home() / ".cache" / "openalpha-sentinel" / "phase3b-final"


class Dependencies(NamedTuple):
    inference_python: Path
    worker_script: Path
    source_path: Path
    model_cache: Path


def _dependencies() -> Dependencies:
    cache_root = Path.home() / ".cache" / "openalpha-sentinel"
    return Dependencies(
        inference_python=(cache_root / "phase2" / "venv" / "Scripts" / "python.exe"),
        worker_script=REPOSITORY_ROOT / "scripts" / "kronos_token_manifold_worker.py",
        source_path=cache_root / "phase2" / "kronos-src",
        model_cache=cache_root / "phase2" / "hf",
    )


OPERATIONS = (
    "roundtrip",
    "audit",
    "canary",
    "classify",
    "conditional",
    "analyze",
    "report",
    "verify",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the locked Sentinel v1.1 compatibility study")
    parser.add_argument("operation", choices=OPERATIONS)
    return parser


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)


def _write_immutable(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise FileExistsError(f"immutable artifact already differs: {path}")
        return
    path.write_bytes(body)


def _require_forecast_seal(origin_root: Path) -> dict[str, Any]:
    forecast_path = origin_root / "forecast.json"
    seal_path = origin_root / "forecast-seal.json"
    if not forecast_path.is_file() or not seal_path.is_file():
        raise ValueError("forecast seal and forecast record are required")
    forecast_bytes = forecast_path.read_bytes()
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(seal, dict):
        raise TypeError("forecast seal must be an object")
    if seal.get("sealed_before_outcome_access") is not True:
        raise ValueError("forecast seal does not precede outcome access")
    actual = hashlib.sha256(forecast_bytes).hexdigest()
    if seal.get("forecast_sha256") != actual:
        raise ValueError("forecast seal hash does not match the forecast record")
    return seal


def _policy_scan(path: Path, payload: object) -> None:
    forbidden_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".csv"}
    if path.suffix.lower() in forbidden_suffixes:
        raise ValueError(f"forbidden committed artifact type: {path.suffix}")
    forbidden_keys = {
        "observations",
        "raw_provider_response",
        "yfinance_cache",
        "model_weights",
        "tokenizer_weights",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            overlap = forbidden_keys.intersection(str(key) for key in value)
            if overlap:
                raise ValueError(f"forbidden public artifact fields: {sorted(overlap)}")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)


def _build_roundtrip_worker_request(
    *,
    tokenizer_repository: str,
    tokenizer_revision: str,
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    if TOKENIZER_REVISIONS.get(tokenizer_repository) != tokenizer_revision:
        raise ValueError("round-trip tokenizer identity is outside the lock")
    if len(windows) != 30:
        raise ValueError("round-trip request requires 30 locked windows")
    return {
        "schema_version": "sentinel-kronos-token-manifold-worker-v1",
        "operation": "roundtrip",
        "experiment_sha256": EXPERIMENT_SHA256,
        "source_revision": SOURCE_REVISION,
        "tokenizer_repository": tokenizer_repository,
        "tokenizer_revision": tokenizer_revision,
        "windows": windows,
    }


def _compact_roundtrip_response(
    response: dict[str, Any],
) -> dict[str, Any]:
    compact = dict(response)
    compact.pop("token_pairs", None)
    return compact


def _build_compatibility_request(
    *,
    origin: Any,
    seed: int,
    observations: list[dict[str, Any]],
    support: dict[str, Any],
) -> dict[str, Any]:
    if origin not in locked_v1_1_origins():
        raise ValueError("compatibility origin is outside the lock")
    if seed not in (1729, 2027, 7919):
        raise ValueError("compatibility seed is outside the lock")
    if len(observations) != 512:
        raise ValueError("compatibility request requires 512 observations")
    model_repository = "NeoQuasar/Kronos-mini"
    tokenizer_repository = "NeoQuasar/Kronos-Tokenizer-2k"
    return {
        "schema_version": "sentinel-kronos-token-manifold-worker-v1",
        "operation": "compatibility",
        "experiment_sha256": EXPERIMENT_SHA256,
        "origin_id": origin.origin_id,
        "symbol": origin.asset,
        "cutoff": origin.cutoff.isoformat(),
        "model_repository": model_repository,
        "model_revision": MODEL_REVISIONS[model_repository],
        "tokenizer_repository": tokenizer_repository,
        "tokenizer_revision": TOKENIZER_REVISIONS[tokenizer_repository],
        "source_revision": SOURCE_REVISION,
        "context_length": 512,
        "forecast_horizon": 5,
        "sampling_seed": seed,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 0,
        "sample_count": 1,
        "candidate_budgets": [64, 256, 1024],
        "support_coarse_counts": support["coarse_counts"],
        "support_fine_counts": support["fine_counts"],
        "support_pair_counts": support["pair_counts"],
        "forecast_sessions": [item.isoformat() for item in origin.forecast_sessions],
        "observations": observations,
    }


def _invoke_worker(
    payload: dict[str, Any],
    dependencies: Dependencies,
    *,
    timeout_seconds: int = 14_400,
) -> dict[str, Any]:
    for path in (
        dependencies.inference_python,
        dependencies.worker_script,
        dependencies.source_path,
        dependencies.model_cache,
    ):
        if not path.exists():
            raise ValueError(f"required worker dependency is missing: {path}")
    completed = subprocess.run(
        [
            str(dependencies.inference_python),
            str(dependencies.worker_script),
            "--source-path",
            str(dependencies.source_path),
            "--cache-path",
            str(dependencies.model_cache),
        ],
        input=_canonical_json_bytes(payload),
        check=False,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"token-manifold worker exited {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace')[-2000:]}"
        )
    response = json.loads(completed.stdout)
    if not isinstance(response, dict) or response.get("status") != "success":
        raise RuntimeError(f"token-manifold worker failed: {response}")
    return response


def _serialize_observations(observations: Any) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": item.timestamp.isoformat(),
            "open": item.open,
            "high": item.high,
            "low": item.low,
            "close": item.close,
            "volume": item.volume,
        }
        for item in observations
    ]


def _support_payload(
    *,
    tokenizer_repository: str,
    tokenizer_revision: str,
    pairs: list[dict[str, int]],
) -> dict[str, Any]:
    coarse = Counter(int(item["coarse"]) for item in pairs)
    fine = Counter(int(item["fine"]) for item in pairs)
    exact = Counter(f"{int(item['coarse'])}:{int(item['fine'])}" for item in pairs)
    payload = {
        "schema_version": "sentinel-v1.1-token-support-v1",
        "experiment_sha256": EXPERIMENT_SHA256,
        "tokenizer_repository": tokenizer_repository,
        "tokenizer_revision": tokenizer_revision,
        "pair_count": len(pairs),
        "unique_pair_count": len(exact),
        "minimum_supported_pair_count": 2,
        "jeffreys_alpha": 0.5,
        "coarse_counts": {str(key): value for key, value in sorted(coarse.items())},
        "fine_counts": {str(key): value for key, value in sorted(fine.items())},
        "pair_counts": dict(sorted(exact.items())),
        "holdout_accessed": False,
    }
    _policy_scan(PUBLIC_ROOT / "token_support" / "support.json", payload)
    return payload


def run_roundtrip() -> dict[str, Any]:
    results_path = PUBLIC_ROOT / "tokenizer_roundtrip" / "results.json"
    manifest_path = PUBLIC_ROOT / "tokenizer_roundtrip" / "manifest.json"
    if results_path.is_file() and manifest_path.is_file():
        return _verify_roundtrip_artifacts(results_path, manifest_path)
    provider = default_yfinance_provider()
    descriptors = []
    window_closes: dict[str, tuple[float, ...]] = {}
    provider_records = []
    for symbol in SUPPORT_SYMBOLS:
        snapshot = provider.fetch(
            MarketDataRequest(
                purpose="forecast_context",
                symbol=symbol,
                start_inclusive=SUPPORT_START,
                end_exclusive=SUPPORT_END_EXCLUSIVE,
                cutoff=date(2024, 6, 28),
                minimum_sessions=1536,
            )
        )
        windows = select_support_windows(
            snapshot.observations,
            window_length=SUPPORT_WINDOW_LENGTH,
            count=SUPPORT_WINDOWS_PER_SYMBOL,
        )
        for index, window in enumerate(windows, start=1):
            window_id = (
                f"{symbol}-{index}-{window[0].session.isoformat()}-{window[-1].session.isoformat()}"
            )
            descriptors.append(
                {
                    "window_id": window_id,
                    "symbol": symbol,
                    "observations": _serialize_observations(window),
                }
            )
            window_closes[window_id] = tuple(item.close for item in window)
        provider_records.append(snapshot.model_dump(mode="json", exclude={"observations"}))
    regimes = _pooled_regime_labels(window_closes)
    dependencies = _dependencies()
    responses = {}
    support_hashes = {}
    for label, repository in (
        ("tokenizer_2k", "NeoQuasar/Kronos-Tokenizer-2k"),
        ("tokenizer_base", "NeoQuasar/Kronos-Tokenizer-base"),
    ):
        response = _invoke_worker(
            _build_roundtrip_worker_request(
                tokenizer_repository=repository,
                tokenizer_revision=TOKENIZER_REVISIONS[repository],
                windows=descriptors,
            ),
            dependencies,
        )
        pairs = response.get("token_pairs")
        if not isinstance(pairs, list) or len(pairs) != 15_360:
            raise ValueError("round-trip worker returned the wrong token-pair count")
        support = _support_payload(
            tokenizer_repository=repository,
            tokenizer_revision=TOKENIZER_REVISIONS[repository],
            pairs=pairs,
        )
        support_path = PUBLIC_ROOT / "token_support" / f"{label}.json"
        support_bytes = _canonical_json_bytes(support)
        _write_immutable(support_path, support_bytes)
        support_hashes[label] = hashlib.sha256(support_bytes).hexdigest()
        compact = _compact_roundtrip_response(response)
        compact["summary"] = _roundtrip_summary(
            compact["windows"],
            regimes,
        )
        responses[label] = compact
    results = {
        "schema_version": "sentinel-v1.1-tokenizer-roundtrip-results-v1",
        "claim_boundary": ("DEVELOPMENT COMPATIBILITY RESEARCH - NOT HOLDOUT EVIDENCE"),
        "experiment_sha256": EXPERIMENT_SHA256,
        "support_corpus": {
            "symbols": list(SUPPORT_SYMBOLS),
            "window_count": 30,
            "window_length": 512,
            "provider_records": provider_records,
        },
        "tokenizers": responses,
        "holdout_accessed": False,
    }
    _policy_scan(results_path, results)
    results_bytes = _canonical_json_bytes(results)
    _write_immutable(results_path, results_bytes)
    manifest = {
        "schema_version": "sentinel-v1.1-roundtrip-manifest-v1",
        "experiment_sha256": EXPERIMENT_SHA256,
        "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
        "support_sha256": support_hashes,
        "holdout_accessed": False,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    _write_immutable(manifest_path, manifest_bytes)
    return {
        "operation": "roundtrip",
        "results_sha256": manifest["results_sha256"],
        "support_sha256": support_hashes,
        "holdout_accessed": False,
    }


def _pooled_regime_labels(
    window_closes: dict[str, tuple[float, ...]],
) -> dict[str, tuple[str | None, ...]]:
    volatility: dict[str, list[float | None]] = {}
    pooled = []
    for window_id, closes in window_closes.items():
        values = np.asarray(closes, dtype=float)
        returns = np.diff(np.log(values))
        rows: list[float | None] = [None] * 20
        for index in range(20, len(values)):
            observed = float(np.std(returns[index - 20 : index], ddof=0))
            rows.append(observed)
            pooled.append(observed)
        volatility[window_id] = rows
    lower, upper = np.quantile(np.asarray(pooled), (1.0 / 3.0, 2.0 / 3.0))
    labels = {}
    for window_id, values in volatility.items():
        labels[window_id] = tuple(
            None
            if value is None
            else "low"
            if value <= lower
            else "middle"
            if value <= upper
            else "high"
            for value in values
        )
    return labels


def _roundtrip_summary(
    windows: list[dict[str, Any]],
    regimes: dict[str, tuple[str | None, ...]],
) -> dict[str, Any]:
    all_rows = [row for window in windows for row in window["rows"]]
    by_instrument = {}
    for symbol in SUPPORT_SYMBOLS:
        rows = [row for window in windows if window["symbol"] == symbol for row in window["rows"]]
        by_instrument[symbol] = _roundtrip_group(rows)
    by_regime = {}
    for regime in ("low", "middle", "high"):
        rows = [
            row
            for window in windows
            for row, label in zip(
                window["rows"],
                regimes[str(window["window_id"])],
            )
            if label == regime
        ]
        by_regime[regime] = _roundtrip_group(rows)
    return {
        "pooled": _roundtrip_group(all_rows),
        "by_instrument": by_instrument,
        "by_volatility_regime": by_regime,
    }


def _roundtrip_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_roundtrip_rows(rows).model_dump(mode="json")
    summary["normalized_error_means"] = {
        field: statistics.fmean(float(row["normalized_errors"][field]) for row in rows)
        for field in ("open", "high", "low", "close", "range")
    }
    return summary


def _summarize_compatibility_steps(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("compatibility summary requires generated steps")
    invalid = [row for row in rows if not bool(row["raw_valid"])]
    unsupported = [row for row in rows if int(row["support"]["exact_pair_count"]) < 2]
    supported = [row for row in rows if int(row["support"]["exact_pair_count"]) >= 2]

    def invalid_rate(group: list[dict[str, Any]]) -> float | None:
        if not group:
            return None
        return sum(not bool(row["raw_valid"]) for row in group) / len(group)

    result: dict[str, Any] = {
        "generated_candle_count": len(rows),
        "raw_invalid_candle_count": len(invalid),
        "raw_invalid_candle_fraction": len(invalid) / len(rows),
        "unsupported_raw_pair_count": len(unsupported),
        "unsupported_raw_pair_fraction": len(unsupported) / len(rows),
        "invalid_raw_unsupported_fraction": (
            sum(int(row["support"]["exact_pair_count"]) < 2 for row in invalid) / len(invalid)
            if invalid
            else None
        ),
        "unsupported_pair_invalid_rate": invalid_rate(unsupported),
        "supported_pair_invalid_rate": invalid_rate(supported),
        "by_horizon_step": {
            str(step): {
                "count": len(group),
                "invalid_fraction": invalid_rate(group),
                "unsupported_fraction": (
                    sum(int(row["support"]["exact_pair_count"]) < 2 for row in group) / len(group)
                ),
            }
            for step in range(1, 6)
            if (group := [row for row in rows if int(row["step"]) == step])
        },
    }
    for budget in (64, 256, 1024):
        masses = [
            mass
            for row in rows
            for mass in row["probability_mass"]
            if int(mass["candidate_budget"]) == budget
        ]
        if not masses:
            continue
        result[f"budget_{budget}"] = {
            "step_count": len(masses),
            "median_considered_mass": statistics.median(
                float(item["considered_probability_mass"]) for item in masses
            ),
            "median_valid_mass_lower": statistics.median(
                float(item["valid_probability_mass_lower"]) for item in masses
            ),
            "median_valid_mass_upper": statistics.median(
                float(item["valid_probability_mass_upper"]) for item in masses
            ),
            "median_supported_mass_lower": statistics.median(
                float(item["supported_probability_mass_lower"]) for item in masses
            ),
            "median_supported_mass_upper": statistics.median(
                float(item["supported_probability_mass_upper"]) for item in masses
            ),
            "median_valid_supported_mass_lower": statistics.median(
                float(item["valid_supported_probability_mass_lower"]) for item in masses
            ),
            "median_valid_supported_mass_upper": statistics.median(
                float(item["valid_supported_probability_mass_upper"]) for item in masses
            ),
        }
    return result


def _verify_roundtrip_artifacts(
    results_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("round-trip manifest must be an object")
    actual = hashlib.sha256(results_path.read_bytes()).hexdigest()
    if manifest.get("results_sha256") != actual:
        raise ValueError("round-trip result hash mismatch")
    for label, expected in manifest.get("support_sha256", {}).items():
        path = PUBLIC_ROOT / "token_support" / f"{label}.json"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"token-support hash mismatch: {label}")
    return {
        "operation": "roundtrip",
        "resumed": True,
        "results_sha256": actual,
        "support_sha256": manifest["support_sha256"],
        "holdout_accessed": False,
    }


def _load_support(label: str) -> dict[str, Any]:
    path = PUBLIC_ROOT / "token_support" / f"{label}.json"
    if not path.is_file():
        raise ValueError("verified tokenizer support is required before audit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("token support must be an object")
    if payload.get("experiment_sha256") != EXPERIMENT_SHA256:
        raise ValueError("token support experiment hash mismatch")
    return payload


def _sealed_v1_raw_hashes(origin_id: str) -> dict[int, str]:
    root = V1_STATE_ROOT / "origins" / origin_id
    _require_forecast_seal(root)
    payload = json.loads((root / "forecast.json").read_text(encoding="utf-8"))
    return {
        int(item["sampling_seed"]): str(item["methods"]["RAW_AUTOREGRESSIVE"]["path_sha256"])
        for item in payload["responses"]
    }


def _compact_step(
    *,
    origin: Any,
    seed: int,
    step: dict[str, Any],
) -> dict[str, Any]:
    return {
        "origin_id": origin.origin_id,
        "asset": origin.asset,
        "cutoff": origin.cutoff.isoformat(),
        "sampling_seed": seed,
        "step": step["step"],
        "raw_token": step["raw_token"],
        "raw_valid": step["raw_valid"],
        "raw_violations": step["raw_violations"],
        "support": step["support"],
        "model_support": step["model_support"],
        "probability_mass": step["probability_mass"],
    }


def _audit_paths() -> tuple[Path, Path, Path]:
    return (
        PUBLIC_ROOT / "token_support" / "generated_steps.jsonl",
        PUBLIC_ROOT / "probability_mass" / "step_bounds.jsonl",
        PUBLIC_ROOT / "token_support" / "compatibility_manifest.json",
    )


def run_audit() -> dict[str, Any]:
    generated_path, mass_path, manifest_path = _audit_paths()
    if all(path.is_file() for path in (generated_path, mass_path, manifest_path)):
        return _verify_audit_artifacts(generated_path, mass_path, manifest_path)
    support = _load_support("tokenizer_2k")
    provider = default_yfinance_provider()
    dependencies = _dependencies()
    rows: list[dict[str, Any]] = []
    provider_records = []
    for origin in locked_v1_1_origins():
        private_origin = PRIVATE_ROOT / "origins" / origin.origin_id
        private_forecast = private_origin / "forecast.json"
        if private_forecast.is_file():
            _require_forecast_seal(private_origin)
            forecast = json.loads(private_forecast.read_text(encoding="utf-8"))
        else:
            snapshot = provider.fetch(
                MarketDataRequest(
                    purpose="forecast_context",
                    symbol=origin.asset,
                    start_inclusive=date(2022, 1, 1),
                    end_exclusive=date.fromordinal(origin.cutoff.toordinal() + 1),
                    cutoff=origin.cutoff,
                    minimum_sessions=512,
                )
            )
            observations = _serialize_observations(snapshot.observations[-512:])
            requests = [
                _build_compatibility_request(
                    origin=origin,
                    seed=seed,
                    observations=observations,
                    support=support,
                )
                for seed in (1729, 2027, 7919)
            ]
            response = _invoke_worker(
                {
                    "schema_version": ("sentinel-kronos-token-manifold-batch-v1"),
                    "requests": requests,
                },
                dependencies,
            )
            expected_hashes = _sealed_v1_raw_hashes(origin.origin_id)
            observed_hashes = {
                int(item["sampling_seed"]): str(item["path_sha256"])
                for item in response["responses"]
            }
            if observed_hashes != expected_hashes:
                raise ValueError(f"raw path parity failed for {origin.origin_id}")
            forecast = {
                "schema_version": "sentinel-v1.1-compatibility-forecast-v1",
                "experiment_sha256": EXPERIMENT_SHA256,
                "origin": {
                    "origin_id": origin.origin_id,
                    "asset": origin.asset,
                    "cutoff": origin.cutoff.isoformat(),
                    "forecast_sessions": [item.isoformat() for item in origin.forecast_sessions],
                },
                "data": snapshot.model_dump(
                    mode="json",
                    exclude={"observations"},
                ),
                "official_v1_raw_path_parity": True,
                "responses": response["responses"],
                "environment": response["environment"],
                "outcome_accessed": False,
                "holdout_accessed": False,
            }
            forecast_bytes = _canonical_json_bytes(forecast)
            _write_immutable(private_forecast, forecast_bytes)
            _write_immutable(
                private_origin / "forecast-seal.json",
                _canonical_json_bytes(
                    {
                        "schema_version": ("sentinel-v1.1-compatibility-forecast-seal-v1"),
                        "forecast_sha256": hashlib.sha256(forecast_bytes).hexdigest(),
                        "sealed_before_outcome_access": True,
                        "holdout_accessed": False,
                    }
                ),
            )
            _require_forecast_seal(private_origin)
        provider_records.append(forecast["data"])
        for response in forecast["responses"]:
            seed = int(response["sampling_seed"])
            rows.extend(
                _compact_step(origin=origin, seed=seed, step=step) for step in response["steps"]
            )
    if len(rows) != 180:
        raise ValueError(f"compatibility audit expected 180 steps, got {len(rows)}")
    rows.sort(
        key=lambda row: (
            str(row["cutoff"]),
            str(row["asset"]),
            int(row["sampling_seed"]),
            int(row["step"]),
        )
    )
    generated_rows = [
        {key: value for key, value in row.items() if key != "probability_mass"} for row in rows
    ]
    mass_rows = [
        {
            "origin_id": row["origin_id"],
            "asset": row["asset"],
            "cutoff": row["cutoff"],
            "sampling_seed": row["sampling_seed"],
            "step": row["step"],
            **mass,
        }
        for row in rows
        for mass in row["probability_mass"]
    ]
    generated_bytes = _canonical_jsonl_bytes(generated_rows)
    mass_bytes = _canonical_jsonl_bytes(mass_rows)
    _write_immutable(generated_path, generated_bytes)
    _write_immutable(mass_path, mass_bytes)
    summary = _summarize_compatibility_steps(rows)
    summary.update(
        {
            "schema_version": "sentinel-v1.1-compatibility-summary-v1",
            "experiment_sha256": EXPERIMENT_SHA256,
            "official_v1_raw_path_parity": True,
            "provider_records": provider_records,
            "holdout_accessed": False,
        }
    )
    summary_path = PUBLIC_ROOT / "token_support" / "generated_summary.json"
    _policy_scan(summary_path, summary)
    summary_bytes = _canonical_json_bytes(summary)
    _write_immutable(summary_path, summary_bytes)
    manifest = {
        "schema_version": "sentinel-v1.1-compatibility-manifest-v1",
        "experiment_sha256": EXPERIMENT_SHA256,
        "generated_steps_sha256": hashlib.sha256(generated_bytes).hexdigest(),
        "step_bounds_sha256": hashlib.sha256(mass_bytes).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "origin_count": 12,
        "path_count": 36,
        "step_count": 180,
        "holdout_accessed": False,
    }
    _write_immutable(manifest_path, _canonical_json_bytes(manifest))
    return {
        "operation": "audit",
        "origin_count": 12,
        "path_count": 36,
        "step_count": 180,
        "raw_invalid_candle_fraction": summary["raw_invalid_candle_fraction"],
        "holdout_accessed": False,
    }


def _verify_audit_artifacts(
    generated_path: Path,
    mass_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "generated_steps_sha256": hashlib.sha256(generated_path.read_bytes()).hexdigest(),
        "step_bounds_sha256": hashlib.sha256(mass_path.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(
            (PUBLIC_ROOT / "token_support" / "generated_summary.json").read_bytes()
        ).hexdigest(),
    }
    if any(manifest.get(key) != value for key, value in checks.items()):
        raise ValueError("compatibility audit artifact hash mismatch")
    return {
        "operation": "audit",
        "resumed": True,
        "origin_count": manifest["origin_count"],
        "path_count": manifest["path_count"],
        "step_count": manifest["step_count"],
        "holdout_accessed": False,
    }


def _canary_disposition(roundtrip: dict[str, Any]) -> dict[str, Any]:
    tokenizers = roundtrip["tokenizers"]
    defects = {
        label: bool(payload["summary"]["pooled"]["material_defect"])
        for label, payload in tokenizers.items()
    }
    if any(defects.values()):
        return {
            "schema_version": "sentinel-v1.1-model-size-canary-v1",
            "status": "not_run_tokenizer_defect_gate",
            "reason": (
                "Encode-decode reconstruction materially violated the grammar; "
                "the preregistered round-trip decision stops further "
                "autoregressive candidate investigation."
            ),
            "tokenizer_material_defects": defects,
            "models_executed": [],
            "holdout_accessed": False,
        }
    return {
        "schema_version": "sentinel-v1.1-model-size-canary-v1",
        "status": "eligible_not_yet_run",
        "tokenizer_material_defects": defects,
        "models_executed": [],
        "holdout_accessed": False,
    }


def _conditional_disposition(
    classification: dict[str, Any],
) -> dict[str, Any]:
    authorized = bool(classification["conditional_method_authorized"])
    return {
        "schema_version": "sentinel-v1.1-conditional-method-gate-v1",
        "root_cause": classification["root_cause"],
        "status": "authorized_not_yet_run" if authorized else "not_authorized",
        "method": "SUPPORT_CONDITIONED_SAMPLING_V0",
        "method_executed": False,
        "reason": (
            "Authorized only for OFF_MANIFOLD_TOKEN_COMBINATIONS, "
            "LOW_VALID_PROBABILITY_MASS, or CANDIDATE_SEARCH_FAILURE."
        ),
        "holdout_accessed": False,
    }


def run_canary() -> dict[str, Any]:
    roundtrip_path = PUBLIC_ROOT / "tokenizer_roundtrip" / "results.json"
    if not roundtrip_path.is_file():
        raise ValueError("round-trip audit is required before the canary gate")
    roundtrip = json.loads(roundtrip_path.read_text(encoding="utf-8"))
    disposition = _canary_disposition(roundtrip)
    path = PUBLIC_ROOT / "model_size_canary" / "status.json"
    _write_immutable(path, _canonical_json_bytes(disposition))
    return {"operation": "canary", **disposition}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"JSONL artifact contains a non-object row: {path}")
    return rows


def run_classify() -> dict[str, Any]:
    roundtrip = json.loads(
        (PUBLIC_ROOT / "tokenizer_roundtrip" / "results.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (PUBLIC_ROOT / "token_support" / "generated_summary.json").read_text(encoding="utf-8")
    )
    mass_rows = [
        row
        for row in _read_jsonl(PUBLIC_ROOT / "probability_mass" / "step_bounds.jsonl")
        if int(row["candidate_budget"]) == 1024
    ]
    if len(mass_rows) != 180:
        raise ValueError("classification requires 180 maximum-budget rows")
    tokenizer_2k = roundtrip["tokenizers"]["tokenizer_2k"]["summary"]["pooled"]
    tokenizer_base = roundtrip["tokenizers"]["tokenizer_base"]["summary"]["pooled"]
    unsupported_count = int(summary["unsupported_raw_pair_count"])
    supported_count = int(summary["generated_candle_count"]) - unsupported_count
    unsupported_rate = summary["unsupported_pair_invalid_rate"]
    supported_rate = summary["supported_pair_invalid_rate"]
    facts = CompatibilityFacts(
        tokenizer_2k_material_defect=bool(tokenizer_2k["material_defect"]),
        tokenizer_base_material_defect=bool(tokenizer_base["material_defect"]),
        tokenizer_base_overwhelmingly_valid=bool(tokenizer_base["overwhelmingly_valid"]),
        mini_material_raw_invalidity=(float(summary["raw_invalid_candle_fraction"]) >= 0.01),
        mini_invalid_candle_rate=float(summary["raw_invalid_candle_fraction"]),
        larger_model_invalid_candle_rates=(),
        median_valid_mass_upper=float(summary["budget_1024"]["median_valid_mass_upper"]),
        low_valid_upper_step_fraction=(
            sum(float(row["valid_probability_mass_upper"]) <= 0.50 for row in mass_rows)
            / len(mass_rows)
        ),
        invalid_raw_unsupported_fraction=float(summary["invalid_raw_unsupported_fraction"] or 0.0),
        unsupported_pair_invalid_rate=float(unsupported_rate or 0.0),
        supported_pair_invalid_rate=float(supported_rate or 0.0),
        unsupported_pair_count=unsupported_count,
        supported_pair_count=supported_count,
        median_valid_mass_lower=float(summary["budget_1024"]["median_valid_mass_lower"]),
        median_valid_supported_mass_upper=float(
            summary["budget_1024"]["median_valid_supported_mass_upper"]
        ),
        median_valid_supported_mass_lower=float(
            summary["budget_1024"]["median_valid_supported_mass_lower"]
        ),
        median_considered_mass=float(summary["budget_1024"]["median_considered_mass"]),
        prior_range_gate_failed=True,
        canary_sufficient=False,
    )
    classification = classify_root_cause(facts)
    payload = {
        "schema_version": "sentinel-v1.1-root-cause-classification-v1",
        "experiment_sha256": EXPERIMENT_SHA256,
        "precedence": [
            "MINI_OR_TOKENIZER_2K_SPECIFIC",
            "TOKENIZER_CONSTRAINT_DEFECT",
            "LOW_VALID_PROBABILITY_MASS",
            "OFF_MANIFOLD_TOKEN_COMBINATIONS",
            "CANDIDATE_SEARCH_FAILURE",
            "MIXED_OR_UNRESOLVED",
        ],
        "facts": facts.model_dump(mode="json"),
        **classification.model_dump(mode="json"),
        "outcomes_used": False,
        "holdout_accessed": False,
    }
    path = PUBLIC_ROOT / "root_cause.json"
    _write_immutable(path, _canonical_json_bytes(payload))
    return {"operation": "classify", **payload}


def run_conditional() -> dict[str, Any]:
    classification = json.loads((PUBLIC_ROOT / "root_cause.json").read_text(encoding="utf-8"))
    disposition = _conditional_disposition(classification)
    if disposition["status"] != "not_authorized":
        raise RuntimeError(
            "authorized support-conditioned inference must be implemented before execution"
        )
    path = PUBLIC_ROOT / "method_comparison" / "conditional_gate.json"
    _write_immutable(path, _canonical_json_bytes(disposition))
    return {"operation": "conditional", **disposition}


def run_analyze() -> dict[str, Any]:
    classification = json.loads((PUBLIC_ROOT / "root_cause.json").read_text(encoding="utf-8"))
    roundtrip = json.loads(
        (PUBLIC_ROOT / "tokenizer_roundtrip" / "results.json").read_text(encoding="utf-8")
    )
    compatibility = json.loads(
        (PUBLIC_ROOT / "token_support" / "generated_summary.json").read_text(encoding="utf-8")
    )
    canary = json.loads(
        (PUBLIC_ROOT / "model_size_canary" / "status.json").read_text(encoding="utf-8")
    )
    conditional = json.loads(
        (PUBLIC_ROOT / "method_comparison" / "conditional_gate.json").read_text(encoding="utf-8")
    )
    tokenizers = roundtrip["tokenizers"]
    probability = {key: value for key, value in compatibility.items() if key.startswith("budget_")}
    probability.update(
        {
            "schema_version": "sentinel-v1.1-probability-mass-summary-v1",
            "estimate_kind": "truncated_lower_and_upper_bounds",
            "experiment_sha256": EXPERIMENT_SHA256,
            "holdout_accessed": False,
        }
    )
    probability_path = PUBLIC_ROOT / "probability_mass" / "summary.json"
    _write_immutable(probability_path, _canonical_json_bytes(probability))
    root_cause = str(classification["root_cause"])
    if root_cause == "TOKENIZER_CONSTRAINT_DEFECT":
        decision = "STOP_CONSTRAINED_DECODING_RESEARCH_SHIP_ASSURANCE_LIBRARY"
    else:
        decision = "CONDITIONAL_METHOD_RESULT_REQUIRED"
    analysis = {
        "schema_version": "sentinel-v1.1-analysis-v1",
        "claim_boundary": ("DEVELOPMENT COMPATIBILITY RESEARCH - NOT HOLDOUT EVIDENCE"),
        "experiment_sha256": EXPERIMENT_SHA256,
        "root_cause": root_cause,
        "matched_rule": classification["matched_rule"],
        "decision": decision,
        "tokenizer_roundtrip": {
            label: payload["summary"]["pooled"] for label, payload in tokenizers.items()
        },
        "generated_support": {
            key: value
            for key, value in compatibility.items()
            if key
            in {
                "generated_candle_count",
                "raw_invalid_candle_count",
                "raw_invalid_candle_fraction",
                "unsupported_raw_pair_count",
                "unsupported_raw_pair_fraction",
                "invalid_raw_unsupported_fraction",
                "unsupported_pair_invalid_rate",
                "supported_pair_invalid_rate",
                "by_horizon_step",
            }
        },
        "probability_mass": probability,
        "model_size_canary": canary,
        "conditional_method": conditional,
        "method_comparison": {
            "status": "not_run_not_authorized",
            "realized_outcomes_accessed": False,
        },
        "practical_fallback": [
            "financial forecast structural validator",
            "immutable raw-output audit layer",
            "deterministic terminal-projection gateway",
            "token-manifold compatibility profiler",
            "model-selection safety check",
        ],
        "future_training_track": (
            "constraint-preserving candle parameterization with open return, "
            "body, nonnegative upper/lower wicks, and nonnegative volume"
        ),
        "holdout_accessed": False,
    }
    path = PUBLIC_ROOT / "analysis.json"
    _policy_scan(path, analysis)
    _write_immutable(path, _canonical_json_bytes(analysis))
    return {
        "operation": "analyze",
        "root_cause": root_cause,
        "decision": decision,
        "holdout_accessed": False,
    }


def run_report() -> dict[str, Any]:
    report = report_state(PUBLIC_ROOT)
    path = PUBLIC_ROOT / "report.md"
    _write_immutable(path, report.encode())
    return {
        "operation": "report",
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "holdout_accessed": False,
    }


def run_verify() -> dict[str, Any]:
    _verify_roundtrip_artifacts(
        PUBLIC_ROOT / "tokenizer_roundtrip" / "results.json",
        PUBLIC_ROOT / "tokenizer_roundtrip" / "manifest.json",
    )
    generated_path, mass_path, audit_manifest = _audit_paths()
    _verify_audit_artifacts(generated_path, mass_path, audit_manifest)
    for origin in locked_v1_1_origins():
        _require_forecast_seal(PRIVATE_ROOT / "origins" / origin.origin_id)
    if any(
        date.fromisoformat(str(row["cutoff"])) >= date(2025, 7, 1)
        for row in _read_jsonl(generated_path)
    ):
        raise ValueError("compatibility artifact crossed the holdout firewall")
    tracked = [
        path
        for path in PUBLIC_ROOT.rglob("*")
        if path.is_file()
        and path.name != "verification.json"
        and path.suffix.lower() in {".json", ".jsonl", ".md", ".yaml", ".sha256"}
    ]
    policy_checked = 0
    for path in tracked:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            _policy_scan(path, payload)
            policy_checked += 1
        elif path.suffix.lower() == ".jsonl":
            for payload in _read_jsonl(path):
                _policy_scan(path, payload)
            policy_checked += 1
    unchanged = (
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "5cbd4e2",
                "--",
                "research/sentinel-v0",
                "research/sentinel-v1",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
        ).returncode
        == 0
    )
    if not unchanged:
        raise ValueError("Sentinel v0 or v1 artifacts changed")
    artifact_hashes = {
        path.relative_to(REPOSITORY_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(tracked)
    }
    verification = {
        "schema_version": "sentinel-v1.1-verification-v1",
        "experiment_sha256": EXPERIMENT_SHA256,
        "artifact_hashes": artifact_hashes,
        "private_origin_seal_count": 12,
        "policy_checked_file_count": policy_checked,
        "v0_v1_unchanged_from_5cbd4e2": True,
        "holdout_accessed": False,
        "verified": True,
    }
    path = PUBLIC_ROOT / "verification.json"
    _write_immutable(path, _canonical_json_bytes(verification))
    return {"operation": "verify", **verification}


def report_state(state_root: Path) -> str:
    analysis_path = state_root / "analysis.json"
    if not analysis_path.is_file():
        raise ValueError("verified v1.1 analysis is required before reporting")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if not isinstance(analysis, dict):
        raise TypeError("v1.1 analysis must be an object")
    if analysis.get("holdout_accessed") is not False:
        raise ValueError("v1.1 analysis does not preserve the holdout firewall")
    tokenizers = analysis.get("tokenizer_roundtrip", {})
    generated = analysis.get("generated_support", {})
    probability = analysis.get("probability_mass", {})
    lines = [
        "# Sentinel v1.1 Token-Manifold Compatibility",
        "",
        f"**{analysis['claim_boundary']}**",
        "",
        "## Conclusion",
        "",
        f"Root cause: {analysis['root_cause']}.",
    ]
    if analysis.get("decision"):
        lines.extend(("", f"Decision: {analysis['decision']}."))
    lines.extend(("", "## Tokenizer round trip", ""))
    for label, summary in tokenizers.items():
        lines.append(
            f"- {label}: {summary['invalid_candle_count']} of "
            f"{summary['reconstructed_candle_count']} reconstructed candles "
            f"invalid ({summary['invalid_candle_fraction']:.4%}); Wilson 95% CI "
            f"[{summary['wilson_lower']:.4%}, {summary['wilson_upper']:.4%}]."
        )
    if generated:
        lines.extend(
            (
                "",
                "## Autoregressive compatibility characterization",
                "",
                (
                    f"- Raw invalid candles: "
                    f"{generated['raw_invalid_candle_count']} of "
                    f"{generated['generated_candle_count']} "
                    f"({generated['raw_invalid_candle_fraction']:.4%})."
                ),
                (
                    f"- Unsupported raw token pairs: "
                    f"{generated['unsupported_raw_pair_count']} "
                    f"({generated['unsupported_raw_pair_fraction']:.4%})."
                ),
                (
                    f"- Invalid raw candles using unsupported pairs: "
                    f"{generated['invalid_raw_unsupported_fraction']:.4%}."
                ),
            )
        )
    budget = probability.get("budget_1024")
    if budget:
        lines.extend(
            (
                "",
                "## Truncated probability-mass bounds",
                "",
                f"- Median considered mass: {budget['median_considered_mass']:.6f}.",
                (
                    f"- Median valid mass: "
                    f"[{budget['median_valid_mass_lower']:.6f}, "
                    f"{budget['median_valid_mass_upper']:.6f}]."
                ),
                (
                    f"- Median valid-and-supported mass: "
                    f"[{budget['median_valid_supported_mass_lower']:.6f}, "
                    f"{budget['median_valid_supported_mass_upper']:.6f}]."
                ),
                "- These are truncated bounds, not exact full-vocabulary mass.",
            )
        )
    lines.extend(
        (
            "",
            "## Governance",
            "",
            (
                "The tokenizer-defect gate did not authorize another decoder. "
                "No realized outcomes were needed for the classification, and "
                "the untouched holdout was not accessed."
            ),
            "",
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    operation = build_parser().parse_args(argv).operation
    if operation == "roundtrip":
        result = run_roundtrip()
    elif operation == "audit":
        result = run_audit()
    elif operation == "canary":
        result = run_canary()
    elif operation == "classify":
        result = run_classify()
    elif operation == "conditional":
        result = run_conditional()
    elif operation == "analyze":
        result = run_analyze()
    elif operation == "report":
        result = run_report()
    elif operation == "verify":
        result = run_verify()
    else:
        raise RuntimeError(f"operation is not implemented yet: {operation}")
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
