from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import exchange_calendars as xcals
import pandas as pd
from openalpha_research import ArtifactRef, LocalArtifactStore

from .contracts import OHLCVObservation
from .structural_validity import AuditCandle, PathValidity, validate_forecast_path


@dataclass(frozen=True)
class Phase25Paths:
    phase2_root: Path
    phase2_5_root: Path

    def __post_init__(self) -> None:
        phase2 = self.phase2_root.resolve()
        phase2_5 = self.phase2_5_root.resolve()
        if phase2 == phase2_5 or phase2 in phase2_5.parents or phase2_5 in phase2.parents:
            raise ValueError("Phase 2 and Phase 2.5 roots must be disjoint")
        object.__setattr__(self, "phase2_root", phase2)
        object.__setattr__(self, "phase2_5_root", phase2_5)


@dataclass(frozen=True)
class PrivateReceipt:
    sha256: str
    path: Path
    size_bytes: int


def phase2_fingerprint(state_root: Path) -> dict[str, Any]:
    root = state_root.resolve()
    descriptors: dict[str, dict[str, object]] = {}
    for name in ("creation.json", "resolved.json"):
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing sealed Phase 2 descriptor: {name}")
        descriptors[name] = _file_summary(path)
    artifact_root = root / "artifacts"
    if not artifact_root.is_dir():
        raise ValueError("missing sealed Phase 2 artifact store")
    artifacts = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            **_file_summary(path),
        }
        for path in sorted(item for item in artifact_root.rglob("*") if item.is_file())
    ]
    return {
        "schema_version": "sentinel-phase2-fingerprint-v0",
        "root": str(root),
        "descriptors": descriptors,
        "artifacts": artifacts,
        "inventory_sha256": _canonical_sha256(
            {
                "descriptors": descriptors,
                "artifacts": artifacts,
            }
        ),
    }


def publish_private_receipt(
    paths: Phase25Paths,
    payload: dict[str, object],
) -> PrivateReceipt:
    paths.phase2_5_root.mkdir(parents=True, exist_ok=True)
    store = LocalArtifactStore(paths.phase2_5_root / "artifacts")
    content = _canonical_json_bytes(payload)
    ref = store.put_bytes(content, media_type="application/json")
    return PrivateReceipt(
        sha256=ref.sha256,
        path=(store.root / ref.relative_path).resolve(),
        size_bytes=ref.size_bytes,
    )


def load_private_receipt(paths: Phase25Paths, sha256: str) -> dict[str, Any]:
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError("receipt SHA-256 must be lowercase hexadecimal")
    store = LocalArtifactStore(paths.phase2_5_root / "artifacts")
    relative_path = f"sha256/{sha256[:2]}/{sha256}"
    path = store.root / Path(relative_path)
    if not path.is_file():
        raise ValueError(f"missing private receipt: {sha256}")
    ref = ArtifactRef(
        sha256=sha256,
        size_bytes=path.stat().st_size,
        media_type="application/json",
        relative_path=relative_path,
    )
    payload = json.loads(store.read_bytes(ref))
    if not isinstance(payload, dict):
        raise TypeError("private receipt payload must be an object")
    return payload


def classify_phase2_5(
    *,
    direct_official_invalid: bool,
    openalpha_output_invalid: bool,
    direct_equals_openalpha: bool,
    repeatable: bool,
    canary_invalid_path_count: int,
) -> str:
    if canary_invalid_path_count < 0:
        raise ValueError("canary invalid path count cannot be negative")
    if not direct_official_invalid and openalpha_output_invalid and not direct_equals_openalpha:
        return "OPENALPHA_INTEGRATION_BUG"
    boundary_isolated = (
        direct_official_invalid
        and openalpha_output_invalid
        and direct_equals_openalpha
        and repeatable
    )
    if boundary_isolated and canary_invalid_path_count > 0:
        return "OFFICIAL_RAW_PATH_STRUCTURAL_INVALIDITY_CONFIRMED"
    if boundary_isolated and canary_invalid_path_count == 0:
        return "NONRECURRING_EDGE_CASE"
    return "AMBIGUOUS"


def next_xnys_sessions(cutoff: date, count: int = 5) -> tuple[date, ...]:
    if count < 1:
        raise ValueError("session count must be positive")
    calendar = xcals.get_calendar("XNYS")
    if not calendar.is_session(cutoff):
        raise ValueError(f"cutoff is not an XNYS session: {cutoff}")
    sessions: list[date] = []
    current = cast(pd.Timestamp, calendar.date_to_session(cutoff))
    for _ in range(count):
        current = calendar.next_session(current)
        sessions.append(current.date())
    return tuple(sessions)


def build_path_comparison(
    *,
    direct_path: list[dict[str, object]],
    provider_path: list[dict[str, object]],
    sealed_path: list[dict[str, object]],
    observed_input_sha256: str,
    sealed_input_sha256: str,
) -> dict[str, object]:
    direct = _normalized_path(direct_path)
    provider = _normalized_path(provider_path)
    sealed = _normalized_path(sealed_path)
    input_match = observed_input_sha256 == sealed_input_sha256
    return {
        "observed_input_sha256": observed_input_sha256,
        "sealed_input_sha256": sealed_input_sha256,
        "exact_comparison_permitted": input_match,
        "direct_path_sha256": _canonical_sha256(direct),
        "provider_path_sha256": _canonical_sha256(provider),
        "sealed_path_sha256": _canonical_sha256(sealed),
        "direct_equals_provider": direct == provider,
        "direct_equals_sealed": direct == sealed if input_match else None,
        "provider_equals_sealed": provider == sealed if input_match else None,
        "first_direct_provider_difference": _first_difference(direct, provider),
        "first_direct_sealed_difference": (
            _first_difference(direct, sealed) if input_match else None
        ),
        "limitation": None if input_match else "NORMALIZED_INPUT_HASH_MISMATCH",
    }


def make_direct_request(
    *,
    audit_case: str,
    observations: Sequence[OHLCVObservation],
    forecast_timestamps: Sequence[date | datetime],
    context_length: int,
    seed: int,
    sample_count: int,
    trace: bool,
) -> dict[str, object]:
    rows = tuple(observations)
    if len(rows) < context_length:
        raise ValueError("insufficient observations for direct request")
    future = tuple(forecast_timestamps)
    if len(future) != 5:
        raise ValueError("direct request requires five forecast timestamps")
    context = rows[-context_length:]
    return {
        "audit_case": audit_case,
        "model_repository": "NeoQuasar/Kronos-mini",
        "model_revision": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
        "tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-2k",
        "tokenizer_revision": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        "source_revision": "67b630e67f6a18c9e9be918d9b4337c960db1e9a",
        "context_length": context_length,
        "sampling_seed": seed,
        "temperature": 1.0,
        "top_p": 0.9,
        "sample_count": sample_count,
        "forecast_horizon": 5,
        "observations": [
            {
                "timestamp": row.timestamp.isoformat(),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
            }
            for row in context
        ],
        "forecast_sessions": [item.isoformat() for item in future],
        "trace": trace,
    }


def run_direct_worker(
    *,
    requests: list[dict[str, object]],
    inference_python: Path,
    worker_script: Path,
    source_path: Path,
    cache_path: Path,
    trace_helper_path: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    command = [
        str(inference_python),
        str(worker_script),
        "--source-path",
        str(source_path),
        "--cache-path",
        str(cache_path),
        "--trace-helper-path",
        str(trace_helper_path),
    ]
    completed = runner(
        command,
        input=json.dumps(
            {
                "schema_version": "sentinel-kronos-phase2_5-worker-v0",
                "requests": requests,
            },
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        text=True,
        capture_output=True,
        timeout=1_800.0,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Phase 2.5 worker process failed with exit {completed.returncode}: "
            f"{completed.stderr[-2_000:]}"
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise TypeError("Phase 2.5 worker returned a non-object")
    if payload.get("status") != "success":
        failure = payload.get("failure")
        raise RuntimeError(f"Phase 2.5 worker failed: {failure}")
    responses = payload.get("responses")
    if not isinstance(responses, list) or len(responses) != len(requests):
        raise RuntimeError("Phase 2.5 worker response count mismatch")
    return payload


def average_path_payloads(
    paths: Sequence[list[dict[str, object]]],
) -> list[dict[str, object]]:
    members = tuple(paths)
    if not members:
        raise ValueError("at least one path is required for averaging")
    horizon = len(members[0])
    if horizon == 0 or any(len(path) != horizon for path in members):
        raise ValueError("paths must have the same nonzero horizon")
    numeric_fields = ("open", "high", "low", "close", "volume")
    include_amount = all("amount" in row for path in members for row in path)
    if include_amount:
        numeric_fields = (*numeric_fields, "amount")
    averaged: list[dict[str, object]] = []
    for step in range(horizon):
        first = members[0][step]
        timestamp = first.get("timestamp")
        if not isinstance(timestamp, str):
            raise TypeError("path timestamp is required")
        if any(path[step].get("timestamp") != timestamp for path in members):
            raise ValueError("path timestamps must match before averaging")
        row: dict[str, object] = {"timestamp": timestamp}
        if "session" in first:
            session = first["session"]
            if any(path[step].get("session") != session for path in members):
                raise ValueError("path sessions must match before averaging")
            row["session"] = session
        for field in numeric_fields:
            numeric_values: list[float] = []
            for path in members:
                value = path[step].get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"path field {field} must be numeric")
                numeric_values.append(float(value))
            row[field] = sum(numeric_values) / len(numeric_values)
        averaged.append(row)
    return averaged


def path_validity_from_payload(
    *,
    path_id: str,
    path: list[dict[str, object]],
    expected_timestamps: Sequence[date | datetime],
    cutoff_close: float,
    cutoff_volume: float,
) -> PathValidity:
    expected = tuple(expected_timestamps)
    intraday = any(isinstance(value, datetime) for value in expected)
    candles: list[AuditCandle] = []
    for row in path:
        timestamp = row.get("timestamp")
        if not isinstance(timestamp, str):
            parsed: date | datetime | None = None
        else:
            parsed_timestamp = pd.Timestamp(timestamp)
            if pd.isna(parsed_timestamp):
                parsed = None
            else:
                parsed = (
                    datetime.fromisoformat(parsed_timestamp.isoformat())
                    if intraday
                    else date.fromisoformat(parsed_timestamp.date().isoformat())
                )
        volume = row.get("volume")
        candles.append(
            AuditCandle(
                session=parsed,
                open=_numeric_path_value(row, "open"),
                high=_numeric_path_value(row, "high"),
                low=_numeric_path_value(row, "low"),
                close=_numeric_path_value(row, "close"),
                volume=(
                    float(volume)
                    if isinstance(volume, (int, float)) and not isinstance(volume, bool)
                    else None
                ),
            )
        )
    return validate_forecast_path(
        path_id=path_id,
        candles=tuple(candles),
        expected_sessions=expected,
        cutoff_close=cutoff_close,
        cutoff_volume=cutoff_volume,
    )


def raw_log_return(path: list[dict[str, object]], *, cutoff_close: float) -> float:
    if not path:
        raise ValueError("return requires a nonempty path")
    final_close = _numeric_path_value(path[-1], "close")
    if cutoff_close <= 0.0 or final_close <= 0.0:
        raise ValueError("return requires positive closes")
    return math.log(final_close / cutoff_close)


def _normalized_path(path: list[dict[str, object]]) -> list[dict[str, object]]:
    required = ("open", "high", "low", "close", "volume")
    result: list[dict[str, object]] = []
    for row in path:
        timestamp = row.get("timestamp")
        if not isinstance(timestamp, str):
            raise TypeError("path row requires timestamp")
        session_value = row.get("session")
        session = (
            str(session_value)
            if session_value is not None
            else pd.Timestamp(timestamp).date().isoformat()
        )
        normalized: dict[str, object] = {
            "session": session,
            "timestamp": pd.Timestamp(timestamp).isoformat(),
        }
        for field in required:
            value = row.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"path row requires numeric {field}")
            normalized[field] = float(value)
        result.append(normalized)
    return result


def _numeric_path_value(row: dict[str, object], field: str) -> float:
    value = row.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"path row requires numeric {field}")
    return float(value)


def _first_difference(
    first: list[dict[str, object]],
    second: list[dict[str, object]],
) -> dict[str, object] | None:
    if len(first) != len(second):
        return {
            "field": "horizon_length",
            "step": 0,
            "direct": len(first),
            "other": len(second),
        }
    fields = ("session", "timestamp", "open", "high", "low", "close", "volume")
    for step, (left, right) in enumerate(zip(first, second), start=1):
        for field in fields:
            if left[field] != right[field]:
                return {
                    "field": field,
                    "step": step,
                    "direct": left[field],
                    "other": right[field],
                }
    return None


def _file_summary(path: Path) -> dict[str, object]:
    value = path.read_bytes()
    return {
        "sha256": hashlib.sha256(value).hexdigest(),
        "size_bytes": len(value),
    }


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
