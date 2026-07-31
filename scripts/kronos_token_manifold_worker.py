from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import contextlib
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
EXPERIMENT_SHA256 = "6dacd9fd0912a77cce9d373d910b7bff9d58b44076d814960a191cdbe183bd76"
MODEL_REVISIONS = {
    "NeoQuasar/Kronos-mini": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
    "NeoQuasar/Kronos-small": "901c26c1332695a2a8f243eb2f37243a37bea320",
    "NeoQuasar/Kronos-base": "2b554741eca47781b64468546e77fef3e85130e6",
}
TOKENIZER_REVISIONS = {
    "NeoQuasar/Kronos-Tokenizer-2k": (
        "26966d0035065a0cae0ebad7af8ece35bc1fb51c"
    ),
    "NeoQuasar/Kronos-Tokenizer-base": (
        "0e0117387f39004a9016484a186a908917e22426"
    ),
}
MODEL_TOKENIZER_PAIRS = {
    "NeoQuasar/Kronos-mini": "NeoQuasar/Kronos-Tokenizer-2k",
    "NeoQuasar/Kronos-small": "NeoQuasar/Kronos-Tokenizer-base",
    "NeoQuasar/Kronos-base": "NeoQuasar/Kronos-Tokenizer-base",
}
ALLOWED_SEEDS = {1729, 2027, 7919}
ALLOWED_ORIGINS = {
    f"sentinel-v1-{symbol}-{cutoff}-h5"
    for symbol in ("SPY", "QQQ")
    for cutoff in (
        "2024-07-05",
        "2024-09-13",
        "2024-11-22",
        "2025-01-31",
        "2025-04-11",
        "2025-06-20",
    )
}
OBSERVATION_FIELDS = {"timestamp", "open", "high", "low", "close", "volume"}
COMPATIBILITY_FIELDS = {
    "schema_version",
    "operation",
    "experiment_sha256",
    "origin_id",
    "symbol",
    "cutoff",
    "model_repository",
    "model_revision",
    "tokenizer_repository",
    "tokenizer_revision",
    "source_revision",
    "context_length",
    "forecast_horizon",
    "sampling_seed",
    "temperature",
    "top_p",
    "top_k",
    "sample_count",
    "candidate_budgets",
    "support_coarse_counts",
    "support_fine_counts",
    "support_pair_counts",
    "forecast_sessions",
    "observations",
}
ROUNDTRIP_FIELDS = {
    "schema_version",
    "operation",
    "experiment_sha256",
    "source_revision",
    "tokenizer_repository",
    "tokenizer_revision",
    "windows",
}
SUPPORT_SYMBOLS = {
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "HYG",
    "GLD",
    "EFA",
    "EEM",
    "XLF",
}
CANDIDATE_DECODE_BATCH = 64


def _validate_request(request: Mapping[str, Any]) -> None:
    if request.get("schema_version") != "sentinel-kronos-token-manifold-worker-v1":
        raise ValueError("worker schema version mismatch")
    operation = request.get("operation")
    if operation not in {"roundtrip", "compatibility"}:
        raise ValueError("worker operation does not match the experiment lock")
    if operation == "roundtrip":
        _validate_roundtrip_request(request)
        return
    if set(request) != COMPATIBILITY_FIELDS:
        unknown = sorted(set(request) - COMPATIBILITY_FIELDS)
        missing = sorted(COMPATIBILITY_FIELDS - set(request))
        raise ValueError(f"unknown fields {unknown}; missing fields {missing}")
    locked = {
        "schema_version": "sentinel-kronos-token-manifold-worker-v1",
        "operation": "compatibility",
        "experiment_sha256": EXPERIMENT_SHA256,
        "source_revision": SOURCE_REVISION,
        "context_length": 512,
        "forecast_horizon": 5,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 0,
        "sample_count": 1,
    }
    for field, expected in locked.items():
        if request.get(field) != expected:
            raise ValueError(f"request field {field} does not match the lock")
    if request.get("candidate_budgets") != [64, 256, 1024]:
        raise ValueError("candidate budgets do not match the experiment lock")
    model_repository = request.get("model_repository")
    tokenizer_repository = request.get("tokenizer_repository")
    if model_repository not in MODEL_TOKENIZER_PAIRS:
        raise ValueError("model repository is not in the experiment lock")
    if tokenizer_repository not in TOKENIZER_REVISIONS:
        raise ValueError("tokenizer repository is not in the experiment lock")
    if MODEL_TOKENIZER_PAIRS[model_repository] != tokenizer_repository:
        raise ValueError("model and tokenizer pairing does not match the lock")
    if request.get("model_revision") != MODEL_REVISIONS[model_repository]:
        raise ValueError("model revision does not match the lock")
    if request.get("tokenizer_revision") != TOKENIZER_REVISIONS[tokenizer_repository]:
        raise ValueError("tokenizer revision does not match the lock")
    if request.get("origin_id") not in ALLOWED_ORIGINS:
        raise ValueError("origin is outside the locked development sample")
    if request.get("sampling_seed") not in ALLOWED_SEEDS:
        raise ValueError("seed is outside the experiment lock")
    observations = request.get("observations")
    if not isinstance(observations, list) or len(observations) != 512:
        raise ValueError("compatibility request requires exactly 512 observations")
    if any(
        not isinstance(row, dict) or set(row) != OBSERVATION_FIELDS
        for row in observations
    ):
        raise ValueError("observations must contain named timestamp and OHLCV fields")
    sessions = request.get("forecast_sessions")
    if not isinstance(sessions, list) or len(sessions) != 5:
        raise ValueError("compatibility request requires five forecast sessions")
    for field in (
        "support_coarse_counts",
        "support_fine_counts",
        "support_pair_counts",
    ):
        counts = request.get(field)
        if not isinstance(counts, dict) or any(
            not isinstance(key, str) or not isinstance(value, int) or value < 0
            for key, value in counts.items()
        ):
            raise ValueError(
                f"{field} must contain nonnegative integer counts"
            )


def _validate_roundtrip_request(request: Mapping[str, Any]) -> None:
    if set(request) != ROUNDTRIP_FIELDS:
        unknown = sorted(set(request) - ROUNDTRIP_FIELDS)
        missing = sorted(ROUNDTRIP_FIELDS - set(request))
        raise ValueError(f"unknown fields {unknown}; missing fields {missing}")
    locked = {
        "schema_version": "sentinel-kronos-token-manifold-worker-v1",
        "operation": "roundtrip",
        "experiment_sha256": EXPERIMENT_SHA256,
        "source_revision": SOURCE_REVISION,
    }
    for field, expected in locked.items():
        if request.get(field) != expected:
            raise ValueError(f"request field {field} does not match the lock")
    repository = request.get("tokenizer_repository")
    if repository not in TOKENIZER_REVISIONS:
        raise ValueError("tokenizer repository is outside the experiment lock")
    if request.get("tokenizer_revision") != TOKENIZER_REVISIONS[repository]:
        raise ValueError("tokenizer revision does not match the lock")
    windows = request.get("windows")
    if not isinstance(windows, list) or len(windows) != 30:
        raise ValueError("round-trip request requires exactly 30 windows")
    identities = []
    symbols = []
    for window in windows:
        if not isinstance(window, dict) or set(window) != {
            "window_id",
            "symbol",
            "observations",
        }:
            raise ValueError("round-trip window fields do not match the lock")
        symbol = window["symbol"]
        observations = window["observations"]
        if symbol not in SUPPORT_SYMBOLS:
            raise ValueError("round-trip symbol is outside the support corpus")
        if not isinstance(observations, list) or len(observations) != 512:
            raise ValueError("round-trip window requires exactly 512 observations")
        if any(
            not isinstance(row, dict) or set(row) != OBSERVATION_FIELDS
            for row in observations
        ):
            raise ValueError("round-trip observations require named OHLCV fields")
        identities.append(window["window_id"])
        symbols.append(symbol)
    if len(set(identities)) != 30:
        raise ValueError("round-trip window identities must be unique")
    if any(symbols.count(symbol) != 3 for symbol in SUPPORT_SYMBOLS):
        raise ValueError("round-trip corpus requires three windows per symbol")


def run_payload(
    payload: object,
    *,
    source_path: Path,
    cache_path: Path,
) -> dict[str, Any]:
    try:
        if not isinstance(payload, dict):
            raise TypeError("worker payload must be an object")
        _validate_payload(payload)
        if payload["schema_version"] == "sentinel-kronos-token-manifold-batch-v1":
            return _run_compatibility_batch(
                payload["requests"],
                source_path,
                cache_path,
            )
        if payload["operation"] == "roundtrip":
            return _run_roundtrip(payload, source_path, cache_path)
        return _run_compatibility(payload, source_path, cache_path)
    except Exception as error:  # noqa: BLE001 - typed external process boundary
        return {
            "status": "failure",
            "failure": {
                "code": "WORKER_FAILED",
                "message": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=8),
            },
        }


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "sentinel-kronos-token-manifold-batch-v1":
        _validate_request(payload)
        return
    if set(payload) != {"schema_version", "requests"}:
        raise ValueError("batch payload has unknown fields")
    requests = payload.get("requests")
    if not isinstance(requests, list) or not 1 <= len(requests) <= 3:
        raise ValueError("compatibility batch requires one to three requests")
    for request in requests:
        if not isinstance(request, dict):
            raise TypeError("compatibility batch requests must be objects")
        _validate_request(request)
    origins = {request["origin_id"] for request in requests}
    if len(origins) != 1:
        raise ValueError("compatibility batch must contain one origin")
    seeds = [request["sampling_seed"] for request in requests]
    if len(set(seeds)) != len(seeds):
        raise ValueError("compatibility batch seeds must be unique")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    args = parser.parse_args()
    result = run_payload(
        json.load(sys.stdin),
        source_path=args.source_path,
        cache_path=args.cache_path,
    )
    sys.stdout.write(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _require_source(source_path: Path) -> None:
    if not source_path.is_dir():
        raise ValueError("pinned Kronos source path is missing")
    revision = subprocess.run(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError("pinned Kronos source revision mismatch")


def _snapshot(
    repository: str,
    revision: str,
    cache_path: Path,
) -> Path:
    from huggingface_hub import snapshot_download

    with contextlib.redirect_stdout(sys.stderr):
        resolved = Path(
            snapshot_download(
                repo_id=repository,
                revision=revision,
                cache_dir=cache_path,
                allow_patterns=("*.json", "*.safetensors"),
            )
        )
    if resolved.name != revision:
        raise ValueError("Hugging Face snapshot did not resolve to the pinned revision")
    files = tuple(path for path in resolved.rglob("*") if path.is_file())
    if not files:
        raise ValueError("pinned snapshot is empty")
    if any(path.suffix.lower() not in {".json", ".safetensors"} for path in files):
        raise ValueError("pinned snapshot contains an unsafe file type")
    return resolved


def _snapshot_inventory(snapshot: Path) -> list[dict[str, Any]]:
    return [
        {
            "filename": path.relative_to(snapshot).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(path for path in snapshot.rglob("*") if path.is_file())
    ]


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _reset_rng(seed: int, *, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tokenizer(
    request: Mapping[str, Any],
    source_path: Path,
    cache_path: Path,
) -> tuple[Any, dict[str, Any], Any, Any, Any]:
    import numpy as np
    import pandas as pd
    import torch

    _require_source(source_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path = _snapshot(
        str(request["tokenizer_repository"]),
        str(request["tokenizer_revision"]),
        cache_path,
    )
    sys.path.insert(0, str(source_path))
    from model import KronosTokenizer

    with contextlib.redirect_stdout(sys.stderr):
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    device = _device(torch)
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    environment = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "operating_system": platform.platform(),
        "device": device,
        "cache_path": str(cache_path.resolve()),
        "cache_size_bytes": _directory_size(cache_path),
        "tokenizer_repository": request["tokenizer_repository"],
        "tokenizer_revision": request["tokenizer_revision"],
        "tokenizer_files": _snapshot_inventory(tokenizer_path),
        "source_revision": SOURCE_REVISION,
        "model_weights_modified": False,
    }
    return tokenizer, environment, np, pd, torch


def _run_roundtrip(
    request: dict[str, Any],
    source_path: Path,
    cache_path: Path,
) -> dict[str, Any]:
    tokenizer, environment, np, _, torch = _load_tokenizer(
        request,
        source_path,
        cache_path,
    )
    started = time.perf_counter()
    windows = []
    all_pairs: list[dict[str, int]] = []
    with torch.inference_mode():
        for descriptor in request["windows"]:
            values = _feature_values(descriptor["observations"], np)
            feature_mean = np.mean(values, axis=0)
            feature_std = np.std(values, axis=0)
            normalized = np.clip(
                (values - feature_mean) / (feature_std + 1e-5),
                -5.0,
                5.0,
            )
            tensor = torch.from_numpy(normalized[np.newaxis, :]).to(
                environment["device"]
            )
            encoded = tokenizer.encode(tensor, half=True)
            decoded = tokenizer.decode(encoded, half=True)
            reconstructed = (
                decoded[0].detach().cpu().numpy() * (feature_std + 1e-5)
                + feature_mean
            )
            coarse = encoded[0][0].detach().cpu().numpy().astype(int)
            fine = encoded[1][0].detach().cpu().numpy().astype(int)
            pairs = [
                {"coarse": int(left), "fine": int(right)}
                for left, right in zip(coarse, fine)
            ]
            all_pairs.extend(pairs)
            row_summaries = [
                _roundtrip_row_summary(observed, rebuilt)
                for observed, rebuilt in zip(values, reconstructed)
            ]
            windows.append(
                {
                    "window_id": descriptor["window_id"],
                    "symbol": descriptor["symbol"],
                    "row_count": len(values),
                    "input_sha256": _canonical_hash(
                        descriptor["observations"]
                    ),
                    "token_pair_count": len(pairs),
                    "token_pairs_sha256": _canonical_hash(pairs),
                    "reconstruction_sha256": _canonical_hash(row_summaries),
                    "rows": row_summaries,
                }
            )
    return {
        "status": "success",
        "operation": "roundtrip",
        "experiment_sha256": EXPERIMENT_SHA256,
        "environment": environment,
        "windows": windows,
        "token_pairs": all_pairs,
        "token_pair_count": len(all_pairs),
        "token_pairs_sha256": _canonical_hash(all_pairs),
        "duration_ms": (time.perf_counter() - started) * 1_000.0,
    }


def _roundtrip_row_summary(
    observed: Sequence[float],
    reconstructed: Sequence[float],
) -> dict[str, Any]:
    original = [float(value) for value in observed]
    rebuilt = [float(value) for value in reconstructed]
    denominator = original[3]
    if denominator <= 0.0 or not all(
        math.isfinite(value) for value in (*original, *rebuilt)
    ):
        raise ValueError("round-trip values require finite positive observed close")
    violations = _candle_violations(rebuilt)
    errors = [
        abs(rebuilt[index] - original[index]) / denominator
        for index in range(4)
    ]
    range_error = abs(
        (rebuilt[1] - rebuilt[2]) - (original[1] - original[2])
    ) / denominator
    return {
        "valid": not violations,
        "violation_codes": violations,
        "normalized_errors": {
            "open": errors[0],
            "high": errors[1],
            "low": errors[2],
            "close": errors[3],
            "range": range_error,
        },
    }


def _load_model_and_tokenizer(
    request: Mapping[str, Any],
    source_path: Path,
    cache_path: Path,
) -> tuple[Any, Any, dict[str, Any], Any, Any, Any]:
    tokenizer, environment, np, pd, torch = _load_tokenizer(
        request,
        source_path,
        cache_path,
    )
    model_path = _snapshot(
        str(request["model_repository"]),
        str(request["model_revision"]),
        cache_path,
    )
    from model import Kronos

    with contextlib.redirect_stdout(sys.stderr):
        model = Kronos.from_pretrained(str(model_path))
    model = model.to(environment["device"])
    model.eval()
    environment.update(
        {
            "model_repository": request["model_repository"],
            "model_revision": request["model_revision"],
            "model_files": _snapshot_inventory(model_path),
            "cache_size_bytes": _directory_size(cache_path),
        }
    )
    return model, tokenizer, environment, np, pd, torch


def _run_compatibility(
    request: dict[str, Any],
    source_path: Path,
    cache_path: Path,
) -> dict[str, Any]:
    model, tokenizer, environment, np, pd, torch = _load_model_and_tokenizer(
        request,
        source_path,
        cache_path,
    )
    return _run_compatibility_loaded(
        request=request,
        model=model,
        tokenizer=tokenizer,
        environment=environment,
        np=np,
        pd=pd,
        torch=torch,
    )


def _run_compatibility_batch(
    requests: list[dict[str, Any]],
    source_path: Path,
    cache_path: Path,
) -> dict[str, Any]:
    model, tokenizer, environment, np, pd, torch = _load_model_and_tokenizer(
        requests[0],
        source_path,
        cache_path,
    )
    responses = [
        _run_compatibility_loaded(
            request=request,
            model=model,
            tokenizer=tokenizer,
            environment=environment,
            np=np,
            pd=pd,
            torch=torch,
        )
        for request in requests
    ]
    return {
        "status": "success",
        "operation": "compatibility_batch",
        "experiment_sha256": EXPERIMENT_SHA256,
        "origin_id": requests[0]["origin_id"],
        "environment": environment,
        "responses": responses,
    }


def _run_compatibility_loaded(
    *,
    request: dict[str, Any],
    model: Any,
    tokenizer: Any,
    environment: dict[str, Any],
    np: Any,
    pd: Any,
    torch: Any,
) -> dict[str, Any]:
    import kronos_constrained_worker as v1_worker

    prepared = v1_worker._prepare_inputs(
        request,
        environment["device"],
        np,
        pd,
        torch,
    )
    _reset_rng(int(request["sampling_seed"]), torch=torch, np=np)
    started = time.perf_counter()
    with torch.inference_mode():
        encoded = tokenizer.encode(torch.clip(prepared["x"], -5.0, 5.0), half=True)
        pre_buffer = encoded[0].clone()
        post_buffer = encoded[1].clone()
        full_stamp = torch.cat(
            [prepared["x_stamp"], prepared["y_stamp"]],
            dim=1,
        )
        generated_pairs = []
        step_records = []
        for index in range(5):
            current_stamp = full_stamp[:, index : index + 512, :].contiguous()
            step = _compatibility_step(
                request=request,
                step=index + 1,
                model=model,
                tokenizer=tokenizer,
                pre_buffer=pre_buffer,
                post_buffer=post_buffer,
                current_stamp=current_stamp,
                x_mean=prepared["x_mean"],
                x_std=prepared["x_std"],
                v1_worker=v1_worker,
                np=np,
                torch=torch,
            )
            selected = (
                int(step["raw_token"]["coarse"]),
                int(step["raw_token"]["fine"]),
            )
            v1_worker._append_pair(pre_buffer, post_buffer, selected, torch)
            generated_pairs.append(
                {"coarse": selected[0], "fine": selected[1]}
            )
            step_records.append(step)
        final_normalized = tokenizer.decode([pre_buffer, post_buffer], half=True)
        final_values = v1_worker._inverse_normalize(
            final_normalized[0, -5:, :].detach().cpu().numpy(),
            prepared["x_mean"],
            prepared["x_std"],
        )
    path = [
        v1_worker._row_payload(
            [float(value) for value in row],
            session,
        )
        for row, session in zip(final_values, request["forecast_sessions"])
    ]
    return {
        "status": "success",
        "operation": "compatibility",
        "experiment_sha256": EXPERIMENT_SHA256,
        "origin_id": request["origin_id"],
        "sampling_seed": request["sampling_seed"],
        "environment": environment,
        "generated_token_pairs": generated_pairs,
        "steps": step_records,
        "path": path,
        "path_sha256": _canonical_hash(path),
        "duration_ms": (time.perf_counter() - started) * 1_000.0,
    }


def _compatibility_step(
    *,
    request: dict[str, Any],
    step: int,
    model: Any,
    tokenizer: Any,
    pre_buffer: Any,
    post_buffer: Any,
    current_stamp: Any,
    x_mean: Any,
    x_std: Any,
    v1_worker: Any,
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    from model.kronos import sample_from_logits

    s1_logits, context = model.decode_s1(
        pre_buffer,
        post_buffer,
        current_stamp,
    )
    s1_last = s1_logits[:, -1, :]
    raw_pre = sample_from_logits(
        s1_last.clone(),
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        sample_logits=True,
    )
    s2_logits = model.decode_s2(context, raw_pre)
    s2_last = s2_logits[:, -1, :]
    raw_post = sample_from_logits(
        s2_last.clone(),
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        sample_logits=True,
    )
    raw_pair = (int(raw_pre.item()), int(raw_post.item()))
    unfiltered_coarse = torch.log_softmax(s1_last, dim=-1)
    unfiltered_fine = torch.log_softmax(s2_last, dim=-1)
    filtered_coarse = torch.exp(
        v1_worker._filtered_log_probabilities(s1_last, torch)
    )
    filtered_fine = torch.exp(
        v1_worker._filtered_log_probabilities(s2_last, torch)
    )
    _, raw_rows = v1_worker._decode_candidate_batch(
        tokenizer=tokenizer,
        pre_buffer=pre_buffer,
        post_buffer=post_buffer,
        pairs=(raw_pair,),
        x_mean=x_mean,
        x_std=x_std,
        torch=torch,
    )
    raw_row = [float(value) for value in raw_rows[0]]
    sides = tuple(
        int(math.sqrt(int(budget))) for budget in request["candidate_budgets"]
    )
    budget_results = _evaluate_candidate_budgets(
        model=model,
        tokenizer=tokenizer,
        context=context,
        s1_last=s1_last,
        pre_buffer=pre_buffer,
        post_buffer=post_buffer,
        x_mean=x_mean,
        x_std=x_std,
        pair_counts=request["support_pair_counts"],
        sides=sides,
        v1_worker=v1_worker,
        np=np,
        torch=torch,
    )
    pair_key = f"{raw_pair[0]}:{raw_pair[1]}"
    return {
        "step": step,
        "raw_token": {"coarse": raw_pair[0], "fine": raw_pair[1]},
        "raw_decoded_candle": raw_row,
        "raw_valid": not _candle_violations(raw_row),
        "raw_violations": _candle_violations(raw_row),
        "support": {
            "coarse_count": int(
                request["support_coarse_counts"].get(str(raw_pair[0]), 0)
            ),
            "fine_count": int(
                request["support_fine_counts"].get(str(raw_pair[1]), 0)
            ),
            "exact_pair_count": int(
                request["support_pair_counts"].get(pair_key, 0)
            ),
        },
        "model_support": {
            "coarse_log_probability": float(
                unfiltered_coarse[0, raw_pair[0]].item()
            ),
            "fine_log_probability": float(
                unfiltered_fine[0, raw_pair[1]].item()
            ),
            "coarse_rank": _rank_of(raw_pair[0], unfiltered_coarse, torch),
            "fine_rank": _rank_of(raw_pair[1], unfiltered_fine, torch),
            "top_p_joint_probability": float(
                filtered_coarse[0, raw_pair[0]].item()
                * filtered_fine[0, raw_pair[1]].item()
            ),
        },
        "probability_mass": budget_results,
    }


def _rank_of(token: int, log_probabilities: Any, torch: Any) -> int:
    selected = log_probabilities[0, token]
    return int(torch.sum(log_probabilities[0] > selected).item()) + 1


def _evaluate_candidate_budgets(
    *,
    model: Any,
    tokenizer: Any,
    context: Any,
    s1_last: Any,
    pre_buffer: Any,
    post_buffer: Any,
    x_mean: Any,
    x_std: Any,
    pair_counts: Mapping[str, int],
    sides: Sequence[int],
    v1_worker: Any,
    np: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    coarse_limit = max(sides)
    fine_limit = max(sides)
    coarse_log = v1_worker._filtered_log_probabilities(s1_last, torch)
    coarse_probabilities = torch.exp(coarse_log)[0]
    _, coarse_ids_tensor = torch.topk(
        coarse_probabilities,
        k=coarse_limit,
        dim=-1,
    )
    coarse_ids = [int(value) for value in coarse_ids_tensor.tolist()]
    repeated_context = context.repeat(coarse_limit, 1, 1)
    coarse_batch = torch.tensor(
        coarse_ids,
        dtype=torch.long,
        device=s1_last.device,
    ).reshape(coarse_limit, 1)
    fine_logits = model.decode_s2(repeated_context, coarse_batch)[:, -1, :]
    fine_log = v1_worker._filtered_log_probabilities(fine_logits, torch)
    fine_probabilities = torch.exp(fine_log)
    fine_by_coarse = {}
    pairs = []
    for row_index, coarse in enumerate(coarse_ids):
        probabilities = fine_probabilities[row_index].detach().cpu().numpy()
        fine_by_coarse[coarse] = probabilities
        fine_ids = np.argsort(-probabilities, kind="stable")[:fine_limit]
        pairs.extend((coarse, int(fine)) for fine in fine_ids)
    decoded_by_pair = {}
    for start in range(0, len(pairs), CANDIDATE_DECODE_BATCH):
        batch = tuple(pairs[start : start + CANDIDATE_DECODE_BATCH])
        _, rows = v1_worker._decode_candidate_batch(
            tokenizer=tokenizer,
            pre_buffer=pre_buffer,
            post_buffer=post_buffer,
            pairs=batch,
            x_mean=x_mean,
            x_std=x_std,
            torch=torch,
        )
        for pair, row in zip(batch, rows):
            decoded_by_pair[f"{pair[0]}:{pair[1]}"] = [
                float(value) for value in row
            ]
    grids = _nested_candidate_grids_from_arrays(
        coarse_probabilities=coarse_probabilities.detach().cpu().numpy(),
        coarse_ids=coarse_ids,
        fine_probabilities_by_coarse=fine_by_coarse,
        decoded_by_pair=decoded_by_pair,
        pair_counts=pair_counts,
        sides=sides,
        np=np,
    )
    return [
        _candidate_mass_result(records, considered_mass, side)
        for side, (records, considered_mass) in grids.items()
    ]


def _candidate_mass_result(
    records: list[dict[str, Any]],
    considered_mass: float,
    side: int,
) -> dict[str, Any]:
    considered_mass = min(1.0, max(0.0, considered_mass))
    valid_mass = sum(
        float(item["joint_probability"]) for item in records if item["valid"]
    )
    supported_mass = sum(
        float(item["joint_probability"])
        for item in records
        if int(item["exact_pair_count"]) >= 2
    )
    valid_supported_mass = sum(
        float(item["joint_probability"])
        for item in records
        if item["valid"] and int(item["exact_pair_count"]) >= 2
    )
    uncovered = max(0.0, 1.0 - considered_mass)
    identifiers = [
        {
            "coarse": int(item["coarse_token"]),
            "fine": int(item["fine_token"]),
        }
        for item in records
    ]
    return {
        "candidate_budget": side * side,
        "coarse_candidates": side,
        "fine_candidates_per_coarse": side,
        "candidate_count": len(records),
        "candidate_pairs_sha256": _canonical_hash(identifiers),
        "considered_probability_mass": considered_mass,
        "uncovered_tail_mass": uncovered,
        "valid_probability_mass_lower": valid_mass,
        "valid_probability_mass_upper": min(1.0, valid_mass + uncovered),
        "supported_probability_mass_lower": supported_mass,
        "supported_probability_mass_upper": min(
            1.0,
            supported_mass + uncovered,
        ),
        "valid_supported_probability_mass_lower": valid_supported_mass,
        "valid_supported_probability_mass_upper": min(
            1.0,
            valid_supported_mass + uncovered,
        ),
        "constraint_tax": (
            -math.log(valid_mass) if valid_mass > 0.0 else None
        ),
        "constraint_tax_status": (
            "COMPUTED_LOWER_BOUND"
            if valid_mass > 0.0
            else "ZERO_ESTIMATED_MASS"
        ),
        "support_constraint_tax": (
            -math.log(valid_supported_mass)
            if valid_supported_mass > 0.0
            else None
        ),
        "support_constraint_tax_status": (
            "COMPUTED_LOWER_BOUND"
            if valid_supported_mass > 0.0
            else "ZERO_ESTIMATED_MASS"
        ),
        "valid_candidate_count": sum(bool(item["valid"]) for item in records),
        "supported_candidate_count": sum(
            int(item["exact_pair_count"]) >= 2 for item in records
        ),
        "valid_supported_candidate_count": sum(
            bool(item["valid"]) and int(item["exact_pair_count"]) >= 2
            for item in records
        ),
        "exact": uncovered <= 1e-12,
    }


def _feature_values(rows: Sequence[Mapping[str, Any]], np: Any) -> Any:
    values = []
    for row in rows:
        try:
            ohlcv = [
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("feature rows require named numeric OHLCV") from exc
        if not all(math.isfinite(value) for value in ohlcv):
            raise ValueError("feature values must be finite")
        amount = ohlcv[4] * sum(ohlcv[:4]) / 4.0
        values.append([*ohlcv, amount])
    if not values:
        raise ValueError("feature preparation requires at least one row")
    return np.asarray(values, dtype=np.float32)


def _top_p_probabilities(logits: Any, *, top_p: float, np: Any) -> Any:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("top-p logits must be a finite vector")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top-p must be finite and in (0, 1]")
    shifted = values - float(np.max(values))
    probabilities = np.exp(shifted)
    probabilities /= float(np.sum(probabilities))
    order = np.argsort(-probabilities, kind="stable")
    ordered = probabilities[order]
    prior_mass = np.cumsum(ordered) - ordered
    keep = prior_mass < top_p
    keep[0] = True
    retained = np.zeros_like(probabilities)
    retained[order[keep]] = probabilities[order[keep]]
    retained /= float(np.sum(retained))
    return retained


def _auxiliary_fraction(
    *,
    experiment_sha256: str,
    origin_id: str,
    seed: int,
    step: int,
) -> float:
    if len(experiment_sha256) != 64 or not origin_id or seed < 0 or step < 1:
        raise ValueError("deterministic selection identity is invalid")
    payload = (
        f"sentinel-v1.1-support-selection|{experiment_sha256}|"
        f"{origin_id}|{seed}|{step}"
    ).encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def _candidate_grid_from_arrays(
    *,
    coarse_probabilities: Any,
    fine_probabilities_by_coarse: Mapping[int, Any],
    decoded_by_pair: Mapping[str, Sequence[float]],
    pair_counts: Mapping[str, int],
    coarse_limit: int,
    fine_limit: int,
    np: Any,
    coarse_ids: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    coarse_values = np.asarray(coarse_probabilities, dtype=float)
    if coarse_values.ndim != 1 or coarse_limit < 1 or fine_limit < 1:
        raise ValueError("candidate grid dimensions are invalid")
    selected_coarse_ids = (
        np.argsort(-coarse_values, kind="stable")[:coarse_limit]
        if coarse_ids is None
        else coarse_ids
    )
    if len(selected_coarse_ids) != coarse_limit:
        raise ValueError("candidate grid coarse identifiers do not match the limit")
    records = []
    for coarse_value in selected_coarse_ids:
        coarse = int(coarse_value)
        fine_values = np.asarray(
            fine_probabilities_by_coarse[coarse],
            dtype=float,
        )
        fine_ids = np.argsort(-fine_values, kind="stable")[:fine_limit]
        for fine_value in fine_ids:
            fine = int(fine_value)
            key = f"{coarse}:{fine}"
            row = [float(value) for value in decoded_by_pair[key]]
            probability = float(coarse_values[coarse] * fine_values[fine])
            violations = _candle_violations(row)
            records.append(
                {
                    "coarse_token": coarse,
                    "fine_token": fine,
                    "joint_probability": probability,
                    "joint_log_probability": (
                        math.log(probability) if probability > 0.0 else None
                    ),
                    "exact_pair_count": int(pair_counts.get(key, 0)),
                    "valid": not violations,
                    "violations": violations,
                    "decoded_candle": row,
                }
            )
    records.sort(
        key=lambda item: (
            -float(item["joint_probability"]),
            int(item["coarse_token"]),
            int(item["fine_token"]),
        )
    )
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    return records, sum(float(item["joint_probability"]) for item in records)


def _nested_candidate_grids_from_arrays(
    *,
    coarse_probabilities: Any,
    coarse_ids: Sequence[int],
    fine_probabilities_by_coarse: Mapping[int, Any],
    decoded_by_pair: Mapping[str, Sequence[float]],
    pair_counts: Mapping[str, int],
    sides: Sequence[int],
    np: Any,
) -> dict[int, tuple[list[dict[str, Any]], float]]:
    if not sides or tuple(sides) != tuple(sorted(set(sides))):
        raise ValueError("candidate grid sides must be unique and increasing")
    if len(coarse_ids) < max(sides):
        raise ValueError("decoded candidate superset is too small")
    return {
        side: _candidate_grid_from_arrays(
            coarse_probabilities=coarse_probabilities,
            fine_probabilities_by_coarse=fine_probabilities_by_coarse,
            decoded_by_pair=decoded_by_pair,
            pair_counts=pair_counts,
            coarse_limit=side,
            fine_limit=side,
            np=np,
            coarse_ids=coarse_ids[:side],
        )
        for side in sides
    }


def _select_supported_record(
    records: Sequence[dict[str, Any]],
    *,
    auxiliary_fraction: float,
    minimum_pair_count: int,
) -> dict[str, Any]:
    if not 0.0 <= auxiliary_fraction < 1.0:
        raise ValueError("auxiliary fraction must be in [0, 1)")
    eligible = [
        item
        for item in records
        if item.get("valid") is True
        and int(item.get("exact_pair_count", 0)) >= minimum_pair_count
    ]
    if not eligible:
        raise ValueError("no valid supported candidate within the declared budget")
    total = sum(float(item["joint_probability"]) for item in eligible)
    threshold = auxiliary_fraction * total
    cumulative = 0.0
    selected = eligible[-1]
    for record in eligible:
        cumulative += float(record["joint_probability"])
        if threshold < cumulative:
            selected = record
            break
    return selected


def _candle_violations(row: Sequence[float]) -> list[str]:
    if len(row) < 5:
        raise ValueError("decoded candle requires OHLCV")
    open_value, high, low, close, volume = (float(value) for value in row[:5])
    values = (open_value, high, low, close, volume)
    if not all(math.isfinite(value) for value in values):
        return ["NONFINITE_OUTPUT"]
    violations = []
    for label, value in (
        ("OPEN", open_value),
        ("HIGH", high),
        ("LOW", low),
        ("CLOSE", close),
    ):
        if value <= 0.0:
            violations.append(f"NONPOSITIVE_{label}")
    if volume < 0.0:
        violations.append("NEGATIVE_VOLUME")
    if high < open_value:
        violations.append("HIGH_BELOW_OPEN")
    if high < close:
        violations.append("HIGH_BELOW_CLOSE")
    if high < low:
        violations.append("HIGH_BELOW_LOW")
    if low > open_value:
        violations.append("LOW_ABOVE_OPEN")
    if low > close:
        violations.append("LOW_ABOVE_CLOSE")
    return violations


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
