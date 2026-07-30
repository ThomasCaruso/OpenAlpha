from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

MODEL_REPOSITORY = "NeoQuasar/Kronos-mini"
MODEL_REVISION = "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"
TOKENIZER_REPOSITORY = "NeoQuasar/Kronos-Tokenizer-2k"
TOKENIZER_REVISION = "26966d0035065a0cae0ebad7af8ece35bc1fb51c"
SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
ALLOWED_CONTEXTS = {128, 256, 512}
ALLOWED_SEEDS = {1729, 2027, 7919}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        _validate_batch(payload)
        result = _run_batch(payload["requests"], args.source_path, args.cache_path)
    except Exception as error:  # noqa: BLE001 - worker must serialize all external failures
        result = {
            "status": "failure",
            "failure": {
                "code": "WORKER_FAILED",
                "message": f"{type(error).__name__}: {error}",
            },
        }
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


def _run_batch(
    requests: list[dict[str, Any]],
    source_path: Path,
    cache_path: Path,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import snapshot_download

    if not source_path.is_dir():
        raise ValueError("pinned Kronos source path is missing")
    cache_path.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        model_path = Path(
            snapshot_download(
                repo_id=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                cache_dir=cache_path,
                allow_patterns=("*.json", "*.safetensors"),
            )
        )
        tokenizer_path = Path(
            snapshot_download(
                repo_id=TOKENIZER_REPOSITORY,
                revision=TOKENIZER_REVISION,
                cache_dir=cache_path,
                allow_patterns=("*.json", "*.safetensors"),
            )
        )
    if model_path.name != MODEL_REVISION or tokenizer_path.name != TOKENIZER_REVISION:
        raise ValueError("Hugging Face snapshot did not resolve to the pinned revision")
    _reject_unsafe_snapshot(model_path)
    _reject_unsafe_snapshot(tokenizer_path)

    sys.path.insert(0, str(source_path))
    from model import Kronos, KronosTokenizer

    with contextlib.redirect_stdout(sys.stderr):
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
        model = Kronos.from_pretrained(str(model_path))
    device = _device(torch)
    tokenizer.eval()
    model.eval()
    environment = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "operating_system": platform.platform(),
        "device": device,
        "cache_path": str(cache_path.resolve()),
        "cache_size_bytes": _directory_size(cache_path),
        "downloaded_files": [
            *_downloaded_files(MODEL_REPOSITORY, MODEL_REVISION, model_path),
            *_downloaded_files(TOKENIZER_REPOSITORY, TOKENIZER_REVISION, tokenizer_path),
        ],
        "official_predictor_derives_amount": True,
    }
    responses = [
        _forecast_one(request, model, tokenizer, device, np, pd, torch)
        for request in requests
    ]
    return {"status": "success", "environment": environment, "responses": responses}


def _forecast_one(
    request: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: str,
    np: Any,
    pd: Any,
    torch: Any,
) -> dict[str, Any]:
    request_id = "req_" + hashlib.sha256(
        json.dumps(
            request,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    started = time.perf_counter()
    shared = {
        "provider_id": "kronos-local",
        "checkpoint_id": f"{MODEL_REPOSITORY}@{MODEL_REVISION}",
        "request_id": request_id,
    }
    try:
        _validate_request(request)
        _reset_rng(int(request["sampling_seed"]), torch, np)
        rows = request["observations"]
        frame = pd.DataFrame(
            [
                {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                for row in rows
            ]
        )
        if "amount" in frame.columns:
            raise ValueError("Sentinel must not supply an amount field")
        x_timestamp = pd.Series(pd.to_datetime([row["timestamp"] for row in rows], utc=True))
        y_timestamp = pd.Series(pd.to_datetime(request["forecast_sessions"], utc=True))
        from model import KronosPredictor

        predictor = KronosPredictor(
            model,
            tokenizer,
            device=device,
            max_context=int(request["context_length"]),
        )
        with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
            predicted = predictor.predict(
                df=frame,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=5,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )
        required = ("open", "high", "low", "close", "volume")
        if len(predicted) != 5 or any(column not in predicted.columns for column in required):
            raise ValueError("Kronos output cannot map to five declared sessions")
        values = predicted.loc[:, list(required)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Kronos output contains non-finite values")
        path = []
        for session, row in zip(request["forecast_sessions"], values):
            open_value, high, low, close, volume = (float(value) for value in row)
            path.append(
                {
                    "session": session,
                    "timestamp": f"{session}T00:00:00+00:00",
                    "open": open_value,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        return {
            **shared,
            "status": "success",
            "inference_duration_ms": (time.perf_counter() - started) * 1_000.0,
            "path": path,
        }
    except Exception as error:  # noqa: BLE001 - preserve each model/provider failure
        return {
            **shared,
            "status": "failure",
            "inference_duration_ms": (time.perf_counter() - started) * 1_000.0,
            "failure": {
                "code": "INFERENCE_FAILED",
                "message": f"{type(error).__name__}: {error}",
            },
        }


def _validate_batch(payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError("worker input must be an object")
    if payload.get("schema_version") != "sentinel-kronos-worker-v0":
        raise ValueError("worker schema version mismatch")
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("worker requires a nonempty request list")
    if len(requests) > 12:
        raise ValueError("worker batch exceeds the Phase 2 bound")
    if any(not isinstance(request, dict) for request in requests):
        raise ValueError("each worker request must be an object")


def _validate_request(request: dict[str, Any]) -> None:
    expected = {
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "source_revision": SOURCE_REVISION,
        "forecast_horizon": 5,
        "temperature": 1.0,
        "top_p": 0.9,
        "sample_count": 1,
    }
    for name, value in expected.items():
        if request.get(name) != value:
            raise ValueError(f"request field {name} does not match the experiment pin")
    context_length = request.get("context_length")
    seed = request.get("sampling_seed")
    observations = request.get("observations")
    forecast_sessions = request.get("forecast_sessions")
    if context_length not in ALLOWED_CONTEXTS:
        raise ValueError("undeclared context length")
    if seed not in ALLOWED_SEEDS:
        raise ValueError("undeclared seed")
    if not isinstance(observations, list) or len(observations) != context_length:
        raise ValueError("request must contain exactly the declared context length")
    if not isinstance(forecast_sessions, list) or len(forecast_sessions) != 5:
        raise ValueError("request must contain exactly five forecast sessions")
    if any("amount" in row for row in observations):
        raise ValueError("request observations must be OHLCV-only")


def _reset_rng(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _reject_unsafe_snapshot(snapshot: Path) -> None:
    allowed = {".json", ".safetensors"}
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    if not files:
        raise ValueError(f"empty model snapshot: {snapshot.name}")
    unsafe = [path.name for path in files if path.suffix.lower() not in allowed]
    if unsafe:
        raise ValueError(f"unsafe files in model snapshot: {unsafe}")


def _downloaded_files(
    repository: str,
    revision: str,
    snapshot: Path,
) -> list[dict[str, object]]:
    result = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        value = path.read_bytes()
        result.append(
            {
                "repository": repository,
                "revision": revision,
                "relative_path": path.relative_to(snapshot).as_posix(),
                "sha256": hashlib.sha256(value).hexdigest(),
                "size_bytes": len(value),
            }
        )
    return result


def _directory_size(root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        total += stat.st_size
    return total


if __name__ == "__main__":
    raise SystemExit(main())
