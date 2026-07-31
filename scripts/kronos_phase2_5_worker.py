from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import contextlib
import hashlib
import importlib.util
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

MODEL_REPOSITORY = "NeoQuasar/Kronos-mini"
MODEL_REVISION = "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"
TOKENIZER_REPOSITORY = "NeoQuasar/Kronos-Tokenizer-2k"
TOKENIZER_REVISION = "26966d0035065a0cae0ebad7af8ece35bc1fb51c"
SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
ALLOWED_CONTEXTS = {128, 256, 512}
ALLOWED_SEEDS = {1729, 2027, 7919}
ALLOWED_SAMPLE_COUNTS = {1, 3, 5}
ALLOWED_CASES = {
    "golden_official_fixture",
    "phase2_spy_20240705",
    "canary_spy_20240927",
    "canary_spy_20250131",
    "canary_spy_20250530",
    "canary_qqq_20240927",
    "canary_qqq_20250131",
    "canary_qqq_20250530",
}
FEATURE_NAMES = ("open", "high", "low", "close", "volume", "amount")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--trace-helper-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        _validate_batch(payload)
        result = _run_batch(
            payload["requests"],
            source_path=args.source_path,
            cache_path=args.cache_path,
            trace_helper_path=args.trace_helper_path,
        )
    except Exception as error:  # noqa: BLE001 - external worker failures are evidence
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
    *,
    source_path: Path,
    cache_path: Path,
    trace_helper_path: Path,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import snapshot_download

    if not source_path.is_dir():
        raise ValueError("pinned Kronos source path is missing")
    if _git_head(source_path) != SOURCE_REVISION:
        raise ValueError("pinned Kronos source revision mismatch")
    trace_helper = _load_trace_helper(trace_helper_path)
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
        raise ValueError("Hugging Face snapshots do not match pinned revisions")
    _reject_unsafe_snapshot(model_path)
    _reject_unsafe_snapshot(tokenizer_path)

    sys.path.insert(0, str(source_path))
    from model import Kronos, KronosTokenizer

    with contextlib.redirect_stdout(sys.stderr):
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
        model = Kronos.from_pretrained(str(model_path))
    tokenizer.eval()
    model.eval()
    device = _device(torch)
    environment = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "operating_system": platform.platform(),
        "device": device,
        "source_revision": SOURCE_REVISION,
        "source_inventory": _source_inventory(source_path),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "model_config": _read_config_summary(model_path / "config.json"),
        "tokenizer_config": _read_config_summary(tokenizer_path / "config.json"),
        "downloaded_files": [
            *_downloaded_files(MODEL_REPOSITORY, MODEL_REVISION, model_path),
            *_downloaded_files(TOKENIZER_REPOSITORY, TOKENIZER_REVISION, tokenizer_path),
        ],
        "cache_path": str(cache_path.resolve()),
        "cache_size_bytes": _directory_size(cache_path),
    }
    responses = [
        _forecast_one(
            request,
            model=model,
            tokenizer=tokenizer,
            device=device,
            np=np,
            pd=pd,
            torch=torch,
            trace_helper=trace_helper,
        )
        for request in requests
    ]
    return {
        "status": "success",
        "environment": environment,
        "responses": responses,
    }


def _forecast_one(
    request: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    np: Any,
    pd: Any,
    torch: Any,
    trace_helper: ModuleType,
) -> dict[str, Any]:
    _validate_request(request)
    _reset_rng(int(request["sampling_seed"]), torch, np)
    started = time.perf_counter()
    frame = pd.DataFrame(
        [
            {name: float(row[name]) for name in FEATURE_NAMES if name in row}
            for row in request["observations"]
        ]
    )
    x_timestamp = pd.Series(
        pd.to_datetime(
            [row["timestamp"] for row in request["observations"]],
            utc=True,
        )
    )
    y_timestamp = pd.Series(pd.to_datetime(request["forecast_sessions"], utc=True))
    frame_for_trace = frame.copy()
    frame_for_trace.index = pd.DatetimeIndex(x_timestamp)
    derived = _derive_official_frame(frame)
    derived_for_trace = derived.copy()
    derived_for_trace.index = pd.DatetimeIndex(x_timestamp)
    trace: dict[str, Any] = {
        "source_functions": [
            "model/kronos.py:KronosPredictor.predict",
            "model/kronos.py:KronosPredictor.generate",
            "model/kronos.py:auto_regressive_inference",
            "model/kronos.py:KronosTokenizer.encode",
            "model/kronos.py:KronosTokenizer.decode",
            "model/kronos.py:calc_time_stamps",
        ],
        "boundaries": [
            trace_helper.dataframe_trace(
                frame_for_trace,
                boundary="official_predictor_input_dataframe",
                units=_units(frame.columns),
                operations=("named_selection", "float_conversion", "copy"),
            ),
            trace_helper.dataframe_trace(
                derived_for_trace,
                boundary="official_dataframe_after_optional_field_derivation",
                units=_units(derived.columns),
                operations=(
                    "named_reorder_open_high_low_close_volume_amount",
                    "derive_amount_as_volume_times_mean_ohlc_when_missing",
                ),
            ),
        ],
    }

    from model import KronosPredictor

    traced_tokenizer = _TracingTokenizer(tokenizer, trace, trace_helper)

    class TracingPredictor(KronosPredictor):
        def generate(
            self,
            x: Any,
            x_stamp: Any,
            y_stamp: Any,
            pred_len: int,
            T: float,
            top_k: int,
            top_p: float,
            sample_count: int,
            verbose: bool,
        ) -> Any:
            trace["boundaries"].extend(
                (
                    trace_helper.array_trace(
                        x,
                        boundary="normalized_feature_tensor",
                        feature_names=FEATURE_NAMES,
                        units=("zscore",) * 6,
                        operations=("float32_cast", "per_feature_zscore", "clip_minus5_plus5"),
                    ),
                    trace_helper.array_trace(
                        x_stamp,
                        boundary="context_timestamp_tensor",
                        feature_names=("minute", "hour", "weekday", "day", "month"),
                        units=("integer_calendar_feature",) * 5,
                        operations=("calc_time_stamps", "float32_cast"),
                    ),
                    trace_helper.array_trace(
                        y_stamp,
                        boundary="forecast_timestamp_tensor",
                        feature_names=("minute", "hour", "weekday", "day", "month"),
                        units=("integer_calendar_feature",) * 5,
                        operations=("calc_time_stamps", "float32_cast"),
                    ),
                )
            )
            result = super().generate(
                x,
                x_stamp,
                y_stamp,
                pred_len,
                T,
                top_k,
                top_p,
                sample_count,
                verbose,
            )
            trace["boundaries"].append(
                trace_helper.array_trace(
                    result,
                    boundary="decoded_normalized_predictor_output",
                    feature_names=FEATURE_NAMES,
                    units=("zscore",) * 6,
                    operations=("decode", "sample_mean_normalized_space"),
                )
            )
            return result

    predictor = TracingPredictor(
        model,
        traced_tokenizer,
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
            sample_count=int(request["sample_count"]),
            verbose=False,
        )
    predicted_trace = predicted.copy()
    predicted_trace.index = pd.DatetimeIndex(predicted.index)
    trace["boundaries"].append(
        trace_helper.dataframe_trace(
            predicted_trace,
            boundary="official_predictor_denormalized_output_dataframe",
            units=_units(predicted.columns),
            operations=("inverse_per_feature_zscore", "named_dataframe_construction"),
        )
    )
    values = _extract_predicted(predicted, request["forecast_sessions"], np, pd)
    path = [
        {
            "timestamp": pd.Timestamp(index).isoformat(),
            **{name: float(value) for name, value in zip(FEATURE_NAMES, row)},
        }
        for index, row in zip(predicted.index, values)
    ]
    return {
        "status": "success",
        "audit_case": request["audit_case"],
        "sampling_seed": request["sampling_seed"],
        "sample_count": request["sample_count"],
        "context_length": request["context_length"],
        "inference_duration_ms": (time.perf_counter() - started) * 1_000.0,
        "path": path,
        "path_sha256": _canonical_sha256(path),
        "trace": trace if request["trace"] else None,
    }


class _TracingTokenizer:
    def __init__(self, inner: Any, trace: dict[str, Any], helper: ModuleType) -> None:
        self._inner = inner
        self._trace = trace
        self._helper = helper

    def to(self, device: str) -> _TracingTokenizer:
        self._inner.to(device)
        return self

    def encode(self, values: Any, half: bool = False) -> Any:
        self._trace["boundaries"].append(
            self._helper.array_trace(
                values,
                boundary="tokenizer_encode_input",
                feature_names=FEATURE_NAMES,
                units=("zscore",) * 6,
                operations=(f"half={str(half).lower()}",),
            )
        )
        encoded = self._inner.encode(values, half=half)
        for label, token_values in zip(("pre", "post"), encoded):
            self._trace["boundaries"].append(
                self._helper.array_trace(
                    token_values,
                    boundary=f"tokenizer_encoded_{label}_tokens",
                    operations=("quantized_token_ids",),
                )
            )
        return encoded

    def decode(self, values: Any, half: bool = False) -> Any:
        for label, token_values in zip(("pre", "post"), values):
            self._trace["boundaries"].append(
                self._helper.array_trace(
                    token_values,
                    boundary=f"tokenizer_decode_{label}_tokens",
                    operations=(f"half={str(half).lower()}",),
                )
            )
        decoded = self._inner.decode(values, half=half)
        self._trace["boundaries"].append(
            self._helper.array_trace(
                decoded,
                boundary="tokenizer_decoded_tensor",
                feature_names=FEATURE_NAMES,
                units=("zscore",) * 6,
                operations=("tokenizer_decode",),
            )
        )
        return decoded


def _validate_batch(payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError("worker input must be an object")
    if payload.get("schema_version") != "sentinel-kronos-phase2_5-worker-v0":
        raise ValueError("worker schema version mismatch")
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("worker requires a nonempty request list")
    if len(requests) > 24:
        raise ValueError("worker batch exceeds the Phase 2.5 bound")
    for request in requests:
        if not isinstance(request, dict):
            raise TypeError("worker requests must be objects")
        _validate_request(request)


def _validate_request(request: dict[str, Any]) -> None:
    expected = {
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "source_revision": SOURCE_REVISION,
        "temperature": 1.0,
        "top_p": 0.9,
        "forecast_horizon": 5,
    }
    for field, value in expected.items():
        if request.get(field) != value:
            raise ValueError(f"request field {field} does not match the audit pin")
    if request.get("audit_case") not in ALLOWED_CASES:
        raise ValueError("request audit_case is not declared")
    if request.get("context_length") not in ALLOWED_CONTEXTS:
        raise ValueError("request context_length is not declared")
    if request.get("sampling_seed") not in ALLOWED_SEEDS:
        raise ValueError("request sampling_seed is not declared")
    if request.get("sample_count") not in ALLOWED_SAMPLE_COUNTS:
        raise ValueError("request sample_count is not declared")
    if not isinstance(request.get("trace"), bool):
        raise TypeError("request trace must be boolean")
    observations = request.get("observations")
    context_length = request.get("context_length")
    if not isinstance(observations, list) or len(observations) != context_length:
        raise ValueError("request observations do not match context_length")
    allowed_fields = {"timestamp", *FEATURE_NAMES}
    required_fields = {"timestamp", "open", "high", "low", "close"}
    timestamps: list[str] = []
    for row in observations:
        if not isinstance(row, dict):
            raise TypeError("observation must be an object")
        if set(row) - allowed_fields or not required_fields <= set(row):
            raise ValueError("observation fields do not match the official schema")
        timestamps.append(str(row["timestamp"]))
    if len(timestamps) != len(set(timestamps)) or timestamps != sorted(timestamps):
        raise ValueError("observation timestamps must be unique and increasing")
    forecast_sessions = request.get("forecast_sessions")
    if not isinstance(forecast_sessions, list) or len(forecast_sessions) != 5:
        raise ValueError("forecast_sessions must contain exactly five timestamps")
    if len(set(forecast_sessions)) != 5:
        raise ValueError("forecast_sessions must be unique")


def _derive_official_frame(frame: Any) -> Any:
    price_columns = ["open", "high", "low", "close"]
    if not all(column in frame.columns for column in price_columns):
        raise ValueError("official price columns are missing")
    result = frame.copy()
    if "volume" not in result.columns:
        result["volume"] = 0.0
        result["amount"] = 0.0
    if "amount" not in result.columns and "volume" in result.columns:
        result["amount"] = result["volume"] * result[price_columns].mean(axis=1)
    return result.loc[:, [*price_columns, "volume", "amount"]]


def _extract_predicted(
    predicted: Any,
    expected_timestamps: list[str],
    np: Any,
    pd: Any,
) -> Any:
    missing = tuple(column for column in FEATURE_NAMES if column not in predicted.columns)
    if missing:
        raise ValueError(f"official predictor missing named columns: {missing}")
    expected = pd.DatetimeIndex(pd.to_datetime(expected_timestamps, utc=True))
    observed = pd.DatetimeIndex(pd.to_datetime(predicted.index, utc=True))
    if observed.has_duplicates or not observed.equals(expected):
        raise ValueError("official predictor output timestamp mismatch")
    values = predicted.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
    if values.shape != (5, 6):
        raise ValueError(f"official predictor output shape mismatch: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("official predictor output contains nonfinite values")
    return values


def _source_inventory(source_path: Path) -> list[dict[str, object]]:
    relative_paths = (
        Path("README.md"),
        Path("model/kronos.py"),
        Path("tests/test_kronos_regression.py"),
    )
    result = []
    for relative in relative_paths:
        path = source_path / relative
        if not path.is_file():
            continue
        content = path.read_bytes()
        result.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    return result


def _units(columns: Any) -> dict[str, str]:
    return {
        str(column): (
            "raw_price"
            if str(column) in {"open", "high", "low", "close"}
            else "shares"
            if str(column) == "volume"
            else "derived_price_times_volume"
        )
        for column in columns
    }


def _load_trace_helper(path: Path) -> ModuleType:
    if not path.is_file():
        raise ValueError("trace helper path is missing")
    spec = importlib.util.spec_from_file_location("openalpha_phase2_5_trace_helper", path)
    if spec is None or spec.loader is None:
        raise ValueError("trace helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _git_head(source_path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _reject_unsafe_snapshot(snapshot: Path) -> None:
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    unsafe = [path.name for path in files if path.suffix.lower() not in {".json", ".safetensors"}]
    if not files or unsafe:
        raise ValueError(f"unsafe or empty model snapshot: {unsafe}")


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


def _read_config_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "d_in",
        "d_model",
        "n_heads",
        "n_layers",
        "max_context",
        "s1_bits",
        "s2_bits",
    }
    return {key: payload[key] for key in sorted(payload) if key in allowed}


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


if __name__ == "__main__":
    raise SystemExit(main())
