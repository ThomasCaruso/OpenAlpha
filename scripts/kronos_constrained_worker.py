from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import contextlib
import ctypes
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MODEL_REPOSITORY = "NeoQuasar/Kronos-mini"
MODEL_REVISION = "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"
TOKENIZER_REPOSITORY = "NeoQuasar/Kronos-Tokenizer-2k"
TOKENIZER_REVISION = "26966d0035065a0cae0ebad7af8ece35bc1fb51c"
SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
ALLOWED_SEEDS = {1729, 2027, 7919}
METHODS = (
    "RAW_AUTOREGRESSIVE",
    "STEPWISE_PROJECT_REENCODE",
    "VALID_CANDIDATE_RESAMPLING",
)
OBSERVATION_FIELDS = {"timestamp", "open", "high", "low", "close", "volume"}


class ConstrainedMethodError(ValueError):
    def __init__(self, code: str, message: str, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.code = code
        self.audit = audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        _validate_batch(payload)
        result = _run_batch(payload["requests"], args.source_path, args.cache_path)
    except Exception as error:  # noqa: BLE001 - serialize the external worker boundary
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

    _require_source(source_path)
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
    tokenizer = tokenizer.to(device)
    model = model.to(device)
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
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "source_revision": SOURCE_REVISION,
        "model_weights_modified": False,
    }
    responses = [
        _forecast_one(
            request=request,
            model=model,
            tokenizer=tokenizer,
            device=device,
            np=np,
            pd=pd,
            torch=torch,
        )
        for request in requests
    ]
    return {"status": "success", "environment": environment, "responses": responses}


def _forecast_one(
    *,
    request: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: str,
    np: Any,
    pd: Any,
    torch: Any,
) -> dict[str, Any]:
    request_id = "req_" + _canonical_hash(request)
    started = time.perf_counter()
    try:
        _validate_request(request)
        prepared = _prepare_inputs(request, device, np, pd, torch)
        method_results = {}
        for method in METHODS:
            _reset_rng(int(request["sampling_seed"]), torch, np)
            method_started = time.perf_counter()
            try:
                method_results[method] = _run_method(
                    method=method,
                    request=request,
                    model=model,
                    tokenizer=tokenizer,
                    prepared=prepared,
                    np=np,
                    torch=torch,
                )
            except Exception as error:  # noqa: BLE001 - isolate each declared method
                code = (
                    error.code
                    if isinstance(error, ConstrainedMethodError)
                    else "METHOD_EXECUTION_FAILED"
                )
                method_results[method] = {
                    "method": method,
                    "status": "hard_failure",
                    "duration_ms": (time.perf_counter() - method_started) * 1_000.0,
                    "failure": {
                        "code": code,
                        "message": f"{type(error).__name__}: {error}",
                    },
                }
                if isinstance(error, ConstrainedMethodError):
                    method_results[method]["audit"] = error.audit
        parity = None
        if request["official_parity_probe"]:
            parity = _official_parity(
                request=request,
                custom_raw=method_results["RAW_AUTOREGRESSIVE"],
                model=model,
                tokenizer=tokenizer,
                device=device,
                np=np,
                pd=pd,
                torch=torch,
            )
        return {
            "status": "success",
            "request_id": request_id,
            "origin_id": request["origin_id"],
            "sampling_seed": request["sampling_seed"],
            "methods": method_results,
            "official_parity": parity,
            "total_duration_ms": (time.perf_counter() - started) * 1_000.0,
        }
    except Exception as error:  # noqa: BLE001 - preserve per-request hard failures
        return {
            "status": "failure",
            "request_id": request_id,
            "origin_id": request.get("origin_id"),
            "sampling_seed": request.get("sampling_seed"),
            "failure": {
                "code": "CONSTRAINED_INFERENCE_FAILED",
                "message": f"{type(error).__name__}: {error}",
            },
            "total_duration_ms": (time.perf_counter() - started) * 1_000.0,
        }


def _prepare_inputs(
    request: dict[str, Any],
    device: str,
    np: Any,
    pd: Any,
    torch: Any,
) -> dict[str, Any]:
    from model.kronos import calc_time_stamps

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
    frame["amount"] = frame["volume"] * frame[["open", "high", "low", "close"]].mean(axis=1)
    x_timestamp = pd.Series(pd.to_datetime([row["timestamp"] for row in rows], utc=True))
    y_timestamp = pd.Series(pd.to_datetime(request["forecast_sessions"], utc=True))
    values = frame[["open", "high", "low", "close", "volume", "amount"]].to_numpy(
        dtype=np.float32
    )
    x_mean = np.mean(values, axis=0)
    x_std = np.std(values, axis=0)
    normalized = np.clip((values - x_mean) / (x_std + 1e-5), -5.0, 5.0)
    x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
    y_stamp = calc_time_stamps(y_timestamp).values.astype(np.float32)
    return {
        "x": torch.from_numpy(normalized[np.newaxis, :]).to(device),
        "x_stamp": torch.from_numpy(x_stamp[np.newaxis, :]).to(device),
        "y_stamp": torch.from_numpy(y_stamp[np.newaxis, :]).to(device),
        "x_mean": x_mean,
        "x_std": x_std,
        "frame": frame,
        "x_timestamp": x_timestamp,
        "y_timestamp": y_timestamp,
    }


def _run_method(
    *,
    method: str,
    request: dict[str, Any],
    model: Any,
    tokenizer: Any,
    prepared: dict[str, Any],
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    from model.kronos import sample_from_logits

    started = time.perf_counter()
    peak_before = _peak_working_set_bytes()
    x = prepared["x"]
    x_stamp = prepared["x_stamp"]
    y_stamp = prepared["y_stamp"]
    with torch.inference_mode():
        x_token = tokenizer.encode(torch.clip(x, -5.0, 5.0), half=True)
        pre_buffer = x_token[0].clone()
        post_buffer = x_token[1].clone()
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
        generated_pairs: list[tuple[int, int]] = []
        steps: list[dict[str, Any]] = []
        first_divergence_step = None
        for index in range(5):
            current_stamp = full_stamp[:, index : index + 512, :].contiguous()
            s1_logits, context = model.decode_s1(pre_buffer, post_buffer, current_stamp)
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
            selected_pair = raw_pair
            step_audit: dict[str, Any] = {
                "step": index + 1,
                "raw_token": {"coarse": raw_pair[0], "fine": raw_pair[1]},
                "selected_token": {"coarse": raw_pair[0], "fine": raw_pair[1]},
                "intervened": False,
            }
            if method != "RAW_AUTOREGRESSIVE":
                raw_window, raw_row = _decode_candidate_batch(
                    tokenizer=tokenizer,
                    pre_buffer=pre_buffer,
                    post_buffer=post_buffer,
                    pairs=(raw_pair,),
                    x_mean=prepared["x_mean"],
                    x_std=prepared["x_std"],
                    torch=torch,
                )
                raw_values = [float(value) for value in raw_row[0]]
                violations = _candle_violations(raw_values)
                step_audit["raw_decoded_candle"] = _row_payload(
                    raw_values, request["forecast_sessions"][index]
                )
                step_audit["raw_violations"] = violations
                if violations and method == "STEPWISE_PROJECT_REENCODE":
                    try:
                        selected_pair, intervention = _stepwise_project_reencode(
                            tokenizer=tokenizer,
                            pre_buffer=pre_buffer,
                            post_buffer=post_buffer,
                            raw_window=raw_window[0],
                            raw_row=raw_values,
                            x_mean=prepared["x_mean"],
                            x_std=prepared["x_std"],
                            cutoff_close=float(request["observations"][-1]["close"]),
                            session=request["forecast_sessions"][index],
                            torch=torch,
                        )
                    except ConstrainedMethodError as error:
                        raise ConstrainedMethodError(
                            error.code,
                            str(error),
                            {
                                "failed_step": index + 1,
                                "completed_steps": steps,
                                "failed_step_audit": {**step_audit, **error.audit},
                            },
                        ) from error
                    step_audit.update(intervention)
                elif violations and method == "VALID_CANDIDATE_RESAMPLING":
                    selected_pair, intervention = _resample_valid_candidate(
                        request=request,
                        step=index + 1,
                        model=model,
                        tokenizer=tokenizer,
                        context=context,
                        s1_logits=s1_last,
                        raw_pair=raw_pair,
                        raw_window=raw_window[0],
                        raw_row=raw_values,
                        pre_buffer=pre_buffer,
                        post_buffer=post_buffer,
                        x_mean=prepared["x_mean"],
                        x_std=prepared["x_std"],
                        session=request["forecast_sessions"][index],
                        torch=torch,
                    )
                    step_audit.update(intervention)
            if selected_pair != raw_pair and first_divergence_step is None:
                first_divergence_step = index + 1
            step_audit["selected_token"] = {
                "coarse": selected_pair[0],
                "fine": selected_pair[1],
            }
            _append_pair(pre_buffer, post_buffer, selected_pair, torch)
            generated_pairs.append(selected_pair)
            steps.append(step_audit)
        final_normalized = tokenizer.decode([pre_buffer, post_buffer], half=True)
        final_values = _inverse_normalize(
            final_normalized[0, -5:, :].detach().cpu().numpy(),
            prepared["x_mean"],
            prepared["x_std"],
        )
    path = [
        _row_payload([float(value) for value in row], session)
        for row, session in zip(final_values, request["forecast_sessions"])
    ]
    final_violations = [
        {
            "step": step,
            "codes": _candle_violations(
                [
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                    row["amount"],
                ]
            ),
        }
        for step, row in enumerate(path, start=1)
    ]
    final_violations = [item for item in final_violations if item["codes"]]
    base = {
        "method": method,
        "duration_ms": (time.perf_counter() - started) * 1_000.0,
        "peak_working_set_bytes": max(peak_before, _peak_working_set_bytes()),
        "generated_token_pairs": [
            {"coarse": coarse, "fine": fine} for coarse, fine in generated_pairs
        ],
        "steps": steps,
        "first_divergence_step": first_divergence_step,
        "final_violations": final_violations,
    }
    if method != "RAW_AUTOREGRESSIVE" and final_violations:
        return {
            **base,
            "status": "hard_failure",
            "failure": {
                "code": "INVALID_FINAL_CONSTRAINED_PATH",
                "message": "full-sequence decode violated the financial grammar",
            },
            "invalid_path": path,
            "path_sha256": _canonical_hash(path),
        }
    return {
        **base,
        "status": "success",
        "path": path,
        "path_sha256": _canonical_hash(path),
    }


def _stepwise_project_reencode(
    *,
    tokenizer: Any,
    pre_buffer: Any,
    post_buffer: Any,
    raw_window: Any,
    raw_row: list[float],
    x_mean: Any,
    x_std: Any,
    cutoff_close: float,
    session: str,
    torch: Any,
) -> tuple[tuple[int, int], dict[str, Any]]:
    projected, changes = _project_price_row(raw_row, cutoff_close=cutoff_close)
    modified = raw_window.clone()
    projected_normalized = (
        torch.as_tensor(projected, dtype=modified.dtype, device=modified.device)
        - torch.as_tensor(x_mean, dtype=modified.dtype, device=modified.device)
    ) / (
        torch.as_tensor(x_std, dtype=modified.dtype, device=modified.device) + 1e-5
    )
    modified[-1, :] = projected_normalized
    reencoded = tokenizer.encode(modified.unsqueeze(0), half=True)
    pair = (int(reencoded[0][0, -1].item()), int(reencoded[1][0, -1].item()))
    _, roundtrip_rows = _decode_candidate_batch(
        tokenizer=tokenizer,
        pre_buffer=pre_buffer,
        post_buffer=post_buffer,
        pairs=(pair,),
        x_mean=x_mean,
        x_std=x_std,
        torch=torch,
    )
    roundtrip = [float(value) for value in roundtrip_rows[0]]
    violations = _candle_violations(roundtrip)
    if violations:
        raise ConstrainedMethodError(
            "ROUNDTRIP_INVALID",
            "stepwise projection re-encode round trip remained invalid: "
            + ",".join(violations),
            {
                "projected_candle": _row_payload(projected, session),
                "projection_changes": changes,
                "projected_token": {"coarse": pair[0], "fine": pair[1]},
                "roundtrip_candle": _row_payload(roundtrip, session),
                "roundtrip_violations": violations,
                "decode_encode_valid": False,
            },
        )
    errors = {
        field: abs(roundtrip[index] - projected[index])
        for index, field in enumerate(("open", "high", "low", "close", "volume", "amount"))
    }
    return pair, {
        "intervened": True,
        "intervention": "STEPWISE_PROJECT_REENCODE",
        "projected_candle": _row_payload(projected, session),
        "projection_changes": changes,
        "projected_token": {"coarse": pair[0], "fine": pair[1]},
        "roundtrip_candle": _row_payload(roundtrip, session),
        "reencoding_absolute_error": errors,
        "decode_encode_valid": True,
    }


def _resample_valid_candidate(
    *,
    request: dict[str, Any],
    step: int,
    model: Any,
    tokenizer: Any,
    context: Any,
    s1_logits: Any,
    raw_pair: tuple[int, int],
    raw_window: Any,
    raw_row: list[float],
    pre_buffer: Any,
    post_buffer: Any,
    x_mean: Any,
    x_std: Any,
    session: str,
    torch: Any,
) -> tuple[tuple[int, int], dict[str, Any]]:
    validation_started = time.perf_counter()
    ranked = _rank_joint_pairs(
        model=model,
        context=context,
        s1_logits=s1_logits,
        maximum=64,
        torch=torch,
    )
    records = [
        {
            "coarse_token": raw_pair[0],
            "fine_token": raw_pair[1],
            "joint_log_probability": _joint_log_probability(
                model=model,
                context=context,
                s1_logits=s1_logits,
                pair=raw_pair,
                torch=torch,
            ),
            "rank": 1,
            "decoded_candle": _row_payload(raw_row, session),
            "valid": False,
            "violations": _candle_violations(raw_row),
        }
    ]
    unique_ranked = [pair for pair in ranked if pair[:2] != raw_pair]
    expansion_count = 0
    selected_record = None
    for budget in (16, 64):
        pairs = tuple((item[0], item[1]) for item in unique_ranked[: budget - 1])
        if pairs:
            _, rows = _decode_candidate_batch(
                tokenizer=tokenizer,
                pre_buffer=pre_buffer,
                post_buffer=post_buffer,
                pairs=pairs,
                x_mean=x_mean,
                x_std=x_std,
                torch=torch,
            )
            records = records[:1]
            for rank, (pair, row, ranked_item) in enumerate(
                zip(pairs, rows, unique_ranked[: budget - 1]),
                start=2,
            ):
                values = [float(value) for value in row]
                violations = _candle_violations(values)
                records.append(
                    {
                        "coarse_token": pair[0],
                        "fine_token": pair[1],
                        "joint_log_probability": ranked_item[2],
                        "rank": rank,
                        "decoded_candle": _row_payload(values, session),
                        "valid": not violations,
                        "violations": violations,
                    }
                )
        valid = [record for record in records if record["valid"]]
        if valid:
            selected_record = _select_valid_record(
                valid,
                fraction=_auxiliary_fraction(
                    str(request["origin_id"]),
                    int(request["sampling_seed"]),
                    step,
                ),
            )
            break
        expansion_count += 1
    if selected_record is not None:
        selected = (
            int(selected_record["coarse_token"]),
            int(selected_record["fine_token"]),
        )
        return selected, {
            "intervened": True,
            "intervention": "VALID_CANDIDATE_RESAMPLING",
            "candidates_considered": len(records),
            "valid_candidate_count": sum(bool(item["valid"]) for item in records),
            "rejection_count": sum(not bool(item["valid"]) for item in records),
            "selected_candidate_rank": selected_record["rank"],
            "search_expansion_count": expansion_count,
            "fallback_used": False,
            "candidate_validation_duration_ms": (
                time.perf_counter() - validation_started
            )
            * 1_000.0,
            "candidates": records,
        }
    fallback_pair, fallback = _stepwise_project_reencode(
        tokenizer=tokenizer,
        pre_buffer=pre_buffer,
        post_buffer=post_buffer,
        raw_window=raw_window,
        raw_row=raw_row,
        x_mean=x_mean,
        x_std=x_std,
        cutoff_close=float(request["observations"][-1]["close"]),
        session=session,
        torch=torch,
    )
    return fallback_pair, {
        **fallback,
        "intervention": "VALID_CANDIDATE_RESAMPLING_FALLBACK_STEPWISE_PROJECT_REENCODE",
        "candidates_considered": len(records),
        "valid_candidate_count": 0,
        "rejection_count": len(records),
        "selected_candidate_rank": None,
        "search_expansion_count": expansion_count,
        "fallback_used": True,
        "candidate_validation_duration_ms": (
            time.perf_counter() - validation_started
        )
        * 1_000.0,
        "candidates": records,
    }


def _rank_joint_pairs(
    *,
    model: Any,
    context: Any,
    s1_logits: Any,
    maximum: int,
    torch: Any,
) -> list[tuple[int, int, float]]:
    coarse_log = _filtered_log_probabilities(s1_logits, torch)
    coarse_count = 8 if maximum > 16 else 4
    coarse_values, coarse_ids = torch.topk(coarse_log, k=coarse_count, dim=-1)
    repeated_context = context.repeat(coarse_count, 1, 1)
    coarse_batch = coarse_ids.reshape(coarse_count, 1)
    fine_logits = model.decode_s2(repeated_context, coarse_batch)[:, -1, :]
    fine_log = _filtered_log_probabilities(fine_logits, torch)
    fine_count = 8 if maximum > 16 else 4
    fine_values, fine_ids = torch.topk(fine_log, k=fine_count, dim=-1)
    pairs = []
    for coarse_index in range(coarse_count):
        coarse = int(coarse_ids[0, coarse_index].item())
        coarse_value = float(coarse_values[0, coarse_index].item())
        for fine_index in range(fine_count):
            fine = int(fine_ids[coarse_index, fine_index].item())
            joint = coarse_value + float(fine_values[coarse_index, fine_index].item())
            if math.isfinite(joint):
                pairs.append((coarse, fine, joint))
    return sorted(pairs, key=lambda item: (-item[2], item[0], item[1]))[:maximum]


def _joint_log_probability(
    *,
    model: Any,
    context: Any,
    s1_logits: Any,
    pair: tuple[int, int],
    torch: Any,
) -> float:
    coarse_log = _filtered_log_probabilities(s1_logits, torch)
    coarse = torch.tensor([[pair[0]]], dtype=torch.long, device=s1_logits.device)
    fine_logits = model.decode_s2(context, coarse)[:, -1, :]
    fine_log = _filtered_log_probabilities(fine_logits, torch)
    return float(coarse_log[0, pair[0]].item() + fine_log[0, pair[1]].item())


def _filtered_log_probabilities(logits: Any, torch: Any) -> Any:
    from model.kronos import top_k_top_p_filtering

    filtered = top_k_top_p_filtering(
        logits.clone(),
        top_k=0,
        top_p=0.9,
    )
    return torch.log_softmax(filtered, dim=-1)


def _decode_candidate_batch(
    *,
    tokenizer: Any,
    pre_buffer: Any,
    post_buffer: Any,
    pairs: tuple[tuple[int, int], ...],
    x_mean: Any,
    x_std: Any,
    torch: Any,
) -> tuple[Any, Any]:
    count = len(pairs)
    pre = torch.roll(pre_buffer.repeat(count, 1), shifts=-1, dims=1)
    post = torch.roll(post_buffer.repeat(count, 1), shifts=-1, dims=1)
    pre[:, -1] = torch.tensor(
        [pair[0] for pair in pairs], dtype=pre.dtype, device=pre.device
    )
    post[:, -1] = torch.tensor(
        [pair[1] for pair in pairs], dtype=post.dtype, device=post.device
    )
    decoded = tokenizer.decode([pre, post], half=True)
    values = _inverse_normalize(
        decoded[:, -1, :].detach().cpu().numpy(),
        x_mean,
        x_std,
    )
    return decoded, values


def _append_pair(
    pre_buffer: Any,
    post_buffer: Any,
    pair: tuple[int, int],
    torch: Any,
) -> None:
    pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
    post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
    pre_buffer[:, -1] = pair[0]
    post_buffer[:, -1] = pair[1]


def _inverse_normalize(values: Any, x_mean: Any, x_std: Any) -> Any:
    return values * (x_std + 1e-5) + x_mean


def _official_parity(
    *,
    request: dict[str, Any],
    custom_raw: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: str,
    np: Any,
    pd: Any,
    torch: Any,
) -> dict[str, Any]:
    from model import KronosPredictor

    _reset_rng(int(request["sampling_seed"]), torch, np)
    prepared = _prepare_inputs(request, device, np, pd, torch)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
        predicted = predictor.predict(
            df=prepared["frame"].drop(columns=["amount"]),
            x_timestamp=prepared["x_timestamp"],
            y_timestamp=prepared["y_timestamp"],
            pred_len=5,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
    path = []
    for session, row in zip(request["forecast_sessions"], predicted.itertuples(index=False)):
        path.append(
            _row_payload(
                [
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                    float(row.volume),
                    float(row.amount),
                ],
                session,
            )
        )
    official_hash = _canonical_hash(path)
    return {
        "official_path_sha256": official_hash,
        "custom_raw_path_sha256": custom_raw.get("path_sha256"),
        "exact_match": official_hash == custom_raw.get("path_sha256"),
        "official_path": path,
    }


def _project_price_row(
    raw: list[float],
    *,
    cutoff_close: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    if len(raw) != 6:
        raise ValueError("projected row must contain OHLC, volume, and amount")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) and value > 0.0 for value in values[:4]):
        raise ValueError("projection requires finite positive OHLC")
    if not math.isfinite(values[4]) or values[4] < 0.0:
        raise ValueError("projection requires finite nonnegative volume")
    if not math.isfinite(values[5]):
        raise ValueError("projection requires finite amount")
    projected = list(values)
    projected[1] = max(values[1], values[0], values[3], values[2])
    projected[2] = min(values[2], values[0], values[3], projected[1])
    changes = []
    for index, field in ((1, "high"), (2, "low")):
        if projected[index] != values[index]:
            absolute = abs(projected[index] - values[index])
            changes.append(
                {
                    "field": field,
                    "original_value": values[index],
                    "projected_value": projected[index],
                    "absolute_adjustment": absolute,
                    "normalized_adjustment": absolute / cutoff_close,
                }
            )
    return projected, changes


def _candle_violations(row: list[float]) -> list[str]:
    if len(row) != 6:
        return ["FEATURE_COUNT_MISMATCH"]
    names = ("OPEN", "HIGH", "LOW", "CLOSE")
    violations = []
    for name, value in zip(names, row[:4]):
        if not math.isfinite(value):
            violations.append(f"NONFINITE_{name}")
        elif value <= 0.0:
            violations.append(f"NONPOSITIVE_{name}")
    if all(math.isfinite(value) for value in row[:4]):
        if row[1] < row[0]:
            violations.append("HIGH_BELOW_OPEN")
        if row[1] < row[3]:
            violations.append("HIGH_BELOW_CLOSE")
        if row[1] < row[2]:
            violations.append("HIGH_BELOW_LOW")
        if row[2] > row[0]:
            violations.append("LOW_ABOVE_OPEN")
        if row[2] > row[3]:
            violations.append("LOW_ABOVE_CLOSE")
    if not math.isfinite(row[4]):
        violations.append("NONFINITE_VOLUME")
    elif row[4] < 0.0:
        violations.append("NEGATIVE_VOLUME")
    return violations


def _row_payload(row: list[float], session: str) -> dict[str, Any]:
    return {
        "session": session,
        "timestamp": f"{session}T00:00:00+00:00",
        "open": row[0],
        "high": row[1],
        "low": row[2],
        "close": row[3],
        "volume": row[4],
        "amount": row[5],
    }


def _select_valid_record(
    records: list[dict[str, Any]],
    *,
    fraction: float,
) -> dict[str, Any]:
    maximum = max(float(record["joint_log_probability"]) for record in records)
    weights = [
        math.exp(float(record["joint_log_probability"]) - maximum) for record in records
    ]
    threshold = fraction * sum(weights)
    cumulative = 0.0
    selected = records[-1]
    for record, weight in zip(records, weights):
        cumulative += weight
        if threshold < cumulative:
            selected = record
            break
    return selected


def _auxiliary_fraction(origin_id: str, seed: int, step: int) -> float:
    payload = f"sentinel-v1-candidate-selection|{origin_id}|{seed}|{step}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def _validate_batch(payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError("worker input must be an object")
    if payload.get("schema_version") != "sentinel-kronos-constrained-worker-v1":
        raise ValueError("worker schema version mismatch")
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("worker requires a nonempty request list")
    if len(requests) > 3:
        raise ValueError("worker batch exceeds the three-seed origin bound")
    for request in requests:
        if not isinstance(request, dict):
            raise TypeError("each worker request must be an object")
        _validate_request(request)


def _validate_request(request: dict[str, Any]) -> None:
    expected = {
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "source_revision": SOURCE_REVISION,
        "context_length": 512,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 0,
        "sample_count": 1,
        "forecast_horizon": 5,
        "candidate_budget_initial": 16,
        "candidate_budget_expanded": 64,
    }
    for name, value in expected.items():
        if request.get(name) != value:
            raise ValueError(f"request field {name} does not match the experiment lock")
    if request.get("sampling_seed") not in ALLOWED_SEEDS:
        raise ValueError("request field sampling_seed does not match the experiment lock")
    if request.get("symbol") not in {"SPY", "QQQ"}:
        raise ValueError("request symbol is outside the locked universe")
    if request.get("methods") != list(METHODS):
        raise ValueError("request methods do not match the experiment lock")
    observations = request.get("observations")
    if not isinstance(observations, list) or len(observations) != 512:
        raise ValueError("request must contain exactly 512 observations")
    if any(not isinstance(row, dict) or set(row) != OBSERVATION_FIELDS for row in observations):
        raise ValueError("observation fields must be timestamp and named OHLCV only")
    sessions = request.get("forecast_sessions")
    if not isinstance(sessions, list) or len(sessions) != 5:
        raise ValueError("request must contain exactly five forecast sessions")
    if request.get("official_parity_probe") not in {True, False}:
        raise ValueError("official_parity_probe must be boolean")


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


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_unsafe_snapshot(snapshot: Path) -> None:
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    if not files:
        raise ValueError(f"empty model snapshot: {snapshot.name}")
    unsafe = [
        path.name
        for path in files
        if path.suffix.lower() not in {".json", ".safetensors"}
    ]
    if unsafe:
        raise ValueError(f"unsafe files in model snapshot: {unsafe}")


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _peak_working_set_bytes() -> int:
    if platform.system() != "Windows":
        return 0

    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    success = get_process_memory_info(
        ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    return int(counters.PeakWorkingSetSize) if success else 0


if __name__ == "__main__":
    raise SystemExit(main())
