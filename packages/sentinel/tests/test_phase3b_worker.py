from __future__ import annotations

import importlib.util
import inspect
import platform
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _worker() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "kronos_constrained_worker.py"
    spec = importlib.util.spec_from_file_location("phase3b_worker_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase 3B worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request() -> dict[str, object]:
    first_timestamp = pd.Timestamp("2022-01-03", tz="UTC")
    observations = [
        {
            "timestamp": (first_timestamp + pd.Timedelta(days=index)).isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
        }
        for index in range(512)
    ]
    return {
        "origin_id": "sentinel-v1-SPY-2024-07-05-h5",
        "symbol": "SPY",
        "cutoff": "2024-07-05",
        "model_repository": "NeoQuasar/Kronos-mini",
        "model_revision": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
        "tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-2k",
        "tokenizer_revision": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        "source_revision": "67b630e67f6a18c9e9be918d9b4337c960db1e9a",
        "context_length": 512,
        "sampling_seed": 1729,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 0,
        "sample_count": 1,
        "forecast_horizon": 5,
        "methods": [
            "RAW_AUTOREGRESSIVE",
            "STEPWISE_PROJECT_REENCODE",
            "VALID_CANDIDATE_RESAMPLING",
        ],
        "candidate_budget_initial": 16,
        "candidate_budget_expanded": 64,
        "observations": observations,
        "forecast_sessions": [
            "2024-07-08",
            "2024-07-09",
            "2024-07-10",
            "2024-07-11",
            "2024-07-12",
        ],
        "official_parity_probe": False,
    }


def test_worker_request_is_pinned_and_bounded() -> None:
    worker = _worker()
    valid = _request()

    worker._validate_batch(
        {
            "schema_version": "sentinel-kronos-constrained-worker-v1",
            "requests": [valid],
        }
    )
    for field, invalid_value in (
        ("context_length", 256),
        ("sample_count", 3),
        ("candidate_budget_initial", 17),
        ("candidate_budget_expanded", 128),
    ):
        with pytest.raises(ValueError, match=field):
            worker._validate_request({**valid, field: invalid_value})


def test_projection_is_auditable_and_preserves_open_close() -> None:
    worker = _worker()
    raw = [100.0, 99.0, 102.0, 101.0, 1_000.0, 100_000.0]

    projected, changes = worker._project_price_row(raw, cutoff_close=100.0)

    assert projected == [100.0, 102.0, 100.0, 101.0, 1_000.0, 100_000.0]
    assert projected[0] == raw[0]
    assert projected[3] == raw[3]
    assert {item["field"] for item in changes} == {"high", "low"}
    assert worker._candle_violations(projected) == []


def test_projection_hard_fails_for_unsupported_invalidity() -> None:
    worker = _worker()

    with pytest.raises(ValueError, match="finite positive OHLC"):
        worker._project_price_row(
            [0.0, 101.0, 99.0, 100.0, 1_000.0, 100_000.0],
            cutoff_close=100.0,
        )
    with pytest.raises(ValueError, match="nonnegative volume"):
        worker._project_price_row(
            [100.0, 101.0, 99.0, 100.0, -1.0, 100_000.0],
            cutoff_close=100.0,
        )


def test_one_constrained_method_failure_does_not_discard_raw_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker()
    request = _request()
    monkeypatch.setattr(worker, "_validate_request", lambda value: None)
    monkeypatch.setattr(worker, "_prepare_inputs", lambda *args: {})
    monkeypatch.setattr(worker, "_reset_rng", lambda *args: None)

    def method_result(*, method: str, **kwargs):
        if method == "STEPWISE_PROJECT_REENCODE":
            raise ValueError("round trip invalid")
        return {"method": method, "status": "success", "path_sha256": method.lower()}

    monkeypatch.setattr(worker, "_run_method", method_result)

    response = worker._forecast_one(
        request=request,
        model=object(),
        tokenizer=object(),
        device="cpu",
        np=object(),
        pd=object(),
        torch=object(),
    )

    assert response["status"] == "success"
    assert response["methods"]["RAW_AUTOREGRESSIVE"]["status"] == "success"
    failed = response["methods"]["STEPWISE_PROJECT_REENCODE"]
    assert failed["status"] == "hard_failure"
    assert failed["failure"]["code"] == "METHOD_EXECUTION_FAILED"


def test_constrained_method_failure_preserves_step_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker()
    request = _request()
    monkeypatch.setattr(worker, "_validate_request", lambda value: None)
    monkeypatch.setattr(worker, "_prepare_inputs", lambda *args: {})
    monkeypatch.setattr(worker, "_reset_rng", lambda *args: None)

    def method_result(*, method: str, **kwargs):
        if method == "STEPWISE_PROJECT_REENCODE":
            raise worker.ConstrainedMethodError(
                "ROUNDTRIP_INVALID",
                "round trip invalid",
                {"failed_step": 1, "projected_token": {"coarse": 4, "fine": 9}},
            )
        return {"method": method, "status": "success", "path_sha256": method.lower()}

    monkeypatch.setattr(worker, "_run_method", method_result)

    response = worker._forecast_one(
        request=request,
        model=object(),
        tokenizer=object(),
        device="cpu",
        np=object(),
        pd=object(),
        torch=object(),
    )

    failed = response["methods"]["STEPWISE_PROJECT_REENCODE"]
    assert failed["failure"]["code"] == "ROUNDTRIP_INVALID"
    assert failed["audit"]["failed_step"] == 1
    assert failed["audit"]["projected_token"] == {"coarse": 4, "fine": 9}


def test_timestamp_helper_is_imported_from_its_defining_pinned_module() -> None:
    worker = _worker()
    prepare_source = inspect.getsource(worker._prepare_inputs)
    method_source = inspect.getsource(worker._run_method)
    filtering_source = inspect.getsource(worker._filtered_log_probabilities)

    assert "from model.kronos import calc_time_stamps" in prepare_source
    assert "from model import calc_time_stamps" not in prepare_source
    assert "from model.kronos import sample_from_logits" in method_source
    assert "from model import sample_from_logits" not in method_source
    assert "from model.kronos import top_k_top_p_filtering" in filtering_source
    assert "from model import top_k_top_p_filtering" not in filtering_source


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows memory API")
def test_peak_working_set_probe_returns_a_real_measurement() -> None:
    assert _worker()._peak_working_set_bytes() > 0
