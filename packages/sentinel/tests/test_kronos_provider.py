from __future__ import annotations

import json
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from openalpha_sentinel.contracts import ForecastRequest, OHLCVObservation
from openalpha_sentinel.providers.kronos import KronosSubprocessClient

CUTOFF = date(2024, 7, 5)
FUTURE = (
    date(2024, 7, 8),
    date(2024, 7, 9),
    date(2024, 7, 10),
    date(2024, 7, 11),
    date(2024, 7, 12),
)


def _request() -> ForecastRequest:
    observations = tuple(
        OHLCVObservation(
            session=CUTOFF - timedelta(days=127 - index),
            timestamp=datetime.combine(
                CUTOFF - timedelta(days=127 - index),
                datetime.min.time(),
                tzinfo=UTC,
            ),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1_000_000.0,
        )
        for index in range(128)
    )
    return ForecastRequest(
        model_repository="NeoQuasar/Kronos-mini",
        model_revision="f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
        tokenizer_repository="NeoQuasar/Kronos-Tokenizer-2k",
        tokenizer_revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        source_revision="67b630e67f6a18c9e9be918d9b4337c960db1e9a",
        symbol="SPY",
        cutoff=CUTOFF,
        forecast_sessions=FUTURE,
        observations=observations,
        context_length=128,
        forecast_horizon=5,
        sampling_seed=1729,
        temperature=1.0,
        top_p=0.9,
        sample_count=1,
        data_snapshot_sha256="a" * 64,
        experiment_sha256="b" * 64,
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
    )


def test_subprocess_request_is_typed_ohlcv_only_and_preserves_one_path(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        request_payload = json.loads(kwargs["input"])["requests"][0]
        assert request_payload["sample_count"] == 1
        assert request_payload["context_length"] == 128
        assert all("amount" not in row for row in request_payload["observations"])
        output = {
            "status": "success",
            "environment": {
                "python_version": "3.11.15",
                "torch_version": "2.13.0",
                "numpy_version": "2.2.6",
                "pandas_version": "2.2.2",
                "operating_system": "Windows-11",
                "device": "cpu",
                "cache_path": str(tmp_path / "cache"),
                "cache_size_bytes": 32_000_000,
                "downloaded_files": [],
                "official_predictor_derives_amount": True,
            },
            "responses": [
                {
                    "status": "success",
                    "provider_id": "kronos-local",
                    "checkpoint_id": "NeoQuasar/Kronos-mini@f4e68697",
                    "request_id": "req-1",
                    "inference_duration_ms": 25.0,
                    "path": [
                        {
                            "session": session.isoformat(),
                            "timestamp": datetime.combine(
                                session, datetime.min.time(), tzinfo=UTC
                            ).isoformat(),
                            "open": 100.0,
                            "high": 101.0,
                            "low": 99.0,
                            "close": 100.5,
                            "volume": 1_000_000.0,
                        }
                        for session in FUTURE
                    ],
                }
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(output), "")

    client = KronosSubprocessClient(
        python_executable=tmp_path / "venv" / "python.exe",
        worker_script=tmp_path / "worker.py",
        source_path=tmp_path / "source",
        cache_path=tmp_path / "cache",
        runner=runner,
        timeout_seconds=60.0,
    )

    result = client.forecast_many((_request(),))

    assert len(calls) == 1
    assert calls[0]["text"] is True
    assert calls[0]["capture_output"] is True
    assert result.environment.device == "cpu"
    assert result.environment.official_predictor_derives_amount is True
    assert len(result.responses[0].generated_paths) == 1
    assert result.responses[0].generated_paths[0].observations[-1].close == 100.5


def test_worker_failure_is_preserved_without_fake_fallback(tmp_path: Path) -> None:
    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        output = {
            "status": "success",
            "environment": {
                "python_version": "3.11.15",
                "torch_version": "2.13.0",
                "numpy_version": "2.2.6",
                "pandas_version": "2.2.2",
                "operating_system": "Windows-11",
                "device": "cpu",
                "cache_path": str(tmp_path / "cache"),
                "cache_size_bytes": 0,
                "downloaded_files": [],
                "official_predictor_derives_amount": True,
            },
            "responses": [
                {
                    "status": "failure",
                    "provider_id": "kronos-local",
                    "checkpoint_id": "NeoQuasar/Kronos-mini@f4e68697",
                    "request_id": "req-2",
                    "inference_duration_ms": 5.0,
                    "failure": {
                        "code": "MODEL_LOAD_FAILED",
                        "message": "checkpoint unavailable",
                    },
                }
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(output), "")

    client = KronosSubprocessClient(
        python_executable=tmp_path / "python.exe",
        worker_script=tmp_path / "worker.py",
        source_path=tmp_path / "source",
        cache_path=tmp_path / "cache",
        runner=runner,
    )

    result = client.forecast_many((_request(),))

    assert result.responses[0].generated_paths == ()
    assert result.responses[0].failure is not None
    assert result.responses[0].failure.code == "MODEL_LOAD_FAILED"
