import importlib.util
from pathlib import Path
from types import ModuleType
from typing import cast

import pandas as pd
import pytest


def _worker() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "kronos_phase2_5_worker.py"
    spec = importlib.util.spec_from_file_location("phase2_5_worker_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase 2.5 worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request() -> dict[str, object]:
    first_timestamp = pd.Timestamp("2023-01-01", tz="UTC")
    observations = [
        {
            "timestamp": (first_timestamp + pd.Timedelta(days=index)).isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
        }
        for index in range(128)
    ]
    return {
        "audit_case": "phase2_spy_20240705",
        "model_repository": "NeoQuasar/Kronos-mini",
        "model_revision": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
        "tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-2k",
        "tokenizer_revision": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        "source_revision": "67b630e67f6a18c9e9be918d9b4337c960db1e9a",
        "context_length": 128,
        "sampling_seed": 1729,
        "temperature": 1.0,
        "top_p": 0.9,
        "sample_count": 1,
        "forecast_horizon": 5,
        "observations": observations,
        "forecast_sessions": [
            "2024-07-08",
            "2024-07-09",
            "2024-07-10",
            "2024-07-11",
            "2024-07-12",
        ],
        "trace": True,
    }


def test_worker_accepts_only_pinned_bounded_requests() -> None:
    worker = _worker()
    valid = _request()

    worker._validate_batch(
        {
            "schema_version": "sentinel-kronos-phase2_5-worker-v0",
            "requests": [valid],
        }
    )
    for field, invalid_value in (
        ("model_revision", "0" * 40),
        ("sample_count", 2),
        ("temperature", 0.8),
    ):
        invalid = {**valid, field: invalid_value}
        with pytest.raises(ValueError, match=field):
            worker._validate_request(invalid)


def test_worker_rejects_large_batches_and_undeclared_observation_fields() -> None:
    worker = _worker()
    valid = _request()

    with pytest.raises(ValueError, match="bound"):
        worker._validate_batch(
            {
                "schema_version": "sentinel-kronos-phase2_5-worker-v0",
                "requests": [valid] * 25,
            }
        )
    invalid = dict(valid)
    raw_observations = cast(list[dict[str, object]], valid["observations"])
    observations = [dict(item) for item in raw_observations]
    observations[0]["raw_response"] = "forbidden"
    invalid["observations"] = observations
    with pytest.raises(ValueError, match="fields"):
        worker._validate_request(invalid)


def test_official_amount_derivation_is_named_and_order_invariant() -> None:
    frame = pd.DataFrame(
        {
            "volume": [10.0],
            "close": [104.0],
            "low": [98.0],
            "high": [106.0],
            "open": [100.0],
        }
    )

    derived = _worker()._derive_official_frame(frame)

    assert derived.columns.tolist() == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert derived.iloc[0]["amount"] == pytest.approx(10.0 * 102.0)


def test_source_inventory_is_read_only_and_content_addressed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    model = source / "model"
    model.mkdir(parents=True)
    first = model / "kronos.py"
    second = source / "README.md"
    first.write_text("predictor", encoding="utf-8")
    second.write_text("contract", encoding="utf-8")
    before = {path: path.read_bytes() for path in (first, second)}

    inventory = _worker()._source_inventory(source)

    assert {item["relative_path"] for item in inventory} == {
        "README.md",
        "model/kronos.py",
    }
    assert all(len(item["sha256"]) == 64 for item in inventory)
    assert {path: path.read_bytes() for path in (first, second)} == before
