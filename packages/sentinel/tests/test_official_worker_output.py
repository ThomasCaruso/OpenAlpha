import importlib.util
from datetime import date
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

SESSIONS = (
    date(2024, 7, 8),
    date(2024, 7, 9),
    date(2024, 7, 10),
    date(2024, 7, 11),
    date(2024, 7, 12),
)


def _worker() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "kronos_inference_worker.py"
    spec = importlib.util.spec_from_file_location("phase2_worker_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase 2 worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predicted_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "volume": [1_000.0 + index for index in range(5)],
            "close": [40.0 + index for index in range(5)],
            "low": [30.0 + index for index in range(5)],
            "high": [20.0 + index for index in range(5)],
            "open": [10.0 + index for index in range(5)],
            "amount": [9_999.0] * 5,
        },
        index=pd.to_datetime(SESSIONS, utc=True),
    )


def test_named_extraction_ignores_physical_column_order() -> None:
    values = _worker()._extract_predicted_ohlcv(
        _predicted_frame(),
        tuple(session.isoformat() for session in SESSIONS),
        np,
        pd,
    )

    assert values.shape == (5, 5)
    assert values[0].tolist() == [10.0, 20.0, 30.0, 40.0, 1_000.0]


@pytest.mark.parametrize(
    "index",
    [
        pd.to_datetime(
            (
                date(2024, 7, 9),
                date(2024, 7, 10),
                date(2024, 7, 11),
                date(2024, 7, 12),
                date(2024, 7, 15),
            ),
            utc=True,
        ),
        pd.to_datetime(
            (
                date(2024, 7, 8),
                date(2024, 7, 8),
                date(2024, 7, 10),
                date(2024, 7, 11),
                date(2024, 7, 12),
            ),
            utc=True,
        ),
    ],
)
def test_shifted_or_duplicate_output_index_is_rejected(index: pd.DatetimeIndex) -> None:
    predicted = _predicted_frame().set_axis(index, axis="index")

    with pytest.raises(ValueError, match="timestamp"):
        _worker()._extract_predicted_ohlcv(
            predicted,
            tuple(session.isoformat() for session in SESSIONS),
            np,
            pd,
        )


def test_missing_named_output_column_is_rejected() -> None:
    predicted = _predicted_frame().drop(columns="high")

    with pytest.raises(ValueError, match="columns"):
        _worker()._extract_predicted_ohlcv(
            predicted,
            tuple(session.isoformat() for session in SESSIONS),
            np,
            pd,
        )
