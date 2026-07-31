import json

import numpy as np
import pandas as pd
from openalpha_sentinel.integration_trace import array_trace, dataframe_trace


def test_dataframe_trace_records_schema_hash_and_safe_summaries() -> None:
    index = pd.to_datetime(
        ("2024-07-03", "2024-07-05"),
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "close": [100.5, 102.0],
            "volume": [1_000.0, 2_000.0],
        },
        index=index,
    )

    first = dataframe_trace(
        frame,
        boundary="normalized_input_dataframe",
        units={"open": "raw_price", "close": "raw_price", "volume": "shares"},
        operations=("named_column_selection", "copy"),
    )
    second = dataframe_trace(
        frame.copy(),
        boundary="normalized_input_dataframe",
        units={"open": "raw_price", "close": "raw_price", "volume": "shares"},
        operations=("named_column_selection", "copy"),
    )

    assert first == second
    assert first["shape"] == [2, 3]
    assert first["dtype"] == "float64"
    assert first["column_order"] == ["open", "close", "volume"]
    assert first["timestamp_index"] == {
        "count": 2,
        "first": "2024-07-03T00:00:00+00:00",
        "last": "2024-07-05T00:00:00+00:00",
        "timezone": "UTC",
    }
    features = {item["name"]: item for item in first["features"]}
    assert features["open"]["minimum"] == 100.0
    assert features["close"]["maximum"] == 102.0
    assert features["volume"]["units"] == "shares"
    assert len(first["sha256"]) == 64
    assert "values" not in json.dumps(first, sort_keys=True)


def test_array_trace_records_nonfinite_counts_without_payload() -> None:
    values = np.asarray(
        [[1.0, np.nan], [np.inf, 4.0]],
        dtype=np.float32,
    )

    trace = array_trace(
        values,
        boundary="decoded_normalized_tensor",
        feature_names=("open", "close"),
        units=("normalized", "normalized"),
        operations=("tokenizer_decode",),
    )
    features = {item["name"]: item for item in trace["features"]}

    assert trace["shape"] == [2, 2]
    assert trace["dtype"] == "float32"
    assert features["open"]["nonfinite_count"] == 1
    assert features["close"]["nonfinite_count"] == 1
    assert features["open"]["minimum"] == 1.0
    assert features["close"]["maximum"] == 4.0
    assert "payload" not in trace
    assert "values" not in trace


def test_array_hash_is_shape_dtype_and_order_sensitive() -> None:
    base = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    reordered = base[:, ::-1].copy()
    reshaped = base.reshape(1, 4)

    base_trace = array_trace(base, boundary="base")

    assert base_trace["sha256"] != array_trace(reordered, boundary="base")["sha256"]
    assert base_trace["sha256"] != array_trace(reshaped, boundary="base")["sha256"]
