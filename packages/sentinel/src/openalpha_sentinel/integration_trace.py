from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd


def dataframe_trace(
    frame: pd.DataFrame,
    *,
    boundary: str,
    units: Mapping[str, str] | None = None,
    operations: Sequence[str] = (),
) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("trace requires a nonempty DataFrame")
    if isinstance(frame.columns, pd.MultiIndex):
        raise TypeError("trace does not accept MultiIndex columns")
    column_order = [str(column) for column in frame.columns]
    numeric = frame.loc[:, list(frame.columns)].to_numpy(dtype=float)
    feature_summaries = [
        _feature_summary(
            name=str(column),
            values=frame[column].to_numpy(dtype=float),
            units=(units or {}).get(str(column)),
        )
        for column in frame.columns
    ]
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    index = cast(Any, index)
    return {
        "boundary": boundary,
        "kind": "dataframe",
        "shape": list(frame.shape),
        "dtype": _frame_dtype(frame),
        "column_order": column_order,
        "timestamp_index": {
            "count": len(index),
            "first": pd.Timestamp(index[0]).isoformat(),
            "last": pd.Timestamp(index[-1]).isoformat(),
            "timezone": str(index.tz),
        },
        "features": feature_summaries,
        "operations": list(operations),
        "sha256": _numeric_hash(
            numeric,
            metadata={
                "columns": column_order,
                "index": [value.isoformat() for value in index],
            },
        ),
    }


def array_trace(
    values: object,
    *,
    boundary: str,
    feature_names: Sequence[str] = (),
    units: Sequence[str] = (),
    operations: Sequence[str] = (),
) -> dict[str, Any]:
    array = _to_numpy(values)
    if array.size == 0:
        raise ValueError("trace requires a nonempty array")
    if feature_names and array.shape[-1] != len(feature_names):
        raise ValueError("feature names do not match final array dimension")
    if units and len(units) != len(feature_names):
        raise ValueError("units do not match feature names")
    if feature_names:
        summaries = [
            _feature_summary(
                name=name,
                values=array[..., index],
                units=units[index] if units else None,
            )
            for index, name in enumerate(feature_names)
        ]
    else:
        summaries = [_feature_summary(name="value", values=array, units=None)]
    return {
        "boundary": boundary,
        "kind": "array",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "column_order": list(feature_names),
        "timestamp_index": None,
        "features": summaries,
        "operations": list(operations),
        "sha256": _numeric_hash(array, metadata={"feature_names": list(feature_names)}),
    }


def _to_numpy(values: object) -> np.ndarray[Any, Any]:
    if isinstance(values, np.ndarray):
        return values
    detach = getattr(values, "detach", None)
    if callable(detach):
        tensor = detach()
        cpu = getattr(tensor, "cpu", None)
        if callable(cpu):
            tensor = cpu()
        numpy_method = getattr(tensor, "numpy", None)
        if callable(numpy_method):
            result = numpy_method()
            if isinstance(result, np.ndarray):
                return result
    return np.asarray(values)


def _feature_summary(
    *,
    name: str,
    values: object,
    units: str | None,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "name": name,
        "dtype": str(np.asarray(values).dtype),
        "units": units,
        "minimum": float(np.min(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
        "nonfinite_count": int(array.size - finite.size),
    }


def _numeric_hash(array: np.ndarray[Any, Any], *, metadata: object) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _frame_dtype(frame: pd.DataFrame) -> str:
    dtypes = tuple(str(dtype) for dtype in frame.dtypes)
    if len(set(dtypes)) == 1:
        return dtypes[0]
    return "mixed[" + ",".join(dtypes) + "]"
