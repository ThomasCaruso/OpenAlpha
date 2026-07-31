from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from .config import BridgeRepresentationConfig
from .errors import BridgeFailure
from .models import FinancialFeatureTensor
from .numerics import NumericalAudit
from .results import ReconstructedSequence
from .validation import BridgeAuditResult


def _float_value(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("canonical Bridge serialization rejects nonfinite floating values")
    return 0.0 if value == 0.0 else value


def _timestamp_value(value: date | datetime) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical Bridge datetime values must be timezone-aware")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return value.isoformat()


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _float_value(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _float_value(float(value))
    if isinstance(value, (date, datetime)):
        return _timestamp_value(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python", exclude_none=False))
    if is_dataclass(value):
        return {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        _canonical_value(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _presence_payload(mask: NDArray[np.bool_]) -> dict[str, Any]:
    canonical = np.asarray(mask, dtype=np.uint8, order="C")
    return {
        "data_hex": canonical.tobytes(order="C").hex(),
        "encoding": "explicit-presence-u8",
        "shape": list(canonical.shape),
    }


def _tensor_payload(
    values: NDArray[np.generic],
    *,
    volume_present: NDArray[np.bool_] | None,
) -> dict[str, Any]:
    raw = np.asarray(values)
    if raw.dtype.name not in ("float32", "float64"):
        raise ValueError("canonical Bridge tensors must use float32 or float64")
    little_dtype = np.dtype("<f4" if raw.dtype.name == "float32" else "<f8")
    canonical = np.array(raw, dtype=little_dtype, copy=True, order="C")
    if volume_present is not None:
        mask = np.asarray(volume_present, dtype=np.bool_)
        if canonical.shape[-1] != 5 or mask.shape != canonical.shape[:-1]:
            raise ValueError("volume presence shape must match tensor leading dimensions")
        volume = canonical[..., 4]
        if np.any(~mask & ~np.isnan(volume)):
            raise ValueError("missing volume must be represented by NaN before canonical encoding")
        volume[~mask] = 0.0
    if not np.isfinite(canonical).all():
        raise ValueError("canonical Bridge tensors reject nonfinite present values")
    canonical[canonical == 0.0] = 0.0
    return {
        "data_hex": canonical.tobytes(order="C").hex(),
        "dtype": raw.dtype.name,
        "encoding": "ieee754-hex",
        "endianness": "little",
        "shape": list(canonical.shape),
    }


def _features_payload(value: FinancialFeatureTensor) -> dict[str, Any]:
    return {
        "configuration_sha256": value.configuration_sha256,
        "missing_volume": (
            None if value.volume_present is None else _presence_payload(value.volume_present)
        ),
        "numerical_warnings": list(value.numerical_warnings),
        "representation_version": value.representation_version,
        "schema_version": "openalpha.bridge.financial-features.v1",
        "single_sequence": value.single_sequence,
        "values": _tensor_payload(value.values, volume_present=value.volume_present),
    }


def bridge_canonical_bytes(value: object) -> bytes:
    if isinstance(value, BridgeRepresentationConfig):
        return value.canonical_bytes()
    if isinstance(value, ReconstructedSequence):
        payload = {
            "candles": _tensor_payload(value.candles, volume_present=value.volume_present),
            "configuration_sha256": value.configuration_sha256,
            "missing_volume": (
                None if value.volume_present is None else _presence_payload(value.volume_present)
            ),
            "numerical_warnings": list(value.numerical_warnings),
            "object_type": "reconstructed_sequence",
            "output_dtype": value.output_dtype,
            "projection_applied": value.projection_applied,
            "representation_version": value.representation_version,
            "schema_version": "openalpha.bridge.artifact.v1",
            "single_sequence": value.single_sequence,
            "transformed_features": _features_payload(value.transformed_features),
        }
        return _json_bytes(payload)
    if isinstance(value, FinancialFeatureTensor):
        payload = {
            "object_type": "financial_feature_tensor",
            "schema_version": "openalpha.bridge.artifact.v1",
            **_features_payload(value),
        }
        return _json_bytes(payload)
    if isinstance(value, BridgeAuditResult):
        return _json_bytes(_canonical_value(value))
    if isinstance(value, NumericalAudit):
        return _json_bytes(_canonical_value(value))
    if isinstance(value, BridgeFailure):
        return _json_bytes(value.model_dump(mode="python", exclude_none=False))
    raise TypeError(f"unsupported Bridge serialization type: {type(value).__name__}")


def bridge_sha256(value: object) -> str:
    return hashlib.sha256(bridge_canonical_bytes(value)).hexdigest()
