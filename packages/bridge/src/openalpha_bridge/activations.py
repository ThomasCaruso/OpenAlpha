from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positive_limit(limit: float) -> None:
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("activation limit must be finite and positive")


def stable_softplus(values: ArrayLike) -> NDArray[np.floating]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "fi":
        raise TypeError("softplus inputs must be real floating values")
    return np.maximum(raw, 0.0) + np.log1p(np.exp(-np.abs(raw)))


def bounded_signed(values: ArrayLike, limit: float) -> NDArray[np.floating]:
    _positive_limit(limit)
    raw = np.asarray(values)
    if raw.dtype.kind not in "fi":
        raise TypeError("bounded signed inputs must be real floating values")
    return limit * np.tanh(raw)


def inverse_bounded_signed(values: ArrayLike, limit: float) -> NDArray[np.floating]:
    _positive_limit(limit)
    targets = np.asarray(values)
    if targets.dtype.kind not in "fi":
        raise TypeError("bounded signed targets must be real floating values")
    scaled = targets / limit
    if np.any(~np.isfinite(scaled)) or np.any(np.abs(scaled) >= 1.0):
        raise ValueError("finite signed targets must lie strictly inside (-limit, limit)")
    return np.arctanh(scaled)


def bounded_nonnegative(values: ArrayLike, limit: float) -> NDArray[np.floating]:
    _positive_limit(limit)
    softplus = stable_softplus(values)
    return (softplus / (limit + softplus)) * limit


def inverse_bounded_nonnegative(values: ArrayLike, limit: float) -> NDArray[np.floating]:
    _positive_limit(limit)
    targets = np.asarray(values)
    if targets.dtype.kind not in "fi":
        raise TypeError("bounded nonnegative targets must be real floating values")
    if np.any(~np.isfinite(targets)) or np.any(targets <= 0.0) or np.any(targets >= limit):
        raise ValueError("finite nonnegative targets must lie strictly inside (0, limit)")
    softplus = limit * targets / (limit - targets)
    result = np.empty_like(softplus)
    large = softplus > 20.0
    result[large] = softplus[large] + np.log1p(-np.exp(-softplus[large]))
    result[~large] = np.log(np.expm1(softplus[~large]))
    return result
