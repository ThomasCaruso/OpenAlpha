"""The control feature sets, built from context rows only.

Every builder takes ``context`` and nothing else. A target row cannot arrive
through the signature, so "the controls could not see the future" is a property
of the type, not a promise in a docstring.

The engineered set is exactly the twelve features the specification enumerates,
in the order it enumerates them. Adding, removing or reordering one would change
the study.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import Final

from openalpha_kronos.model.input import OFFICIAL_COLUMNS, OfficialRow

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .spec import (
    CONTEXT_CANDLES,
    ENGINEERED_FEATURE_NAMES,
    ENGINEERED_RANGE_WINDOWS,
    ENGINEERED_RETURN_LAGS,
    ENGINEERED_VOLATILITY_WINDOWS,
    ENGINEERED_VOLUME_ZSCORE_WINDOW,
    FEATURE_SET_DIMENSIONS,
)

__all__ = [
    "CONTROL_BUILDERS",
    "engineered_features",
    "feature_names",
    "intercept_only",
    "raw_ohlcv",
]

_EPSILON: Final[float] = 1e-12


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(category=FailureCategory.INVALID_CONFIGURATION, code=code, message=message)
    )


def _require_context(context: tuple[OfficialRow, ...]) -> None:
    if len(context) != CONTEXT_CANDLES:
        raise _fail(
            "PROBE_CONTROL_CONTEXT_LENGTH_MISMATCH",
            f"{len(context)} context rows, expected exactly {CONTEXT_CANDLES}",
        )
    for row in context:
        for value in row.channels():
            if not math.isfinite(value):
                raise _fail(
                    "PROBE_CONTROL_NON_FINITE_INPUT",
                    "the context contains a non-finite value",
                )


def _log_returns(context: tuple[OfficialRow, ...]) -> list[float]:
    out: list[float] = []
    for index in range(1, len(context)):
        previous, current = context[index - 1].close, context[index].close
        if previous <= 0.0 or current <= 0.0:
            raise _fail(
                "PROBE_CONTROL_NON_POSITIVE_CLOSE",
                "a context close is not positive, so its log return is undefined",
            )
        out.append(math.log(current / previous))
    return out


def raw_ohlcv(context: tuple[OfficialRow, ...]) -> tuple[float, ...]:
    """The 40x6 context window, flattened in the official column order."""
    _require_context(context)
    values = tuple(value for row in context for value in row.channels())
    expected = FEATURE_SET_DIMENSIONS["raw_ohlcv"]
    if len(values) != expected:
        raise _fail(
            "PROBE_CONTROL_DIMENSION_MISMATCH",
            f"raw_ohlcv produced {len(values)} values, expected {expected}",
        )
    return values


def engineered_features(context: tuple[OfficialRow, ...]) -> tuple[float, ...]:
    """Twelve simple return, volatility, range and volume features.

    Every window looks strictly backwards from the final context row, so no
    feature uses information the model would not have had at the origin.
    """
    _require_context(context)
    returns = _log_returns(context)

    values: list[float] = []

    # Trailing log return at each declared lag, measured back from the anchor.
    anchor = context[-1].close
    for lag in ENGINEERED_RETURN_LAGS:
        past = context[-1 - lag].close
        if past <= 0.0:
            raise _fail(
                "PROBE_CONTROL_NON_POSITIVE_CLOSE",
                f"the close {lag} sessions back is not positive",
            )
        values.append(math.log(anchor / past))

    # Realized volatility: population standard deviation of the trailing returns.
    for window in ENGINEERED_VOLATILITY_WINDOWS:
        recent = returns[-window:]
        values.append(statistics.pstdev(recent) if len(recent) > 1 else 0.0)

    # Mean high-low range, normalized by close so it is scale free.
    for window in ENGINEERED_RANGE_WINDOWS:
        recent = context[-window:]
        ranges = [
            (row.high - row.low) / row.close if row.close > 0.0 else 0.0 for row in recent
        ]
        values.append(statistics.fmean(ranges))

    # Volume z-score over its window, against the same window's statistics.
    volume_window = context[-ENGINEERED_VOLUME_ZSCORE_WINDOW:]
    volumes = [row.volume for row in volume_window]
    mean = statistics.fmean(volumes)
    spread = statistics.pstdev(volumes) if len(volumes) > 1 else 0.0
    values.append((context[-1].volume - mean) / (spread + _EPSILON))

    expected = FEATURE_SET_DIMENSIONS["engineered_features"]
    if len(values) != expected:
        raise _fail(
            "PROBE_CONTROL_DIMENSION_MISMATCH",
            f"engineered_features produced {len(values)} values, expected {expected}",
        )
    for value in values:
        if not math.isfinite(value):
            raise _fail(
                "PROBE_CONTROL_NON_FINITE_OUTPUT",
                "an engineered feature is not finite",
            )
    return tuple(values)


def intercept_only(context: tuple[OfficialRow, ...]) -> tuple[float, ...]:
    """No features. The estimator predicts the training-partition mean."""
    _require_context(context)
    return ()


#: Every control the specification names, keyed by its declared id.
CONTROL_BUILDERS: Final[dict[str, Callable[[tuple[OfficialRow, ...]], tuple[float, ...]]]] = {
    "raw_ohlcv": raw_ohlcv,
    "engineered_features": engineered_features,
    "intercept_only": intercept_only,
}


def feature_names(feature_set: str) -> tuple[str, ...]:
    """Human-readable column names, recorded in the fit artifact."""
    if feature_set == "raw_ohlcv":
        return tuple(
            f"{column}_t{index - CONTEXT_CANDLES}"
            for index in range(CONTEXT_CANDLES)
            for column in OFFICIAL_COLUMNS
        )
    if feature_set == "engineered_features":
        return ENGINEERED_FEATURE_NAMES
    if feature_set == "intercept_only":
        return ()
    if feature_set == "kronos_hidden_state":
        return tuple(
            f"hidden_{index}" for index in range(FEATURE_SET_DIMENSIONS["kronos_hidden_state"])
        )
    raise _fail("PROBE_UNKNOWN_FEATURE_SET", f"unknown feature set {feature_set!r}")
