"""Forecast error against the known development target.

The primary metric is the mean absolute error of one-step log close-to-close
returns, which is scale free and therefore comparable across methods. A
forecast containing a non-finite or non-positive close has no defined log
return; that is reported as an undefined metric rather than silently dropped,
because dropping it would make a broken forecast look accurate on whatever
steps survived.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from .validity import DecodedCandle

__all__ = ["ForecastError", "forecast_error", "mean_of"]


class ForecastError(BaseModel):
    """All three metrics, or an explicit statement that they are undefined."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    defined: bool
    #: Primary. None when any close is non-finite or non-positive.
    close_return_mae: float | None = None
    close_mae: float | None = None
    full_ohlc_mae: float | None = None
    scored_steps: int = 0
    undefined_reason: str | None = None


def _log_returns(closes: list[float], anchor: float) -> list[float] | None:
    previous = anchor
    out: list[float] = []
    for close in closes:
        if not math.isfinite(close) or close <= 0.0 or previous <= 0.0:
            return None
        out.append(math.log(close / previous))
        previous = close
    return out


def forecast_error(
    predicted: tuple[DecodedCandle, ...],
    actual: tuple[DecodedCandle, ...],
    *,
    anchor_close: float,
) -> ForecastError:
    """Compare a predicted path against the known target path.

    ``anchor_close`` is the final observed context close, so the first scored
    step has a return like every other step rather than being skipped.
    """
    if len(predicted) != len(actual):
        return ForecastError(
            defined=False,
            undefined_reason=(
                f"predicted {len(predicted)} steps against {len(actual)} target steps"
            ),
        )
    if not predicted:
        return ForecastError(defined=False, undefined_reason="no steps to score")
    if not math.isfinite(anchor_close) or anchor_close <= 0.0:
        return ForecastError(defined=False, undefined_reason="anchor close is not usable")

    fields = ("open", "high", "low", "close")
    for candle in predicted:
        for name in fields:
            if not math.isfinite(getattr(candle, name)):
                return ForecastError(
                    defined=False,
                    scored_steps=len(predicted),
                    undefined_reason=f"predicted {name} is not finite",
                )

    predicted_returns = _log_returns([c.close for c in predicted], anchor_close)
    actual_returns = _log_returns([c.close for c in actual], anchor_close)
    if predicted_returns is None or actual_returns is None:
        return ForecastError(
            defined=False,
            scored_steps=len(predicted),
            undefined_reason="a close is not positive, so its log return is undefined",
        )

    steps = len(predicted)
    close_return_mae = sum(
        abs(p - a) for p, a in zip(predicted_returns, actual_returns, strict=True)
    ) / steps
    close_mae = sum(abs(p.close - a.close) for p, a in zip(predicted, actual, strict=True)) / steps
    ohlc_mae = sum(
        abs(getattr(p, name) - getattr(a, name))
        for p, a in zip(predicted, actual, strict=True)
        for name in fields
    ) / (steps * len(fields))

    return ForecastError(
        defined=True,
        close_return_mae=close_return_mae,
        close_mae=close_mae,
        full_ohlc_mae=ohlc_mae,
        scored_steps=steps,
    )


def mean_of(values: list[float]) -> float | None:
    """Arithmetic mean, or None when there is nothing to average."""
    return sum(values) / len(values) if values else None
