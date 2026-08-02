"""Error metrics over the six official channels.

Two return metrics, deliberately distinguished:

* ``forecast_error`` scores a predicted path against the known target. Its
  first transition uses the final observed context close, which the model
  genuinely had.
* ``reconstruction_error`` scores a round trip against its own input. Every
  transition is internal to the sequence, so no external close is supplied.
  v1 anchored the first transition on an actual close, which mixed
  reconstruction error with an externally provided truth and made a lossy
  round trip look better than it was.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from .official_input import OFFICIAL_COLUMNS, OfficialRow

__all__ = [
    "ForecastError",
    "ReconstructionError",
    "forecast_error",
    "mean_of",
    "reconstruction_error",
]

_PRICE_FIELDS = ("open", "high", "low", "close")


class ForecastError(BaseModel):
    """Metrics against a known target, or an explicit statement of absence."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    defined: bool
    close_return_mae: float | None = None
    close_mae: float | None = None
    full_ohlcva_mae: float | None = None
    scored_steps: int = 0
    undefined_reason: str | None = None


class ReconstructionError(BaseModel):
    """Round-trip metrics. Returns come from internal transitions only."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    defined: bool
    full_sequence_ohlcva_mae: float | None = None
    per_column_mae: dict[str, float] | None = None
    #: Over transitions 1..n-1. No external anchor.
    internal_return_mae: float | None = None
    transitions_scored: int = 0
    rows_scored: int = 0
    undefined_reason: str | None = None


def _finite(rows: tuple[OfficialRow, ...]) -> bool:
    return all(math.isfinite(value) for row in rows for value in row.channels())


def _internal_returns(rows: tuple[OfficialRow, ...]) -> list[float] | None:
    """log(close[t] / close[t-1]) for t in 1..n-1."""
    out: list[float] = []
    for index in range(1, len(rows)):
        previous = rows[index - 1].close
        current = rows[index].close
        if not math.isfinite(previous) or not math.isfinite(current):
            return None
        if previous <= 0.0 or current <= 0.0:
            return None
        out.append(math.log(current / previous))
    return out


def reconstruction_error(
    reconstructed: tuple[OfficialRow, ...], actual: tuple[OfficialRow, ...]
) -> ReconstructionError:
    """Full six-channel error plus an internally anchored return error."""
    if len(reconstructed) != len(actual):
        return ReconstructionError(
            defined=False,
            undefined_reason=(
                f"reconstructed {len(reconstructed)} rows against {len(actual)} actual rows"
            ),
        )
    if len(reconstructed) < 2:
        return ReconstructionError(
            defined=False, undefined_reason="fewer than two rows, so no transition exists"
        )
    if not _finite(reconstructed):
        return ReconstructionError(
            defined=False,
            rows_scored=len(reconstructed),
            undefined_reason="the reconstruction contains a non-finite value",
        )

    rows = len(reconstructed)
    per_column: dict[str, float] = {}
    for index, name in enumerate(OFFICIAL_COLUMNS):
        per_column[name] = (
            sum(
                abs(r.channels()[index] - a.channels()[index])
                for r, a in zip(reconstructed, actual, strict=True)
            )
            / rows
        )
    overall = sum(per_column.values()) / len(OFFICIAL_COLUMNS)

    predicted_returns = _internal_returns(reconstructed)
    actual_returns = _internal_returns(actual)
    if predicted_returns is None or actual_returns is None:
        return ReconstructionError(
            defined=False,
            rows_scored=rows,
            full_sequence_ohlcva_mae=overall,
            per_column_mae=per_column,
            undefined_reason="a close is not positive, so its log return is undefined",
        )
    transitions = len(predicted_returns)
    return_mae = (
        sum(abs(p - a) for p, a in zip(predicted_returns, actual_returns, strict=True))
        / transitions
    )
    return ReconstructionError(
        defined=True,
        full_sequence_ohlcva_mae=overall,
        per_column_mae=per_column,
        internal_return_mae=return_mae,
        transitions_scored=transitions,
        rows_scored=rows,
    )


def _anchored_returns(rows: tuple[OfficialRow, ...], anchor: float) -> list[float] | None:
    previous = anchor
    out: list[float] = []
    for row in rows:
        close = row.close
        if not math.isfinite(close) or close <= 0.0 or previous <= 0.0:
            return None
        out.append(math.log(close / previous))
        previous = close
    return out


def forecast_error(
    predicted: tuple[OfficialRow, ...],
    actual: tuple[OfficialRow, ...],
    *,
    anchor_close: float,
) -> ForecastError:
    """Score a forecast against the known target.

    ``anchor_close`` is the final observed context close. It is legitimate here
    and not in reconstruction: the model had that value, so scoring the first
    predicted transition against it measures a forecast the model could make.
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
    if not _finite(predicted):
        return ForecastError(
            defined=False,
            scored_steps=len(predicted),
            undefined_reason="the prediction contains a non-finite value",
        )

    predicted_returns = _anchored_returns(predicted, anchor_close)
    actual_returns = _anchored_returns(actual, anchor_close)
    if predicted_returns is None or actual_returns is None:
        return ForecastError(
            defined=False,
            scored_steps=len(predicted),
            undefined_reason="a close is not positive, so its log return is undefined",
        )

    steps = len(predicted)
    close_return_mae = (
        sum(abs(p - a) for p, a in zip(predicted_returns, actual_returns, strict=True)) / steps
    )
    close_mae = sum(abs(p.close - a.close) for p, a in zip(predicted, actual, strict=True)) / steps
    ohlcva_mae = sum(
        abs(p.channels()[index] - a.channels()[index])
        for p, a in zip(predicted, actual, strict=True)
        for index in range(len(OFFICIAL_COLUMNS))
    ) / (steps * len(OFFICIAL_COLUMNS))

    return ForecastError(
        defined=True,
        close_return_mae=close_return_mae,
        close_mae=close_mae,
        full_ohlcva_mae=ohlcva_mae,
        scored_steps=steps,
    )


def mean_of(values: list[float]) -> float | None:
    """Arithmetic mean, or None when there is nothing to average."""
    return sum(values) / len(values) if values else None


class DirectionalAccuracy(BaseModel):
    """Sign agreement between predicted and actual one-step log returns.

    Ties are counted, never silently folded into either the numerator or the
    denominator. A step whose predicted or actual return is exactly zero has no
    direction to agree about, so it is excluded from the ratio and reported.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    defined: bool
    scored_steps: int = 0
    #: Steps where both returns are nonzero, the only ones the ratio uses.
    directional_steps: int = 0
    agreeing_steps: int = 0
    #: Agreements divided by directional_steps. None when nothing is scorable.
    accuracy: float | None = None
    predicted_zero_steps: int = 0
    actual_zero_steps: int = 0
    both_zero_steps: int = 0
    undefined_reason: str | None = None


class PersistenceComparison(BaseModel):
    """A candidate measured against the zero-return persistence baseline.

    Persistence holds the final observed context close for every target step,
    so its predicted log return is exactly zero at every step. Skill is
    ``1 - candidate / persistence``: positive beats the baseline, zero matches
    it, negative is worse. No economic or trading interpretation is attached.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    label: str
    defined: bool
    candidate_close_return_mae: float | None = None
    candidate_close_mae: float | None = None
    persistence_close_return_mae: float | None = None
    persistence_close_mae: float | None = None
    #: 1 - candidate/persistence, on each metric.
    close_return_skill: float | None = None
    close_level_skill: float | None = None
    directional: DirectionalAccuracy | None = None
    scored_steps: int = 0
    undefined_reason: str | None = None


def persistence_forecast(
    actual: tuple[OfficialRow, ...], *, anchor_close: float
) -> tuple[OfficialRow, ...]:
    """The baseline path: every close equals the final observed context close.

    Only the close matters for the metrics that use it; the other price
    channels are carried from the anchor so the row stays a well formed
    six-channel row.
    """
    return tuple(
        row.model_copy(
            update={
                "open": anchor_close,
                "high": anchor_close,
                "low": anchor_close,
                "close": anchor_close,
            }
        )
        for row in actual
    )


def directional_accuracy(
    predicted: tuple[OfficialRow, ...],
    actual: tuple[OfficialRow, ...],
    *,
    anchor_close: float,
) -> DirectionalAccuracy:
    """Sign agreement, with every tie category counted separately."""
    if len(predicted) != len(actual) or not predicted:
        return DirectionalAccuracy(
            defined=False, undefined_reason="predicted and actual lengths differ or are empty"
        )
    predicted_returns = _anchored_returns(predicted, anchor_close)
    actual_returns = _anchored_returns(actual, anchor_close)
    if predicted_returns is None or actual_returns is None:
        return DirectionalAccuracy(
            defined=False,
            scored_steps=len(predicted),
            undefined_reason="a close is not positive, so its log return is undefined",
        )

    agreeing = directional = predicted_zero = actual_zero = both_zero = 0
    for p, a in zip(predicted_returns, actual_returns, strict=True):
        p_zero, a_zero = p == 0.0, a == 0.0
        if p_zero and a_zero:
            both_zero += 1
        if p_zero:
            predicted_zero += 1
        if a_zero:
            actual_zero += 1
        if p_zero or a_zero:
            continue
        directional += 1
        if (p > 0.0) == (a > 0.0):
            agreeing += 1

    return DirectionalAccuracy(
        defined=True,
        scored_steps=len(predicted_returns),
        directional_steps=directional,
        agreeing_steps=agreeing,
        accuracy=(agreeing / directional) if directional else None,
        predicted_zero_steps=predicted_zero,
        actual_zero_steps=actual_zero,
        both_zero_steps=both_zero,
    )


def _skill(candidate: float | None, baseline: float | None) -> float | None:
    """1 - candidate/baseline. None when the baseline cannot normalise it."""
    if candidate is None or baseline is None or baseline == 0.0:
        return None
    return 1.0 - candidate / baseline


def compare_to_persistence(
    predicted: tuple[OfficialRow, ...] | None,
    actual: tuple[OfficialRow, ...],
    *,
    anchor_close: float,
    label: str,
) -> PersistenceComparison:
    """Score a candidate and the persistence baseline on the same steps."""
    if predicted is None:
        return PersistenceComparison(
            label=label, defined=False, undefined_reason="the candidate path is absent"
        )
    candidate = forecast_error(predicted, actual, anchor_close=anchor_close)
    baseline_path = persistence_forecast(actual, anchor_close=anchor_close)
    baseline = forecast_error(baseline_path, actual, anchor_close=anchor_close)
    if not candidate.defined or not baseline.defined:
        return PersistenceComparison(
            label=label,
            defined=False,
            candidate_close_return_mae=candidate.close_return_mae,
            persistence_close_return_mae=baseline.close_return_mae,
            scored_steps=candidate.scored_steps,
            undefined_reason=candidate.undefined_reason or baseline.undefined_reason,
        )
    return PersistenceComparison(
        label=label,
        defined=True,
        candidate_close_return_mae=candidate.close_return_mae,
        candidate_close_mae=candidate.close_mae,
        persistence_close_return_mae=baseline.close_return_mae,
        persistence_close_mae=baseline.close_mae,
        close_return_skill=_skill(candidate.close_return_mae, baseline.close_return_mae),
        close_level_skill=_skill(candidate.close_mae, baseline.close_mae),
        directional=directional_accuracy(predicted, actual, anchor_close=anchor_close),
        scored_steps=candidate.scored_steps,
    )


def spearman(x: list[float], y: list[float]) -> float | None:
    """Descriptive rank correlation. Local to one origin; never generalized."""
    if len(x) != len(y) or len(x) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            average = (index + stop) / 2.0 + 1.0
            for position in range(index, stop + 1):
                out[order[position]] = average
            index = stop + 1
        return out

    rx, ry = ranks(x), ranks(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0.0 or dy == 0.0:
        return None
    return numerator / (dx * dy)


__all__ += [
    "DirectionalAccuracy",
    "PersistenceComparison",
    "compare_to_persistence",
    "directional_accuracy",
    "persistence_forecast",
    "spearman",
]
