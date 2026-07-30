from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import date, datetime
from itertools import pairwise
from typing import Literal

import numpy as np
from openalpha_research.artifacts import Sha256
from pydantic import Field, model_validator

from .contracts import FrozenModel, OHLCVObservation
from .ensemble import EnsembleResult

DiagnosticStatus = Literal["available", "not_computable"]
DIAGNOSTIC_NAMES = (
    "DIRECTIONAL_AGREEMENT",
    "RETURN_DISPERSION",
    "PATH_DISPERSION",
    "CONTEXT_DIRECTION_AGREEMENT",
    "CONTEXT_RETURN_SPREAD",
    "BASELINE_DISAGREEMENT",
    "RECENT_VOLATILITY",
    "VOLATILITY_CHANGE",
    "TREND_STRENGTH",
    "GAP_OR_OUTLIER_SCORE",
    "HISTORICAL_ANALOGUE_DISTANCE",
    "ANALOGUE_OUTCOME_DISPERSION",
    "RECENT_MODEL_ERROR",
    "HORIZON_PATH_DIVERGENCE",
    "UNCERTAINTY_MISCALIBRATION",
)
CONFIGURATION_SHA256 = hashlib.sha256(
    ("sentinel-v0.3:" + ",".join(DIAGNOSTIC_NAMES)).encode()
).hexdigest()


class DiagnosticValue(FrozenModel):
    name: str = Field(min_length=1, max_length=64)
    status: DiagnosticStatus
    value: float | None
    units: str = Field(min_length=1, max_length=64)
    reason: str | None = Field(default=None, min_length=1, max_length=128)
    available_before_outcome: bool

    @model_validator(mode="after")
    def require_availability_shape(self) -> DiagnosticValue:
        if self.status == "available":
            if self.value is None or self.reason is not None:
                raise ValueError("available diagnostic requires a value and no missingness reason")
            if not math.isfinite(self.value):
                raise ValueError("diagnostic value must be finite")
        elif self.value is not None or self.reason is None:
            raise ValueError("not-computable diagnostic requires null value and a reason")
        return self


class DiagnosticVector(FrozenModel):
    symbol: Literal["SPY", "QQQ"]
    cutoff: date
    causal_through: date
    created_at: datetime
    values: tuple[DiagnosticValue, ...]
    input_sha256: Sha256
    configuration_sha256: Sha256
    action: None = None


def compute_phase_2_diagnostics(
    *,
    symbol: Literal["SPY", "QQQ"],
    cutoff: date,
    context: tuple[OHLCVObservation, ...],
    ensemble: EnsembleResult,
    created_at: datetime,
) -> DiagnosticVector:
    if len(context) < 512:
        raise ValueError("diagnostics require at least 512 causal observations")
    sessions = tuple(row.session for row in context)
    if sessions[-1] != cutoff or any(session > cutoff for session in sessions):
        raise ValueError("diagnostic context must end exactly at cutoff")
    if any(current >= following for current, following in pairwise(sessions)):
        raise ValueError("diagnostic context must be strictly increasing")
    if ensemble.cutoff_close != context[-1].close:
        raise ValueError("ensemble cutoff close does not match diagnostic context")

    closes = np.asarray([row.close for row in context], dtype=float)
    opens = np.asarray([row.open for row in context], dtype=float)
    returns = np.log(closes[1:] / closes[:-1])
    if not np.isfinite(returns).all():
        raise ValueError("causal returns must be finite")
    recent20 = returns[-20:]
    recent60 = returns[-60:]
    volatility20_daily = float(np.std(recent20, ddof=1))
    volatility60_daily = float(np.std(recent60, ddof=1))

    individual_returns = np.asarray(
        [item.predicted_log_return for item in ensemble.individual_predicted_log_returns],
        dtype=float,
    )
    cumulative_paths = np.asarray(
        [
            [math.log(row.close / ensemble.cutoff_close) for row in member.path.observations]
            for member in ensemble.members
        ],
        dtype=float,
    )
    path_std = np.std(cumulative_paths, axis=0, ddof=1)
    steps = np.arange(1.0, 6.0)
    direction_agreement = _modal_share(np.sign(individual_returns))
    return_dispersion_bps = float(np.std(individual_returns, ddof=1) * 10_000.0)
    if volatility20_daily > 0:
        path_dispersion = float(np.mean(path_std / (volatility20_daily * np.sqrt(steps))))
    else:
        path_dispersion = None
    context_returns = np.asarray(
        [summary.predicted_log_return for summary in ensemble.context_summaries],
        dtype=float,
    )
    context_direction_agreement = _modal_share(np.sign(context_returns))
    context_return_spread_bps = float(abs(context_returns[0] - context_returns[2]) * 10_000.0)
    baseline_disagreement_bps = float(
        abs(ensemble.canonical_predicted_log_return) * 10_000.0
    )
    recent_volatility = float(volatility20_daily * math.sqrt(252.0))
    volatility_change = (
        float(abs(math.log(volatility20_daily / volatility60_daily)))
        if volatility20_daily > 0 and volatility60_daily > 0
        else None
    )
    denominator = float(np.sum(np.abs(recent20)))
    trend_strength = float(abs(np.sum(recent20)) / denominator) if denominator > 0 else None
    gap_score = _gap_or_outlier_score(opens, closes, returns)
    analogue_distance, analogue_dispersion = _analogue_diagnostics(closes)
    horizon_divergence = float(np.polyfit(steps, path_std, 1)[0])

    available: dict[str, tuple[float | None, str]] = {
        "DIRECTIONAL_AGREEMENT": (direction_agreement, "share"),
        "RETURN_DISPERSION": (return_dispersion_bps, "basis_points"),
        "PATH_DISPERSION": (path_dispersion, "volatility_normalized_ratio"),
        "CONTEXT_DIRECTION_AGREEMENT": (context_direction_agreement, "share"),
        "CONTEXT_RETURN_SPREAD": (context_return_spread_bps, "basis_points"),
        "BASELINE_DISAGREEMENT": (baseline_disagreement_bps, "basis_points"),
        "RECENT_VOLATILITY": (recent_volatility, "annualized_fraction"),
        "VOLATILITY_CHANGE": (volatility_change, "absolute_log_ratio"),
        "TREND_STRENGTH": (trend_strength, "ratio"),
        "GAP_OR_OUTLIER_SCORE": (gap_score, "robust_z"),
        "HISTORICAL_ANALOGUE_DISTANCE": (analogue_distance, "normalized_distance"),
        "ANALOGUE_OUTCOME_DISPERSION": (analogue_dispersion, "log_return"),
        "HORIZON_PATH_DIVERGENCE": (horizon_divergence, "log_return_per_step"),
    }
    values: list[DiagnosticValue] = []
    for name in DIAGNOSTIC_NAMES:
        if name == "RECENT_MODEL_ERROR":
            values.append(
                _missing(name, "log_return", "NO_PRIOR_RESOLVED_FORECASTS")
            )
        elif name == "UNCERTAINTY_MISCALIBRATION":
            values.append(
                _missing(name, "calibration_error", "NO_PRIOR_CALIBRATION_SAMPLE")
            )
        else:
            value, units = available[name]
            if value is None:
                values.append(_missing(name, units, "ZERO_OR_INSUFFICIENT_DENOMINATOR"))
            else:
                values.append(
                    DiagnosticValue(
                        name=name,
                        status="available",
                        value=value,
                        units=units,
                        reason=None,
                        available_before_outcome=True,
                    )
                )
    input_payload = {
        "context": [(row.session.isoformat(), row.close) for row in context],
        "paths": [member.path.canonical_sha256 for member in ensemble.members],
    }
    input_sha256 = hashlib.sha256(
        json.dumps(input_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return DiagnosticVector(
        symbol=symbol,
        cutoff=cutoff,
        causal_through=cutoff,
        created_at=created_at,
        values=tuple(values),
        input_sha256=input_sha256,
        configuration_sha256=CONFIGURATION_SHA256,
        action=None,
    )


def _missing(name: str, units: str, reason: str) -> DiagnosticValue:
    return DiagnosticValue(
        name=name,
        status="not_computable",
        value=None,
        units=units,
        reason=reason,
        available_before_outcome=True,
    )


def _modal_share(signs: np.ndarray) -> float:
    counts = Counter(float(value) for value in signs)
    return max(counts.values()) / len(signs)


def _gap_or_outlier_score(
    opens: np.ndarray,
    closes: np.ndarray,
    returns: np.ndarray,
) -> float | None:
    gaps = np.log(opens[1:] / closes[:-1])
    baseline_returns = returns[-272:-20]
    baseline_gaps = gaps[-272:-20]
    if len(baseline_returns) < 252 or len(baseline_gaps) < 252:
        return None
    scores: list[float] = []
    for recent, baseline in ((returns[-20:], baseline_returns), (gaps[-20:], baseline_gaps)):
        median = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - median)))
        if mad == 0:
            return None
        scores.extend(abs(0.6744897501960817 * (value - median) / mad) for value in recent)
    return float(max(scores))


def _analogue_diagnostics(closes: np.ndarray) -> tuple[float | None, float | None]:
    returns = np.log(closes[1:] / closes[:-1])
    current = returns[-20:]
    current_std = float(np.std(current, ddof=1))
    if current_std == 0:
        return None, None
    current_normalized = (current - np.mean(current)) / current_std
    candidates: list[tuple[float, set[int], float]] = []
    final_index = len(closes) - 1
    for end_index in range(20, final_index - 5):
        candidate = returns[end_index - 20 : end_index]
        candidate_std = float(np.std(candidate, ddof=1))
        if candidate_std == 0:
            continue
        normalized = (candidate - np.mean(candidate)) / candidate_std
        distance = float(np.linalg.norm(current_normalized - normalized) / math.sqrt(20.0))
        outcome = float(math.log(closes[end_index + 5] / closes[end_index]))
        occupied = set(range(end_index - 20, end_index + 6))
        candidates.append((distance, occupied, outcome))
    selected: list[tuple[float, float]] = []
    used: set[int] = set()
    for distance, occupied, outcome in sorted(candidates, key=lambda item: item[0]):
        if occupied & used:
            continue
        selected.append((distance, outcome))
        used.update(occupied)
        if len(selected) == 10:
            break
    if len(selected) < 10:
        return None, None
    return (
        float(np.mean([item[0] for item in selected])),
        float(np.std([item[1] for item in selected], ddof=1)),
    )
