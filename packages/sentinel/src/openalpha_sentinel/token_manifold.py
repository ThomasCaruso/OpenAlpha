from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import Field

from .contracts import FrozenModel


class PairSupport(FrozenModel):
    coarse_count: int = Field(ge=0)
    fine_count: int = Field(ge=0)
    exact_pair_count: int = Field(ge=0)
    exact_pair_frequency: float = Field(ge=0.0, le=1.0)
    smoothed_probability: float = Field(gt=0.0, le=1.0)
    smoothed_surprisal: float = Field(ge=0.0)
    empirically_supported: bool


class TokenSupportIndex(FrozenModel):
    total_pairs: int = Field(ge=1)
    vocab_s1: int = Field(ge=1)
    vocab_s2: int = Field(ge=1)
    minimum_pair_count: int = Field(ge=1)
    alpha: float = Field(gt=0.0)
    coarse_counts: dict[int, int]
    fine_counts: dict[int, int]
    pair_counts: dict[str, int]

    def describe(self, coarse: int, fine: int) -> PairSupport:
        count = self.pair_counts.get(_pair_key(coarse, fine), 0)
        probability = (count + self.alpha) / (
            self.total_pairs + self.alpha * self.vocab_s1 * self.vocab_s2
        )
        return PairSupport(
            coarse_count=self.coarse_counts.get(coarse, 0),
            fine_count=self.fine_counts.get(fine, 0),
            exact_pair_count=count,
            exact_pair_frequency=count / self.total_pairs,
            smoothed_probability=probability,
            smoothed_surprisal=-math.log(probability),
            empirically_supported=count >= self.minimum_pair_count,
        )


class RoundTripSummary(FrozenModel):
    reconstructed_candle_count: int = Field(ge=1)
    invalid_candle_count: int = Field(ge=0)
    invalid_candle_fraction: float = Field(ge=0.0, le=1.0)
    wilson_lower: float = Field(ge=0.0, le=1.0)
    wilson_upper: float = Field(ge=0.0, le=1.0)
    material_defect: bool
    overwhelmingly_valid: bool
    violation_categories: dict[str, int]


class CausalVolatilityRegime(FrozenModel):
    status: Literal["computable", "not_computable"]
    value: float | None
    regime: Literal["low", "middle", "high"] | None
    reason: str | None


def select_support_windows[T](
    values: Sequence[T],
    *,
    window_length: int,
    count: int,
) -> tuple[tuple[T, ...], ...]:
    if window_length < 1 or count < 1:
        raise ValueError("window length and count must be positive")
    required = window_length * count
    if len(values) < required:
        raise ValueError(f"support corpus requires at least {required:,} rows")
    retained = tuple(values[-required:])
    return tuple(
        retained[index : index + window_length]
        for index in range(0, required, window_length)
    )


def build_token_support(
    token_pairs: Sequence[tuple[int, int]],
    *,
    vocab_s1: int,
    vocab_s2: int,
    minimum_pair_count: int = 2,
    alpha: float = 0.5,
) -> TokenSupportIndex:
    pairs = tuple(token_pairs)
    if not pairs:
        raise ValueError("token support requires at least one pair")
    if vocab_s1 < 1 or vocab_s2 < 1:
        raise ValueError("vocabulary sizes must be positive")
    if minimum_pair_count < 1 or not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("support floor and smoothing alpha must be positive")
    if any(
        coarse < 0 or coarse >= vocab_s1 or fine < 0 or fine >= vocab_s2
        for coarse, fine in pairs
    ):
        raise ValueError("token pair falls outside the declared vocabulary")
    return TokenSupportIndex(
        total_pairs=len(pairs),
        vocab_s1=vocab_s1,
        vocab_s2=vocab_s2,
        minimum_pair_count=minimum_pair_count,
        alpha=alpha,
        coarse_counts=dict(Counter(coarse for coarse, _ in pairs)),
        fine_counts=dict(Counter(fine for _, fine in pairs)),
        pair_counts=dict(Counter(_pair_key(*pair) for pair in pairs)),
    )


def wilson_interval(
    *,
    successes: int,
    observations: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if observations < 1 or successes < 0 or successes > observations:
        raise ValueError("Wilson counts are invalid")
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("Wilson z must be finite and positive")
    proportion = successes / observations
    denominator = 1.0 + z * z / observations
    center = (proportion + z * z / (2.0 * observations)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / observations
            + z * z / (4.0 * observations * observations)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def summarize_roundtrip_rows(
    rows: Sequence[Mapping[str, Any]],
) -> RoundTripSummary:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("round-trip summary requires at least one row")
    invalid = tuple(row for row in materialized if row.get("valid") is not True)
    categories: Counter[str] = Counter()
    for row in invalid:
        codes = row.get("violation_codes")
        if not isinstance(codes, (tuple, list)):
            raise TypeError("invalid round-trip row requires violation codes")
        categories.update(str(code) for code in codes)
    fraction = len(invalid) / len(materialized)
    lower, upper = wilson_interval(
        successes=len(invalid),
        observations=len(materialized),
    )
    return RoundTripSummary(
        reconstructed_candle_count=len(materialized),
        invalid_candle_count=len(invalid),
        invalid_candle_fraction=fraction,
        wilson_lower=lower,
        wilson_upper=upper,
        material_defect=fraction >= 0.01 and lower >= 0.005,
        overwhelmingly_valid=fraction <= 0.001 and upper <= 0.0025,
        violation_categories=dict(sorted(categories.items())),
    )


def normalized_reconstruction_errors(
    *,
    observed: Mapping[str, float],
    reconstructed: Mapping[str, float],
) -> dict[str, float]:
    fields = ("open", "high", "low", "close")
    try:
        observed_values = {field: float(observed[field]) for field in fields}
        reconstructed_values = {
            field: float(reconstructed[field]) for field in fields
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reconstruction requires named OHLC values") from exc
    values = tuple(observed_values.values()) + tuple(reconstructed_values.values())
    denominator = observed_values["close"]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("reconstruction values must be finite")
    if denominator <= 0.0:
        raise ValueError("observed close must be positive")
    result = {
        field: abs(reconstructed_values[field] - observed_values[field])
        / denominator
        for field in fields
    }
    observed_range = observed_values["high"] - observed_values["low"]
    reconstructed_range = reconstructed_values["high"] - reconstructed_values["low"]
    result["range"] = abs(reconstructed_range - observed_range) / denominator
    return result


def causal_volatility_regimes(
    closes: Sequence[float],
    *,
    window: int,
) -> tuple[CausalVolatilityRegime, ...]:
    values = np.asarray(tuple(closes), dtype=float)
    if window < 2:
        raise ValueError("volatility window must be at least two")
    if len(values) <= window:
        raise ValueError("volatility regimes require more observations than the window")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("close values must be finite and positive")
    log_returns = np.diff(np.log(values))
    volatilities: list[float | None] = [None] * window
    for index in range(window, len(values)):
        sample = log_returns[index - window : index]
        volatilities.append(float(np.std(sample, ddof=0)))
    computable = np.asarray(
        [value for value in volatilities if value is not None],
        dtype=float,
    )
    lower, upper = np.quantile(computable, (1.0 / 3.0, 2.0 / 3.0))
    result = []
    for value in volatilities:
        if value is None:
            result.append(
                CausalVolatilityRegime(
                    status="not_computable",
                    value=None,
                    regime=None,
                    reason="INSUFFICIENT_CAUSAL_RETURNS",
                )
            )
            continue
        regime: Literal["low", "middle", "high"]
        if value <= lower:
            regime = "low"
        elif value <= upper:
            regime = "middle"
        else:
            regime = "high"
        result.append(
            CausalVolatilityRegime(
                status="computable",
                value=value,
                regime=regime,
                reason=None,
            )
        )
    return tuple(result)


def _pair_key(coarse: int, fine: int) -> str:
    return f"{coarse}:{fine}"
