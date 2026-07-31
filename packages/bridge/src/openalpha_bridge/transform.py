from __future__ import annotations

import math
from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import BridgeRepresentationConfig, VolumeMode
from .errors import BridgeFailure, BridgeTransformError, FailureCategory
from .models import FinancialFeatureTensor

_PRICE_FIELDS = ("open", "high", "low", "close")


def _display_value(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "Infinity"
    if value == -math.inf:
        return "-Infinity"
    return float(value)


class BridgeFinancialTransform:
    """Framework-neutral Phase 1 target transform and inverse boundary."""

    def __init__(self, config: BridgeRepresentationConfig) -> None:
        self.config = config

    def encode_targets(
        self,
        *,
        candles: ArrayLike,
        initial_previous_close: ArrayLike,
        volume_present: ArrayLike | None = None,
    ) -> FinancialFeatureTensor:
        source, single_sequence = self._normalize_candles(candles)
        batch_size, time_steps, feature_count = source.shape
        expected_features = len(self.config.supported_feature_order)
        if feature_count != expected_features:
            self._fail(
                category=FailureCategory.INVALID_SHAPE,
                code="FEATURE_COUNT_MISMATCH",
                field="candles",
                observed_value=float(feature_count),
                lower_bound=float(expected_features),
                upper_bound=float(expected_features),
                message=f"expected {expected_features} ordered features, observed {feature_count}",
            )
        if time_steps == 0:
            self._fail(
                category=FailureCategory.INVALID_SHAPE,
                code="EMPTY_SEQUENCE",
                field="candles",
                message="at least one candle is required",
            )
        if time_steps > self.config.maximum_decode_steps:
            self._fail(
                category=FailureCategory.OUT_OF_DOMAIN,
                code="SEQUENCE_LENGTH_ABOVE_LIMIT",
                field="time",
                observed_value=float(time_steps),
                upper_bound=float(self.config.maximum_decode_steps),
                message="sequence length exceeds configured supported limit",
            )

        anchors = self._normalize_anchors(
            initial_previous_close,
            batch_size=batch_size,
            single_sequence=single_sequence,
        )
        presence = self._normalize_volume_presence(
            volume_present,
            source=source,
            single_sequence=single_sequence,
        )
        self._validate_anchor_domain(anchors)
        self._validate_source(source, presence)

        encoded_features = 4 if self.config.volume_mode is VolumeMode.PRICE_ONLY else 5
        encoded = np.empty((batch_size, time_steps, encoded_features), dtype=np.float64)
        for sequence_index in range(batch_size):
            previous_close = float(anchors[sequence_index])
            for candle_index in range(time_steps):
                row = source[sequence_index, candle_index]
                open_value, high_value, low_value, close_value = (
                    float(row[index]) for index in range(4)
                )
                log_previous = math.log(previous_close)
                log_open = math.log(open_value)
                log_high = math.log(high_value)
                log_low = math.log(low_value)
                log_close = math.log(close_value)
                gap = log_open - log_previous
                body = log_close - log_open
                upper = log_high - max(log_open, log_close)
                lower = min(log_open, log_close) - log_low
                upper = self._canonical_nonnegative(upper)
                lower = self._canonical_nonnegative(lower)

                self._validate_channel(
                    value=gap,
                    limit=self.config.gap_return_limit,
                    sequence_index=sequence_index,
                    candle_index=candle_index,
                    field="gap",
                    signed=True,
                )
                self._validate_channel(
                    value=body,
                    limit=self.config.body_return_limit,
                    sequence_index=sequence_index,
                    candle_index=candle_index,
                    field="body",
                    signed=True,
                )
                self._validate_channel(
                    value=upper,
                    limit=self.config.upper_wick_limit,
                    sequence_index=sequence_index,
                    candle_index=candle_index,
                    field="upper",
                    signed=False,
                )
                self._validate_channel(
                    value=lower,
                    limit=self.config.lower_wick_limit,
                    sequence_index=sequence_index,
                    candle_index=candle_index,
                    field="lower",
                    signed=False,
                )
                encoded[sequence_index, candle_index, :4] = (gap, body, upper, lower)
                if encoded_features == 5:
                    assert presence is not None
                    if presence[sequence_index, candle_index]:
                        log_volume = math.log1p(float(row[4]))
                        self._validate_channel(
                            value=log_volume,
                            limit=self.config.volume_limit,
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            field="log1p_volume",
                            signed=False,
                        )
                        encoded[sequence_index, candle_index, 4] = log_volume
                    else:
                        encoded[sequence_index, candle_index, 4] = math.nan
                previous_close = close_value

        output_values = encoded[0] if single_sequence else encoded
        output_presence: NDArray[np.bool_] | None
        if presence is None:
            output_presence = None
        else:
            output_presence = presence[0] if single_sequence else presence
        return FinancialFeatureTensor(
            values=output_values,
            volume_present=output_presence,
            single_sequence=single_sequence,
            representation_version=self.config.representation_version,
            configuration_sha256=self.config.canonical_sha256,
        )

    def _normalize_candles(self, candles: ArrayLike) -> tuple[NDArray[np.float64], bool]:
        raw = np.asarray(candles)
        if raw.dtype.kind not in "fiu":
            self._fail(
                category=FailureCategory.INVALID_INPUT,
                code="NONNUMERIC_CANDLES",
                field="candles",
                message="candles must contain real numeric values",
            )
        if raw.ndim not in (2, 3):
            self._fail(
                category=FailureCategory.INVALID_SHAPE,
                code="CANDLE_RANK_MISMATCH",
                field="candles",
                observed_value=float(raw.ndim),
                lower_bound=2.0,
                upper_bound=3.0,
                message="candles must have shape [time,features] or [batch,time,features]",
            )
        single_sequence = raw.ndim == 2
        result = np.asarray(raw, dtype=np.float64)
        return (result[np.newaxis, ...] if single_sequence else result), single_sequence

    def _normalize_anchors(
        self,
        anchors: ArrayLike,
        *,
        batch_size: int,
        single_sequence: bool,
    ) -> NDArray[np.float64]:
        raw = np.asarray(anchors)
        if raw.dtype.kind not in "fiu" or raw.ndim > 1:
            self._fail(
                category=FailureCategory.INVALID_SHAPE,
                code="PREVIOUS_CLOSE_SHAPE",
                field="initial_previous_close",
                message="previous closes must be a scalar for one sequence or exact [batch] vector",
            )
        if single_sequence:
            if raw.ndim == 0:
                result = np.asarray([raw.item()], dtype=np.float64)
            elif raw.shape == (1,):
                result = np.asarray(raw, dtype=np.float64)
            else:
                self._fail(
                    category=FailureCategory.INVALID_SHAPE,
                    code="PREVIOUS_CLOSE_SHAPE",
                    field="initial_previous_close",
                    message="single-sequence previous close must be scalar or shape [1]",
                )
        else:
            if raw.ndim != 1 or raw.shape != (batch_size,):
                self._fail(
                    category=FailureCategory.INVALID_SHAPE,
                    code="PREVIOUS_CLOSE_SHAPE",
                    field="initial_previous_close",
                    observed_value=float(raw.size),
                    lower_bound=float(batch_size),
                    upper_bound=float(batch_size),
                    message="batch previous closes must have exact shape [batch]",
                )
            result = np.asarray(raw, dtype=np.float64)
        return result

    def _normalize_volume_presence(
        self,
        value: ArrayLike | None,
        *,
        source: NDArray[np.float64],
        single_sequence: bool,
    ) -> NDArray[np.bool_] | None:
        batch_size, time_steps, _ = source.shape
        mode = self.config.volume_mode
        if mode is VolumeMode.PRICE_ONLY:
            if value is not None:
                self._fail(
                    category=FailureCategory.INVALID_INPUT,
                    code="VOLUME_MASK_UNEXPECTED",
                    field="volume_present",
                    message="PRICE_ONLY calls must not provide a volume mask",
                )
            return None
        if mode is VolumeMode.VOLUME_REQUIRED:
            if value is not None:
                self._fail(
                    category=FailureCategory.INVALID_INPUT,
                    code="VOLUME_MASK_UNEXPECTED",
                    field="volume_present",
                    message="VOLUME_REQUIRED implies every candle is present",
                )
            return np.ones((batch_size, time_steps), dtype=np.bool_)
        if value is None:
            self._fail(
                category=FailureCategory.INVALID_INPUT,
                code="VOLUME_MASK_REQUIRED",
                field="volume_present",
                message="VOLUME_OPTIONAL requires an explicit presence mask",
            )
        raw = np.asarray(value)
        if raw.dtype.kind != "b":
            self._fail(
                category=FailureCategory.INVALID_INPUT,
                code="VOLUME_MASK_DTYPE",
                field="volume_present",
                message="volume presence mask must have boolean dtype",
            )
        expected_shape = (time_steps,) if single_sequence else (batch_size, time_steps)
        if raw.shape != expected_shape:
            self._fail(
                category=FailureCategory.INVALID_SHAPE,
                code="VOLUME_MASK_SHAPE",
                field="volume_present",
                message=f"volume mask must have exact shape {expected_shape}",
            )
        return np.asarray(raw[np.newaxis, ...] if single_sequence else raw, dtype=np.bool_)

    def _validate_anchor_domain(self, anchors: NDArray[np.float64]) -> None:
        for sequence_index, item in enumerate(anchors):
            value = float(item)
            if not math.isfinite(value):
                self._fail(
                    category=FailureCategory.INVALID_INPUT,
                    code="NONFINITE_PREVIOUS_CLOSE",
                    sequence_index=sequence_index,
                    field="initial_previous_close",
                    observed_value=_display_value(value),
                    message="initial previous close must be finite",
                )
            if value <= 0.0:
                self._fail(
                    category=FailureCategory.INVALID_INPUT,
                    code="NONPOSITIVE_PREVIOUS_CLOSE",
                    sequence_index=sequence_index,
                    field="initial_previous_close",
                    observed_value=value,
                    lower_bound=0.0,
                    message="initial previous close must be positive",
                )
            self._validate_price_domain(
                value,
                sequence_index=sequence_index,
                candle_index=None,
                field="initial_previous_close",
                code_prefix="PREVIOUS_CLOSE",
            )

    def _validate_source(
        self,
        source: NDArray[np.float64],
        presence: NDArray[np.bool_] | None,
    ) -> None:
        for sequence_index in range(source.shape[0]):
            for candle_index in range(source.shape[1]):
                row = source[sequence_index, candle_index]
                for field_index, field in enumerate(_PRICE_FIELDS):
                    value = float(row[field_index])
                    if not math.isfinite(value):
                        self._fail(
                            category=FailureCategory.INVALID_INPUT,
                            code=f"NONFINITE_{field.upper()}",
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            field=field,
                            observed_value=_display_value(value),
                            message=f"source {field} must be finite",
                        )
                    if value <= 0.0:
                        self._fail(
                            category=FailureCategory.INVALID_INPUT,
                            code=f"NONPOSITIVE_{field.upper()}",
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            field=field,
                            observed_value=value,
                            lower_bound=0.0,
                            message=f"source {field} must be positive",
                        )

                open_value, high_value, low_value, close_value = (
                    float(row[index]) for index in range(4)
                )
                ordering_checks = (
                    (high_value < open_value, "HIGH_BELOW_OPEN", "high", high_value),
                    (high_value < close_value, "HIGH_BELOW_CLOSE", "high", high_value),
                    (high_value < low_value, "HIGH_BELOW_LOW", "high", high_value),
                    (low_value > open_value, "LOW_ABOVE_OPEN", "low", low_value),
                    (low_value > close_value, "LOW_ABOVE_CLOSE", "low", low_value),
                )
                for invalid, code, field, value in ordering_checks:
                    if invalid:
                        self._fail(
                            category=FailureCategory.INVALID_INPUT,
                            code=code,
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            field=field,
                            observed_value=value,
                            message="source candle violates OHLC ordering",
                        )
                for field_index, field in enumerate(_PRICE_FIELDS):
                    self._validate_price_domain(
                        float(row[field_index]),
                        sequence_index=sequence_index,
                        candle_index=candle_index,
                        field=field,
                        code_prefix="PRICE",
                    )

                if presence is None:
                    continue
                volume = float(row[4])
                is_present = bool(presence[sequence_index, candle_index])
                if is_present:
                    if not math.isfinite(volume):
                        code = (
                            "PRESENT_VOLUME_NONFINITE"
                            if self.config.volume_mode is VolumeMode.VOLUME_OPTIONAL
                            else "NONFINITE_VOLUME"
                        )
                        self._fail(
                            category=FailureCategory.INVALID_INPUT,
                            code=code,
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            field="volume",
                            observed_value=_display_value(volume),
                            message="present source volume must be finite",
                        )
                    if volume < 0.0:
                        self._fail(
                            category=FailureCategory.INVALID_INPUT,
                            code="NEGATIVE_VOLUME",
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            field="volume",
                            observed_value=volume,
                            lower_bound=0.0,
                            message="present source volume must be nonnegative",
                        )
                    if volume > self.config.maximum_volume:
                        self._fail(
                            category=FailureCategory.OUT_OF_DOMAIN,
                            code="VOLUME_ABOVE_LIMIT",
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            field="volume",
                            observed_value=volume,
                            upper_bound=self.config.maximum_volume,
                            message="source volume exceeds configured supported limit",
                        )
                elif not math.isnan(volume):
                    self._fail(
                        category=FailureCategory.INVALID_INPUT,
                        code="MISSING_VOLUME_MUST_USE_NAN",
                        sequence_index=sequence_index,
                        candle_index=candle_index,
                        field="volume",
                        observed_value=_display_value(volume),
                        message="missing optional volume must use NaN storage plus a false presence mask",
                    )

    def _validate_price_domain(
        self,
        value: float,
        *,
        sequence_index: int,
        candle_index: int | None,
        field: str,
        code_prefix: str,
    ) -> None:
        if value < self.config.minimum_price:
            self._fail(
                category=FailureCategory.OUT_OF_DOMAIN,
                code=f"{code_prefix}_BELOW_LIMIT",
                sequence_index=sequence_index,
                candle_index=candle_index,
                field=field,
                observed_value=value,
                lower_bound=self.config.minimum_price,
                message=f"{field} is below configured supported price limit",
            )
        if value > self.config.maximum_price:
            self._fail(
                category=FailureCategory.OUT_OF_DOMAIN,
                code=f"{code_prefix}_ABOVE_LIMIT",
                sequence_index=sequence_index,
                candle_index=candle_index,
                field=field,
                observed_value=value,
                upper_bound=self.config.maximum_price,
                message=f"{field} is above configured supported price limit",
            )

    def _canonical_nonnegative(self, value: float) -> float:
        if value < 0.0 and abs(value) <= self.config.numerical_epsilon:
            return 0.0
        return 0.0 if value == 0.0 else value

    def _validate_channel(
        self,
        *,
        value: float,
        limit: float,
        sequence_index: int,
        candle_index: int,
        field: str,
        signed: bool,
    ) -> None:
        if not math.isfinite(value):
            self._fail(
                category=FailureCategory.UNSUPPORTED_NUMERICAL,
                code=f"NONFINITE_{field.upper()}",
                sequence_index=sequence_index,
                candle_index=candle_index,
                field=field,
                observed_value=_display_value(value),
                message="transformed channel must be finite",
            )
        lower = -limit if signed else 0.0
        if value < lower:
            self._fail(
                category=FailureCategory.OUT_OF_DOMAIN,
                code=f"{field.upper()}_BELOW_LIMIT",
                sequence_index=sequence_index,
                candle_index=candle_index,
                field=field,
                observed_value=value,
                lower_bound=lower,
                message=f"transformed {field} is below configured supported limit",
            )
        if value > limit:
            self._fail(
                category=FailureCategory.OUT_OF_DOMAIN,
                code=f"{field.upper()}_ABOVE_LIMIT",
                sequence_index=sequence_index,
                candle_index=candle_index,
                field=field,
                observed_value=value,
                upper_bound=limit,
                message=f"transformed {field} exceeds configured supported limit",
            )

    def _fail(
        self,
        *,
        category: FailureCategory,
        code: str,
        message: str,
        sequence_index: int | None = None,
        candle_index: int | None = None,
        field: str | None = None,
        observed_value: float | str | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> NoReturn:
        raise BridgeTransformError(
            BridgeFailure(
                category=category,
                code=code,
                sequence_index=sequence_index,
                candle_index=candle_index,
                field=field,
                observed_value=observed_value,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                message=message,
            )
        )
