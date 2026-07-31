from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .activations import bounded_nonnegative, bounded_signed
from .config import VolumeMode
from .errors import FailureCategory
from .models import FinancialFeatureTensor
from .results import ReconstructedSequence
from .transform import BridgeFinancialTransform as _TargetTransform
from .transform import _display_value
from .validation import BridgeAuditResult, audit_bridge_output

_HEAD_FIELDS = ("gap", "body", "upper", "lower", "log1p_volume")


class BridgeFinancialTransform(_TargetTransform):
    """Complete Phase 1 financial transform, activation, and causal inverse."""

    def decode_head_outputs(
        self,
        *,
        raw_outputs: ArrayLike,
        initial_previous_close: ArrayLike,
        volume_present: ArrayLike | None = None,
    ) -> ReconstructedSequence:
        raw, single_sequence = self._normalize_candles(raw_outputs)
        if raw.shape[2] != 5:
            self._fail(
                category=FailureCategory.INVALID_SHAPE,
                code="HEAD_FEATURE_COUNT_MISMATCH",
                field="raw_outputs",
                observed_value=float(raw.shape[2]),
                lower_bound=5.0,
                upper_bound=5.0,
                message="raw Bridge head output must have exactly five channels",
            )
        self._validate_time_length(raw.shape[1])
        for sequence_index in range(raw.shape[0]):
            for candle_index in range(raw.shape[1]):
                for field_index, field in enumerate(_HEAD_FIELDS):
                    value = float(raw[sequence_index, candle_index, field_index])
                    if not math.isfinite(value):
                        self._fail(
                            category=FailureCategory.INVALID_INPUT,
                            code="NONFINITE_HEAD_OUTPUT",
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            field=field,
                            observed_value=_display_value(value),
                            message="all five raw Bridge head channels must be finite",
                        )

        presence = self._decode_volume_presence(
            volume_present,
            batch_size=raw.shape[0],
            time_steps=raw.shape[1],
            single_sequence=single_sequence,
        )
        dtype = np.dtype(self.config.output_dtype)
        dtype_maximum = float(np.finfo(dtype).max)
        for sequence_index in range(raw.shape[0]):
            for candle_index in range(raw.shape[1]):
                for field_index, field in enumerate(_HEAD_FIELDS):
                    value = float(raw[sequence_index, candle_index, field_index])
                    if abs(value) > dtype_maximum:
                        self._fail(
                            category=FailureCategory.UNSUPPORTED_NUMERICAL,
                            code="HEAD_OUTPUT_OUTSIDE_DTYPE_RANGE",
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            field=field,
                            observed_value=value,
                            lower_bound=-dtype_maximum,
                            upper_bound=dtype_maximum,
                            message="raw head channel cannot be represented in output dtype",
                        )
        output_features = 4 if self.config.volume_mode is VolumeMode.PRICE_ONLY else 5
        mapped = np.empty((raw.shape[0], raw.shape[1], output_features), dtype=dtype)
        raw_runtime = np.asarray(raw, dtype=dtype)
        mapped[..., 0] = bounded_signed(raw_runtime[..., 0], self.config.gap_return_limit)
        mapped[..., 1] = bounded_signed(raw_runtime[..., 1], self.config.body_return_limit)
        mapped[..., 2] = bounded_nonnegative(raw_runtime[..., 2], self.config.upper_wick_limit)
        mapped[..., 3] = bounded_nonnegative(raw_runtime[..., 3], self.config.lower_wick_limit)
        warnings: tuple[str, ...] = ()
        if output_features == 5:
            assert presence is not None
            volume_values = bounded_nonnegative(raw_runtime[..., 4], self.config.volume_limit)
            mapped[..., 4] = np.where(presence, volume_values, np.nan)
        else:
            warnings = ("PRICE_ONLY_RAW_VOLUME_CHANNEL_IGNORED",)

        return self._decode_mapped(
            mapped=mapped,
            presence=presence,
            single_sequence=single_sequence,
            initial_previous_close=initial_previous_close,
            warnings=warnings,
        )

    def decode_features(
        self,
        *,
        transformed_features: FinancialFeatureTensor | ArrayLike,
        initial_previous_close: ArrayLike,
        volume_present: ArrayLike | None = None,
    ) -> ReconstructedSequence:
        warnings: tuple[str, ...] = ()
        feature_tensor_input = isinstance(transformed_features, FinancialFeatureTensor)
        if feature_tensor_input:
            if transformed_features.configuration_sha256 != self.config.canonical_sha256:
                self._fail(
                    category=FailureCategory.INVALID_CONFIGURATION,
                    code="CONFIGURATION_HASH_MISMATCH",
                    field="transformed_features",
                    message="feature tensor configuration hash does not match this transform",
                )
            if transformed_features.representation_version != self.config.representation_version:
                self._fail(
                    category=FailureCategory.INVALID_CONFIGURATION,
                    code="REPRESENTATION_VERSION_MISMATCH",
                    field="transformed_features",
                    message="feature tensor representation version does not match this transform",
                )
            if volume_present is not None:
                self._fail(
                    category=FailureCategory.INVALID_INPUT,
                    code="VOLUME_MASK_DUPLICATED",
                    field="volume_present",
                    message="feature tensor already carries volume presence metadata",
                )
            raw_values = transformed_features.values
            volume_present = transformed_features.volume_present
            warnings = transformed_features.numerical_warnings
        else:
            raw_values = transformed_features

        mapped, single_sequence = self._normalize_candles(raw_values)
        expected_features = 4 if self.config.volume_mode is VolumeMode.PRICE_ONLY else 5
        if mapped.shape[2] != expected_features:
            self._fail(
                category=FailureCategory.INVALID_SHAPE,
                code="FEATURE_COUNT_MISMATCH",
                field="transformed_features",
                observed_value=float(mapped.shape[2]),
                lower_bound=float(expected_features),
                upper_bound=float(expected_features),
                message=f"expected {expected_features} transformed financial channels",
            )
        self._validate_time_length(mapped.shape[1])
        presence = self._decode_volume_presence(
            volume_present,
            batch_size=mapped.shape[0],
            time_steps=mapped.shape[1],
            single_sequence=single_sequence,
            allow_required_metadata=feature_tensor_input,
        )
        self._validate_mapped_channels(mapped, presence)
        return self._decode_mapped(
            mapped=np.asarray(mapped, dtype=np.dtype(self.config.output_dtype)),
            presence=presence,
            single_sequence=single_sequence,
            initial_previous_close=initial_previous_close,
            warnings=warnings,
        )

    def _validate_time_length(self, time_steps: int) -> None:
        if time_steps == 0:
            self._fail(
                category=FailureCategory.INVALID_SHAPE,
                code="EMPTY_SEQUENCE",
                field="time",
                message="at least one transformed candle is required",
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

    def audit(
        self,
        decoded: ReconstructedSequence,
        *,
        timestamps: object | None = None,
        expected_timestamps: object | None = None,
        expected_sequence_length: int | None = None,
    ) -> BridgeAuditResult:
        return audit_bridge_output(
            config=self.config,
            candles=decoded.candles,
            volume_present=decoded.volume_present,
            timestamps=timestamps,
            expected_timestamps=expected_timestamps,
            expected_sequence_length=expected_sequence_length,
        )

    def _decode_volume_presence(
        self,
        value: ArrayLike | None,
        *,
        batch_size: int,
        time_steps: int,
        single_sequence: bool,
        allow_required_metadata: bool = False,
    ) -> NDArray[np.bool_] | None:
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
                if not allow_required_metadata:
                    self._fail(
                        category=FailureCategory.INVALID_INPUT,
                        code="VOLUME_MASK_UNEXPECTED",
                        field="volume_present",
                        message="VOLUME_REQUIRED calls must not provide a volume mask",
                    )
                supplied = np.asarray(value)
                expected = (time_steps,) if single_sequence else (batch_size, time_steps)
                if supplied.dtype.kind != "b" or supplied.shape != expected or not supplied.all():
                    self._fail(
                        category=FailureCategory.INVALID_INPUT,
                        code="VOLUME_MASK_UNEXPECTED",
                        field="volume_present",
                        message="VOLUME_REQUIRED metadata must be absent or exactly all true",
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
        expected = (time_steps,) if single_sequence else (batch_size, time_steps)
        if raw.dtype.kind != "b":
            self._fail(
                category=FailureCategory.INVALID_INPUT,
                code="VOLUME_MASK_DTYPE",
                field="volume_present",
                message="volume presence mask must have boolean dtype",
            )
        if raw.shape != expected:
            self._fail(
                category=FailureCategory.INVALID_SHAPE,
                code="VOLUME_MASK_SHAPE",
                field="volume_present",
                message=f"volume mask must have exact shape {expected}",
            )
        return np.asarray(raw[np.newaxis, ...] if single_sequence else raw, dtype=np.bool_)

    def _validate_mapped_channels(
        self,
        mapped: NDArray[np.float64],
        presence: NDArray[np.bool_] | None,
    ) -> None:
        limits = (
            self.config.gap_return_limit,
            self.config.body_return_limit,
            self.config.upper_wick_limit,
            self.config.lower_wick_limit,
        )
        for sequence_index in range(mapped.shape[0]):
            for candle_index in range(mapped.shape[1]):
                for field_index, (field, limit) in enumerate(zip(_HEAD_FIELDS[:4], limits)):
                    self._validate_mapped_channel_strict(
                        value=float(mapped[sequence_index, candle_index, field_index]),
                        limit=limit,
                        signed=field_index < 2,
                        sequence_index=sequence_index,
                        candle_index=candle_index,
                        field=field,
                    )
                if mapped.shape[2] == 5:
                    assert presence is not None
                    value = float(mapped[sequence_index, candle_index, 4])
                    if presence[sequence_index, candle_index]:
                        self._validate_mapped_channel_strict(
                            value=value,
                            limit=self.config.volume_limit,
                            signed=False,
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            field="log1p_volume",
                        )
                    elif not math.isnan(value):
                        self._fail(
                            category=FailureCategory.INVALID_INPUT,
                            code="MISSING_VOLUME_MUST_USE_NAN",
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            field="log1p_volume",
                            observed_value=_display_value(value),
                            message="missing transformed volume must be NaN under a false mask",
                        )

    def _validate_mapped_channel_strict(
        self,
        *,
        value: float,
        limit: float,
        signed: bool,
        sequence_index: int,
        candle_index: int,
        field: str,
    ) -> None:
        if not math.isfinite(value):
            self._fail(
                category=FailureCategory.INVALID_INPUT,
                code=f"NONFINITE_{field.upper()}",
                sequence_index=sequence_index,
                candle_index=candle_index,
                field=field,
                observed_value=_display_value(value),
                message="transformed financial channel must be finite",
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
                message="transformed financial channel is below its exact bound",
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
                message="transformed financial channel exceeds its exact bound",
            )

    def _decode_mapped(
        self,
        *,
        mapped: NDArray[np.floating],
        presence: NDArray[np.bool_] | None,
        single_sequence: bool,
        initial_previous_close: ArrayLike,
        warnings: tuple[str, ...],
    ) -> ReconstructedSequence:
        batch_size, time_steps, feature_count = mapped.shape
        anchors = self._normalize_anchors(
            initial_previous_close,
            batch_size=batch_size,
            single_sequence=single_sequence,
        )
        self._validate_anchor_domain(anchors)
        dtype = np.dtype(self.config.output_dtype)
        dtype_type = cast(type[np.float32] | type[np.float64], dtype.type)
        candles = np.empty((batch_size, time_steps, feature_count), dtype=dtype)
        for sequence_index in range(batch_size):
            previous_close = dtype_type(anchors[sequence_index])
            for candle_index in range(time_steps):
                row = mapped[sequence_index, candle_index]
                log_previous = dtype_type(np.log(previous_close))
                log_open = dtype_type(log_previous + dtype_type(row[0]))
                log_close = dtype_type(log_open + dtype_type(row[1]))
                log_high = dtype_type(max(log_open, log_close) + dtype_type(row[2]))
                log_low = dtype_type(min(log_open, log_close) - dtype_type(row[3]))
                logs = (log_open, log_high, log_low, log_close)
                for field, log_value in zip(("open", "high", "low", "close"), logs):
                    self._validate_log_value(
                        float(log_value),
                        dtype=dtype,
                        sequence_index=sequence_index,
                        candle_index=candle_index,
                        field=field,
                    )
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    prices = np.asarray([np.exp(value) for value in logs], dtype=dtype)
                if not np.isfinite(prices).all() or not (prices > 0.0).all():
                    self._fail(
                        category=FailureCategory.UNSUPPORTED_NUMERICAL,
                        code="PRICE_EXPONENTIATION_FAILED",
                        sequence_index=sequence_index,
                        candle_index=candle_index,
                        field="ohlc",
                        message="guarded exponential did not produce finite positive OHLC",
                    )
                candles[sequence_index, candle_index, :4] = prices
                if feature_count == 5:
                    assert presence is not None
                    if presence[sequence_index, candle_index]:
                        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                            volume = dtype_type(np.expm1(dtype_type(row[4])))
                        if not math.isfinite(float(volume)) or volume < 0.0:
                            self._fail(
                                category=FailureCategory.UNSUPPORTED_NUMERICAL,
                                code="VOLUME_EXPM1_FAILED",
                                sequence_index=sequence_index,
                                candle_index=candle_index,
                                field="volume",
                                observed_value=_display_value(float(volume)),
                                message="expm1 did not produce finite nonnegative volume",
                            )
                        candles[sequence_index, candle_index, 4] = volume
                    else:
                        candles[sequence_index, candle_index, 4] = math.nan
                if not self._constructed_row_valid(
                    candles[sequence_index, candle_index], feature_count
                ):
                    self._fail(
                        category=FailureCategory.UNSUPPORTED_NUMERICAL,
                        code="CONSTRUCTION_CONTRACT_VIOLATION",
                        sequence_index=sequence_index,
                        candle_index=candle_index,
                        field="ohlcv",
                        message="runtime arithmetic violated the hard financial construction",
                    )
                previous_close = candles[sequence_index, candle_index, 3]

        output_candles = candles[0] if single_sequence else candles
        output_presence = (
            None if presence is None else (presence[0] if single_sequence else presence)
        )
        output_mapped = mapped[0] if single_sequence else mapped
        transformed = FinancialFeatureTensor(
            values=output_mapped,
            volume_present=output_presence,
            single_sequence=single_sequence,
            representation_version=self.config.representation_version,
            configuration_sha256=self.config.canonical_sha256,
            numerical_warnings=warnings,
        )
        return ReconstructedSequence(
            candles=output_candles,
            transformed_features=transformed,
            volume_present=output_presence,
            single_sequence=single_sequence,
            output_dtype=self.config.output_dtype,
            representation_version=self.config.representation_version,
            configuration_sha256=self.config.canonical_sha256,
            numerical_warnings=warnings,
        )

    def _validate_log_value(
        self,
        value: float,
        *,
        dtype: np.dtype,
        sequence_index: int,
        candle_index: int,
        field: str,
    ) -> None:
        if not math.isfinite(value):
            self._fail(
                category=FailureCategory.UNSUPPORTED_NUMERICAL,
                code="NONFINITE_LOG_PRICE",
                sequence_index=sequence_index,
                candle_index=candle_index,
                field=field,
                observed_value=_display_value(value),
                message="log-price recursion produced a nonfinite value",
            )
        guard_low, guard_high = self.config.log_price_guard
        if value < guard_low or value > guard_high:
            self._fail(
                category=FailureCategory.UNSUPPORTED_NUMERICAL,
                code="LOG_PRICE_OUTSIDE_GUARD",
                sequence_index=sequence_index,
                candle_index=candle_index,
                field=field,
                observed_value=value,
                lower_bound=guard_low,
                upper_bound=guard_high,
                message="log-price recursion left the configured scientific guard",
            )
        finfo = np.finfo(dtype)
        smallest = np.nextafter(dtype.type(0.0), dtype.type(1.0), dtype=dtype)
        dtype_low = float(np.log(smallest))
        dtype_high = float(np.log(finfo.max))
        if value < dtype_low or value >= dtype_high:
            self._fail(
                category=FailureCategory.UNSUPPORTED_NUMERICAL,
                code="LOG_PRICE_OUTSIDE_DTYPE_RANGE",
                sequence_index=sequence_index,
                candle_index=candle_index,
                field=field,
                observed_value=value,
                lower_bound=dtype_low,
                upper_bound=dtype_high,
                message=f"log price cannot exponentiate to a nonzero finite {dtype.name}",
            )

    @staticmethod
    def _constructed_row_valid(row: NDArray[np.floating], feature_count: int) -> bool:
        open_value, high_value, low_value, close_value = (float(row[index]) for index in range(4))
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (open_value, high_value, low_value, close_value)
        ):
            return False
        if high_value < open_value or high_value < close_value or high_value < low_value:
            return False
        if low_value > open_value or low_value > close_value:
            return False
        if feature_count == 5:
            volume = float(row[4])
            if not math.isnan(volume) and (not math.isfinite(volume) or volume < 0.0):
                return False
        return True
