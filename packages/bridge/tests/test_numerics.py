from __future__ import annotations

import math

import numpy as np
import pytest
from openalpha_bridge import (
    BridgeFinancialTransform,
    BridgeRepresentationConfig,
    BridgeTransformError,
    VolumeMode,
    bridge_canonical_bytes,
    numerical_roundtrip_audit,
    roundtrip_tolerance,
)


def _config(dtype: str, length: int) -> BridgeRepresentationConfig:
    return BridgeRepresentationConfig(
        volume_mode=VolumeMode.VOLUME_REQUIRED,
        output_dtype=dtype,  # type: ignore[arg-type]
        maximum_decode_steps=length,
    )


def _supported_features(length: int) -> np.ndarray:
    index = np.arange(length, dtype=np.float64)
    features = np.zeros((length, 5), dtype=np.float64)
    features[:, 0] = 0.001 * np.sin(index * 0.17)
    features[:, 1] = 0.002 * np.cos(index * 0.11)
    features[:, 2] = 0.003 * (1.0 + np.sin(index * 0.07))
    features[:, 3] = 0.003 * (1.0 + np.cos(index * 0.05))
    features[:, 4] = np.log1p(1000.0 + index)
    return features


@pytest.mark.parametrize("length", [1, 5, 128, 512, 2048])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_supported_long_sequence_roundtrip_meets_locked_dtype_tolerances(
    length: int,
    dtype: str,
) -> None:
    transform = BridgeFinancialTransform(_config(dtype, length))
    reference_transform = BridgeFinancialTransform(_config("float64", length))
    source = reference_transform.decode_features(
        transformed_features=_supported_features(length),
        initial_previous_close=100.0,
    )
    targets = transform.encode_targets(
        candles=source.candles,
        initial_previous_close=100.0,
    )
    reconstructed = transform.decode_features(
        transformed_features=targets,
        initial_previous_close=100.0,
    )

    audit = numerical_roundtrip_audit(
        reference=source.candles,
        reconstructed=reconstructed.candles,
        volume_present=reconstructed.volume_present,
    )

    assert audit.dtype == dtype
    assert audit.sequence_length == length
    assert audit.passes is True
    assert audit.maximum_absolute_error >= 0.0
    assert audit.maximum_relative_error <= roundtrip_tolerance(dtype, "close").relative
    assert audit.maximum_log_error <= roundtrip_tolerance(dtype, "close").log_space
    assert (
        audit.maximum_recursive_close_drift
        <= roundtrip_tolerance(dtype, "close").recursive_relative
    )
    assert b'"schema_version":"openalpha.bridge.numerical-audit.v1"' in bridge_canonical_bytes(
        audit
    )


def test_locked_tolerances_are_dtype_and_field_specific() -> None:
    reference_price = roundtrip_tolerance("float64", "open")
    runtime_price = roundtrip_tolerance("float32", "open")
    runtime_volume = roundtrip_tolerance("float32", "volume")

    assert reference_price.absolute == 1.0e-12
    assert reference_price.relative == 1.0e-12
    assert reference_price.log_space == 1.0e-12
    assert runtime_price.absolute == 1.0e-6
    assert runtime_price.relative == 2.0e-5
    assert runtime_price.log_space == 1.0e-4
    assert runtime_price.recursive_relative == 1.0e-4
    assert runtime_volume.absolute == 1.0e-3
    assert runtime_volume.relative == 5.0e-5


def test_float32_precision_loss_is_measured_not_hidden() -> None:
    transform = BridgeFinancialTransform(_config("float32", 5))
    features = _supported_features(5)
    reference = BridgeFinancialTransform(_config("float64", 5)).decode_features(
        transformed_features=features,
        initial_previous_close=100.0,
    )
    runtime = transform.decode_features(
        transformed_features=features,
        initial_previous_close=100.0,
    )
    audit = numerical_roundtrip_audit(
        reference=reference.candles,
        reconstructed=runtime.candles,
        volume_present=runtime.volume_present,
    )
    assert audit.maximum_absolute_error > 0.0
    assert audit.passes is True


@pytest.mark.parametrize(
    ("channel", "outside"),
    [
        (0, math.inf),
        (0, -math.inf),
        (1, math.inf),
        (2, -math.inf),
        (3, math.inf),
        (4, 710.0),
    ],
)
def test_unsafe_exponential_and_expm1_attempts_fail_before_output(
    channel: int,
    outside: float,
) -> None:
    transform = BridgeFinancialTransform(_config("float64", 1))
    features = np.zeros((1, 5), dtype=np.float64)
    features[0, channel] = outside
    with pytest.raises(BridgeTransformError):
        transform.decode_features(
            transformed_features=features,
            initial_previous_close=100.0,
        )


def test_values_exactly_at_caps_are_supported_and_next_float_outside_fails() -> None:
    config = _config("float64", 1)
    transform = BridgeFinancialTransform(config)
    at_caps = np.array(
        [
            [
                config.gap_return_limit,
                -config.body_return_limit,
                config.upper_wick_limit,
                config.lower_wick_limit,
                config.volume_limit,
            ]
        ]
    )
    result = transform.decode_features(
        transformed_features=at_caps,
        initial_previous_close=100.0,
    )
    assert np.isfinite(result.candles).all()
    assert result.candles[0, 4] == pytest.approx(1.0e15, rel=1.0e-12)

    for channel, limit in enumerate(
        (
            config.gap_return_limit,
            config.body_return_limit,
            config.upper_wick_limit,
            config.lower_wick_limit,
            config.volume_limit,
        )
    ):
        outside = np.zeros((1, 5), dtype=np.float64)
        outside[0, channel] = np.nextafter(limit, math.inf)
        with pytest.raises(BridgeTransformError):
            transform.decode_features(
                transformed_features=outside,
                initial_previous_close=100.0,
            )


def test_recursive_float32_guard_fails_explicitly_near_dtype_boundary() -> None:
    config = _config("float32", 64)
    features = np.zeros((64, 5), dtype=np.float64)
    features[:, :2] = config.gap_return_limit
    with pytest.raises(BridgeTransformError, match="LOG_PRICE_OUTSIDE_DTYPE_RANGE"):
        BridgeFinancialTransform(config).decode_features(
            transformed_features=features,
            initial_previous_close=config.maximum_price,
        )
