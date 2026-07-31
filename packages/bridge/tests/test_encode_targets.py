from __future__ import annotations

import math

import numpy as np
import pytest
from openalpha_bridge import (
    BridgeFinancialTransform,
    BridgeRepresentationConfig,
    BridgeTransformError,
    FailureCategory,
    VolumeMode,
)


def _required_transform(*, output_dtype: str = "float32") -> BridgeFinancialTransform:
    return BridgeFinancialTransform(
        BridgeRepresentationConfig(
            volume_mode=VolumeMode.VOLUME_REQUIRED,
            output_dtype=output_dtype,  # type: ignore[arg-type]
        )
    )


def test_encode_single_sequence_uses_exact_log_difference_formulas() -> None:
    candles = np.array(
        [
            [102.0, 108.0, 99.0, 105.0, 10.0],
            [104.0, 107.0, 100.0, 101.0, 0.0],
        ],
        dtype=np.float64,
    )

    encoded = _required_transform().encode_targets(
        candles=candles,
        initial_previous_close=100.0,
    )

    expected = np.array(
        [
            [
                math.log(102.0) - math.log(100.0),
                math.log(105.0) - math.log(102.0),
                math.log(108.0) - math.log(105.0),
                math.log(102.0) - math.log(99.0),
                math.log1p(10.0),
            ],
            [
                math.log(104.0) - math.log(105.0),
                math.log(101.0) - math.log(104.0),
                math.log(107.0) - math.log(104.0),
                math.log(101.0) - math.log(100.0),
                0.0,
            ],
        ]
    )
    np.testing.assert_array_equal(encoded.values, expected)
    np.testing.assert_array_equal(encoded.volume_present, np.array([True, True]))
    assert encoded.values.shape == (2, 5)
    assert encoded.single_sequence is True
    assert encoded.values.dtype == np.float64
    assert encoded.values.flags.writeable is False
    assert encoded.volume_present is not None
    assert encoded.volume_present.flags.writeable is False
    assert encoded.representation_version == "openalpha.bridge.financial.v1"
    assert encoded.configuration_sha256 == _required_transform().config.canonical_sha256
    assert encoded.numerical_warnings == ()


def test_encode_batch_keeps_each_sequence_anchor_and_history_separate() -> None:
    candles = np.array(
        [
            [[101.0, 102.0, 99.0, 100.0, 1.0], [110.0, 111.0, 108.0, 109.0, 2.0]],
            [[202.0, 205.0, 198.0, 200.0, 3.0], [190.0, 201.0, 189.0, 199.0, 4.0]],
        ],
        dtype=np.float64,
    )
    anchors = np.array([100.0, 200.0], dtype=np.float64)

    encoded = _required_transform().encode_targets(
        candles=candles,
        initial_previous_close=anchors,
    )

    assert encoded.values.shape == (2, 2, 5)
    assert encoded.single_sequence is False
    assert encoded.values[0, 0, 0] == pytest.approx(math.log(101.0 / 100.0))
    assert encoded.values[0, 1, 0] == pytest.approx(math.log(110.0 / 100.0))
    assert encoded.values[1, 0, 0] == pytest.approx(math.log(202.0 / 200.0))
    assert encoded.values[1, 1, 0] == pytest.approx(math.log(190.0 / 200.0))


@pytest.mark.parametrize(
    ("row", "field", "code"),
    [
        ([math.nan, 2.0, 0.5, 1.0, 1.0], "open", "NONFINITE_OPEN"),
        ([1.0, math.inf, 0.5, 1.0, 1.0], "high", "NONFINITE_HIGH"),
        ([1.0, 2.0, -math.inf, 1.0, 1.0], "low", "NONFINITE_LOW"),
        ([1.0, 2.0, 0.5, 0.0, 1.0], "close", "NONPOSITIVE_CLOSE"),
        ([0.0, 2.0, 0.5, 1.0, 1.0], "open", "NONPOSITIVE_OPEN"),
        ([-1.0, 2.0, 0.5, 1.0, 1.0], "open", "NONPOSITIVE_OPEN"),
        ([1.0, 0.9, 0.5, 1.0, 1.0], "high", "HIGH_BELOW_OPEN"),
        ([1.0, 1.5, 0.5, 2.0, 1.0], "high", "HIGH_BELOW_CLOSE"),
        ([1.0, 2.0, 1.1, 1.0, 1.0], "low", "LOW_ABOVE_OPEN"),
        ([2.0, 2.0, 1.5, 1.0, 1.0], "low", "LOW_ABOVE_CLOSE"),
        ([1.0, 2.0, 0.5, 1.0, -1.0], "volume", "NEGATIVE_VOLUME"),
        ([1.0, 2.0, 0.5, 1.0, math.inf], "volume", "NONFINITE_VOLUME"),
    ],
)
def test_invalid_source_candles_fail_before_target_construction(
    row: list[float],
    field: str,
    code: str,
) -> None:
    with pytest.raises(BridgeTransformError) as caught:
        _required_transform().encode_targets(
            candles=np.array([row]),
            initial_previous_close=1.0,
        )

    failure = caught.value.failures[0]
    assert failure.category is FailureCategory.INVALID_INPUT
    assert failure.code == code
    assert failure.sequence_index == 0
    assert failure.candle_index == 0
    assert failure.field == field
    assert failure.observed_value is not None


@pytest.mark.parametrize(
    ("row", "anchor", "field", "code", "upper"),
    [
        ([5.0, 5.0, 5.0, 5.0, 1.0], 1.0, "gap", "GAP_ABOVE_LIMIT", math.log(4.0)),
        ([1.0, 5.0, 1.0, 5.0, 1.0], 1.0, "body", "BODY_ABOVE_LIMIT", math.log(4.0)),
        ([1.0, 5.0, 1.0, 1.0, 1.0], 1.0, "upper", "UPPER_ABOVE_LIMIT", math.log(4.0)),
        ([1.0, 1.0, 0.19, 1.0, 1.0], 1.0, "lower", "LOWER_ABOVE_LIMIT", math.log(4.0)),
        ([1.0, 1.0, 1.0, 1.0, 1.0e15 + 1.0], 1.0, "volume", "VOLUME_ABOVE_LIMIT", 1.0e15),
        (
            [1.0e12 + 1.0, 1.0e12 + 1.0, 1.0e12 + 1.0, 1.0e12 + 1.0, 1.0],
            1.0e12,
            "open",
            "PRICE_ABOVE_LIMIT",
            1.0e12,
        ),
    ],
)
def test_valid_but_out_of_domain_source_reports_exact_location_and_bound(
    row: list[float],
    anchor: float,
    field: str,
    code: str,
    upper: float,
) -> None:
    with pytest.raises(BridgeTransformError) as caught:
        _required_transform().encode_targets(
            candles=np.array([row], dtype=np.float64),
            initial_previous_close=anchor,
        )

    failure = caught.value.failures[0]
    assert failure.category is FailureCategory.OUT_OF_DOMAIN
    assert failure.code == code
    assert failure.sequence_index == 0
    assert failure.candle_index == 0
    assert failure.field == field
    assert failure.observed_value is not None
    assert failure.upper_bound == upper


def test_out_of_domain_initial_anchor_is_not_clipped() -> None:
    candles = np.array([[1.0, 1.0, 1.0, 1.0, 0.0]])
    with pytest.raises(BridgeTransformError) as caught:
        _required_transform().encode_targets(
            candles=candles,
            initial_previous_close=1.0e-13,
        )
    failure = caught.value.failures[0]
    assert failure.category is FailureCategory.OUT_OF_DOMAIN
    assert failure.code == "PREVIOUS_CLOSE_BELOW_LIMIT"
    assert failure.field == "initial_previous_close"
    assert failure.observed_value == 1.0e-13
    assert failure.lower_bound == 1.0e-12


def test_target_value_one_float_above_exact_gap_cap_fails_without_tolerance_clipping() -> None:
    transform = _required_transform(output_dtype="float64")
    limit = transform.config.gap_return_limit
    just_outside = np.nextafter(limit, math.inf)
    price = math.exp(just_outside)
    candle = np.array([[price, price, price, price, 0.0]], dtype=np.float64)

    with pytest.raises(BridgeTransformError, match="GAP_ABOVE_LIMIT"):
        transform.encode_targets(candles=candle, initial_previous_close=1.0)


def test_required_optional_and_price_only_volume_are_distinct() -> None:
    required = _required_transform()
    with pytest.raises(BridgeTransformError, match="FEATURE_COUNT_MISMATCH"):
        required.encode_targets(
            candles=np.array([[1.0, 1.0, 1.0, 1.0]]),
            initial_previous_close=1.0,
        )

    optional = BridgeFinancialTransform(BridgeRepresentationConfig())
    candles = np.array(
        [[1.0, 1.0, 1.0, 1.0, math.nan], [1.0, 1.0, 1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    encoded = optional.encode_targets(
        candles=candles,
        initial_previous_close=1.0,
        volume_present=np.array([False, True]),
    )
    assert math.isnan(float(encoded.values[0, 4]))
    assert encoded.values[1, 4] == 0.0
    np.testing.assert_array_equal(encoded.volume_present, [False, True])

    price_only = BridgeFinancialTransform(
        BridgeRepresentationConfig(
            volume_mode=VolumeMode.PRICE_ONLY,
            supported_feature_order=("open", "high", "low", "close"),
        )
    )
    price_encoded = price_only.encode_targets(
        candles=np.array([[1.0, 1.0, 1.0, 1.0]]),
        initial_previous_close=1.0,
    )
    assert price_encoded.values.shape == (1, 4)
    assert price_encoded.volume_present is None


def test_optional_volume_requires_an_exact_boolean_presence_mask_and_nan_missing_values() -> None:
    transform = BridgeFinancialTransform(BridgeRepresentationConfig())
    candles = np.array([[1.0, 1.0, 1.0, 1.0, math.nan]])

    with pytest.raises(BridgeTransformError, match="VOLUME_MASK_REQUIRED"):
        transform.encode_targets(candles=candles, initial_previous_close=1.0)
    with pytest.raises(BridgeTransformError, match="VOLUME_MASK_DTYPE"):
        transform.encode_targets(
            candles=candles,
            initial_previous_close=1.0,
            volume_present=np.array([0]),
        )
    with pytest.raises(BridgeTransformError, match="VOLUME_MASK_SHAPE"):
        transform.encode_targets(
            candles=candles,
            initial_previous_close=1.0,
            volume_present=np.array([[False]]),
        )
    with pytest.raises(BridgeTransformError, match="MISSING_VOLUME_MUST_USE_NAN"):
        transform.encode_targets(
            candles=np.array([[1.0, 1.0, 1.0, 1.0, 0.0]]),
            initial_previous_close=1.0,
            volume_present=np.array([False]),
        )
    with pytest.raises(BridgeTransformError, match="PRESENT_VOLUME_NONFINITE"):
        transform.encode_targets(
            candles=candles,
            initial_previous_close=1.0,
            volume_present=np.array([True]),
        )
