from __future__ import annotations

import math

import numpy as np
import pytest
from openalpha_bridge import (
    BridgeFinancialTransform,
    BridgeRepresentationConfig,
    BridgeTransformError,
    VolumeMode,
)


def _transform(mode: VolumeMode) -> BridgeFinancialTransform:
    order = (
        ("open", "high", "low", "close")
        if mode is VolumeMode.PRICE_ONLY
        else ("open", "high", "low", "close", "volume")
    )
    return BridgeFinancialTransform(
        BridgeRepresentationConfig(volume_mode=mode, supported_feature_order=order)
    )


@pytest.mark.parametrize(
    "shape",
    [
        (5,),
        (1, 1, 1, 5),
        (2, 3, 4),
        (2, 3, 6),
    ],
)
def test_raw_head_output_dimensions_are_unambiguous(shape: tuple[int, ...]) -> None:
    with pytest.raises(BridgeTransformError):
        _transform(VolumeMode.VOLUME_REQUIRED).decode_head_outputs(
            raw_outputs=np.zeros(shape),
            initial_previous_close=1.0 if len(shape) == 2 else np.ones(2),
        )


def test_batch_previous_close_never_broadcasts_scalar_or_wrong_length() -> None:
    raw = np.zeros((2, 3, 5))
    transform = _transform(VolumeMode.VOLUME_REQUIRED)
    with pytest.raises(BridgeTransformError, match="PREVIOUS_CLOSE_SHAPE"):
        transform.decode_head_outputs(raw_outputs=raw, initial_previous_close=100.0)
    with pytest.raises(BridgeTransformError, match="PREVIOUS_CLOSE_SHAPE"):
        transform.decode_head_outputs(
            raw_outputs=raw,
            initial_previous_close=np.array([100.0]),
        )
    with pytest.raises(BridgeTransformError, match="PREVIOUS_CLOSE_SHAPE"):
        transform.decode_head_outputs(
            raw_outputs=raw,
            initial_previous_close=np.array([[100.0], [200.0]]),
        )


def test_single_sequence_rejects_multiple_previous_closes() -> None:
    with pytest.raises(BridgeTransformError, match="PREVIOUS_CLOSE_SHAPE"):
        _transform(VolumeMode.VOLUME_REQUIRED).decode_head_outputs(
            raw_outputs=np.zeros((2, 5)),
            initial_previous_close=np.array([100.0, 200.0]),
        )


def test_volume_required_produces_volume_without_a_mask() -> None:
    result = _transform(VolumeMode.VOLUME_REQUIRED).decode_head_outputs(
        raw_outputs=np.zeros((2, 5)),
        initial_previous_close=100.0,
    )
    assert result.candles.shape == (2, 5)
    assert result.volume_present is not None
    np.testing.assert_array_equal(result.volume_present, [True, True])
    assert (result.candles[:, 4] > 0.0).all()


def test_volume_optional_preserves_missing_and_zero_as_distinct_states() -> None:
    transform = _transform(VolumeMode.VOLUME_OPTIONAL)
    raw = np.zeros((3, 5))
    present = np.array([False, True, True])
    result = transform.decode_head_outputs(
        raw_outputs=raw,
        initial_previous_close=100.0,
        volume_present=present,
    )

    assert math.isnan(float(result.candles[0, 4]))
    assert result.candles[1, 4] > 0.0
    assert result.candles[2, 4] > 0.0
    np.testing.assert_array_equal(result.volume_present, present)

    exact_zero_features = np.zeros((3, 5))
    exact_zero_features[0, 4] = math.nan
    exact_zero = transform.decode_features(
        transformed_features=exact_zero_features,
        initial_previous_close=100.0,
        volume_present=present,
    )
    assert math.isnan(float(exact_zero.candles[0, 4]))
    assert exact_zero.candles[1, 4] == 0.0
    assert exact_zero.volume_present is not None
    assert bool(exact_zero.volume_present[1]) is True


def test_optional_volume_mask_is_required_and_never_broadcasts() -> None:
    transform = _transform(VolumeMode.VOLUME_OPTIONAL)
    raw = np.zeros((2, 3, 5))
    for mask in (None, np.array([True, False, True]), np.ones((2, 1), dtype=bool)):
        with pytest.raises(BridgeTransformError):
            transform.decode_head_outputs(
                raw_outputs=raw,
                initial_previous_close=np.array([100.0, 200.0]),
                volume_present=mask,
            )


def test_price_only_omits_volume_and_labels_ignored_raw_channel() -> None:
    result = _transform(VolumeMode.PRICE_ONLY).decode_head_outputs(
        raw_outputs=np.zeros((2, 5)),
        initial_previous_close=100.0,
    )
    assert result.candles.shape == (2, 4)
    assert result.transformed_features.values.shape == (2, 4)
    assert result.volume_present is None
    assert result.numerical_warnings == ("PRICE_ONLY_RAW_VOLUME_CHANNEL_IGNORED",)


def test_price_only_and_required_reject_unexpected_volume_masks() -> None:
    mask = np.array([True])
    for mode in (VolumeMode.PRICE_ONLY, VolumeMode.VOLUME_REQUIRED):
        with pytest.raises(BridgeTransformError, match="VOLUME_MASK_UNEXPECTED"):
            _transform(mode).decode_head_outputs(
                raw_outputs=np.zeros((1, 5)),
                initial_previous_close=100.0,
                volume_present=mask,
            )


def test_transformed_feature_shapes_are_exact() -> None:
    transform = _transform(VolumeMode.VOLUME_REQUIRED)
    with pytest.raises(BridgeTransformError, match="FEATURE_COUNT_MISMATCH"):
        transform.decode_features(
            transformed_features=np.zeros((2, 4)),
            initial_previous_close=100.0,
        )
    with pytest.raises(BridgeTransformError, match="CANDLE_RANK_MISMATCH"):
        transform.decode_features(
            transformed_features=np.zeros((5,)),
            initial_previous_close=100.0,
        )
