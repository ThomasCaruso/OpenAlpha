from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from openalpha_bridge import (
    BridgeFinancialTransform,
    BridgeRepresentationConfig,
    BridgeTransformError,
    VolumeMode,
    bounded_nonnegative,
    bounded_signed,
    inverse_bounded_nonnegative,
    inverse_bounded_signed,
    stable_softplus,
)


def _config(*, dtype: str = "float64", steps: int = 64) -> BridgeRepresentationConfig:
    return BridgeRepresentationConfig(
        output_dtype=dtype,  # type: ignore[arg-type]
        volume_mode=VolumeMode.VOLUME_REQUIRED,
        maximum_decode_steps=steps,
    )


def _assert_financial_grammar(candles: np.ndarray) -> None:
    assert np.isfinite(candles).all()
    assert (candles[..., :4] > 0.0).all()
    assert (candles[..., 1] >= candles[..., 0]).all()
    assert (candles[..., 1] >= candles[..., 3]).all()
    assert (candles[..., 1] >= candles[..., 2]).all()
    assert (candles[..., 2] <= candles[..., 0]).all()
    assert (candles[..., 2] <= candles[..., 3]).all()
    assert (candles[..., 4] >= 0.0).all()


def test_versioned_activation_formulas_and_inverses() -> None:
    signed_limit = math.log(4.0)
    nonnegative_limit = math.log(4.0)
    raw = np.array([-3.0, -0.25, 0.0, 0.25, 3.0], dtype=np.float64)

    np.testing.assert_array_equal(stable_softplus(raw), np.logaddexp(0.0, raw))
    signed = bounded_signed(raw, signed_limit)
    np.testing.assert_array_equal(signed, signed_limit * np.tanh(raw))
    np.testing.assert_allclose(inverse_bounded_signed(signed, signed_limit), raw, rtol=1e-13)

    softplus = stable_softplus(raw)
    nonnegative = bounded_nonnegative(raw, nonnegative_limit)
    np.testing.assert_allclose(
        nonnegative,
        nonnegative_limit * softplus / (nonnegative_limit + softplus),
        rtol=2.0e-16,
    )
    np.testing.assert_allclose(
        inverse_bounded_nonnegative(nonnegative, nonnegative_limit),
        raw,
        rtol=1e-12,
        atol=1e-12,
    )
    assert (nonnegative >= 0.0).all()
    assert (nonnegative < nonnegative_limit).all()


def test_activation_saturation_is_bounded_without_post_output_clipping() -> None:
    limit = math.log(4.0)
    raw = np.array([-1.0e6, -100.0, 100.0, 1.0e6], dtype=np.float64)
    signed = bounded_signed(raw, limit)
    nonnegative = bounded_nonnegative(raw, limit)

    assert signed[0] == -limit
    assert signed[-1] == limit
    assert (nonnegative >= 0.0).all()
    assert (nonnegative <= limit).all()
    assert nonnegative[0] == 0.0
    assert nonnegative[-1] < limit


def test_activation_mapping_and_inverse_are_stable_at_finite_extremes() -> None:
    limit = math.log(4.0)
    largest = np.finfo(np.float64).max
    mapped = bounded_nonnegative(np.array([largest]), limit)
    assert np.isfinite(mapped).all()
    assert 0.0 <= mapped[0] <= limit

    near_cap = np.nextafter(limit, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        raw = inverse_bounded_nonnegative(np.array([near_cap]), limit)
    assert np.isfinite(raw).all()
    np.testing.assert_allclose(bounded_nonnegative(raw, limit), near_cap, rtol=1.0e-12)


def test_decode_head_outputs_exposes_transformed_features_and_valid_candles() -> None:
    transform = BridgeFinancialTransform(_config())
    raw = np.array(
        [
            [0.1, 0.2, -1.0, -2.0, -3.0],
            [-0.1, -0.2, 1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )

    result = transform.decode_head_outputs(
        raw_outputs=raw,
        initial_previous_close=100.0,
    )

    expected = np.column_stack(
        (
            bounded_signed(raw[:, 0], transform.config.gap_return_limit),
            bounded_signed(raw[:, 1], transform.config.body_return_limit),
            bounded_nonnegative(raw[:, 2], transform.config.upper_wick_limit),
            bounded_nonnegative(raw[:, 3], transform.config.lower_wick_limit),
            bounded_nonnegative(raw[:, 4], transform.config.volume_limit),
        )
    )
    np.testing.assert_array_equal(result.transformed_features.values, expected)
    assert result.candles.shape == (2, 5)
    assert result.candles.dtype == np.float64
    assert result.candles.flags.writeable is False
    assert result.output_dtype == "float64"
    assert result.projection_applied is False
    _assert_financial_grammar(result.candles)


def test_forward_inverse_roundtrip_uses_previous_reconstructed_close() -> None:
    transform = BridgeFinancialTransform(_config())
    candles = np.array(
        [
            [102.0, 108.0, 99.0, 105.0, 10.0],
            [104.0, 107.0, 100.0, 101.0, 0.0],
            [99.0, 103.0, 98.0, 102.0, 1.0e6],
        ],
        dtype=np.float64,
    )
    encoded = transform.encode_targets(candles=candles, initial_previous_close=100.0)

    decoded = transform.decode_features(
        transformed_features=encoded,
        initial_previous_close=100.0,
    )

    np.testing.assert_allclose(decoded.candles, candles, rtol=1.0e-12, atol=1.0e-12)
    _assert_financial_grammar(decoded.candles)


def test_future_feature_perturbation_cannot_change_earlier_output() -> None:
    transform = BridgeFinancialTransform(_config())
    features = np.zeros((5, 5), dtype=np.float64)
    base = transform.decode_features(
        transformed_features=features,
        initial_previous_close=100.0,
    )
    changed = features.copy()
    changed[4, 0] = 0.2
    changed[4, 1] = -0.1
    perturbed = transform.decode_features(
        transformed_features=changed,
        initial_previous_close=100.0,
    )

    np.testing.assert_array_equal(base.candles[:4], perturbed.candles[:4])
    assert not np.array_equal(base.candles[4], perturbed.candles[4])


def test_batch_reconstruction_never_crosses_sequence_anchors() -> None:
    transform = BridgeFinancialTransform(_config())
    features = np.zeros((2, 3, 5), dtype=np.float64)
    result = transform.decode_features(
        transformed_features=features,
        initial_previous_close=np.array([10.0, 1000.0]),
    )

    np.testing.assert_allclose(result.candles[0, :, :4], 10.0, rtol=1.0e-12)
    np.testing.assert_allclose(result.candles[1, :, :4], 1000.0, rtol=1.0e-12)
    np.testing.assert_array_equal(result.candles[..., 4], 0.0)
    _assert_financial_grammar(result.candles)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_head_output_is_a_typed_no_output_failure(bad: float) -> None:
    raw = np.zeros((1, 5), dtype=np.float64)
    raw[0, 2] = bad
    with pytest.raises(BridgeTransformError) as caught:
        BridgeFinancialTransform(_config()).decode_head_outputs(
            raw_outputs=raw,
            initial_previous_close=100.0,
        )
    failure = caught.value.failures[0]
    assert failure.code == "NONFINITE_HEAD_OUTPUT"
    assert failure.sequence_index == 0
    assert failure.candle_index == 0
    assert failure.field == "upper"


def test_head_output_outside_float32_range_fails_at_exact_channel_before_cast() -> None:
    raw = np.zeros((1, 5), dtype=np.float64)
    raw[0, 3] = 1.0e100
    transform = BridgeFinancialTransform(_config(dtype="float32"))

    with pytest.raises(BridgeTransformError) as caught:
        transform.decode_head_outputs(raw_outputs=raw, initial_previous_close=100.0)

    failure = caught.value.failures[0]
    assert failure.code == "HEAD_OUTPUT_OUTSIDE_DTYPE_RANGE"
    assert failure.sequence_index == 0
    assert failure.candle_index == 0
    assert failure.field == "lower"


def test_guard_failure_emits_no_nonfinite_or_zero_candle() -> None:
    config = _config(dtype="float32", steps=64)
    transform = BridgeFinancialTransform(config)
    features = np.zeros((64, 5), dtype=np.float64)
    features[:, 0] = config.gap_return_limit
    features[:, 1] = config.body_return_limit

    with pytest.raises(BridgeTransformError) as caught:
        transform.decode_features(
            transformed_features=features,
            initial_previous_close=config.maximum_price,
        )
    assert caught.value.failures[0].code in {
        "LOG_PRICE_OUTSIDE_DTYPE_RANGE",
        "LOG_PRICE_OUTSIDE_GUARD",
    }


def test_decode_features_rejects_values_immediately_outside_channel_caps() -> None:
    config = _config()
    transform = BridgeFinancialTransform(config)
    features = np.zeros((1, 5), dtype=np.float64)
    features[0, 0] = np.nextafter(config.gap_return_limit, math.inf)

    with pytest.raises(BridgeTransformError, match="GAP_ABOVE_LIMIT"):
        transform.decode_features(
            transformed_features=features,
            initial_previous_close=100.0,
        )
