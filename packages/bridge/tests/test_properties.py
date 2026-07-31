from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from openalpha_bridge import (
    BridgeFinancialTransform,
    BridgeRepresentationConfig,
    VolumeMode,
)

FINITE_HEAD = st.floats(
    min_value=-1.0e4,
    max_value=1.0e4,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=True,
    width=32,
)
SAFE_ANCHOR = st.floats(
    min_value=1.0e-6,
    max_value=1.0e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)


def _assert_grammar(candles: np.ndarray) -> None:
    assert np.isfinite(candles).all()
    assert (candles[..., :4] > 0.0).all()
    assert (candles[..., 1] >= candles[..., 0]).all()
    assert (candles[..., 1] >= candles[..., 3]).all()
    assert (candles[..., 1] >= candles[..., 2]).all()
    assert (candles[..., 2] <= candles[..., 0]).all()
    assert (candles[..., 2] <= candles[..., 3]).all()
    assert (candles[..., 4] >= 0.0).all()


@settings(
    max_examples=10_000,
    derandomize=True,
    database=None,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)
@given(
    raw=st.tuples(FINITE_HEAD, FINITE_HEAD, FINITE_HEAD, FINITE_HEAD, FINITE_HEAD),
    anchor=SAFE_ANCHOR,
)
def test_ten_thousand_finite_head_outputs_emit_only_valid_candles(
    raw: tuple[float, float, float, float, float],
    anchor: float,
) -> None:
    transform = BridgeFinancialTransform(
        BridgeRepresentationConfig(
            volume_mode=VolumeMode.VOLUME_REQUIRED,
            output_dtype="float32",
        )
    )
    result = transform.decode_head_outputs(
        raw_outputs=np.asarray([raw], dtype=np.float32),
        initial_previous_close=anchor,
    )

    _assert_grammar(result.candles)
    assert transform.audit(result, expected_sequence_length=1).valid is True


@settings(max_examples=250, derandomize=True, database=None, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    time_steps=st.integers(min_value=1, max_value=8),
    raw_value=st.floats(
        min_value=-5.0,
        max_value=5.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)
def test_generated_batches_keep_sequences_independent_and_valid(
    batch_size: int,
    time_steps: int,
    raw_value: float,
) -> None:
    transform = BridgeFinancialTransform(
        BridgeRepresentationConfig(
            volume_mode=VolumeMode.VOLUME_REQUIRED,
            maximum_decode_steps=8,
        )
    )
    raw = np.full((batch_size, time_steps, 5), raw_value, dtype=np.float32)
    anchors = np.geomspace(1.0, 1000.0, batch_size, dtype=np.float64)
    result = transform.decode_head_outputs(
        raw_outputs=raw,
        initial_previous_close=anchors,
    )

    _assert_grammar(result.candles)
    assert transform.audit(result, expected_sequence_length=time_steps).valid is True
    if batch_size > 1 and raw_value == 0.0:
        assert not math.isclose(
            float(result.candles[0, 0, 0]),
            float(result.candles[-1, 0, 0]),
        )


@settings(max_examples=250, derandomize=True, database=None, deadline=None)
@given(
    time_steps=st.integers(min_value=1, max_value=8),
    missing=st.booleans(),
    body=st.floats(
        min_value=-0.1,
        max_value=0.1,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)
def test_optional_volume_properties_never_conflate_missing_with_zero(
    time_steps: int,
    missing: bool,
    body: float,
) -> None:
    transform = BridgeFinancialTransform(BridgeRepresentationConfig(maximum_decode_steps=8))
    features = np.zeros((time_steps, 5), dtype=np.float64)
    features[:, 1] = body
    present = np.full(time_steps, not missing, dtype=bool)
    if missing:
        features[:, 4] = math.nan
    result = transform.decode_features(
        transformed_features=features,
        initial_previous_close=100.0,
        volume_present=present,
    )

    assert transform.audit(result, expected_sequence_length=time_steps).valid is True
    if missing:
        assert np.isnan(result.candles[:, 4]).all()
    else:
        np.testing.assert_array_equal(result.candles[:, 4], 0.0)
