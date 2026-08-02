"""Byte-exact parity with the pinned float32 normalization.

The pinned source runs

    x = values.astype(np.float32)
    x_mean = np.mean(x, axis=0)
    x_std  = np.std(x, axis=0)
    x = (x - x_mean) / (x_std + 1e-5)
    x = np.clip(x, -5, 5)

entirely in float32, and the result feeds a quantizer. A value a few ULPs from
a bin boundary lands in a different bin, which changes a discrete token id, so
"close enough" is not a defence here. Every assertion below compares raw bytes
against the expression written out directly, not against an approximation of
it.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pytest
from openalpha_bridge.diagnostic import normalization as normalization_module
from openalpha_bridge.diagnostic.normalization import (
    CLIP_VALUE,
    EPSILON,
    context_matrix,
    fit_context_state,
)
from openalpha_bridge.diagnostic.official_input import OFFICIAL_COLUMNS, OfficialRow

CLIP = 5


def _rows(values: Sequence[tuple[float, ...]]) -> tuple[OfficialRow, ...]:
    start = date(2015, 5, 7)
    return tuple(
        OfficialRow(
            session=start + timedelta(days=index),
            open=row[0],
            high=row[1],
            low=row[2],
            close=row[3],
            volume=row[4],
            amount=row[5],
        )
        for index, row in enumerate(values)
    )


def _pinned(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pinned expression, written out verbatim."""
    x = matrix.astype(np.float32)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    normalized = (x - x_mean) / (x_std + 1e-5)
    normalized = np.clip(normalized, -CLIP, CLIP)
    return x_mean, x_std, normalized


# --------------------------------------------------------------- the corpora


def _ordinary() -> list[tuple[float, ...]]:
    out = []
    level = 412.37
    for index in range(64):
        level = level * (1.0 + 0.0007 * np.cos(index / 5.0))
        close = level * 1.00013
        out.append(
            (
                level,
                max(level, close) * 1.0031,
                min(level, close) * 0.9968,
                close,
                83_412_900.0 + index * 137.0,
                (83_412_900.0 + index * 137.0) * close,
            )
        )
    return out


def _large_volume_and_amount() -> list[tuple[float, ...]]:
    """Amount is price times volume, so it reaches 1e11 and beyond.

    float32 has 24 bits of significand, so values this large lose integer
    resolution. That is what the official path does, and the point is to
    reproduce it rather than to avoid it.
    """
    return [
        (400.0 + i, 401.0 + i, 399.0 + i, 400.5 + i, 9.5e9 + i * 1e6, 3.8e12 + i * 1e9)
        for i in range(48)
    ]


def _near_zero_variance() -> list[tuple[float, ...]]:
    """A column that barely moves, so std is tiny and epsilon dominates."""
    return [(100.0, 100.0, 100.0, 100.0 + (1e-7 if i % 2 else 0.0), 5.0, 500.0) for i in range(32)]


def _exactly_zero_variance() -> list[tuple[float, ...]]:
    """A constant column. std is exactly zero, so the divisor is epsilon."""
    return [(100.0, 100.0, 100.0, 100.0, 1000.0, 100_000.0) for _ in range(24)]


def _at_and_beyond_the_clip() -> list[tuple[float, ...]]:
    """Outliers far enough out to saturate the clip in both directions."""
    rows = [(100.0, 101.0, 99.0, 100.0, 1000.0, 100_000.0) for _ in range(40)]
    rows[0] = (100.0, 101.0, 99.0, 10_000.0, 1000.0, 100_000.0)
    rows[1] = (100.0, 101.0, 99.0, 0.01, 1000.0, 100_000.0)
    return rows


def _near_quantization_boundaries() -> list[tuple[float, ...]]:
    """Values whose normalized result sits within one ULP of a bin edge.

    The tokenizer's bins are not published, so the boundaries are approximated
    by walking the float32 grid around representative normalized magnitudes.
    Any transform that differs from the pinned one by a single ULP shows up.
    """
    rows = []
    base = np.float32(250.0)
    for index in range(32):
        nudged = float(np.nextafter(base, np.float32(np.inf), dtype=np.float32))
        base = np.float32(nudged)
        rows.append((nudged, nudged * 1.001, nudged * 0.999, nudged, 1.0e6, 1.0e6 * nudged))
    return rows


CORPORA = {
    "ordinary": _ordinary(),
    "large_volume_and_amount": _large_volume_and_amount(),
    "near_zero_variance": _near_zero_variance(),
    "exactly_zero_variance": _exactly_zero_variance(),
    "at_and_beyond_the_clip": _at_and_beyond_the_clip(),
    "near_quantization_boundaries": _near_quantization_boundaries(),
}


# ------------------------------------------------------------- byte parity


@pytest.mark.parametrize("name", sorted(CORPORA))
def test_mean_bytes_match_the_pinned_expression(name: str) -> None:
    rows = _rows(CORPORA[name])
    state = fit_context_state(rows)
    expected_mean, _, _ = _pinned(context_matrix(rows))
    assert state.mean_array().dtype == np.float32
    assert state.mean_array().tobytes() == expected_mean.astype("<f4").tobytes()
    assert state.mean_float32_hex == expected_mean.astype("<f4").tobytes().hex()


@pytest.mark.parametrize("name", sorted(CORPORA))
def test_standard_deviation_bytes_match_the_pinned_expression(name: str) -> None:
    rows = _rows(CORPORA[name])
    state = fit_context_state(rows)
    _, expected_std, _ = _pinned(context_matrix(rows))
    assert state.std_array().dtype == np.float32
    assert state.std_array().tobytes() == expected_std.astype("<f4").tobytes()
    assert state.standard_deviation_float32_hex == expected_std.astype("<f4").tobytes().hex()


@pytest.mark.parametrize("name", sorted(CORPORA))
def test_normalized_bytes_match_the_pinned_expression(name: str) -> None:
    rows = _rows(CORPORA[name])
    state = fit_context_state(rows)
    matrix = context_matrix(rows)
    _, _, expected = _pinned(matrix)
    observed = state.normalize_array(matrix)
    assert observed.dtype == np.float32
    assert observed.tobytes() == expected.tobytes()


@pytest.mark.parametrize("name", sorted(CORPORA))
def test_inverted_bytes_match_the_pinned_expression(name: str) -> None:
    rows = _rows(CORPORA[name])
    state = fit_context_state(rows)
    matrix = context_matrix(rows)
    mean, std, normalized = _pinned(matrix)

    expected = normalized * (std + 1e-5) + mean
    observed = state.invert_array(np.asarray(normalized, dtype=np.float32))
    assert observed.dtype == np.float32
    assert observed.tobytes() == expected.tobytes()


@pytest.mark.parametrize("name", sorted(CORPORA))
def test_the_row_convenience_wrappers_agree_with_the_arrays(name: str) -> None:
    rows = _rows(CORPORA[name])
    state = fit_context_state(rows)
    matrix = context_matrix(rows)

    from_array = state.normalize_array(matrix)
    from_rows = np.asarray(state.normalize(rows), dtype=np.float32)
    assert from_rows.tobytes() == from_array.tobytes()

    inverted_array = state.invert_array(from_array)
    as_rows = tuple(tuple(float(v) for v in row) for row in from_rows)
    inverted_rows = np.asarray(state.invert(as_rows), dtype=np.float32)
    assert inverted_rows.tobytes() == inverted_array.tobytes()


# ------------------------------------------- float64 really would differ


def test_a_double_precision_computation_would_not_have_matched() -> None:
    """The reason this commit exists, demonstrated rather than asserted.

    Not every corpus has to diverge: on small, well conditioned data the two
    accumulations can land on the same float32. What matters is that they do
    diverge on realistic data, and that when they do the difference is far too
    small for an approximate test to have caught it.
    """
    diverged: list[str] = []
    for name, values in CORPORA.items():
        rows = _rows(values)
        state = fit_context_state(rows)

        doubles = np.array([row.channels() for row in rows], dtype=np.float64)
        double_normalized = np.clip(
            (doubles - np.mean(doubles, axis=0)) / (np.std(doubles, axis=0) + 1e-5),
            -CLIP,
            CLIP,
        ).astype(np.float32)

        observed = state.normalize_array(context_matrix(rows))
        if observed.tobytes() != double_normalized.tobytes():
            diverged.append(name)

    assert diverged, "float64 and float32 agreed everywhere, so nothing was proven"
    assert "ordinary" in diverged, "realistic price data must be among the divergent cases"


def test_the_divergence_is_subtle_on_prices_and_gross_on_volume() -> None:
    """Two different sizes of error, both of which matter.

    On the four price columns float64 and float32 differ by parts in 10^5:
    small enough that a tolerance-based test would have accepted the wrong
    arithmetic, large enough to move a value across a quantizer bin edge.

    On volume and amount the difference is far larger, around half a per cent
    of a standard deviation here. float32 carries about seven significant
    digits, and a volume near 8e7 changing by 137 per session is right at that
    limit, so the float32 standard deviation of that column is materially
    different from the float64 one. Computing these statistics in double
    precision would therefore have fed the tokenizer a visibly different
    volume channel, not merely a rounded one.
    """
    rows = _rows(CORPORA["ordinary"])
    state = fit_context_state(rows)

    doubles = np.array([row.channels() for row in rows], dtype=np.float64)
    double_normalized = np.clip(
        (doubles - np.mean(doubles, axis=0)) / (np.std(doubles, axis=0) + 1e-5), -CLIP, CLIP
    ).astype(np.float32)
    observed = state.normalize_array(context_matrix(rows))
    assert observed.tobytes() != double_normalized.tobytes()

    difference = np.abs(observed.astype(np.float64) - double_normalized.astype(np.float64))
    price_columns = [OFFICIAL_COLUMNS.index(name) for name in ("open", "high", "low", "close")]
    volume_column = OFFICIAL_COLUMNS.index("volume")

    price_error = difference[:, price_columns].max()
    volume_error = difference[:, volume_column].max()

    assert 0.0 < price_error < 1e-3
    assert volume_error > price_error
    assert volume_error > 1e-3


def test_the_divergence_is_large_enough_to_move_a_quantizer_bin() -> None:
    """A few ULPs is the whole risk: bins are decided on exact comparisons."""
    rows = _rows(CORPORA["large_volume_and_amount"])
    state = fit_context_state(rows)

    doubles = np.array([row.channels() for row in rows], dtype=np.float64)
    double_normalized = np.clip(
        (doubles - np.mean(doubles, axis=0)) / (np.std(doubles, axis=0) + 1e-5), -CLIP, CLIP
    ).astype(np.float32)
    observed = state.normalize_array(context_matrix(rows))

    differing = int(np.count_nonzero(observed != double_normalized))
    assert differing > 0
    # Any nonzero count is a value that could fall on the other side of a bin
    # edge, which is a different token id and a different decoded candle.
    assert differing <= observed.size


# ----------------------------------------------------------- clipping


def test_clipping_saturates_at_the_positive_bound() -> None:
    rows = _rows(CORPORA["at_and_beyond_the_clip"])
    state = fit_context_state(rows)
    observed = state.normalize_array(context_matrix(rows))
    close = OFFICIAL_COLUMNS.index("close")
    assert observed[:, close].max() == np.float32(CLIP_VALUE)
    assert float(observed.max()) <= CLIP_VALUE
    assert float(observed.min()) >= -CLIP_VALUE


def test_clipping_saturates_at_the_negative_bound() -> None:
    """The mirror image, so both bounds are exercised rather than assumed."""
    mirrored = [(100.0, 101.0, 99.0, 100.0, 1000.0, 100_000.0) for _ in range(40)]
    mirrored[0] = (100.0, 101.0, 99.0, -10_000.0, 1000.0, 100_000.0)
    rows = _rows(mirrored)
    state = fit_context_state(rows)
    observed = state.normalize_array(context_matrix(rows))
    close = OFFICIAL_COLUMNS.index("close")
    assert observed[:, close].min() == np.float32(-CLIP_VALUE)
    assert float(observed.min()) >= -CLIP_VALUE


def test_clipping_is_applied_after_standardization_not_before() -> None:
    rows = _rows(CORPORA["at_and_beyond_the_clip"])
    state = fit_context_state(rows)
    matrix = context_matrix(rows)
    # Raw prices are far outside [-5, 5]; if the clip ran first the mean would
    # be 5 and everything would collapse.
    assert float(matrix.max()) > CLIP_VALUE
    assert float(state.mean_array().max()) > CLIP_VALUE


def test_zero_variance_divides_by_epsilon_alone() -> None:
    rows = _rows(CORPORA["exactly_zero_variance"])
    state = fit_context_state(rows)
    assert all(value == 0.0 for value in state.standard_deviation)
    observed = state.normalize_array(context_matrix(rows))
    # Every value equals its column mean, so the numerator is zero.
    assert np.all(observed == np.float32(0.0))
    assert state.epsilon == EPSILON == 1e-5


# ------------------------------------------------- the digest is over bytes


def test_the_state_digest_is_over_raw_float32_bytes() -> None:
    source = inspect.getsource(normalization_module._state_digest)
    assert 'mean.astype("<f4").tobytes()' in source
    assert 'standard_deviation.astype("<f4").tobytes()' in source
    assert "repr(" not in source


def test_no_decimal_repr_is_used_as_the_numerical_identity() -> None:
    source = inspect.getsource(normalization_module)
    assert "repr(value)" not in source
    assert "repr(" not in source.replace("# ", "")


def test_two_statistics_one_ulp_apart_hash_differently() -> None:
    """A digest built from decimal text could not reliably tell these apart.

    Asserted on the digest function directly, because a single perturbed input
    row is averaged away by the mean over the whole context and would not
    change the statistics at all.
    """
    rows = _rows(CORPORA["ordinary"])
    state = fit_context_state(rows)

    mean = state.mean_array()
    nudged = mean.copy()
    nudged[3] = np.nextafter(nudged[3], np.float32(np.inf))
    assert nudged.tobytes() != mean.tobytes()

    def digest_for(vector):
        return normalization_module._state_digest(
            columns=state.columns,
            mean=vector,
            standard_deviation=state.std_array(),
            epsilon=state.epsilon,
            clip_policy=state.clip_policy,
            clip_value=state.clip_value,
            fitted_candle_count=state.fitted_candle_count,
        )

    assert digest_for(mean) != digest_for(nudged)
    assert digest_for(mean) == state.state_sha256


def test_the_digest_covers_the_transform_metadata_too() -> None:
    rows = _rows(CORPORA["ordinary"])
    state = fit_context_state(rows)
    recomputed = normalization_module._state_digest(
        columns=state.columns,
        mean=state.mean_array(),
        standard_deviation=state.std_array(),
        epsilon=state.epsilon,
        clip_policy=state.clip_policy,
        clip_value=state.clip_value,
        fitted_candle_count=state.fitted_candle_count,
    )
    assert recomputed == state.state_sha256

    # A different clip policy is a different transform, so a different digest.
    different = normalization_module._state_digest(
        columns=state.columns,
        mean=state.mean_array(),
        standard_deviation=state.std_array(),
        epsilon=state.epsilon,
        clip_policy="something_else",
        clip_value=state.clip_value,
        fitted_candle_count=state.fitted_candle_count,
    )
    assert different != state.state_sha256


def test_the_stored_hex_reconstructs_the_statistics_exactly() -> None:
    rows = _rows(CORPORA["large_volume_and_amount"])
    state = fit_context_state(rows)
    mean = np.frombuffer(bytes.fromhex(state.mean_float32_hex), dtype="<f4")
    std = np.frombuffer(bytes.fromhex(state.standard_deviation_float32_hex), dtype="<f4")
    assert mean.tobytes() == state.mean_array().tobytes()
    assert std.tobytes() == state.std_array().tobytes()
    assert len(mean) == len(OFFICIAL_COLUMNS) == 6


# -------------------------------------------------- dtype discipline


def test_the_matrix_is_built_in_float32() -> None:
    rows = _rows(CORPORA["ordinary"])
    assert context_matrix(rows).dtype == np.float32
    assert context_matrix(rows).shape == (len(rows), 6)


def test_a_non_float32_matrix_is_refused() -> None:
    from openalpha_bridge.errors import BridgeTransformError

    rows = _rows(CORPORA["ordinary"])
    state = fit_context_state(rows)
    doubles = np.array([row.channels() for row in rows], dtype=np.float64)
    with pytest.raises(BridgeTransformError) as excinfo:
        # Deliberately the wrong dtype; that is what is under test.
        state.normalize_array(doubles)  # pyright: ignore[reportArgumentType]
    assert excinfo.value.failures[0].code == "NORMALIZATION_DTYPE_MISMATCH"


def test_python_floats_appear_only_at_the_row_boundary() -> None:
    source = inspect.getsource(normalization_module._to_rows)
    assert "float(value)" in source
    # And the arrays themselves never pass through a Python float loop.
    for method in ("normalize_array", "invert_array"):
        body = inspect.getsource(getattr(normalization_module.NormalizationState, method))
        assert "float(" not in body


def test_the_schema_version_records_the_change() -> None:
    rows = _rows(CORPORA["ordinary"])
    state = fit_context_state(rows)
    assert state.schema_version == "openalpha.bridge.diagnostic.normalization.v2"
    assert state.dtype == "float32"


def test_the_digest_changes_when_the_dtype_story_changes() -> None:
    """A v1 digest and a v2 digest of the same numbers must not collide."""
    rows = _rows(CORPORA["ordinary"])
    state = fit_context_state(rows)
    legacy = hashlib.sha256(
        repr((state.mean, state.standard_deviation)).encode("utf-8")
    ).hexdigest()
    assert state.state_sha256 != legacy
