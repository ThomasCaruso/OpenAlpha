from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
from openalpha_bridge import (
    BridgeFinancialTransform,
    BridgeRepresentationConfig,
    VolumeMode,
    audit_bridge_output,
)


def _required() -> BridgeFinancialTransform:
    return BridgeFinancialTransform(
        BridgeRepresentationConfig(
            volume_mode=VolumeMode.VOLUME_REQUIRED,
            output_dtype="float64",
        )
    )


def test_every_constructed_output_passes_the_independent_sentinel_adapter() -> None:
    transform = _required()
    decoded = transform.decode_head_outputs(
        raw_outputs=np.array(
            [
                [0.1, -0.2, -4.0, 3.0, -8.0],
                [-0.3, 0.4, 2.0, -2.0, 5.0],
            ]
        ),
        initial_previous_close=100.0,
    )

    audit = transform.audit(decoded)

    assert audit.valid is True
    assert audit.sequence_count == 1
    assert audit.candle_count == 2
    assert audit.violations == ()
    assert len(audit.sentinel_results) == 1
    assert audit.sentinel_results[0].valid is True
    assert audit.timestamps_supplied is False


def test_optional_missing_volume_is_none_to_sentinel_not_numeric_zero() -> None:
    transform = BridgeFinancialTransform(BridgeRepresentationConfig())
    present = np.array([False, True])
    decoded = transform.decode_features(
        transformed_features=np.array([[0.0, 0.0, 0.0, 0.0, np.nan], [0.0, 0.0, 0.0, 0.0, 0.0]]),
        initial_previous_close=100.0,
        volume_present=present,
    )

    audit = transform.audit(decoded)

    assert audit.valid is True
    assert audit.volume_present_count == 1
    assert audit.volume_missing_count == 1


def test_validator_reports_exact_financial_grammar_violations_without_repair() -> None:
    config = BridgeRepresentationConfig(
        volume_mode=VolumeMode.VOLUME_REQUIRED,
        output_dtype="float64",
    )
    invalid = np.array([[100.0, 99.0, 101.0, 102.0, -1.0]])

    audit = audit_bridge_output(
        config=config,
        candles=invalid,
        volume_present=np.array([True]),
        expected_sequence_length=1,
    )

    assert audit.valid is False
    codes = {violation.code for violation in audit.violations}
    assert "HIGH_BELOW_OPEN" in codes
    assert "HIGH_BELOW_CLOSE" in codes
    assert "HIGH_BELOW_LOW" in codes
    assert "LOW_ABOVE_OPEN" in codes
    assert "LOW_ABOVE_CLOSE" not in codes
    assert "NEGATIVE_VOLUME" in codes
    np.testing.assert_array_equal(invalid, [[100.0, 99.0, 101.0, 102.0, -1.0]])


def test_validator_checks_feature_count_and_exact_sequence_length() -> None:
    config = BridgeRepresentationConfig(
        volume_mode=VolumeMode.VOLUME_REQUIRED,
        output_dtype="float64",
    )
    wrong_features = audit_bridge_output(
        config=config,
        candles=np.ones((2, 4)),
        expected_sequence_length=2,
    )
    assert wrong_features.valid is False
    assert wrong_features.violations[0].code == "FEATURE_COUNT_MISMATCH"

    wrong_length = audit_bridge_output(
        config=config,
        candles=np.ones((2, 5)),
        volume_present=np.ones(2, dtype=bool),
        expected_sequence_length=3,
    )
    assert wrong_length.valid is False
    assert any(item.code == "SEQUENCE_LENGTH_MISMATCH" for item in wrong_length.violations)


def test_timestamp_alignment_duplicates_and_strict_order_are_structured() -> None:
    config = BridgeRepresentationConfig(
        volume_mode=VolumeMode.VOLUME_REQUIRED,
        output_dtype="float64",
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    expected = [start + timedelta(days=step) for step in range(3)]
    duplicate_unordered = [expected[1], expected[1], expected[0]]
    candles = np.ones((3, 5))

    audit = audit_bridge_output(
        config=config,
        candles=candles,
        volume_present=np.ones(3, dtype=bool),
        timestamps=duplicate_unordered,
        expected_timestamps=expected,
        expected_sequence_length=3,
    )

    assert audit.valid is False
    codes = {item.code for item in audit.violations}
    assert "DUPLICATE_TIMESTAMP" in codes
    assert "TIMESTAMP_NOT_STRICTLY_INCREASING" in codes
    assert "UNEXPECTED_TIMESTAMP" in codes
    assert audit.timestamps_supplied is True


def test_empty_output_is_invalid_even_without_an_expected_length() -> None:
    config = BridgeRepresentationConfig(
        volume_mode=VolumeMode.VOLUME_REQUIRED,
        output_dtype="float64",
    )
    audit = audit_bridge_output(config=config, candles=np.empty((0, 5)))
    assert audit.valid is False
    assert audit.violations[0].code == "EMPTY_SEQUENCE"


def test_incomparable_timestamp_types_produce_a_violation_instead_of_type_error() -> None:
    config = BridgeRepresentationConfig(
        volume_mode=VolumeMode.VOLUME_REQUIRED,
        output_dtype="float64",
    )
    audit = audit_bridge_output(
        config=config,
        candles=np.ones((2, 5)),
        timestamps=[date(2026, 1, 1), datetime(2026, 1, 2, tzinfo=UTC)],
    )
    assert audit.valid is False
    assert any(item.code == "INCOMPARABLE_TIMESTAMPS" for item in audit.violations)


def test_optional_missing_volume_requires_nan_storage_not_numeric_zero() -> None:
    audit = audit_bridge_output(
        config=BridgeRepresentationConfig(),
        candles=np.ones((1, 5)),
        volume_present=np.array([False]),
    )
    assert audit.valid is False
    assert any(item.code == "MISSING_VOLUME_MUST_USE_NAN" for item in audit.violations)


def test_batch_timestamps_and_volumes_must_match_without_broadcasting() -> None:
    config = BridgeRepresentationConfig()
    candles = np.ones((2, 3, 5))
    candles[..., 4] = np.nan

    audit = audit_bridge_output(
        config=config,
        candles=candles,
        volume_present=np.zeros((1, 3), dtype=bool),
        timestamps=[
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            datetime(2026, 1, 3, tzinfo=UTC),
        ],
        expected_sequence_length=3,
    )

    assert audit.valid is False
    codes = {item.code for item in audit.violations}
    assert "VOLUME_MASK_SHAPE" in codes
    assert "TIMESTAMP_SHAPE" in codes
