from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise

import pytest
from openalpha_bridge.calendars import (
    CalendarName,
    sessions_in_half_open_range,
    xnys_holidays,
)
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.windowing import (
    CONTEXT_PREFIX_LENGTH,
    EXAMPLE_LENGTH,
    SCORED_SUFFIX_LENGTH,
    Partition,
    assert_nonoverlapping_targets,
    assert_preprocessing_fit_partitions,
    build_scored_sequences,
    feasibility_row,
    score_mask,
    score_mask_sha256,
    sequence_id,
    validate_example_window,
)

# Locked periods from research/bridge-v0/experiment.yaml.
TRAIN_PERIOD = (date(2010, 1, 1), date(2022, 1, 1))
VALIDATION_PERIOD = (date(2022, 1, 1), date(2023, 1, 1))
TEST_PERIOD = (date(2023, 1, 1), date(2024, 6, 29))
EXTERNAL_PERIOD = (date(2024, 7, 1), date(2025, 7, 1))

# Committed in research/bridge-v0/phase2-amendment-2-context-prefix.yaml.
SCORE_MASK_SHA256 = "2fe5b1b3c66dfd7c7e8af2612d69d3c2337a4a0f6dd3896a4d7c2f69911f3711"


def _sessions(period: tuple[date, date]) -> tuple[date, ...]:
    return sessions_in_half_open_range(*period)


def _valid_example() -> tuple[tuple[date, ...], tuple[Partition, ...]]:
    """A validation example: training-history prefix, validation-only suffix."""
    start = date(2020, 1, 1)
    stamps = tuple(start + timedelta(days=offset) for offset in range(EXAMPLE_LENGTH))
    membership = (Partition.TRAIN,) * CONTEXT_PREFIX_LENGTH + (
        Partition.VALIDATION,
    ) * SCORED_SUFFIX_LENGTH
    return stamps, membership


# --------------------------------------------------------------- score mask


def test_score_mask_shape_and_hash_are_locked() -> None:
    mask = score_mask()
    assert len(mask) == EXAMPLE_LENGTH
    assert sum(mask) == SCORED_SUFFIX_LENGTH
    assert not any(mask[:CONTEXT_PREFIX_LENGTH])
    assert all(mask[CONTEXT_PREFIX_LENGTH:])
    assert score_mask_sha256() == SCORE_MASK_SHA256


def test_prefix_and_suffix_lengths_sum_to_the_example_length() -> None:
    assert CONTEXT_PREFIX_LENGTH + SCORED_SUFFIX_LENGTH == EXAMPLE_LENGTH
    assert (CONTEXT_PREFIX_LENGTH, SCORED_SUFFIX_LENGTH) == (448, 64)


# ------------------------------------------------------- calendar behaviour


def test_xnys_excludes_weekends_and_observed_holidays() -> None:
    sessions = set(_sessions((date(2024, 1, 1), date(2025, 1, 1))))
    assert date(2024, 1, 1) not in sessions  # New Year's Day
    assert date(2024, 3, 29) not in sessions  # Good Friday
    assert date(2024, 6, 19) not in sessions  # Juneteenth
    assert date(2024, 11, 28) not in sessions  # Thanksgiving
    assert date(2024, 1, 6) not in sessions  # Saturday
    assert date(2024, 1, 7) not in sessions  # Sunday
    assert date(2024, 1, 2) in sessions
    assert len(sessions) == 252


def test_juneteenth_is_only_observed_from_2022() -> None:
    assert date(2021, 6, 18) in set(_sessions((date(2021, 6, 1), date(2021, 7, 1))))
    assert date(2022, 6, 20) not in set(_sessions((date(2022, 6, 1), date(2022, 7, 1))))


def test_ad_hoc_closures_are_excluded() -> None:
    assert date(2012, 10, 29) in xnys_holidays(2012)
    assert date(2012, 10, 30) in xnys_holidays(2012)
    assert date(2018, 12, 5) in xnys_holidays(2018)
    assert date(2025, 1, 9) in xnys_holidays(2025)


def test_crypto_calendar_includes_weekends_unlike_xnys() -> None:
    period = (date(2024, 1, 1), date(2024, 1, 15))
    crypto = sessions_in_half_open_range(*period, calendar=CalendarName.CRYPTO_24_7)
    assert len(crypto) == 14
    assert date(2024, 1, 6) in crypto  # Saturday
    assert date(2024, 1, 7) in crypto  # Sunday
    assert len(crypto) > len(_sessions(period))


def test_inverted_range_fails_explicitly() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        sessions_in_half_open_range(date(2024, 1, 2), date(2024, 1, 1))
    assert excinfo.value.failures[0].code == "INVERTED_DATE_RANGE"


# ------------------------------------------- feasibility under the amendment


@pytest.mark.parametrize(
    ("partition", "period", "expected_sessions", "expected_sequences"),
    [
        (Partition.TRAIN, TRAIN_PERIOD, 3021, 40),
        (Partition.VALIDATION, VALIDATION_PERIOD, 251, 3),
        (Partition.RECONSTRUCTION_TEST, TEST_PERIOD, 374, 5),
        (Partition.EXTERNAL_LATER, EXTERNAL_PERIOD, 250, 3),
    ],
)
def test_committed_feasibility_counts(
    partition: Partition,
    period: tuple[date, date],
    expected_sessions: int,
    expected_sequences: int,
) -> None:
    history: tuple[date, ...] = ()
    if partition is not Partition.TRAIN:
        history = _sessions((TRAIN_PERIOD[0], period[0]))

    row = feasibility_row(
        symbol="SPY",
        interval="1d",
        partition=partition,
        declared_start=period[0],
        declared_end=period[1],
        partition_sessions=_sessions(period),
        history_sessions=history,
    )
    assert row.expected_sessions == expected_sessions
    assert row.nonoverlapping_scored_sequences == expected_sequences
    assert row.scored_candles == expected_sequences * SCORED_SUFFIX_LENGTH
    assert row.feasible


def test_pooled_test_corpus_meets_the_locked_minima() -> None:
    history = _sessions((TRAIN_PERIOD[0], TEST_PERIOD[0]))
    scored_per_symbol = feasibility_row(
        symbol="SPY",
        interval="1d",
        partition=Partition.RECONSTRUCTION_TEST,
        declared_start=TEST_PERIOD[0],
        declared_end=TEST_PERIOD[1],
        partition_sessions=_sessions(TEST_PERIOD),
        history_sessions=history,
    ).scored_candles

    assert scored_per_symbol == 320
    assert scored_per_symbol * 14 == 4480  # training symbols
    assert scored_per_symbol * 6 == 1920  # unseen symbols, minimum 500
    assert scored_per_symbol * 20 == 6400  # declared pool, minimum 5000
    assert scored_per_symbol * 20 >= 5000
    assert scored_per_symbol * 6 >= 500


def test_validation_prefix_reaches_back_into_training_history() -> None:
    specs = build_scored_sequences(
        symbol="SPY",
        interval="1d",
        partition=Partition.VALIDATION,
        partition_sessions=_sessions(VALIDATION_PERIOD),
        history_sessions=_sessions((TRAIN_PERIOD[0], VALIDATION_PERIOD[0])),
    )
    first = specs[0]
    assert first.target_start >= VALIDATION_PERIOD[0]
    assert first.target_end < VALIDATION_PERIOD[1]
    # The read-only prefix legitimately precedes the validation partition.
    assert first.prefix_start < VALIDATION_PERIOD[0]
    assert first.prefix_end < first.target_start


def test_training_partition_supplies_its_own_warm_up() -> None:
    sessions = _sessions(TRAIN_PERIOD)
    specs = build_scored_sequences(
        symbol="SPY",
        interval="1d",
        partition=Partition.TRAIN,
        partition_sessions=sessions,
        history_sessions=(),
    )
    assert specs[0].target_start == sessions[CONTEXT_PREFIX_LENGTH]
    assert specs[0].prefix_start == sessions[0]
    for spec in specs:
        assert spec.target_start >= TRAIN_PERIOD[0]
        assert spec.target_end < TRAIN_PERIOD[1]


def test_history_must_precede_the_partition() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        build_scored_sequences(
            symbol="SPY",
            interval="1d",
            partition=Partition.VALIDATION,
            partition_sessions=(date(2022, 1, 3), date(2022, 1, 4)),
            history_sessions=(date(2022, 6, 1),),
        )
    assert excinfo.value.failures[0].code == "HISTORY_NOT_STRICTLY_EARLIER"


# --------------------------------------------------- overlap and identifiers


def test_evaluation_targets_are_nonoverlapping() -> None:
    specs = build_scored_sequences(
        symbol="SPY",
        interval="1d",
        partition=Partition.RECONSTRUCTION_TEST,
        partition_sessions=_sessions(TEST_PERIOD),
        history_sessions=_sessions((TRAIN_PERIOD[0], TEST_PERIOD[0])),
    )
    assert_nonoverlapping_targets(specs)
    for earlier, later in pairwise(specs):
        assert earlier.target_end < later.target_start


def test_overlapping_stride_is_rejected_for_evaluation() -> None:
    specs = build_scored_sequences(
        symbol="SPY",
        interval="1d",
        partition=Partition.RECONSTRUCTION_TEST,
        partition_sessions=_sessions(TEST_PERIOD),
        history_sessions=_sessions((TRAIN_PERIOD[0], TEST_PERIOD[0])),
        stride=32,
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        assert_nonoverlapping_targets(specs)
    assert excinfo.value.failures[0].code == "OVERLAPPING_EVALUATION_TARGET"


def test_sequence_ids_are_deterministic_and_distinct() -> None:
    kwargs = {
        "symbol": "SPY",
        "interval": "1d",
        "partition": Partition.RECONSTRUCTION_TEST,
        "target_start": date(2023, 1, 3),
        "target_end": date(2023, 4, 4),
    }
    assert sequence_id(**kwargs) == sequence_id(**kwargs)
    assert sequence_id(**{**kwargs, "symbol": "QQQ"}) != sequence_id(**kwargs)
    assert sequence_id(**{**kwargs, "partition": Partition.VALIDATION}) != sequence_id(**kwargs)


def test_sequence_records_required_provenance_fields() -> None:
    spec = build_scored_sequences(
        symbol="SPY",
        interval="1d",
        partition=Partition.VALIDATION,
        partition_sessions=_sessions(VALIDATION_PERIOD),
        history_sessions=_sessions((TRAIN_PERIOD[0], VALIDATION_PERIOD[0])),
    )[0]
    assert spec.prefix_length == CONTEXT_PREFIX_LENGTH
    assert spec.suffix_length == SCORED_SUFFIX_LENGTH
    assert spec.partition is Partition.VALIDATION
    assert spec.symbol == "SPY"
    assert spec.interval == "1d"
    assert spec.target_overlaps_other_target is False


# ------------------------------------------------- deliberate leakage tests


def test_well_formed_example_passes() -> None:
    stamps, membership = _valid_example()
    validate_example_window(
        session_timestamps=stamps,
        partition=Partition.VALIDATION,
        partition_membership=membership,
    )


def test_future_context_candle_fails() -> None:
    stamps, membership = _valid_example()
    corrupted = (*stamps[:100], stamps[CONTEXT_PREFIX_LENGTH] + timedelta(days=5), *stamps[101:])
    with pytest.raises(BridgeTransformError) as excinfo:
        validate_example_window(
            session_timestamps=corrupted,
            partition=Partition.VALIDATION,
            partition_membership=membership,
        )
    assert excinfo.value.failures[0].code == "FUTURE_CONTEXT_CANDLE"


def test_prefix_starting_after_the_target_fails() -> None:
    stamps, membership = _valid_example()
    target_start = stamps[CONTEXT_PREFIX_LENGTH]
    corrupted = (
        *stamps[: CONTEXT_PREFIX_LENGTH - 1],
        target_start + timedelta(days=1),
        *stamps[CONTEXT_PREFIX_LENGTH:],
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        validate_example_window(
            session_timestamps=corrupted,
            partition=Partition.VALIDATION,
            partition_membership=membership,
        )
    assert excinfo.value.failures[0].code == "FUTURE_CONTEXT_CANDLE"


def test_validation_target_inside_a_training_example_fails() -> None:
    stamps, _ = _valid_example()
    membership = (Partition.TRAIN,) * (EXAMPLE_LENGTH - 1) + (Partition.VALIDATION,)
    with pytest.raises(BridgeTransformError) as excinfo:
        validate_example_window(
            session_timestamps=stamps,
            partition=Partition.TRAIN,
            partition_membership=membership,
        )
    assert excinfo.value.failures[0].code == "SCORED_TARGET_OUTSIDE_PARTITION"


def test_held_out_candle_anywhere_in_a_training_example_fails() -> None:
    stamps, _ = _valid_example()
    membership: list[Partition] = [Partition.TRAIN] * EXAMPLE_LENGTH
    membership[10] = Partition.VALIDATION  # inside the read-only prefix
    with pytest.raises(BridgeTransformError) as excinfo:
        validate_example_window(
            session_timestamps=stamps,
            partition=Partition.TRAIN,
            partition_membership=tuple(membership),
        )
    assert excinfo.value.failures[0].code == "HELD_OUT_CANDLE_IN_TRAINING_EXAMPLE"


def test_test_target_inside_a_validation_example_fails() -> None:
    stamps, _ = _valid_example()
    membership = (
        (Partition.TRAIN,) * CONTEXT_PREFIX_LENGTH
        + (Partition.VALIDATION,) * (SCORED_SUFFIX_LENGTH - 1)
        + (Partition.RECONSTRUCTION_TEST,)
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        validate_example_window(
            session_timestamps=stamps,
            partition=Partition.VALIDATION,
            partition_membership=membership,
        )
    assert excinfo.value.failures[0].code == "SCORED_TARGET_OUTSIDE_PARTITION"


def test_incorrect_score_mask_fails() -> None:
    stamps, membership = _valid_example()
    shifted = (False,) * (CONTEXT_PREFIX_LENGTH - 1) + (True,) * (SCORED_SUFFIX_LENGTH + 1)
    with pytest.raises(BridgeTransformError) as excinfo:
        validate_example_window(
            session_timestamps=stamps,
            partition=Partition.VALIDATION,
            partition_membership=membership,
            mask=shifted,
        )
    assert excinfo.value.failures[0].code == "INVALID_SCORED_SUFFIX_LENGTH"


def test_score_mask_of_correct_size_but_wrong_positions_fails() -> None:
    stamps, membership = _valid_example()
    scattered = (True,) * SCORED_SUFFIX_LENGTH + (False,) * CONTEXT_PREFIX_LENGTH
    with pytest.raises(BridgeTransformError) as excinfo:
        validate_example_window(
            session_timestamps=stamps,
            partition=Partition.VALIDATION,
            partition_membership=membership,
            mask=scattered,
        )
    assert excinfo.value.failures[0].code == "INVALID_SCORE_MASK"


@pytest.mark.parametrize("suffix_length", [63, 65])
def test_off_by_one_suffix_fails(suffix_length: int) -> None:
    stamps, membership = _valid_example()
    prefix_length = EXAMPLE_LENGTH - suffix_length
    mask = (False,) * prefix_length + (True,) * suffix_length
    with pytest.raises(BridgeTransformError) as excinfo:
        validate_example_window(
            session_timestamps=stamps,
            partition=Partition.VALIDATION,
            partition_membership=membership,
            mask=mask,
        )
    assert excinfo.value.failures[0].code == "INVALID_SCORED_SUFFIX_LENGTH"


@pytest.mark.parametrize("length", [511, 513])
def test_wrong_example_length_fails(length: int) -> None:
    start = date(2020, 1, 1)
    stamps = tuple(start + timedelta(days=offset) for offset in range(length))
    membership = (Partition.VALIDATION,) * length
    with pytest.raises(BridgeTransformError) as excinfo:
        validate_example_window(
            session_timestamps=stamps,
            partition=Partition.VALIDATION,
            partition_membership=membership,
        )
    assert excinfo.value.failures[0].code == "INVALID_EXAMPLE_LENGTH"


def test_non_monotonic_context_fails() -> None:
    stamps, membership = _valid_example()
    corrupted = (*stamps[:10], stamps[8], *stamps[11:])
    with pytest.raises(BridgeTransformError) as excinfo:
        validate_example_window(
            session_timestamps=corrupted,
            partition=Partition.VALIDATION,
            partition_membership=membership,
        )
    assert excinfo.value.failures[0].code == "NON_MONOTONIC_CAUSAL_ORDER"


# ------------------------------------------------ training-only preprocessing


def test_preprocessing_fitted_on_training_only_passes() -> None:
    assert_preprocessing_fit_partitions(frozenset({Partition.TRAIN}))


@pytest.mark.parametrize(
    "partition",
    [Partition.VALIDATION, Partition.RECONSTRUCTION_TEST, Partition.EXTERNAL_LATER],
)
def test_preprocessing_fitted_on_held_out_partition_fails(partition: Partition) -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        assert_preprocessing_fit_partitions(frozenset({Partition.TRAIN, partition}))
    assert excinfo.value.failures[0].code == "PREPROCESSING_FIT_ON_HELD_OUT_PARTITION"
