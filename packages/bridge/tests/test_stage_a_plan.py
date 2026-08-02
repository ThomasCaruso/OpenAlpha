"""The Stage A audit expectation is derived from locks, never from the report."""

from __future__ import annotations

import inspect
from datetime import date

import pytest
from openalpha_bridge.calendars import sessions_in_half_open_range
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2 import stage_a_plan as plan_module
from openalpha_bridge.phase2 import stage_a_verify
from openalpha_bridge.phase2.pipeline import LOCKED_PERIODS
from openalpha_bridge.phase2.stage_a_plan import (
    STAGE_A_LOCKED_MAXIMUM_CANDLES,
    STAGE_A_LOCKED_REQUEST_PERIOD,
    STAGE_A_LOCKED_SYMBOLS,
    stage_a_expected_sequence_ids,
)
from openalpha_bridge.windowing import EXAMPLE_LENGTH, Partition


def test_the_locked_stage_a_scope_matches_the_amendment() -> None:
    """These come from phase2-preregistration-amendment.yaml, stage_a."""
    assert STAGE_A_LOCKED_SYMBOLS == ("SPY",)
    assert STAGE_A_LOCKED_REQUEST_PERIOD == ("2015-05-04", "2023-01-01")
    assert STAGE_A_LOCKED_MAXIMUM_CANDLES == 2000


def test_the_expectation_is_non_empty_and_bounded() -> None:
    identifiers = stage_a_expected_sequence_ids()
    assert identifiers, "an empty expectation would be satisfied by any report"

    start = max(
        date.fromisoformat(STAGE_A_LOCKED_REQUEST_PERIOD[0]),
        date.fromisoformat(LOCKED_PERIODS[Partition.TRAIN][0]),
    )
    end = min(
        date.fromisoformat(STAGE_A_LOCKED_REQUEST_PERIOD[1]),
        date.fromisoformat(LOCKED_PERIODS[Partition.TRAIN][1]),
    )
    sessions = sessions_in_half_open_range(start, end)[:STAGE_A_LOCKED_MAXIMUM_CANDLES]
    assert len(identifiers) == (len(sessions) - EXAMPLE_LENGTH) // 64 + 1


def test_stage_a_never_reaches_beyond_the_training_period() -> None:
    """The locked request period runs to 2023-01-01, past the training end."""
    assert date.fromisoformat(STAGE_A_LOCKED_REQUEST_PERIOD[1]) > date.fromisoformat(
        LOCKED_PERIODS[Partition.TRAIN][1]
    )
    truncated = stage_a_expected_sequence_ids()
    # Recomputing with a training period that ends earlier must shrink the set,
    # which proves the intersection is real rather than decorative.
    narrower = dict(LOCKED_PERIODS)
    narrower[Partition.TRAIN] = ("2010-01-01", "2019-01-01")
    assert stage_a_expected_sequence_ids(narrower) < truncated


def test_a_period_that_does_not_intersect_fails_closed() -> None:
    disjoint = dict(LOCKED_PERIODS)
    disjoint[Partition.TRAIN] = ("2010-01-01", "2010-06-01")
    with pytest.raises(BridgeTransformError) as excinfo:
        stage_a_expected_sequence_ids(disjoint)
    assert excinfo.value.failures[0].code in {
        "STAGE_A_PLAN_EMPTY_PERIOD",
        "STAGE_A_PLAN_NO_TRAINING_SEQUENCES",
    }


def test_a_missing_training_period_fails_closed() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        stage_a_expected_sequence_ids({})
    assert excinfo.value.failures[0].code == "STAGE_A_PLAN_NO_TRAINING_PERIOD"


def test_the_derivation_reads_no_report() -> None:
    """The signature has nowhere to put one."""
    parameters = inspect.signature(stage_a_expected_sequence_ids).parameters
    assert set(parameters) == {"locked_periods"}


def test_the_verifier_takes_its_expectations_only_from_the_plan() -> None:
    parameters = inspect.signature(stage_a_verify.verify_stage_a_report).parameters
    assert set(parameters) == {"report", "plan", "cache", "expected_tensor_specification"}
    # The previous signature let a caller pass identities derived from the
    # report itself, which made the sequence comparison unfailable.
    for gone in ("expected_sequence_ids", "run_id", "assets", "provider_identity"):
        assert gone not in parameters


def test_the_verifier_source_never_derives_expectations_from_the_report() -> None:
    source = inspect.getsource(stage_a_verify.verify_stage_a_report)
    assert "for record in report.sequences)" not in source
    assert "expected_sequence_ids=frozenset" not in source
    # Every check is (name, observed_from_report, expected_from_plan). The shard
    # identity check used to take its expectation from the report's own
    # provider_client_version, so it could not detect a shard written by a
    # different client.
    assert "                report.provider_client_version,\n" not in source
    assert "                plan.provider_client_version,\n" in source


def test_the_runner_does_not_resolve_assets_twice() -> None:
    from openalpha_bridge.cloud import runner as runner_module

    source = inspect.getsource(runner_module)
    assert "self._kronos.resolve_assets()" not in source
    assert "pipeline.resolved_assets()" in source
    # Exactly one resolution, performed by the pipeline stage. The audit reads
    # that result back rather than resolving again.
    assert source.count("resolve_assets()") == 1
    assert source.count("resolved_assets()") == 1


def test_the_runner_does_not_pass_report_derived_identities() -> None:
    from openalpha_bridge.cloud import runner as runner_module

    source = inspect.getsource(runner_module)
    assert "expected_sequence_ids=" not in source


def test_the_plan_module_never_imports_the_report_type() -> None:
    source = inspect.getsource(plan_module)
    assert "StageAReport" not in source


def test_a_report_covering_the_wrong_sequences_is_now_rejected(tmp_path) -> None:
    """The regression this whole change exists to fix.

    Under the old signature the caller passed the report's own sequence ids as
    the expectation, so a report covering one arbitrary window verified exactly
    as well as one covering the locked set. The runner now derives the
    expectation from the locks, so the short report is caught.
    """
    from openalpha_bridge.cloud.objectstore import InMemoryObjectStore
    from test_cloud_adapter import _measurements, _runner

    store = InMemoryObjectStore()
    runner = _runner(store, tmp_path)
    complete = runner._stage_a_report
    assert complete is not None
    assert {record.sequence_id for record in complete.sequences} == (
        stage_a_expected_sequence_ids()
    )

    # Drop a single window. Everything else about the report stays genuine:
    # real shards, real hashes, real identity.
    runner._stage_a_report = complete.model_copy(update={"sequences": complete.sequences[:-1]})
    result = runner.execute(
        holder="worker-1", confirm_open_test_partition=True, measurements=_measurements()
    )
    assert result.blocker_code == "STAGE_A_UNEXPECTED_SEQUENCES"
