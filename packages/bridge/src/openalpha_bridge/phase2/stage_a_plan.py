"""What Stage A was supposed to do, derived without reading what it did.

The verifier used to take ``expected_sequence_ids`` from the report it was
auditing. Any set of sequences the report happened to contain therefore matched
the set it was checked against, so that check could never fail. A report is not
its own specification.

This module builds the specification independently: sequence identities from the
locked periods, the locked calendar and the configured symbols; every identity
and version constant from locked configuration; resolved assets from the one
resolution the run already performed. Nothing here reads the report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..calendars import sessions_in_half_open_range
from ..config import BridgeRepresentationConfig
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..windowing import (
    CONTEXT_PREFIX_LENGTH,
    SCORED_SUFFIX_LENGTH,
    Partition,
    build_scored_sequences,
    score_mask_sha256,
)
from .cache import CACHE_SCHEMA_VERSION
from .kronos import BRIDGE_INPUT_DIMENSION, SOURCE_SPEC, ResolvedAssets
from .pipeline import LOCKED_PERIODS, Phase2Config

__all__ = [
    "STAGE_A_LOCKED_MAXIMUM_CANDLES",
    "STAGE_A_LOCKED_REQUEST_PERIOD",
    "STAGE_A_LOCKED_SYMBOLS",
    "StageAExecutionPlan",
    "build_stage_a_plan",
    "stage_a_expected_sequence_ids",
]

# Locked in phase2-preregistration-amendment.yaml under stage_a. Stage A is a
# pipeline smoke over SPY, not the whole training corpus, so the expectation is
# derived from these rather than from the training period.
STAGE_A_LOCKED_SYMBOLS: tuple[str, ...] = ("SPY",)
STAGE_A_LOCKED_REQUEST_PERIOD: tuple[str, str] = ("2015-05-04", "2023-01-01")
STAGE_A_LOCKED_MAXIMUM_CANDLES = 2000


class StageAExecutionPlan(BaseModel):
    """The independently derived expectation Stage A evidence is audited against."""

    model_config = ConfigDict(
        allow_inf_nan=False, arbitrary_types_allowed=True, extra="forbid", frozen=True, strict=True
    )

    schema_version: Literal["openalpha.bridge.phase2.stage_a_plan.v1"] = (
        "openalpha.bridge.phase2.stage_a_plan.v1"
    )

    # Run identity, from the active run rather than from the report.
    run_id: str
    experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    amendment_sha256: tuple[str, ...]
    evidence_class: str
    source_commit: str | None

    # Provider identity, from the provider object that is actually installed.
    provider_name: str
    provider_client_version: str

    # Locked structural constants.
    score_mask_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    representation_version: str
    cache_schema_version: str
    bridge_input_dimension: int
    prefix_length: int
    suffix_length: int
    official_source_revision: str
    official_source_file_sha256: dict[str, str]

    # Derived from locked periods, the locked calendar and configured symbols.
    expected_sequence_ids: frozenset[str]

    # Resolved once by the run; never re-resolved for the audit.
    assets: ResolvedAssets

    @property
    def expected_sequence_count(self) -> int:
        return len(self.expected_sequence_ids)


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


def stage_a_expected_sequence_ids(
    locked_periods: dict[Partition, tuple[str, str]] | None = None,
) -> frozenset[str]:
    """Every sequence Stage A is locked to produce, derived from the locks alone.

    Sessions come from the intersection of the locked Stage A request period and
    the locked training period, because Stage A may not touch validation,
    reconstruction-test or external data. The locked candle maximum is applied
    on top. No report is consulted.
    """
    periods = locked_periods if locked_periods is not None else LOCKED_PERIODS
    training = periods.get(Partition.TRAIN)
    if training is None:
        raise _fail(
            "STAGE_A_PLAN_NO_TRAINING_PERIOD",
            "the locked configuration defines no training period",
        )

    requested_start = datetime.fromisoformat(STAGE_A_LOCKED_REQUEST_PERIOD[0]).date()
    requested_end = datetime.fromisoformat(STAGE_A_LOCKED_REQUEST_PERIOD[1]).date()
    training_start = datetime.fromisoformat(training[0]).date()
    training_end = datetime.fromisoformat(training[1]).date()

    start = max(requested_start, training_start)
    end = min(requested_end, training_end)
    if start >= end:
        raise _fail(
            "STAGE_A_PLAN_EMPTY_PERIOD",
            "the locked Stage A request period does not intersect the training period",
        )

    sessions = sessions_in_half_open_range(start, end)[:STAGE_A_LOCKED_MAXIMUM_CANDLES]

    identifiers: set[str] = set()
    for symbol in STAGE_A_LOCKED_SYMBOLS:
        for sequence in build_scored_sequences(
            symbol=symbol,
            interval="1d",
            partition=Partition.TRAIN,
            partition_sessions=sessions,
            history_sessions=(),
        ):
            identifiers.add(sequence.sequence_id)
    if not identifiers:
        raise _fail(
            "STAGE_A_PLAN_NO_TRAINING_SEQUENCES",
            "the locked Stage A period and symbols imply no scored sequences",
        )
    return frozenset(identifiers)


def build_stage_a_plan(
    *,
    run_id: str,
    experiment_sha256: str,
    amendment_sha256: tuple[str, ...],
    evidence_class: str,
    source_commit: str | None,
    provider_name: str,
    provider_client_version: str,
    config: Phase2Config,
    assets: ResolvedAssets,
    locked_periods: dict[Partition, tuple[str, str]] | None = None,
    expected_sequence_ids: frozenset[str] | None = None,
) -> StageAExecutionPlan:
    """Derive the plan from locked configuration and the live run.

    ``expected_sequence_ids`` exists only so tests can pin a narrower set. In
    production it is omitted and the identities come from
    ``stage_a_expected_sequence_ids``, which reads the locks and never the
    report.
    """
    identifiers = (
        expected_sequence_ids
        if expected_sequence_ids is not None
        else stage_a_expected_sequence_ids(locked_periods)
    )
    if not identifiers:
        raise _fail(
            "STAGE_A_PLAN_NO_TRAINING_SEQUENCES",
            "the plan would expect no sequences, so any report would satisfy it",
        )
    return StageAExecutionPlan(
        run_id=run_id,
        experiment_sha256=experiment_sha256,
        amendment_sha256=tuple(amendment_sha256),
        evidence_class=evidence_class,
        source_commit=source_commit,
        provider_name=provider_name,
        provider_client_version=provider_client_version,
        score_mask_sha256=score_mask_sha256(),
        representation_version=BridgeRepresentationConfig().representation_version,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        bridge_input_dimension=BRIDGE_INPUT_DIMENSION,
        prefix_length=CONTEXT_PREFIX_LENGTH,
        suffix_length=SCORED_SUFFIX_LENGTH,
        official_source_revision=SOURCE_SPEC.revision,
        official_source_file_sha256=dict(SOURCE_SPEC.files),
        expected_sequence_ids=frozenset(identifiers),
        assets=assets,
    )
