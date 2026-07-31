from __future__ import annotations

import math
from datetime import date

from pydantic import Field, model_validator

from .contracts import ForecastPath, FrozenModel, OHLCVObservation
from .development_serialization import canonical_json_bytes, sha256_bytes

EXPECTED_COMBINATIONS = frozenset(
    (context, seed)
    for context in (128, 256, 512)
    for seed in (1729, 2027, 7919)
)


class EnsembleMember(FrozenModel):
    context_length: int
    sampling_seed: int
    path: ForecastPath
    inference_duration_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def require_declared_configuration(self) -> EnsembleMember:
        if (self.context_length, self.sampling_seed) not in EXPECTED_COMBINATIONS:
            raise ValueError("undeclared ensemble context/seed combination")
        return self


class IndividualPredictedReturn(FrozenModel):
    context_length: int
    sampling_seed: int
    predicted_log_return: float


class ContextReturnSummary(FrozenModel):
    context_length: int
    averaged_close_path: tuple[float, ...] = Field(min_length=5, max_length=5)
    predicted_log_return: float


class EnsembleResult(FrozenModel):
    cutoff_close: float = Field(gt=0.0)
    forecast_sessions: tuple[date, ...] = Field(min_length=5, max_length=5)
    members: tuple[EnsembleMember, ...] = Field(min_length=9, max_length=9)
    individual_predicted_log_returns: tuple[IndividualPredictedReturn, ...] = Field(
        min_length=9, max_length=9
    )
    context_summaries: tuple[ContextReturnSummary, ...] = Field(min_length=3, max_length=3)
    canonical_close_path: tuple[float, ...] = Field(min_length=5, max_length=5)
    canonical_predicted_log_return: float
    baseline_close_path: tuple[float, ...] = Field(min_length=5, max_length=5)
    baseline_predicted_log_return: float

    @model_validator(mode="after")
    def require_finite_outputs(self) -> EnsembleResult:
        values = (
            *self.canonical_close_path,
            self.canonical_predicted_log_return,
            *(item.predicted_log_return for item in self.individual_predicted_log_returns),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ensemble values must be finite")
        return self


def average_forecast_paths(
    paths: tuple[ForecastPath, ...],
    *,
    path_id: str,
) -> ForecastPath:
    if not paths:
        raise ValueError('at least one forecast path is required')
    expected = tuple(
        (row.session, row.timestamp) for row in paths[0].observations
    )
    if any(
        tuple((row.session, row.timestamp) for row in path.observations) != expected
        for path in paths[1:]
    ):
        raise ValueError('forecast path sessions and timestamps must match')
    count = float(len(paths))
    observations = tuple(
        OHLCVObservation(
            session=paths[0].observations[step].session,
            timestamp=paths[0].observations[step].timestamp,
            open=sum(path.observations[step].open for path in paths) / count,
            high=sum(path.observations[step].high for path in paths) / count,
            low=sum(path.observations[step].low for path in paths) / count,
            close=sum(path.observations[step].close for path in paths) / count,
            volume=sum(path.observations[step].volume for path in paths) / count,
        )
        for step in range(len(paths[0].observations))
    )
    canonical_sha256 = sha256_bytes(
        canonical_json_bytes([row.model_dump(mode='json') for row in observations])
    )
    return ForecastPath(
        path_id=path_id,
        observations=observations,
        canonical_sha256=canonical_sha256,
    )


def assemble_ensemble(
    *,
    cutoff_close: float,
    forecast_sessions: tuple[date, ...],
    members: tuple[EnsembleMember, ...],
) -> EnsembleResult:
    if not math.isfinite(cutoff_close) or cutoff_close <= 0:
        raise ValueError("cutoff_close must be finite and positive")
    if len(members) != 9:
        raise ValueError("ensemble requires exactly nine individual paths")
    combinations = {(member.context_length, member.sampling_seed) for member in members}
    if combinations != EXPECTED_COMBINATIONS or len(combinations) != len(members):
        raise ValueError("ensemble must contain the exact context/seed Cartesian product")
    ordered = tuple(
        sorted(members, key=lambda item: (item.context_length, item.sampling_seed))
    )
    for member in ordered:
        sessions = tuple(row.session for row in member.path.observations)
        if sessions != forecast_sessions:
            raise ValueError("forecast path sessions do not match declared XNYS sessions")
        if any(row.close <= 0 for row in member.path.observations):
            raise ValueError("forecast close values must be positive")

    individual = tuple(
        IndividualPredictedReturn(
            context_length=member.context_length,
            sampling_seed=member.sampling_seed,
            predicted_log_return=math.log(member.path.observations[-1].close / cutoff_close),
        )
        for member in ordered
    )
    context_summaries: list[ContextReturnSummary] = []
    for context in (128, 256, 512):
        context_members = tuple(item for item in ordered if item.context_length == context)
        averaged = tuple(
            sum(member.path.observations[step].close for member in context_members) / 3.0
            for step in range(5)
        )
        context_summaries.append(
            ContextReturnSummary(
                context_length=context,
                averaged_close_path=averaged,
                predicted_log_return=math.log(averaged[-1] / cutoff_close),
            )
        )
    canonical = context_summaries[-1]
    return EnsembleResult(
        cutoff_close=cutoff_close,
        forecast_sessions=forecast_sessions,
        members=ordered,
        individual_predicted_log_returns=individual,
        context_summaries=tuple(context_summaries),
        canonical_close_path=canonical.averaged_close_path,
        canonical_predicted_log_return=canonical.predicted_log_return,
        baseline_close_path=(cutoff_close,) * 5,
        baseline_predicted_log_return=0.0,
    )
