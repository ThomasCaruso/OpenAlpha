from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import numpy as np
from openalpha_research.artifacts import Sha256
from pydantic import Field, field_validator

from .contracts import FrozenModel, OHLCVObservation
from .development_serialization import canonical_json_bytes, sha256_bytes
from .diagnostics import (
    DIAGNOSTIC_NAMES,
    DiagnosticValue,
    DiagnosticVector,
    compute_phase_2_diagnostics,
)
from .ensemble import EnsembleResult

DEVELOPMENT_DIAGNOSTIC_NAMES = tuple(
    name for name in DIAGNOSTIC_NAMES if name != 'UNCERTAINTY_MISCALIBRATION'
)
DEVELOPMENT_CONFIGURATION_SHA256 = sha256_bytes(
    canonical_json_bytes(
        {
            'version': 'sentinel-v0.4',
            'diagnostics': list(DEVELOPMENT_DIAGNOSTIC_NAMES),
            'recent_model_error_window': 8,
        }
    )
)


class ResolvedForecastError(FrozenModel):
    origin_id: str = Field(min_length=1, max_length=128)
    asset: Literal['SPY', 'QQQ']
    forecast_cutoff: date
    outcome_end: date
    resolved_at: datetime
    absolute_error: float = Field(ge=0.0)
    resolved_evidence_sha256: Sha256

    @field_validator('resolved_at')
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('resolved_at must be timezone-aware')
        return value


def recent_model_error(errors: tuple[float, ...]) -> DiagnosticValue:
    if len(errors) < 8:
        return DiagnosticValue(
            name='RECENT_MODEL_ERROR',
            status='not_computable',
            value=None,
            units='log_return',
            reason='INSUFFICIENT_PRIOR_RESOLVED_FORECASTS',
            available_before_outcome=True,
        )
    recent = np.asarray(errors[-8:], dtype=float)
    if not np.isfinite(recent).all() or np.any(recent < 0):
        raise ValueError('recent model errors must be finite and nonnegative')
    return DiagnosticValue(
        name='RECENT_MODEL_ERROR',
        status='available',
        value=float(np.mean(recent)),
        units='log_return',
        reason=None,
        available_before_outcome=True,
    )


def compute_development_nonstructural_diagnostics(
    *,
    symbol: Literal['SPY', 'QQQ'],
    cutoff: date,
    context: tuple[OHLCVObservation, ...],
    ensemble: EnsembleResult,
    created_at: datetime,
    prior_resolved: tuple[ResolvedForecastError, ...],
) -> DiagnosticVector:
    eligible = tuple(item for item in prior_resolved if item.asset == symbol)
    for item in eligible:
        if item.outcome_end > cutoff:
            raise ValueError('prior outcome must resolve no later than the current cutoff')
        if item.resolved_at > created_at:
            raise ValueError('prior outcome seal must predate diagnostic creation')
    ordered = tuple(
        sorted(
            eligible,
            key=lambda item: (item.outcome_end, item.forecast_cutoff, item.origin_id),
        )
    )
    base = compute_phase_2_diagnostics(
        symbol=symbol,
        cutoff=cutoff,
        context=context,
        ensemble=ensemble,
        created_at=created_at,
    )
    recent = recent_model_error(tuple(item.absolute_error for item in ordered))
    values = tuple(
        recent if item.name == 'RECENT_MODEL_ERROR' else item
        for item in base.values
        if item.name != 'UNCERTAINTY_MISCALIBRATION'
    )
    if tuple(item.name for item in values) != DEVELOPMENT_DIAGNOSTIC_NAMES:
        raise ValueError('development diagnostic order does not match Sentinel v0.4')
    input_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                'phase_2_input_sha256': base.input_sha256,
                'prior_resolved': [
                    {
                        'origin_id': item.origin_id,
                        'outcome_end': item.outcome_end.isoformat(),
                        'absolute_error': item.absolute_error,
                        'resolved_evidence_sha256': item.resolved_evidence_sha256,
                    }
                    for item in ordered
                ],
            }
        )
    )
    return DiagnosticVector(
        symbol=symbol,
        cutoff=cutoff,
        causal_through=cutoff,
        created_at=created_at,
        values=values,
        input_sha256=input_sha256,
        configuration_sha256=DEVELOPMENT_CONFIGURATION_SHA256,
        action=None,
    )
