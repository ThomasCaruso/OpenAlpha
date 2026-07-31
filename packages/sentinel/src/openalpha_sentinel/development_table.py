from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from openalpha_research.artifacts import Sha256
from pydantic import Field

from .contracts import FrozenModel
from .development_manifest import DevelopmentOrigin
from .development_serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


class DevelopmentTable(FrozenModel):
    schema_version: str
    rows: tuple[dict[str, Any], ...]
    completed_rows: tuple[dict[str, Any], ...]
    failed_rows: tuple[dict[str, Any], ...]
    failure_threshold: float
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    origin_sha256: tuple[Sha256, ...]
    jsonl_bytes: bytes
    content_sha256: Sha256


def build_development_table(
    origins: Sequence[DevelopmentOrigin],
    terminal_records: Mapping[str, Mapping[str, Any]],
) -> DevelopmentTable:
    identifiers = tuple(origin.origin_id for origin in origins)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('development origin identifiers must be unique')
    if set(terminal_records) != set(identifiers):
        missing = sorted(set(identifiers) - set(terminal_records))
        extra = sorted(set(terminal_records) - set(identifiers))
        raise ValueError(f'terminal population mismatch: missing={missing}, extra={extra}')
    normalized = tuple(
        _normalize_record(origin, terminal_records[origin.origin_id])
        for origin in origins
    )
    completed_without_labels = tuple(
        row for row in normalized if row['terminal_status'] == 'completed'
    )
    if not completed_without_labels:
        raise ValueError('development table has no completed origins')
    errors = np.asarray(
        [float(row['kronos_absolute_error']) for row in completed_without_labels],
        dtype=float,
    )
    if not np.isfinite(errors).all():
        raise ValueError('development forecast errors must be finite')
    threshold = float(np.quantile(errors, 0.75, method='linear'))
    rows = tuple(
        {
            **row,
            'failure_label': (
                float(row['kronos_absolute_error']) >= threshold
                if row['terminal_status'] == 'completed'
                else None
            ),
        }
        for row in normalized
    )
    completed = tuple(row for row in rows if row['terminal_status'] == 'completed')
    failed = tuple(row for row in rows if row['terminal_status'] == 'failed')
    jsonl = canonical_jsonl_bytes(rows)
    rebuilt = canonical_jsonl_bytes(tuple(dict(row) for row in rows))
    if rebuilt != jsonl:
        raise ValueError('development table serialization is not deterministic')
    return DevelopmentTable(
        schema_version='sentinel-phase3a-development-table-v1',
        rows=rows,
        completed_rows=completed,
        failed_rows=failed,
        failure_threshold=threshold,
        completed_count=len(completed),
        failed_count=len(failed),
        origin_sha256=tuple(sha256_bytes(canonical_json_bytes(row)) for row in rows),
        jsonl_bytes=jsonl,
        content_sha256=sha256_bytes(jsonl),
    )


def _normalize_record(
    origin: DevelopmentOrigin,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    status = record.get('terminal_status')
    if status == 'failed':
        nested_origin = record.get('origin')
        if not isinstance(nested_origin, Mapping):
            raise ValueError('terminal failure is missing its origin')
        if nested_origin.get('origin_id') != origin.origin_id:
            raise ValueError('terminal failure origin identity mismatch')
        return {
            'schema_version': 'sentinel-phase3a-development-row-v1',
            'terminal_status': 'failed',
            'analysis_status': 'unavailable',
            'origin_id': origin.origin_id,
            'asset': origin.asset,
            'cutoff': origin.cutoff.isoformat(),
            'horizon_end': origin.forecast_sessions[-1].isoformat(),
            'failure_code': record.get('code'),
            'failure_stage': record.get('stage'),
            'terminal_record_sha256': record.get('record_sha256'),
            'diagnostics': {},
            'diagnostic_missingness': {},
            'diagnostic_missing_reasons': {},
            'kronos_absolute_error': None,
            'baseline_absolute_error': None,
            'direction_correct': None,
            'deployability_label': None,
        }
    if status != 'completed' or record.get('origin_id') != origin.origin_id:
        raise ValueError('completed terminal record identity or status mismatch')
    structural = _require_mapping(record.get('structural_diagnostics'), 'structural diagnostics')
    reliability = _require_mapping(record.get('reliability_features'), 'reliability features')
    diagnostics: dict[str, Any] = {}
    missingness: dict[str, bool] = {}
    reasons: dict[str, Any] = {}
    for name, payload in (*structural.items(), *reliability.items()):
        entry = _require_mapping(payload, f'diagnostic {name}')
        available = entry.get('status') == 'available'
        diagnostics[str(name)] = entry.get('value') if available else None
        missingness[str(name)] = not available
        reasons[str(name)] = entry.get('reason') if not available else None
    forecast_error = _require_mapping(record.get('forecast_error'), 'forecast error')
    return {
        **dict(record),
        'schema_version': 'sentinel-phase3a-development-row-v1',
        'analysis_status': 'available',
        'diagnostics': diagnostics,
        'diagnostic_missingness': missingness,
        'diagnostic_missing_reasons': reasons,
        'kronos_absolute_error': forecast_error.get('kronos_absolute_error'),
        'baseline_absolute_error': forecast_error.get('baseline_absolute_error'),
        'direction_correct': forecast_error.get('direction_correct'),
        'deployability_label': forecast_error.get('deployability_label'),
    }


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping')
    return value
