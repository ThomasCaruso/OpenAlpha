from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Literal, cast

import exchange_calendars as xcals
import pandas as pd
from openalpha_research.artifacts import Sha256
from pydantic import Field, model_validator

from .contracts import FrozenModel
from .development_serialization import canonical_json_bytes, sha256_bytes

EXPERIMENT_SHA256 = 'fd50c208f465ce99750b11d4de5d5bc8c391bd73cb9ff450037c596c3e37d8bc'
DEVELOPMENT_START = date(2024, 7, 1)
DEVELOPMENT_END = date(2025, 6, 30)
HOLDOUT_START = date(2025, 7, 1)
ASSETS: tuple[Literal['SPY', 'QQQ'], ...] = ('SPY', 'QQQ')


class DevelopmentOrigin(FrozenModel):
    origin_id: str = Field(pattern=r'^sentinel-v0-(SPY|QQQ)-\d{4}-\d{2}-\d{2}-h5$')
    asset: Literal['SPY', 'QQQ']
    cutoff: date
    forecast_sessions: tuple[date, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode='after')
    def require_consistent_identity(self) -> DevelopmentOrigin:
        expected = f'sentinel-v0-{self.asset}-{self.cutoff.isoformat()}-h5'
        if self.origin_id != expected:
            raise ValueError('origin_id does not match asset and cutoff')
        if self.forecast_sessions[0] <= self.cutoff:
            raise ValueError('forecast sessions must follow cutoff')
        return self


class DevelopmentManifest(FrozenModel):
    schema_version: Literal['sentinel-phase3a-manifest-v1']
    experiment_sha256: Sha256
    development_start: date
    development_end: date
    holdout_start: date
    calendar: Literal['XNYS']
    assets: tuple[Literal['SPY', 'QQQ'], ...] = Field(min_length=2, max_length=2)
    cutoffs: tuple[date, ...] = Field(min_length=52, max_length=52)
    origins: tuple[DevelopmentOrigin, ...] = Field(min_length=104, max_length=104)
    stale_target_override: Literal[True]
    stale_target_note: str = Field(min_length=1)
    canonical_sha256: Sha256

    @model_validator(mode='after')
    def require_locked_order(self) -> DevelopmentManifest:
        if self.assets != ASSETS:
            raise ValueError('asset order must be SPY then QQQ')
        if tuple(sorted(self.cutoffs)) != self.cutoffs:
            raise ValueError('cutoffs must be chronological')
        expected = tuple((cutoff, asset) for cutoff in self.cutoffs for asset in ASSETS)
        actual = tuple((origin.cutoff, origin.asset) for origin in self.origins)
        if actual != expected:
            raise ValueError('origin order does not match locked population')
        return self


def _xnys_sessions(start: date, end: date) -> tuple[date, ...]:
    calendar = xcals.get_calendar('XNYS')
    timestamps = calendar.sessions_in_range(start.isoformat(), end.isoformat())
    return tuple(cast(pd.Timestamp, timestamp).date() for timestamp in timestamps)


@lru_cache(maxsize=1)
def locked_development_cutoffs() -> tuple[date, ...]:
    sessions = _xnys_sessions(
        DEVELOPMENT_START - timedelta(days=8),
        DEVELOPMENT_END + timedelta(days=8),
    )
    weekly_last: dict[tuple[int, int], date] = {}
    for session in sessions:
        iso = session.isocalendar()
        weekly_last[(iso.year, iso.week)] = session
    return tuple(
        session
        for _, session in sorted(weekly_last.items())
        if DEVELOPMENT_START <= session <= DEVELOPMENT_END
    )


def require_development_cutoff(cutoff: date) -> date:
    if cutoff >= HOLDOUT_START:
        raise ValueError('holdout cutoff is prohibited during Phase 3A')
    if cutoff not in locked_development_cutoffs():
        raise ValueError('cutoff is not a locked development cutoff')
    return cutoff


def require_outcome_sessions(cutoff: date) -> tuple[date, ...]:
    require_development_cutoff(cutoff)
    sessions = _xnys_sessions(cutoff + timedelta(days=1), cutoff + timedelta(days=14))
    if len(sessions) < 5:
        raise ValueError('five XNYS outcome sessions are unavailable')
    return sessions[:5]


def _manifest_payload() -> dict[str, object]:
    cutoffs = locked_development_cutoffs()
    origins = tuple(
        DevelopmentOrigin(
            origin_id=f'sentinel-v0-{asset}-{cutoff.isoformat()}-h5',
            asset=asset,
            cutoff=cutoff,
            forecast_sessions=require_outcome_sessions(cutoff),
        )
        for cutoff in cutoffs
        for asset in ASSETS
    )
    return {
        'schema_version': 'sentinel-phase3a-manifest-v1',
        'experiment_sha256': EXPERIMENT_SHA256,
        'development_start': DEVELOPMENT_START,
        'development_end': DEVELOPMENT_END,
        'holdout_start': HOLDOUT_START,
        'calendar': 'XNYS',
        'assets': ASSETS,
        'cutoffs': cutoffs,
        'origins': origins,
        'stale_target_override': True,
        'stale_target_note': (
            'Sentinel v0.4 target_usable_per_asset.minimum=100 is stale for the locked '
            'weekly one-year development sample; the explicit 52-cutoff design governs.'
        ),
    }


@lru_cache(maxsize=1)
def build_development_manifest() -> DevelopmentManifest:
    payload = _manifest_payload()
    origins = cast(tuple[DevelopmentOrigin, ...], payload['origins'])
    canonical_payload = {
        **payload,
        'development_start': DEVELOPMENT_START.isoformat(),
        'development_end': DEVELOPMENT_END.isoformat(),
        'holdout_start': HOLDOUT_START.isoformat(),
        'assets': list(ASSETS),
        'cutoffs': [item.isoformat() for item in locked_development_cutoffs()],
        'origins': [item.model_dump(mode='json') for item in origins],
    }
    return DevelopmentManifest.model_validate(
        {**payload, 'canonical_sha256': sha256_bytes(canonical_json_bytes(canonical_payload))}
    )
