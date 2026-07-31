from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from openalpha_research.artifacts import Sha256
from openalpha_research.errors import ArtifactConflictError
from pydantic import Field

from .contracts import FrozenModel
from .development_manifest import (
    DevelopmentManifest,
    DevelopmentOrigin,
    build_development_manifest,
)
from .development_serialization import atomic_write_bytes, canonical_json_bytes


class OriginExecutionSummary(FrozenModel):
    schema_version: Literal['sentinel-phase3a-origin-execution-v1']
    origin_id: str
    terminal_status: Literal['completed', 'failed']
    request_success_count: int = Field(ge=0)
    request_count: int = Field(ge=0)
    ensemble_latency_seconds: float = Field(ge=0.0)
    cache_bytes: int = Field(ge=0)
    process_stable: bool
    terminal_record_sha256: Sha256


class PilotOriginRecord(FrozenModel):
    origin_id: str
    request_success_count: int = Field(ge=0)
    request_count: int = Field(gt=0)
    ensemble_latency_seconds: float = Field(ge=0.0)
    process_stable: bool


class PilotGate(FrozenModel):
    schema_version: Literal['sentinel-phase3a-pilot-gate-v1']
    origin_count: int
    request_success_rate: float = Field(ge=0.0, le=1.0)
    median_ensemble_latency_seconds: float = Field(ge=0.0)
    cache_bytes: int = Field(ge=0)
    hosted_cost_per_cutoff: float = Field(ge=0.0)
    deterministic_replay_supported: bool
    process_stable: bool
    continue_automatically: bool
    failed_criteria: tuple[str, ...]


class DevelopmentPreflightReceipt(FrozenModel):
    schema_version: Literal['sentinel-phase3a-preflight-v1']
    manifest_sha256: Sha256
    experiment_sha256: Sha256
    cutoff_count: Literal[52]
    origin_count: Literal[104]
    first_cutoff: str
    last_cutoff: str
    network_accessed: Literal[False]
    inference_accessed: Literal[False]


class PilotExecutionReceipt(FrozenModel):
    schema_version: Literal['sentinel-phase3a-pilot-execution-v1']
    pilot_origin_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    gate: PilotGate
    continued_automatically: bool
    terminal_origin_count: int = Field(ge=0)


ExecutionCallable = Callable[[DevelopmentOrigin], OriginExecutionSummary]
TerminalVerificationCallable = Callable[
    [DevelopmentOrigin, OriginExecutionSummary],
    bool,
]


class DevelopmentRunner:
    def __init__(
        self,
        state_root: Path | str,
        *,
        terminal_verifier: TerminalVerificationCallable,
    ) -> None:
        configured = Path(state_root)
        configured.mkdir(parents=True, exist_ok=True)
        self.state_root = configured.resolve(strict=True)
        self.terminal_verifier = terminal_verifier

    def run_origins(
        self,
        origins: Sequence[DevelopmentOrigin],
        execute: ExecutionCallable,
        *,
        max_new: int | None = None,
    ) -> tuple[OriginExecutionSummary, ...]:
        results: list[OriginExecutionSummary] = []
        new_count = 0
        for origin in origins:
            existing = self._load_marker(origin)
            if existing is not None:
                results.append(existing)
                continue
            if max_new is not None and new_count >= max_new:
                continue
            summary = execute(origin)
            if summary.origin_id != origin.origin_id:
                raise ValueError('origin execution summary identity mismatch')
            self._write_marker(origin, summary)
            results.append(summary)
            new_count += 1
        return tuple(results)

    def verify_terminal_markers(self, origins: Sequence[DevelopmentOrigin]) -> bool:
        for origin in origins:
            marker = self._load_marker(origin)
            if marker is None or marker.origin_id != origin.origin_id:
                return False
        return True

    def pending_origins(
        self,
        origins: Sequence[DevelopmentOrigin],
    ) -> tuple[DevelopmentOrigin, ...]:
        return tuple(origin for origin in origins if self._load_marker(origin) is None)

    def run_pilot(
        self,
        origins: Sequence[DevelopmentOrigin],
        execute: ExecutionCallable,
        *,
        cache_bytes: Callable[[], int],
        hosted_cost_per_cutoff: float,
        deterministic_replay_supported: bool,
    ) -> PilotExecutionReceipt:
        pilot_origins = self._pilot_origins(origins)
        summaries = self.run_origins(pilot_origins, execute)
        if len(summaries) != 10:
            raise ValueError('pilot did not produce ten terminal operational records')
        records = tuple(
            PilotOriginRecord(
                origin_id=item.origin_id,
                request_success_count=item.request_success_count,
                request_count=item.request_count,
                ensemble_latency_seconds=item.ensemble_latency_seconds,
                process_stable=item.process_stable,
            )
            for item in summaries
        )
        gate = evaluate_pilot(
            records,
            cache_bytes=cache_bytes(),
            hosted_cost_per_cutoff=hosted_cost_per_cutoff,
            deterministic_replay_supported=deterministic_replay_supported,
        )
        _write_immutable(
            self.state_root / 'pilot-gate.json',
            canonical_json_bytes(gate.model_dump(mode='json')),
        )
        if gate.continue_automatically:
            self.run_origins(origins, execute)
        terminal_count = sum(self._load_marker(origin) is not None for origin in origins)
        receipt = PilotExecutionReceipt(
            schema_version='sentinel-phase3a-pilot-execution-v1',
            pilot_origin_ids=tuple(origin.origin_id for origin in pilot_origins),
            gate=gate,
            continued_automatically=gate.continue_automatically,
            terminal_origin_count=terminal_count,
        )
        _write_immutable(
            self.state_root / 'pilot-execution.json',
            canonical_json_bytes(receipt.model_dump(mode='json')),
        )
        return receipt

    def _pilot_origins(
        self,
        origins: Sequence[DevelopmentOrigin],
    ) -> tuple[DevelopmentOrigin, ...]:
        membership_path = self.state_root / 'pilot-origins.json'
        by_id = {origin.origin_id: origin for origin in origins}
        if membership_path.exists():
            payload = json.loads(membership_path.read_bytes())
            identifiers = payload.get('origin_ids') if isinstance(payload, dict) else None
            if not isinstance(identifiers, list) or len(identifiers) != 10:
                raise ValueError('pilot membership record is invalid')
            if any(identifier not in by_id for identifier in identifiers):
                raise ValueError('pilot membership is outside the declared population')
            return tuple(by_id[identifier] for identifier in identifiers)
        selected = self.pending_origins(origins)[:10]
        if len(selected) != 10:
            raise ValueError('fewer than ten unprocessed origins remain for the pilot')
        _write_immutable(
            membership_path,
            canonical_json_bytes(
                {
                    'schema_version': 'sentinel-phase3a-pilot-membership-v1',
                    'origin_ids': [origin.origin_id for origin in selected],
                }
            ),
        )
        return selected

    def _marker_path(self, origin: DevelopmentOrigin) -> Path:
        return self.state_root / 'origins' / origin.origin_id / 'runner-terminal.json'

    def _load_marker(self, origin: DevelopmentOrigin) -> OriginExecutionSummary | None:
        path = self._marker_path(origin)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise ValueError('runner terminal marker is not a regular file')
        marker = OriginExecutionSummary.model_validate_json(path.read_bytes())
        if marker.origin_id != origin.origin_id:
            raise ValueError('runner terminal marker identity mismatch')
        if not self.terminal_verifier(origin, marker):
            raise ValueError('runner terminal record failed verification')
        return marker

    def _write_marker(
        self,
        origin: DevelopmentOrigin,
        summary: OriginExecutionSummary,
    ) -> None:
        _write_immutable(
            self._marker_path(origin),
            canonical_json_bytes(summary.model_dump(mode='json')),
        )


def evaluate_pilot(
    records: tuple[PilotOriginRecord, ...],
    *,
    cache_bytes: int,
    hosted_cost_per_cutoff: float,
    deterministic_replay_supported: bool,
) -> PilotGate:
    if len(records) != 10:
        raise ValueError('operational pilot requires exactly ten origin records')
    request_count = sum(item.request_count for item in records)
    success_rate = sum(item.request_success_count for item in records) / request_count
    median_latency = float(
        statistics.median(item.ensemble_latency_seconds for item in records)
    )
    process_stable = all(item.process_stable for item in records)
    failures: list[str] = []
    if success_rate < 0.95:
        failures.append('REQUEST_SUCCESS_RATE_BELOW_0_95')
    if median_latency > 600.0:
        failures.append('MEDIAN_ENSEMBLE_LATENCY_ABOVE_600_SECONDS')
    if cache_bytes > 2_147_483_648:
        failures.append('CACHE_ABOVE_TWO_GIB')
    if hosted_cost_per_cutoff > 0.50:
        failures.append('HOSTED_EQUIVALENT_COST_ABOVE_0_50_USD')
    if not process_stable:
        failures.append('PROCESS_NOT_RESUMABLY_STABLE')
    if not deterministic_replay_supported:
        failures.append('DETERMINISTIC_REPLAY_UNSUPPORTED')
    return PilotGate(
        schema_version='sentinel-phase3a-pilot-gate-v1',
        origin_count=len(records),
        request_success_rate=success_rate,
        median_ensemble_latency_seconds=median_latency,
        cache_bytes=cache_bytes,
        hosted_cost_per_cutoff=hosted_cost_per_cutoff,
        deterministic_replay_supported=deterministic_replay_supported,
        process_stable=process_stable,
        continue_automatically=not failures,
        failed_criteria=tuple(failures),
    )


def preflight(state_root: Path | str) -> DevelopmentPreflightReceipt:
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_development_manifest()
    _write_immutable(
        root / 'development-manifest.json',
        canonical_json_bytes(manifest.model_dump(mode='json')),
    )
    receipt = DevelopmentPreflightReceipt(
        schema_version='sentinel-phase3a-preflight-v1',
        manifest_sha256=manifest.canonical_sha256,
        experiment_sha256=manifest.experiment_sha256,
        cutoff_count=52,
        origin_count=104,
        first_cutoff=manifest.cutoffs[0].isoformat(),
        last_cutoff=manifest.cutoffs[-1].isoformat(),
        network_accessed=False,
        inference_accessed=False,
    )
    _write_immutable(
        root / 'preflight.json',
        canonical_json_bytes(receipt.model_dump(mode='json')),
    )
    return receipt


def verify_offline_state(state_root: Path | str) -> dict[str, object]:
    root = Path(state_root)
    manifest_path = root / 'development-manifest.json'
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError('sealed development manifest is missing')
    observed = DevelopmentManifest.model_validate_json(manifest_path.read_bytes())
    expected = build_development_manifest()
    verified = observed == expected and manifest_path.read_bytes() == canonical_json_bytes(
        expected.model_dump(mode='json')
    )
    if not verified:
        raise ValueError('sealed development manifest failed offline verification')
    return {
        'manifest_verified': True,
        'manifest_sha256': expected.canonical_sha256,
        'network_accessed': False,
        'inference_accessed': False,
    }


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ArtifactConflictError('immutable Phase 3A runner record differs')
        return
    atomic_write_bytes(path, payload)
