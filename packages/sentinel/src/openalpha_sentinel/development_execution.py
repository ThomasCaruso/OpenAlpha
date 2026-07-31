from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from openalpha_research import EnvironmentMetadata, LocalArtifactStore
from openalpha_research.artifacts import Sha256

from .contracts import ForecastResponse, FrozenModel
from .development_diagnostics import ResolvedForecastError
from .development_manifest import DevelopmentOrigin
from .development_origin import (
    DevelopmentForecastCreation,
    MarketDataProvider,
    build_development_forecast_requests,
    create_development_forecast,
)
from .development_resolution import (
    DevelopmentResolvedOrigin,
    TerminalFailureRecord,
    record_terminal_failure,
    resolve_development_outcome,
)
from .development_runner import OriginExecutionSummary
from .development_serialization import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
)
from .evidence import EvidenceContext, verify_creation, verify_resolved_evidence
from .inference_cache import InferenceCache
from .market_data import MarketDataRequest
from .providers.kronos import InferenceBatchResult, InferenceEnvironment


class InferenceClient(Protocol):
    def forecast_many(self, requests: tuple) -> InferenceBatchResult: ...


class ReproducibilityProbe(FrozenModel):
    schema_version: Literal['sentinel-phase3a-reproducibility-probe-v1']
    origin_id: str
    context_length: Literal[512]
    sampling_seed: Literal[1729]
    request_count: Literal[2]
    request_success_count: int
    first_path_sha256: Sha256
    second_path_sha256: Sha256
    output_artifact_sha256: tuple[Sha256, Sha256]
    deterministic_replay_supported: bool
    environment: InferenceEnvironment
    created_at: datetime
    receipt_sha256: Sha256


class ExecutionEnvironmentRecord(FrozenModel):
    schema_version: Literal['sentinel-phase3a-execution-environment-v1']
    code_commit: str
    code_dirty: bool
    diff_sha256: Sha256 | None
    dependency_lock_sha256: Sha256
    inference: InferenceEnvironment
    source_revision: str
    state_root_class: Literal['private_external_cache']
    created_at: datetime


def run_reproducibility_probe(
    *,
    origin: DevelopmentOrigin,
    state_root: Path,
    market_provider: MarketDataProvider,
    inference_client: InferenceClient,
    inference_cache: InferenceCache,
    clock: Callable[[], datetime],
) -> ReproducibilityProbe:
    receipt_path = state_root / 'reproducibility-probe.json'
    if receipt_path.exists():
        probe = ReproducibilityProbe.model_validate_json(receipt_path.read_bytes())
        _verify_probe(state_root, probe)
        return probe
    created_at = clock()
    snapshot = market_provider.fetch(
        MarketDataRequest(
            purpose='forecast_context',
            symbol=origin.asset,
            start_inclusive=date(2022, 6, 1),
            end_exclusive=origin.cutoff + timedelta(days=1),
            cutoff=origin.cutoff,
            minimum_sessions=512,
        )
    )
    requests = build_development_forecast_requests(
        origin=origin,
        snapshot=snapshot,
        created_at=created_at,
    )
    request = next(
        item
        for item in requests
        if item.context_length == 512 and item.sampling_seed == 1729
    )
    first = inference_client.forecast_many((request,))
    second = inference_client.forecast_many((request,))
    first_response = _successful_response(first)
    second_response = _successful_response(second)
    first_path = first_response.generated_paths[0]
    second_path = second_response.generated_paths[0]
    store = LocalArtifactStore(state_root / 'reproducibility-probe-artifacts')
    first_ref = store.put_bytes(
        canonical_json_bytes(first_response.model_dump(mode='json')),
        media_type='application/json',
    )
    second_ref = store.put_bytes(
        canonical_json_bytes(second_response.model_dump(mode='json')),
        media_type='application/json',
    )
    deterministic = first_path.canonical_sha256 == second_path.canonical_sha256
    if deterministic:
        inference_cache.put(request, first_response)
    unsigned = ReproducibilityProbe(
        schema_version='sentinel-phase3a-reproducibility-probe-v1',
        origin_id=origin.origin_id,
        context_length=512,
        sampling_seed=1729,
        request_count=2,
        request_success_count=2,
        first_path_sha256=first_path.canonical_sha256,
        second_path_sha256=second_path.canonical_sha256,
        output_artifact_sha256=(first_ref.sha256, second_ref.sha256),
        deterministic_replay_supported=deterministic,
        environment=first.environment,
        created_at=created_at,
        receipt_sha256='0' * 64,
    )
    canonical_body = unsigned.model_dump(mode='json', exclude={'receipt_sha256'})
    probe = ReproducibilityProbe(
        schema_version=unsigned.schema_version,
        origin_id=unsigned.origin_id,
        context_length=unsigned.context_length,
        sampling_seed=unsigned.sampling_seed,
        request_count=unsigned.request_count,
        request_success_count=unsigned.request_success_count,
        first_path_sha256=unsigned.first_path_sha256,
        second_path_sha256=unsigned.second_path_sha256,
        output_artifact_sha256=unsigned.output_artifact_sha256,
        deterministic_replay_supported=unsigned.deterministic_replay_supported,
        environment=unsigned.environment,
        created_at=unsigned.created_at,
        receipt_sha256=sha256_bytes(canonical_json_bytes(canonical_body)),
    )
    _write_immutable(
        receipt_path,
        canonical_json_bytes(probe.model_dump(mode='json')),
    )
    _verify_probe(state_root, probe)
    return probe


class RealDevelopmentExecutor:
    def __init__(
        self,
        *,
        state_root: Path,
        repository_root: Path,
        market_provider: MarketDataProvider,
        inference_client: InferenceClient,
        inference_cache: InferenceCache,
        inference_environment: InferenceEnvironment,
        model_cache: Path,
        source_revision: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.state_root = state_root.resolve(strict=True)
        self.repository_root = repository_root.resolve(strict=True)
        self.market_provider = market_provider
        self.inference_client = inference_client
        self.inference_cache = inference_cache
        self.inference_environment = inference_environment
        self.model_cache = model_cache.resolve(strict=True)
        self.source_revision = source_revision
        self.clock = clock
        environment_path = self.state_root / 'execution-environment.json'
        if environment_path.exists():
            self.environment_record = ExecutionEnvironmentRecord.model_validate_json(
                environment_path.read_bytes()
            )
            if (
                self.environment_record.inference != inference_environment
                or self.environment_record.source_revision != source_revision
            ):
                raise ValueError('resumed execution environment differs from sealed record')
        else:
            self.environment_record = _execution_environment(
                repository_root=self.repository_root,
                inference=inference_environment,
                source_revision=source_revision,
                created_at=clock(),
            )
            _write_immutable(
                environment_path,
                canonical_json_bytes(self.environment_record.model_dump(mode='json')),
            )

    def __call__(self, origin: DevelopmentOrigin) -> OriginExecutionSummary:
        origin_root = self.state_root / 'origins' / origin.origin_id
        summary_path = origin_root / 'origin-execution.json'
        if summary_path.exists():
            summary = OriginExecutionSummary.model_validate_json(summary_path.read_bytes())
            if not self.verify_terminal_summary(origin, summary):
                raise ValueError('resumed origin execution failed terminal verification')
            return summary
        started = time.perf_counter()
        creation: DevelopmentForecastCreation | None = None
        stage = 'FORECAST'
        try:
            context = self._evidence_context(origin)
            store = LocalArtifactStore(origin_root / 'artifacts')
            creation = self._load_or_create(origin, store, context)
            forecast_latency = time.perf_counter() - started
            stage = 'OUTCOME'
            resolved = self._load_or_resolve(creation, store, context)
            analysis_path = origin_root / 'analysis-row.json'
            terminal_sha256 = sha256_bytes(analysis_path.read_bytes())
            summary = OriginExecutionSummary(
                schema_version='sentinel-phase3a-origin-execution-v1',
                origin_id=origin.origin_id,
                terminal_status='completed',
                request_success_count=9,
                request_count=9,
                ensemble_latency_seconds=forecast_latency,
                cache_bytes=self.cache_bytes(),
                process_stable=bool(resolved.chain_verified),
                terminal_record_sha256=terminal_sha256,
            )
        except Exception as error:  # noqa: BLE001 - terminal record preserves every failure
            failure = self._load_or_record_failure(
                origin=origin,
                stage=stage,
                error=error,
                creation=creation,
            )
            summary = OriginExecutionSummary(
                schema_version='sentinel-phase3a-origin-execution-v1',
                origin_id=origin.origin_id,
                terminal_status='failed',
                request_success_count=9 if creation is not None else 0,
                request_count=9,
                ensemble_latency_seconds=time.perf_counter() - started,
                cache_bytes=self.cache_bytes(),
                process_stable=True,
                terminal_record_sha256=failure.record_sha256,
            )
        _write_immutable(summary_path, canonical_json_bytes(summary.model_dump(mode='json')))
        if not self.verify_terminal_summary(origin, summary):
            raise ValueError('new origin execution failed terminal verification')
        return summary

    def cache_bytes(self) -> int:
        return _tree_bytes(self.model_cache) + _tree_bytes(self.inference_cache.root)

    def verify_terminal_summary(
        self,
        origin: DevelopmentOrigin,
        summary: OriginExecutionSummary,
    ) -> bool:
        try:
            if summary.origin_id != origin.origin_id:
                return False
            origin_root = self.state_root / 'origins' / origin.origin_id
            summary_path = origin_root / 'origin-execution.json'
            if (
                not summary_path.is_file()
                or summary_path.is_symlink()
                or summary_path.read_bytes()
                != canonical_json_bytes(summary.model_dump(mode='json'))
            ):
                return False
            if summary.terminal_status == 'completed':
                analysis_path = origin_root / 'analysis-row.json'
                resolved_path = origin_root / 'resolved.json'
                artifact_root = origin_root / 'artifacts'
                if any(
                    not path.is_file() or path.is_symlink()
                    for path in (analysis_path, resolved_path)
                ) or not artifact_root.is_dir():
                    return False
                analysis_bytes = analysis_path.read_bytes()
                if sha256_bytes(analysis_bytes) != summary.terminal_record_sha256:
                    return False
                resolved = DevelopmentResolvedOrigin.model_validate_json(
                    resolved_path.read_bytes()
                )
                if (
                    resolved.origin != origin
                    or canonical_json_bytes(resolved.analysis_row) != analysis_bytes
                    or not resolved.chain_verified
                ):
                    return False
                store = LocalArtifactStore(artifact_root)
                return verify_resolved_evidence(store, resolved.resolved)
            failure_path = origin_root / 'terminal-failure.json'
            if not failure_path.is_file() or failure_path.is_symlink():
                return False
            failure = TerminalFailureRecord.model_validate_json(failure_path.read_bytes())
            payload = failure.model_dump(mode='json')
            body = {
                key: value
                for key, value in payload.items()
                if key != 'record_sha256'
            }
            return (
                failure.origin == origin
                and failure.record_sha256 == summary.terminal_record_sha256
                and sha256_bytes(canonical_json_bytes(body)) == failure.record_sha256
            )
        except Exception:  # noqa: BLE001 - invalid terminal evidence must fail closed
            return False

    def _load_or_create(
        self,
        origin: DevelopmentOrigin,
        store: LocalArtifactStore,
        context: EvidenceContext,
    ) -> DevelopmentForecastCreation:
        path = self.state_root / 'origins' / origin.origin_id / 'creation.json'
        if path.exists():
            payload = json.loads(path.read_bytes())
            if not isinstance(payload, dict):
                raise TypeError('creation descriptor must be a mapping')
            payload['cache_hits'] = 9
            payload['cache_misses'] = 0
            creation = DevelopmentForecastCreation.model_validate_json(
                json.dumps(payload, allow_nan=False)
            )
            if not verify_creation(store, creation.creation):
                raise ValueError('resumed forecast creation failed verification')
            return creation

        def forecast_many(requests):
            return self.inference_client.forecast_many(requests).responses

        return create_development_forecast(
            origin=origin,
            state_root=self.state_root,
            market_provider=self.market_provider,
            forecast_many=forecast_many,
            cache=self.inference_cache,
            artifact_store=store,
            evidence_context=context,
            prior_resolved=self._prior_resolved(origin),
            clock=self.clock,
        )

    def _load_or_resolve(
        self,
        creation: DevelopmentForecastCreation,
        store: LocalArtifactStore,
        context: EvidenceContext,
    ) -> DevelopmentResolvedOrigin:
        path = self.state_root / 'origins' / creation.origin.origin_id / 'resolved.json'
        if path.exists():
            resolved = DevelopmentResolvedOrigin.model_validate_json(path.read_bytes())
            if not resolved.chain_verified:
                raise ValueError('resumed outcome chain is not verified')
            return resolved
        return resolve_development_outcome(
            creation=creation,
            state_root=self.state_root,
            market_provider=self.market_provider,
            artifact_store=store,
            evidence_context=context,
            clock=self.clock,
        )

    def _load_or_record_failure(
        self,
        *,
        origin: DevelopmentOrigin,
        stage: str,
        error: Exception,
        creation: DevelopmentForecastCreation | None,
    ) -> TerminalFailureRecord:
        path = self.state_root / 'origins' / origin.origin_id / 'terminal-failure.json'
        if path.exists():
            return TerminalFailureRecord.model_validate_json(path.read_bytes())
        attempts = (
            (creation.creation.seal_ref.sha256,) if creation is not None else ()
        )
        return record_terminal_failure(
            origin=origin,
            state_root=self.state_root,
            code=f'{stage}_FAILURE',
            stage=stage,
            message=f'{type(error).__name__}: {error}',
            attempt_sha256=attempts,
            occurred_at=self.clock(),
        )

    def _evidence_context(self, origin: DevelopmentOrigin) -> EvidenceContext:
        path = self.state_root / 'origins' / origin.origin_id / 'evidence-context.json'
        if path.exists():
            return EvidenceContext.model_validate_json(path.read_bytes())
        record = self.environment_record
        context = EvidenceContext(
            run_id=origin.origin_id,
            attempt_id='attempt-01',
            code_commit=record.code_commit,
            code_dirty=record.code_dirty,
            diff_sha256=record.diff_sha256,
            dependency_lock_sha256=record.dependency_lock_sha256,
            environment=EnvironmentMetadata(
                os_name=platform.system(),
                os_version=platform.version(),
                architecture=platform.machine(),
                python_version=platform.python_version(),
                node_version=None,
                dependency_lock_sha256=record.dependency_lock_sha256,
                container_image=None,
                hardware=(
                    f'repository_device=cpu; inference_device={record.inference.device}'
                ),
            ),
        )
        _write_immutable(path, canonical_json_bytes(context.model_dump(mode='json')))
        return context

    def _prior_resolved(
        self,
        origin: DevelopmentOrigin,
    ) -> tuple[ResolvedForecastError, ...]:
        resolved_errors = []
        origins_root = self.state_root / 'origins'
        if not origins_root.exists():
            return ()
        for path in origins_root.glob('*/resolved.json'):
            resolved = DevelopmentResolvedOrigin.model_validate_json(path.read_bytes())
            previous = resolved.origin
            if previous.cutoff >= origin.cutoff or previous.forecast_sessions[-1] > origin.cutoff:
                continue
            row = resolved.analysis_row
            forecast_error = row.get('forecast_error')
            if not isinstance(forecast_error, dict):
                raise TypeError('resolved analysis row is missing forecast error')
            resolved_errors.append(
                ResolvedForecastError(
                    origin_id=previous.origin_id,
                    asset=previous.asset,
                    forecast_cutoff=previous.cutoff,
                    outcome_end=previous.forecast_sessions[-1],
                    resolved_at=resolved.resolved.outcome_event.occurred_at,
                    absolute_error=float(forecast_error['kronos_absolute_error']),
                    resolved_evidence_sha256=resolved.resolved.manifest_ref.sha256,
                )
            )
        return tuple(
            sorted(
                resolved_errors,
                key=lambda item: (item.outcome_end, item.forecast_cutoff, item.origin_id),
            )
        )


def _execution_environment(
    *,
    repository_root: Path,
    inference: InferenceEnvironment,
    source_revision: str,
    created_at: datetime,
) -> ExecutionEnvironmentRecord:
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError('execution environment timestamp must be timezone-aware')
    commit = _git(repository_root, 'rev-parse', 'HEAD').decode().strip()
    dirty = bool(_git(repository_root, 'status', '--porcelain').strip())
    diff_sha256 = _working_tree_sha256(repository_root) if dirty else None
    lock_sha256 = hashlib.sha256((repository_root / 'uv.lock').read_bytes()).hexdigest()
    return ExecutionEnvironmentRecord(
        schema_version='sentinel-phase3a-execution-environment-v1',
        code_commit=commit,
        code_dirty=dirty,
        diff_sha256=diff_sha256,
        dependency_lock_sha256=lock_sha256,
        inference=inference,
        source_revision=source_revision,
        state_root_class='private_external_cache',
        created_at=created_at,
    )


def _working_tree_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_git(repository_root, 'diff', '--binary', 'HEAD'))
    untracked = _git(
        repository_root,
        'ls-files',
        '--others',
        '--exclude-standard',
        '-z',
    ).split(b'\0')
    for raw_path in sorted(path for path in untracked if path):
        relative = raw_path.decode('utf-8')
        digest.update(relative.encode('utf-8'))
        digest.update((repository_root / relative).read_bytes())
    return digest.hexdigest()


def _git(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ['git', '-C', str(repository_root), *arguments],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob('*') if path.is_file())


def _successful_response(batch: InferenceBatchResult) -> ForecastResponse:
    if len(batch.responses) != 1:
        raise ValueError('replay probe requires exactly one response')
    response = batch.responses[0]
    if response.failure is not None or len(response.generated_paths) != 1:
        raise ValueError('replay probe did not return one successful path')
    return response


def _verify_probe(state_root: Path, probe: ReproducibilityProbe) -> None:
    payload = probe.model_dump(mode='json')
    body = {key: value for key, value in payload.items() if key != 'receipt_sha256'}
    if sha256_bytes(canonical_json_bytes(body)) != probe.receipt_sha256:
        raise ValueError('reproducibility probe receipt hash mismatch')
    root = state_root / 'reproducibility-probe-artifacts' / 'sha256'
    for digest in probe.output_artifact_sha256:
        path = root / digest[:2] / digest
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('reproducibility probe output artifact failed verification')


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError('immutable reproducibility probe differs')
        return
    atomic_write_bytes(path, payload)
