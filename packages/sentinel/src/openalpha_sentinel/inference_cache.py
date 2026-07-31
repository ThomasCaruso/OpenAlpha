from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Self

from openalpha_research import ArtifactRef, LocalArtifactStore
from openalpha_research.artifacts import Sha256
from openalpha_research.errors import ArtifactConflictError, ArtifactIntegrityError
from pydantic import computed_field

from .contracts import ForecastRequest, ForecastResponse, FrozenModel
from .development_serialization import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
)


class InferenceCacheKey(FrozenModel):
    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    source_revision: str
    symbol: str
    cutoff: date
    forecast_sessions: tuple[date, ...]
    data_snapshot_sha256: Sha256
    context_length: int
    forecast_horizon: int
    sampling_seed: int
    temperature: float
    top_p: float
    sample_count: int

    @computed_field(return_type=str)
    @property
    def canonical_sha256(self) -> str:
        payload = self.model_dump(mode='json', exclude_computed_fields=True)
        return sha256_bytes(canonical_json_bytes(payload))

    @classmethod
    def from_request(cls, request: ForecastRequest) -> Self:
        return cls(
            model_repository=request.model_repository,
            model_revision=request.model_revision,
            tokenizer_repository=request.tokenizer_repository,
            tokenizer_revision=request.tokenizer_revision,
            source_revision=request.source_revision,
            symbol=request.symbol,
            cutoff=request.cutoff,
            forecast_sessions=request.forecast_sessions,
            data_snapshot_sha256=request.data_snapshot_sha256,
            context_length=request.context_length,
            forecast_horizon=request.forecast_horizon,
            sampling_seed=request.sampling_seed,
            temperature=request.temperature,
            top_p=request.top_p,
            sample_count=request.sample_count,
        )


class CachedInference(FrozenModel):
    schema_version: Literal['sentinel-inference-cache-v1']
    key: InferenceCacheKey
    response_ref: ArtifactRef
    path_sha256: Sha256


class InferenceCache:
    def __init__(self, root: Path | str) -> None:
        configured = Path(root)
        if configured.exists() and configured.is_symlink():
            raise ValueError('inference cache root cannot be a symlink')
        configured.mkdir(parents=True, exist_ok=True)
        self.root = configured.resolve(strict=True)
        self.artifact_store = LocalArtifactStore(self.root / 'artifacts')
        self.index_root = self.root / 'index'

    def put(self, request: ForecastRequest, response: ForecastResponse) -> CachedInference:
        key = InferenceCacheKey.from_request(request)
        path_sha256 = self._validate_response(request, response)
        response_ref = self.artifact_store.put_bytes(
            canonical_json_bytes(response.model_dump(mode='json')),
            media_type='application/json',
        )
        descriptor = CachedInference(
            schema_version='sentinel-inference-cache-v1',
            key=key,
            response_ref=response_ref,
            path_sha256=path_sha256,
        )
        descriptor_bytes = canonical_json_bytes(
            descriptor.model_dump(mode='json', exclude_computed_fields=True)
        )
        index_path = self._index_path(key.canonical_sha256)
        if index_path.exists():
            if index_path.is_symlink() or index_path.read_bytes() != descriptor_bytes:
                raise ArtifactConflictError('inference cache key has conflicting content')
        else:
            atomic_write_bytes(index_path, descriptor_bytes)
        return descriptor

    def get_verified(self, request: ForecastRequest) -> ForecastResponse | None:
        expected_key = InferenceCacheKey.from_request(request)
        index_path = self._index_path(expected_key.canonical_sha256)
        if not index_path.exists():
            return None
        if index_path.is_symlink() or not index_path.is_file():
            raise ArtifactIntegrityError('inference cache index is not a regular file')
        try:
            descriptor = CachedInference.model_validate_json(index_path.read_bytes())
        except Exception as error:
            raise ArtifactIntegrityError('inference cache descriptor is invalid') from error
        if descriptor.key.canonical_sha256 != expected_key.canonical_sha256:
            raise ArtifactIntegrityError('inference cache request identity mismatch')
        response_bytes = self.artifact_store.read_bytes(descriptor.response_ref)
        try:
            response = ForecastResponse.model_validate_json(response_bytes)
        except Exception as error:
            raise ArtifactIntegrityError('cached inference response is invalid') from error
        observed_path_sha256 = self._validate_response(request, response)
        if observed_path_sha256 != descriptor.path_sha256:
            raise ArtifactIntegrityError('cached forecast path hash mismatch')
        return response

    def _index_path(self, key_sha256: str) -> Path:
        if len(key_sha256) != 64 or any(char not in '0123456789abcdef' for char in key_sha256):
            raise ValueError('cache key must be a lowercase SHA-256')
        return self.index_root / key_sha256[:2] / f'{key_sha256}.json'

    @staticmethod
    def _validate_response(request: ForecastRequest, response: ForecastResponse) -> str:
        if response.failure is not None or len(response.generated_paths) != 1:
            raise ValueError('only successful individual inference paths are cacheable')
        accepted_checkpoints = {
            request.model_revision,
            f'{request.model_repository}@{request.model_revision}',
        }
        if response.checkpoint_id not in accepted_checkpoints:
            raise ValueError('response checkpoint does not match request model revision')
        path = response.generated_paths[0]
        sessions = tuple(row.session for row in path.observations)
        if sessions != request.forecast_sessions:
            raise ValueError('response sessions do not match request sessions')
        observed_sha256 = sha256_bytes(
            canonical_json_bytes(
                [row.model_dump(mode='json') for row in path.observations]
            )
        )
        if observed_sha256 != path.canonical_sha256:
            raise ArtifactIntegrityError('forecast path canonical hash mismatch')
        return observed_sha256
