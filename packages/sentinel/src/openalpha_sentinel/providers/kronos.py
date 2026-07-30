from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from openalpha_research.artifacts import Sha256
from pydantic import Field

from ..contracts import (
    ForecastFailure,
    ForecastPath,
    ForecastRequest,
    ForecastResponse,
    FrozenModel,
    OHLCVObservation,
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


class KronosProviderError(RuntimeError):
    """Failure of the isolated inference process or its typed protocol."""


class DownloadedFile(FrozenModel):
    repository: str = Field(min_length=1, max_length=256)
    revision: str = Field(min_length=40, max_length=64)
    relative_path: str = Field(min_length=1, max_length=1_024)
    sha256: Sha256
    size_bytes: int = Field(ge=0)


class InferenceEnvironment(FrozenModel):
    python_version: str = Field(min_length=1, max_length=64)
    torch_version: str = Field(min_length=1, max_length=64)
    numpy_version: str = Field(min_length=1, max_length=64)
    pandas_version: str = Field(min_length=1, max_length=64)
    operating_system: str = Field(min_length=1, max_length=512)
    device: str = Field(min_length=1, max_length=128)
    cache_path: str = Field(min_length=1, max_length=1_024)
    cache_size_bytes: int = Field(ge=0)
    downloaded_files: tuple[DownloadedFile, ...]
    official_predictor_derives_amount: bool


class InferenceBatchResult(FrozenModel):
    environment: InferenceEnvironment
    responses: tuple[ForecastResponse, ...]


class KronosSubprocessClient:
    def __init__(
        self,
        *,
        python_executable: Path,
        worker_script: Path,
        source_path: Path,
        cache_path: Path,
        runner: Runner = subprocess.run,
        timeout_seconds: float = 3_600.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._python_executable = python_executable
        self._worker_script = worker_script
        self._source_path = source_path
        self._cache_path = cache_path
        self._runner = runner
        self._timeout_seconds = timeout_seconds

    def forecast_many(
        self,
        requests: tuple[ForecastRequest, ...],
    ) -> InferenceBatchResult:
        if not requests:
            raise ValueError("at least one forecast request is required")
        request_payload = {
            "schema_version": "sentinel-kronos-worker-v0",
            "requests": [request.model_dump(mode="json") for request in requests],
        }
        command = [
            str(self._python_executable),
            str(self._worker_script),
            "--source-path",
            str(self._source_path),
            "--cache-path",
            str(self._cache_path),
        ]
        try:
            completed = self._runner(
                command,
                input=json.dumps(
                    request_payload,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                text=True,
                capture_output=True,
                check=False,
                timeout=self._timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise KronosProviderError(f"Kronos worker failed to execute: {error}") from error
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise KronosProviderError(
                f"Kronos worker exited {completed.returncode}: {stderr[:2_000]}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise KronosProviderError("Kronos worker returned invalid JSON") from error
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise KronosProviderError("Kronos worker batch did not complete")
        environment_payload = payload.get("environment")
        response_payloads = payload.get("responses")
        if not isinstance(environment_payload, dict) or not isinstance(response_payloads, list):
            raise KronosProviderError("Kronos worker response is missing typed batch fields")
        if len(response_payloads) != len(requests):
            raise KronosProviderError("Kronos worker response count mismatch")
        environment = InferenceEnvironment.model_validate_json(
            json.dumps(environment_payload, allow_nan=False)
        )
        responses = tuple(
            self._parse_response(request, response_payload)
            for request, response_payload in zip(requests, response_payloads)
        )
        return InferenceBatchResult(environment=environment, responses=responses)

    @staticmethod
    def _parse_response(
        request: ForecastRequest,
        payload: object,
    ) -> ForecastResponse:
        if not isinstance(payload, dict):
            raise KronosProviderError("Kronos worker response entry must be an object")
        provider_id, checkpoint_id, request_id, duration = _shared_response_fields(payload)
        if payload.get("status") == "failure":
            failure_payload = payload.get("failure")
            if not isinstance(failure_payload, dict):
                raise KronosProviderError("failed Kronos response is missing failure details")
            return ForecastResponse(
                provider_id=provider_id,
                checkpoint_id=checkpoint_id,
                request_id=request_id,
                inference_duration_ms=duration,
                generated_paths=(),
                failure=ForecastFailure.model_validate_json(
                    json.dumps(failure_payload, allow_nan=False)
                ),
            )
        if payload.get("status") != "success":
            raise KronosProviderError("unknown Kronos worker response status")
        rows = payload.get("path")
        if not isinstance(rows, list) or len(rows) != 5:
            raise KronosProviderError("successful Kronos response must contain five rows")
        observations = tuple(
            OHLCVObservation.model_validate_json(json.dumps(row, allow_nan=False))
            for row in rows
        )
        if tuple(row.session for row in observations) != request.forecast_sessions:
            raise KronosProviderError("Kronos output sessions do not match request")
        canonical_payload = [row.model_dump(mode="json") for row in observations]
        canonical_bytes = json.dumps(
            canonical_payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        path = ForecastPath(
            path_id=(
                f"kronos-{request.context_length}-{request.sampling_seed}-"
                f"{request_id}"
            ),
            observations=observations,
            canonical_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
        )
        return ForecastResponse(
            provider_id=provider_id,
            checkpoint_id=checkpoint_id,
            request_id=request_id,
            inference_duration_ms=duration,
            generated_paths=(path,),
            failure=None,
        )


def _shared_response_fields(payload: Mapping[str, object]) -> tuple[str, str, str, float]:
    required = ("provider_id", "checkpoint_id", "request_id", "inference_duration_ms")
    if any(name not in payload for name in required):
        raise KronosProviderError("Kronos response is missing provenance fields")
    provider_id = payload["provider_id"]
    checkpoint_id = payload["checkpoint_id"]
    request_id = payload["request_id"]
    duration = payload["inference_duration_ms"]
    if not all(isinstance(value, str) for value in (provider_id, checkpoint_id, request_id)):
        raise KronosProviderError("Kronos response provenance fields must be strings")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        raise KronosProviderError("Kronos response duration must be numeric")
    assert isinstance(provider_id, str)
    assert isinstance(checkpoint_id, str)
    assert isinstance(request_id, str)
    return provider_id, checkpoint_id, request_id, float(duration)
