"""S3-compatible object store for immutable Phase 2 run artifacts.

The default deployment target is Cloudflare R2: S3-compatible, no egress fees,
and no AWS account. ``boto3`` is resolved lazily so the base environment never
needs a cloud SDK.

Every write is conditional. ``put_immutable`` fails when the key already exists,
which is what makes the one-time test-opening record and the append-only journal
safe against two workers racing on the same run.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.identity import canonical_json
from ..phase2.optional import BridgeExtra, require_module
from ..phase2.states import EvidenceClass

__all__ = [
    "ARTIFACT_PREFIX",
    "InMemoryObjectStore",
    "ObjectMetadata",
    "ObjectStore",
    "S3CompatibleObjectStore",
    "experiment_prefix",
    "run_prefix",
]

ARTIFACT_PREFIX = "openalpha/bridge-phase2"
_SYNTHETIC_PREFIX = "openalpha-synthetic/bridge-phase2"


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def _base_prefix(evidence_class: EvidenceClass) -> str:
    return (
        ARTIFACT_PREFIX
        if evidence_class is EvidenceClass.REAL_PHASE2
        else _SYNTHETIC_PREFIX
    )


def run_prefix(run_id: str, evidence_class: EvidenceClass) -> str:
    """Key prefix for one run. Synthetic runs live under a separate root."""
    return f"{_base_prefix(evidence_class)}/runs/{run_id}"


def experiment_prefix(experiment_hash: str, evidence_class: EvidenceClass) -> str:
    return f"{_base_prefix(evidence_class)}/experiments/{experiment_hash}"


class ObjectMetadata(BaseModel):
    """Provenance carried by every stored object."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    experiment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_journal_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    evidence_class: EvidenceClass


class StoredObject(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False, arbitrary_types_allowed=True, extra="forbid", frozen=True, strict=True
    )

    key: str
    body: bytes
    metadata: ObjectMetadata


class ObjectStore(Protocol):
    """Minimal conditional-write object store."""

    def put_immutable(self, key: str, body: bytes, metadata: ObjectMetadata) -> ObjectMetadata:
        """Write only when ``key`` does not exist. Raises OBJECT_ALREADY_EXISTS otherwise."""
        ...

    def put_overwrite(self, key: str, body: bytes, metadata: ObjectMetadata) -> ObjectMetadata: ...

    def get(self, key: str) -> StoredObject: ...

    def exists(self, key: str) -> bool: ...

    def list_keys(self, prefix: str) -> tuple[str, ...]: ...

    def delete(self, key: str) -> None: ...


def _metadata_for(
    body: bytes,
    *,
    schema_version: str,
    run_id: str,
    experiment_hash: str,
    evidence_class: EvidenceClass,
    prior_journal_sha256: str | None = None,
    created_at: datetime | None = None,
) -> ObjectMetadata:
    return ObjectMetadata(
        schema_version=schema_version,
        run_id=run_id,
        experiment_hash=experiment_hash,
        content_sha256=hashlib.sha256(body).hexdigest(),
        prior_journal_sha256=prior_journal_sha256,
        created_at=created_at or datetime.now(UTC),
        evidence_class=evidence_class,
    )


class InMemoryObjectStore:
    """Thread-safe in-memory store used by tests and synthetic runs."""

    def __init__(self) -> None:
        self._objects: dict[str, StoredObject] = {}
        self._lock = threading.Lock()

    def put_immutable(self, key: str, body: bytes, metadata: ObjectMetadata) -> ObjectMetadata:
        with self._lock:
            if key in self._objects:
                raise _fail(
                    "OBJECT_ALREADY_EXISTS",
                    f"immutable object already exists and cannot be replaced: {key}",
                    field=key,
                )
            self._objects[key] = StoredObject(key=key, body=body, metadata=metadata)
        return metadata

    def put_overwrite(self, key: str, body: bytes, metadata: ObjectMetadata) -> ObjectMetadata:
        with self._lock:
            self._objects[key] = StoredObject(key=key, body=body, metadata=metadata)
        return metadata

    def get(self, key: str) -> StoredObject:
        with self._lock:
            stored = self._objects.get(key)
        if stored is None:
            raise _fail("OBJECT_NOT_FOUND", f"no object at {key}", field=key)
        observed = hashlib.sha256(stored.body).hexdigest()
        if observed != stored.metadata.content_sha256:
            raise _fail("OBJECT_HASH_MISMATCH", f"content hash mismatch at {key}", field=key)
        return stored

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._objects

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(k for k in self._objects if k.startswith(prefix)))

    def delete(self, key: str) -> None:
        with self._lock:
            self._objects.pop(key, None)


class S3CompatibleObjectStore:
    """Cloudflare R2 or any S3-compatible endpoint. Resolves ``boto3`` lazily.

    Immutability uses ``IfNoneMatch: "*"``, which R2 and S3 both honour by
    failing a conditional write when the key exists.
    """

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str = "auto",
        stage: str = "cloud-storage",
    ) -> None:
        self._bucket = bucket
        self._endpoint_url = endpoint_url
        self._region_name = region_name
        self._stage = stage
        self._client: Any | None = None

    def _boto(self) -> Any:
        if self._client is None:
            boto3 = require_module("boto3", stage=self._stage, extra=BridgeExtra.BRIDGE_CLOUD)
            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                region_name=self._region_name,
            )
        return self._client

    @staticmethod
    def _headers(metadata: ObjectMetadata) -> dict[str, str]:
        return {
            "schema-version": metadata.schema_version,
            "run-id": metadata.run_id,
            "experiment-hash": metadata.experiment_hash,
            "content-sha256": metadata.content_sha256,
            "prior-journal-sha256": metadata.prior_journal_sha256 or "",
            "created-at": metadata.created_at.isoformat(),
            "evidence-class": metadata.evidence_class.value,
        }

    def put_immutable(self, key: str, body: bytes, metadata: ObjectMetadata) -> ObjectMetadata:
        client = self._boto()
        try:
            client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                Metadata=self._headers(metadata),
                IfNoneMatch="*",
            )
        except Exception as error:
            if _is_precondition_failure(error):
                raise _fail(
                    "OBJECT_ALREADY_EXISTS",
                    f"immutable object already exists and cannot be replaced: {key}",
                    field=key,
                ) from error
            raise _fail(
                "OBJECT_STORE_WRITE_FAILED",
                f"conditional write failed for {key} ({type(error).__name__})",
                field=key,
            ) from error
        return metadata

    def put_overwrite(self, key: str, body: bytes, metadata: ObjectMetadata) -> ObjectMetadata:
        self._boto().put_object(
            Bucket=self._bucket, Key=key, Body=body, Metadata=self._headers(metadata)
        )
        return metadata

    def get(self, key: str) -> StoredObject:
        client = self._boto()
        try:
            response = client.get_object(Bucket=self._bucket, Key=key)
        except Exception as error:
            raise _fail("OBJECT_NOT_FOUND", f"no object at {key}", field=key) from error
        body = response["Body"].read()
        raw = response.get("Metadata", {})
        metadata = ObjectMetadata(
            schema_version=raw.get("schema-version", "unknown"),
            run_id=raw.get("run-id", "unknown"),
            experiment_hash=raw.get("experiment-hash", "0" * 64),
            content_sha256=raw.get("content-sha256", hashlib.sha256(body).hexdigest()),
            prior_journal_sha256=raw.get("prior-journal-sha256") or None,
            created_at=datetime.fromisoformat(
                raw.get("created-at", datetime.now(UTC).isoformat())
            ),
            evidence_class=EvidenceClass(raw.get("evidence-class", EvidenceClass.REAL_PHASE2.value)),
        )
        observed = hashlib.sha256(body).hexdigest()
        if observed != metadata.content_sha256:
            raise _fail("OBJECT_HASH_MISMATCH", f"content hash mismatch at {key}", field=key)
        return StoredObject(key=key, body=body, metadata=metadata)

    def exists(self, key: str) -> bool:
        try:
            self._boto().head_object(Bucket=self._bucket, Key=key)
        except Exception:  # noqa: BLE001
            return False
        return True

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        client = self._boto()
        keys: list[str] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = client.list_objects_v2(**kwargs)
            keys.extend(item["Key"] for item in response.get("Contents", ()))
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
        return tuple(sorted(keys))

    def delete(self, key: str) -> None:
        self._boto().delete_object(Bucket=self._bucket, Key=key)

    def signed_url(self, key: str, *, expires_seconds: int = 3600) -> str:
        return self._boto().generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )


def _is_precondition_failure(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"PreconditionFailed", "ConditionalRequestConflict"} or status == 412:
            return True
    return "PreconditionFailed" in type(error).__name__


def put_json(
    store: ObjectStore,
    key: str,
    payload: Any,
    *,
    schema_version: str,
    run_id: str,
    experiment_hash: str,
    evidence_class: EvidenceClass,
    immutable: bool = True,
    prior_journal_sha256: str | None = None,
    created_at: datetime | None = None,
) -> ObjectMetadata:
    """Serialize canonically and store, immutably by default."""
    body = canonical_json(payload)
    metadata = _metadata_for(
        body,
        schema_version=schema_version,
        run_id=run_id,
        experiment_hash=experiment_hash,
        evidence_class=evidence_class,
        prior_journal_sha256=prior_journal_sha256,
        created_at=created_at,
    )
    writer = store.put_immutable if immutable else store.put_overwrite
    return writer(key, body, metadata)


def get_json(store: ObjectStore, key: str) -> Any:
    import json

    return json.loads(store.get(key).body.decode("utf-8"))


def get_model[ModelT: BaseModel](store: ObjectStore, key: str, model: type[ModelT]) -> ModelT:
    """Read and validate in JSON mode.

    Models are declared ``strict=True``, which in Python mode rejects the ISO
    strings that JSON round-tripping produces. Validating the raw bytes keeps
    strictness while still parsing timestamps.
    """
    return model.model_validate_json(store.get(key).body)
