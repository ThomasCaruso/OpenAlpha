from __future__ import annotations

import hashlib
import importlib
import json
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .failures import FailureCategory, ResearchFailure, ResearchFailureError
from .identity import canonical_json

__all__ = [
    "InMemoryObjectStore",
    "ObjectMetadata",
    "ObjectStore",
    "S3CompatibleObjectStore",
    "StoredObject",
    "get_json",
    "get_model",
    "put_json",
]

def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def _optional_module(name: str, *, extra: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise _fail(
            "OPTIONAL_DEPENDENCY_MISSING",
            f"install openalpha-research-core[{extra}] to use {name}",
            field=name,
        ) from error


class ObjectMetadata(BaseModel):
    """Caller-owned provenance carried with a stored object."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    experiment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_journal_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    evidence_class: str = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class StoredObject(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    key: str
    body: bytes
    metadata: ObjectMetadata


class ObjectStore(Protocol):
    def put_immutable(
        self,
        key: str,
        body: bytes,
        metadata: ObjectMetadata,
    ) -> ObjectMetadata: ...

    def put_overwrite(
        self,
        key: str,
        body: bytes,
        metadata: ObjectMetadata,
    ) -> ObjectMetadata: ...

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
    evidence_class: str,
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


def _verify_body_metadata(key: str, body: bytes, metadata: ObjectMetadata) -> None:
    observed = hashlib.sha256(body).hexdigest()
    if observed != metadata.content_sha256:
        raise _fail("OBJECT_HASH_MISMATCH", f"content hash mismatch at {key}", field=key)


class InMemoryObjectStore:
    """Thread-safe object store for tests and local execution."""

    def __init__(self) -> None:
        self._objects: dict[str, StoredObject] = {}
        self._lock = threading.Lock()

    def put_immutable(
        self,
        key: str,
        body: bytes,
        metadata: ObjectMetadata,
    ) -> ObjectMetadata:
        with self._lock:
            if key in self._objects:
                raise _fail(
                    "OBJECT_ALREADY_EXISTS",
                    f"immutable object already exists and cannot be replaced: {key}",
                    field=key,
                )
            _verify_body_metadata(key, body, metadata)
            self._objects[key] = StoredObject(key=key, body=body, metadata=metadata)
        return metadata

    def put_overwrite(
        self,
        key: str,
        body: bytes,
        metadata: ObjectMetadata,
    ) -> ObjectMetadata:
        _verify_body_metadata(key, body, metadata)
        with self._lock:
            self._objects[key] = StoredObject(key=key, body=body, metadata=metadata)
        return metadata

    def get(self, key: str) -> StoredObject:
        with self._lock:
            stored = self._objects.get(key)
        if stored is None:
            raise _fail("OBJECT_NOT_FOUND", f"no object at {key}", field=key)
        _verify_body_metadata(key, stored.body, stored.metadata)
        return stored

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._objects

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(key for key in self._objects if key.startswith(prefix)))

    def delete(self, key: str) -> None:
        with self._lock:
            self._objects.pop(key, None)


class S3CompatibleObjectStore:
    """S3-compatible storage with lazy SDK loading and conditional writes."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str = "auto",
    ) -> None:
        self._bucket = bucket
        self._endpoint_url = endpoint_url
        self._region_name = region_name
        self._client: Any | None = None

    @property
    def bucket(self) -> str:
        return self._bucket

    def _boto(self) -> Any:
        if self._client is None:
            boto3 = _optional_module("boto3", extra="s3")
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
            "evidence-class": metadata.evidence_class,
        }

    def put_immutable(
        self,
        key: str,
        body: bytes,
        metadata: ObjectMetadata,
    ) -> ObjectMetadata:
        _verify_body_metadata(key, body, metadata)
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

    def put_overwrite(
        self,
        key: str,
        body: bytes,
        metadata: ObjectMetadata,
    ) -> ObjectMetadata:
        _verify_body_metadata(key, body, metadata)
        self._boto().put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            Metadata=self._headers(metadata),
        )
        return metadata

    def get(self, key: str) -> StoredObject:
        client = self._boto()
        try:
            response = client.get_object(Bucket=self._bucket, Key=key)
            body = response["Body"].read()
        except Exception as error:
            if _is_not_found(error):
                raise _fail("OBJECT_NOT_FOUND", f"no object at {key}", field=key) from error
            raise _request_failure("read", key, error) from error
        raw = response.get("Metadata", {})
        required = (
            "schema-version",
            "run-id",
            "experiment-hash",
            "content-sha256",
            "created-at",
            "evidence-class",
        )
        if not isinstance(raw, dict) or any(not raw.get(name) for name in required):
            raise _fail(
                "OBJECT_METADATA_INVALID",
                f"required provenance metadata is missing at {key}",
                field=key,
            )
        try:
            metadata = ObjectMetadata(
                schema_version=raw["schema-version"],
                run_id=raw["run-id"],
                experiment_hash=raw["experiment-hash"],
                content_sha256=raw["content-sha256"],
                prior_journal_sha256=raw.get("prior-journal-sha256") or None,
                created_at=datetime.fromisoformat(raw["created-at"]),
                evidence_class=raw["evidence-class"],
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise _fail(
                "OBJECT_METADATA_INVALID",
                f"provenance metadata is invalid at {key}",
                field=key,
            ) from error
        _verify_body_metadata(key, body, metadata)
        return StoredObject(key=key, body=body, metadata=metadata)

    def exists(self, key: str) -> bool:
        client = self._boto()
        try:
            client.head_object(Bucket=self._bucket, Key=key)
        except Exception as error:
            if _is_not_found(error):
                return False
            raise _request_failure("metadata read", key, error) from error
        return True

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        keys: list[str] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            arguments: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if token:
                arguments["ContinuationToken"] = token
            try:
                response = self._boto().list_objects_v2(**arguments)
            except Exception as error:
                raise _request_failure("list", prefix, error) from error
            contents = response.get("Contents", ())
            if not isinstance(contents, Sequence) or isinstance(contents, (str, bytes)):
                raise _fail(
                    "OBJECT_STORE_RESPONSE_INVALID",
                    f"object listing returned invalid contents for {prefix}",
                    field=prefix,
                )
            page_keys: list[str] = []
            for item in contents:
                if not isinstance(item, Mapping) or not isinstance(item.get("Key"), str):
                    raise _fail(
                        "OBJECT_STORE_RESPONSE_INVALID",
                        f"object listing returned an invalid key entry for {prefix}",
                        field=prefix,
                    )
                page_keys.append(item["Key"])
            keys.extend(page_keys)
            if not response.get("IsTruncated"):
                return tuple(sorted(keys))
            next_token = response.get("NextContinuationToken")
            if (
                not isinstance(next_token, str)
                or not next_token
                or next_token in seen_tokens
            ):
                raise _fail(
                    "OBJECT_STORE_PAGINATION_INVALID",
                    f"object listing returned an invalid continuation token for {prefix}",
                    field=prefix,
                )
            seen_tokens.add(next_token)
            token = next_token

    def delete(self, key: str) -> None:
        self._boto().delete_object(Bucket=self._bucket, Key=key)

    def signed_url(self, key: str, *, expires_seconds: int = 3600) -> str:
        return str(
            self._boto().generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_seconds,
            )
        )


def _is_precondition_failure(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"PreconditionFailed", "ConditionalRequestConflict"} or status == 412:
            return True
    return "PreconditionFailed" in type(error).__name__


def _is_not_found(error: Exception) -> bool:
    response = getattr(error, "response", None)
    code = ""
    status: object = None
    if isinstance(response, dict):
        raw_error = response.get("Error")
        if isinstance(raw_error, dict):
            code = str(raw_error.get("Code", ""))
        raw_metadata = response.get("ResponseMetadata")
        if isinstance(raw_metadata, dict):
            status = raw_metadata.get("HTTPStatusCode")
    return (
        code in {"NoSuchKey", "NotFound", "404"}
        or status == 404
        or type(error).__name__ in {"NoSuchKey", "NotFound"}
        or str(error).strip() == "404"
    )


def _request_failure(operation: str, key: str, error: Exception) -> ResearchFailureError:
    return _fail(
        "OBJECT_STORE_REQUEST_FAILED",
        f"object store {operation} failed for {key} ({type(error).__name__})",
        field=key,
    )


def put_json(
    store: ObjectStore,
    key: str,
    payload: Any,
    *,
    schema_version: str,
    run_id: str,
    experiment_hash: str,
    evidence_class: str,
    immutable: bool = True,
    prior_journal_sha256: str | None = None,
    created_at: datetime | None = None,
) -> ObjectMetadata:
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
    return json.loads(store.get(key).body.decode("utf-8"))


def get_model[ModelT: BaseModel](
    store: ObjectStore,
    key: str,
    model: type[ModelT],
) -> ModelT:
    return model.model_validate_json(store.get(key).body)
