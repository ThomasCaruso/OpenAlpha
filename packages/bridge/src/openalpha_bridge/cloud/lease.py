"""Exclusive run lease.

A worker must hold a lease before starting or resuming a run, so a cloud retry
or a duplicate spawn cannot advance the same scientific run twice. The lease is
an object-store key acquired with a conditional create; expiry allows recovery
after an interrupted worker dies without releasing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.states import EvidenceClass
from .objectstore import ObjectStore, get_model, put_json, run_prefix

__all__ = ["DEFAULT_LEASE_SECONDS", "RunLease", "acquire_lease", "lease_key", "release_lease"]

DEFAULT_LEASE_SECONDS = 3600


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


def lease_key(run_id: str, evidence_class: EvidenceClass) -> str:
    return f"{run_prefix(run_id, evidence_class)}/identity/lease.json"


class RunLease(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.lease.v1"] = (
        "openalpha.bridge.phase2.lease.v1"
    )
    run_id: str = Field(min_length=1)
    holder: str = Field(min_length=1)
    acquired_at: datetime
    expires_at: datetime
    cloud_execution_id: str | None = None

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


def acquire_lease(
    store: ObjectStore,
    *,
    run_id: str,
    experiment_hash: str,
    evidence_class: EvidenceClass,
    holder: str,
    now: datetime,
    duration_seconds: int = DEFAULT_LEASE_SECONDS,
    cloud_execution_id: str | None = None,
) -> RunLease:
    """Acquire, or take over an expired lease. Fails when another holder is live."""
    key = lease_key(run_id, evidence_class)
    lease = RunLease(
        run_id=run_id,
        holder=holder,
        acquired_at=now,
        expires_at=now + timedelta(seconds=duration_seconds),
        cloud_execution_id=cloud_execution_id,
    )

    if store.exists(key):
        existing = get_model(store, key, RunLease)
        if existing.holder == holder:
            # Re-entrant: the same worker renews rather than conflicts.
            put_json(
                store,
                key,
                lease.model_dump(mode="json"),
                schema_version=lease.schema_version,
                run_id=run_id,
                experiment_hash=experiment_hash,
                evidence_class=evidence_class,
                immutable=False,
                created_at=now,
            )
            return lease
        if not existing.is_expired(now):
            raise _fail(
                "RUN_LEASE_HELD",
                (
                    f"run {run_id} is leased by {existing.holder} until "
                    f"{existing.expires_at.isoformat()}"
                ),
            )
        put_json(
            store,
            key,
            lease.model_dump(mode="json"),
            schema_version=lease.schema_version,
            run_id=run_id,
            experiment_hash=experiment_hash,
            evidence_class=evidence_class,
            immutable=False,
            created_at=now,
        )
        return lease

    try:
        put_json(
            store,
            key,
            lease.model_dump(mode="json"),
            schema_version=lease.schema_version,
            run_id=run_id,
            experiment_hash=experiment_hash,
            evidence_class=evidence_class,
            immutable=True,
            created_at=now,
        )
    except BridgeTransformError as error:
        # Lost a race between exists() and the conditional create.
        raise _fail("RUN_LEASE_HELD", f"run {run_id} was leased concurrently") from error
    return lease


def release_lease(
    store: ObjectStore,
    *,
    run_id: str,
    evidence_class: EvidenceClass,
    holder: str,
) -> None:
    """Release only a lease this holder owns."""
    key = lease_key(run_id, evidence_class)
    if not store.exists(key):
        return
    existing = get_model(store, key, RunLease)
    if existing.holder != holder:
        raise _fail(
            "RUN_LEASE_NOT_HELD",
            f"{holder} cannot release a lease held by {existing.holder}",
        )
    store.delete(key)


def current_lease(
    store: ObjectStore, *, run_id: str, evidence_class: EvidenceClass
) -> RunLease | None:
    key = lease_key(run_id, evidence_class)
    if not store.exists(key):
        return None
    return get_model(store, key, RunLease)
