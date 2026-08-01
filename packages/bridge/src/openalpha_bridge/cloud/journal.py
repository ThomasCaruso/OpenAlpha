"""Append-only run journal in cloud object storage.

Each entry is written to its own immutable key and chains to the previous entry
by hash. A conflicting write means another worker advanced the run, which the
lease is meant to prevent and which this layer detects regardless.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.identity import canonical_sha256
from ..phase2.states import EvidenceClass, Phase2State
from .objectstore import ObjectStore, get_model, put_json, run_prefix
from .redaction import redact

__all__ = ["CloudJournal", "CloudJournalEntry", "journal_key"]


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


def journal_key(run_id: str, evidence_class: EvidenceClass, sequence: int) -> str:
    return f"{run_prefix(run_id, evidence_class)}/journal/{sequence:06d}.json"


class CloudJournalEntry(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.cloud_journal.v1"] = (
        "openalpha.bridge.phase2.cloud_journal.v1"
    )
    sequence: int = Field(ge=1)
    run_id: str
    experiment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_class: EvidenceClass
    state: Phase2State
    stage: str | None = None
    occurred_at: datetime
    reason: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    prior_entry_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @property
    def entry_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class CloudJournal:
    """Reader and appender for a single run's journal."""

    def __init__(
        self,
        store: ObjectStore,
        *,
        run_id: str,
        experiment_hash: str,
        evidence_class: EvidenceClass,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._experiment_hash = experiment_hash
        self._evidence_class = evidence_class

    def entries(self) -> tuple[CloudJournalEntry, ...]:
        prefix = f"{run_prefix(self._run_id, self._evidence_class)}/journal/"
        keys = self._store.list_keys(prefix)
        return tuple(
            get_model(self._store, key, CloudJournalEntry) for key in keys
        )

    def latest(self) -> CloudJournalEntry | None:
        found = self.entries()
        return found[-1] if found else None

    def current_state(self) -> Phase2State:
        latest = self.latest()
        return latest.state if latest else Phase2State.CREATED

    def append(
        self,
        state: Phase2State,
        *,
        occurred_at: datetime,
        stage: str | None = None,
        reason: str | None = None,
        progress: dict[str, Any] | None = None,
    ) -> CloudJournalEntry:
        previous = self.latest()
        sequence = (previous.sequence + 1) if previous else 1

        entry = CloudJournalEntry(
            sequence=sequence,
            run_id=self._run_id,
            experiment_hash=self._experiment_hash,
            evidence_class=self._evidence_class,
            state=state,
            stage=stage,
            occurred_at=occurred_at,
            reason=redact(reason) if reason else None,
            progress=redact(progress or {}),
            prior_entry_sha256=previous.entry_sha256 if previous else None,
        )

        key = journal_key(self._run_id, self._evidence_class, sequence)
        try:
            put_json(
                self._store,
                key,
                entry.model_dump(mode="json"),
                schema_version=entry.schema_version,
                run_id=self._run_id,
                experiment_hash=self._experiment_hash,
                evidence_class=self._evidence_class,
                immutable=True,
                prior_journal_sha256=entry.prior_entry_sha256,
                created_at=occurred_at,
            )
        except BridgeTransformError as error:
            raise _fail(
                "CONCURRENT_JOURNAL_WRITE",
                (
                    f"journal entry {sequence} for run {self._run_id} already exists; "
                    "another worker advanced this run"
                ),
            ) from error
        return entry

    def verify_chain(self) -> bool:
        """Every entry must reference its predecessor's hash."""
        previous: CloudJournalEntry | None = None
        for entry in self.entries():
            expected = previous.entry_sha256 if previous else None
            if entry.prior_entry_sha256 != expected:
                return False
            if previous and entry.sequence != previous.sequence + 1:
                return False
            previous = entry
        return True
