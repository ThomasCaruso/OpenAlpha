"""External feature cache for Phase 2 examples.

Cached shards live outside Git. Writes are atomic with interrupted-write
recovery, shard names are deterministic, contents are hashed, and partitions are
physically separated so test shards cannot be loaded before the test gate opens.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..windowing import CONTEXT_PREFIX_LENGTH, EXAMPLE_LENGTH, SCORED_SUFFIX_LENGTH, Partition
from .identity import canonical_json, canonical_sha256

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "MAXIMUM_CACHE_BYTES",
    "CacheIdentity",
    "CachedExample",
    "FeatureCache",
    "ShardRef",
]

CACHE_SCHEMA_VERSION = "openalpha.bridge.phase2.cache.v2"
#: Hard cap from experiment.yaml maximum_temporary_cache_bytes.
MAXIMUM_CACHE_BYTES = 10_737_418_240

_TEMP_SUFFIX = ".tmp"


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class CacheIdentity(BaseModel):
    """Everything a shard must bind to be reusable evidence.

    A shard is only interchangeable with another if every one of these matches.
    Anything less and a stale shard from a different experiment, amendment,
    source revision, or provider could silently be reused.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.cache_identity.v1"] = (
        "openalpha.bridge.phase2.cache_identity.v1"
    )
    run_id: str = Field(min_length=1)
    experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    amendment_sha256: tuple[str, ...]
    score_mask_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(min_length=7)
    evidence_class: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    provider_client_version: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    interval: str = Field(min_length=1)
    partition: Partition
    prefix_start: str
    prefix_end: str
    target_start: str
    target_end: str
    candle_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    official_source_repository: str
    official_source_revision: str
    official_source_file_sha256: dict[str, str]
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_config_sha256: str | None
    tokenizer_weights_sha256: str | None
    frozen_parameter_sha256: str | None
    feature_schema_version: str = CACHE_SCHEMA_VERSION
    representation_version: str
    tensor_specification: dict[str, str]

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class CachedExample(BaseModel):
    """One cached example: only the locked inputs Bridge needs."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal["openalpha.bridge.phase2.cache.v2"] = CACHE_SCHEMA_VERSION
    identity: CacheIdentity
    sequence_id: str = Field(min_length=16, max_length=16)
    symbol: str
    interval: str
    partition: Partition
    coarse_ids: NDArray[np.int64]
    fine_ids: NDArray[np.int64]
    bipolar_latent: NDArray[np.float32]
    frozen_hidden: NDArray[np.float32]
    causal_features: NDArray[np.float32]
    constrained_targets: NDArray[np.float32]
    initial_previous_close: float
    volume_mask: NDArray[np.bool_]
    sequence_mask: NDArray[np.bool_]
    score_mask: NDArray[np.bool_]
    source_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def validate_shapes(self) -> None:
        checks: tuple[tuple[str, tuple[int, ...]], ...] = (
            ("coarse_ids", (EXAMPLE_LENGTH,)),
            ("fine_ids", (EXAMPLE_LENGTH,)),
            ("bipolar_latent", (EXAMPLE_LENGTH, 20)),
            ("frozen_hidden", (EXAMPLE_LENGTH, 256)),
            ("causal_features", (EXAMPLE_LENGTH, 13)),
            ("constrained_targets", (SCORED_SUFFIX_LENGTH, 5)),
            ("volume_mask", (EXAMPLE_LENGTH,)),
            ("sequence_mask", (EXAMPLE_LENGTH,)),
            ("score_mask", (EXAMPLE_LENGTH,)),
        )
        for name, expected in checks:
            observed = tuple(getattr(self, name).shape)
            if observed != expected:
                raise _fail(
                    "INVALID_CACHED_SHAPE",
                    f"{name} must have shape {expected}, got {observed}",
                    field=name,
                )
        mask = self.score_mask
        if bool(mask[:CONTEXT_PREFIX_LENGTH].any()) or not bool(mask[CONTEXT_PREFIX_LENGTH:].all()):
            raise _fail("INVALID_CACHED_SCORE_MASK", "cached score mask is not the locked 448/64")

    def to_arrays(self) -> dict[str, Any]:
        return {
            "coarse_ids": self.coarse_ids,
            "fine_ids": self.fine_ids,
            "bipolar_latent": self.bipolar_latent,
            "frozen_hidden": self.frozen_hidden,
            "causal_features": self.causal_features,
            "constrained_targets": self.constrained_targets,
            "volume_mask": self.volume_mask,
            "sequence_mask": self.sequence_mask,
            "score_mask": self.score_mask,
        }

    @property
    def provenance(self) -> dict[str, str | float]:
        return {
            "schema_version": self.schema_version,
            "sequence_id": self.sequence_id,
            "symbol": self.symbol,
            "interval": self.interval,
            "partition": self.partition.value,
            "initial_previous_close": self.initial_previous_close,
            "source_data_sha256": self.source_data_sha256,
            "identity_sha256": self.identity.identity_sha256,
        }


class ShardRef(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.shard.v1"] = (
        "openalpha.bridge.phase2.shard.v1"
    )
    partition: Partition
    sequence_id: str
    relative_path: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class FeatureCache:
    """Partition-separated, content-addressed shard store outside Git."""

    def __init__(
        self,
        root: Path | str,
        *,
        maximum_bytes: int = MAXIMUM_CACHE_BYTES,
        test_gate_open: bool = False,
        repository_root: Path | None = None,
    ) -> None:
        if maximum_bytes > MAXIMUM_CACHE_BYTES:
            raise _fail(
                "CACHE_CAP_EXCEEDS_LOCK",
                f"maximum_bytes {maximum_bytes} exceeds the locked cap {MAXIMUM_CACHE_BYTES}",
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # The cache must not live inside the project worktree. Unrelated Git
        # repositories elsewhere on the filesystem are irrelevant, so only the
        # supplied repository root is consulted.
        if (self.root / ".git").exists() or (
            repository_root is not None and _inside_directory(self.root, repository_root)
        ):
            raise _fail(
                "CACHE_INSIDE_GIT",
                f"feature cache root is inside the repository worktree: {self.root}",
            )
        self.maximum_bytes = maximum_bytes
        self._test_gate_open = test_gate_open

    def open_test_gate(self) -> None:
        self._test_gate_open = True

    def _partition_dir(self, partition: Partition) -> Path:
        return self.root / partition.value

    def shard_path(self, partition: Partition, sequence_id: str) -> Path:
        return self._partition_dir(partition) / f"{sequence_id}.npz"

    def _assert_readable(self, partition: Partition) -> None:
        if partition is Partition.RECONSTRUCTION_TEST and not self._test_gate_open:
            raise _fail(
                "TEST_SHARD_LOAD_BEFORE_GATE",
                "reconstruction-test shards cannot be loaded before the test gate opens",
            )

    def _assert_writable(self, partition: Partition) -> None:
        """Materializing test features is itself an access of the partition."""
        if partition is Partition.RECONSTRUCTION_TEST and not self._test_gate_open:
            raise _fail(
                "TEST_SHARD_WRITE_BEFORE_GATE",
                "reconstruction-test shards cannot be written before the test gate opens",
            )

    def current_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.rglob("*.npz") if path.is_file())

    @staticmethod
    def _assert_identity_consistent(example: CachedExample) -> None:
        """The bound identity must describe this very example."""
        identity = example.identity
        mismatches: list[str] = []
        if identity.symbol != example.symbol:
            mismatches.append("symbol")
        if identity.interval != example.interval:
            mismatches.append("interval")
        if identity.partition is not example.partition:
            mismatches.append("partition")
        if identity.candle_data_sha256 != example.source_data_sha256:
            mismatches.append("candle_data_sha256")
        if identity.feature_schema_version != CACHE_SCHEMA_VERSION:
            mismatches.append("feature_schema_version")
        if mismatches:
            raise _fail(
                "SHARD_IDENTITY_MISMATCH",
                f"cache identity does not describe this example: {', '.join(mismatches)}",
            )

    def write(self, example: CachedExample) -> ShardRef:
        example.validate_shapes()
        self._assert_writable(example.partition)
        self._assert_identity_consistent(example)
        target = self.shard_path(example.partition, example.sequence_id)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = _encode(example)

        # An existing path is reusable only when its complete canonical content
        # is byte-identical. Anything else is a collision, never an overwrite.
        if target.exists():
            existing = target.read_bytes()
            if existing == payload:
                return ShardRef(
                    partition=example.partition,
                    sequence_id=example.sequence_id,
                    relative_path=f"{example.partition.value}/{target.name}",
                    content_sha256=hashlib.sha256(existing).hexdigest(),
                    size_bytes=len(existing),
                )
            raise _fail(
                "SHARD_IDENTITY_COLLISION",
                (
                    f"{example.sequence_id} already exists with different content or "
                    "identity; a shard is never overwritten"
                ),
                field=example.sequence_id,
            )
        if self.current_bytes() + len(payload) > self.maximum_bytes:
            raise _fail(
                "CACHE_SIZE_LIMIT_EXCEEDED",
                f"writing {example.sequence_id} would exceed {self.maximum_bytes} bytes",
            )

        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}{_TEMP_SUFFIX}")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

        return ShardRef(
            partition=example.partition,
            sequence_id=example.sequence_id,
            relative_path=f"{example.partition.value}/{target.name}",
            content_sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def read(self, ref: ShardRef) -> CachedExample:
        self._assert_readable(ref.partition)
        path = self.root / ref.relative_path
        if not path.is_file():
            raise _fail("MISSING_SHARD", f"cache shard not found: {ref.relative_path}")
        payload = path.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != ref.content_sha256:
            raise _fail(
                "SHARD_HASH_MISMATCH",
                f"{ref.relative_path} expected {ref.content_sha256}, observed {observed}",
            )
        example = _decode(payload)
        # A decoded shard is only usable if it still satisfies the locked shape
        # contract and its bound identity still describes it.
        example.validate_shapes()
        self._assert_identity_consistent(example)
        return example

    def recover_interrupted_writes(self) -> tuple[str, ...]:
        """Delete partial temporaries left by an interrupted run."""
        removed: list[str] = []
        for path in self.root.rglob(f"*{_TEMP_SUFFIX}"):
            if path.is_file():
                path.unlink()
                removed.append(path.name)
        return tuple(sorted(removed))

    def cleanup(self, partition: Partition | None = None) -> int:
        """Remove cached shards. Returns the number of files removed."""
        targets: Iterator[Path]
        targets = (
            self._partition_dir(partition).rglob("*.npz")
            if partition is not None
            else self.root.rglob("*.npz")
        )
        count = 0
        for path in list(targets):
            if path.is_file():
                path.unlink()
                count += 1
        return count


def _inside_directory(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _encode(example: CachedExample) -> bytes:
    import io

    buffer = io.BytesIO()
    np.savez(buffer, **example.to_arrays())
    arrays = buffer.getvalue()
    header = canonical_json(
        {
            **example.provenance,
            "identity": example.identity.model_dump(mode="json"),
            "arrays_sha256": hashlib.sha256(arrays).hexdigest(),
        }
    )
    return len(header).to_bytes(8, "big") + header + arrays


def _decode(payload: bytes) -> CachedExample:
    import io

    if len(payload) < 8:
        raise _fail("CORRUPT_SHARD", "shard is shorter than its header length prefix")
    header_length = int.from_bytes(payload[:8], "big")
    if header_length <= 0 or 8 + header_length > len(payload):
        raise _fail("CORRUPT_SHARD", "shard header length is out of range")

    header = json.loads(payload[8 : 8 + header_length].decode("utf-8"))
    arrays_bytes = payload[8 + header_length :]
    if hashlib.sha256(arrays_bytes).hexdigest() != header["arrays_sha256"]:
        raise _fail("CORRUPT_SHARD", "shard array payload does not match its recorded hash")

    with np.load(io.BytesIO(arrays_bytes)) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}

    return CachedExample(
        identity=CacheIdentity.model_validate_json(json.dumps(header["identity"])),
        sequence_id=header["sequence_id"],
        symbol=header["symbol"],
        interval=header["interval"],
        partition=Partition(header["partition"]),
        initial_previous_close=float(header["initial_previous_close"]),
        source_data_sha256=header["source_data_sha256"],
        coarse_ids=arrays["coarse_ids"],
        fine_ids=arrays["fine_ids"],
        bipolar_latent=arrays["bipolar_latent"],
        frozen_hidden=arrays["frozen_hidden"],
        causal_features=arrays["causal_features"],
        constrained_targets=arrays["constrained_targets"],
        volume_mask=arrays["volume_mask"],
        sequence_mask=arrays["sequence_mask"],
        score_mask=arrays["score_mask"],
    )


def manifest_sha256(refs: tuple[ShardRef, ...]) -> str:
    return canonical_sha256([ref.model_dump(mode="json") for ref in sorted(refs, key=lambda r: r.sequence_id)])
