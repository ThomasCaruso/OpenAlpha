"""Cloud adapter for the Phase 2 execution engine.

The scientific engine in :mod:`openalpha_bridge.phase2` stays cloud-agnostic.
This package adds only what a managed deployment needs: an authenticated control
service, object-store persistence, an exclusive run lease, a cloud-native
one-time test gate, and secret redaction.

Importing this package pulls in no cloud SDK. ``boto3``, ``fastapi``, and
``modal`` are resolved lazily by the components that use them.
"""

from __future__ import annotations

from .auth import AuthContext, TokenAuthenticator, log_admin_action
from .identity import CloudRunIdentity, assert_idempotent_match
from .journal import CloudJournal, CloudJournalEntry
from .lease import RunLease, acquire_lease, current_lease, release_lease
from .models import (
    ArtifactManifestResponse,
    CancelRunRequest,
    CreateRunRequest,
    ExecutionMode,
    LogsResponse,
    ResumeRunRequest,
    RunCreatedResponse,
    RunStatusResponse,
)
from .objectstore import (
    InMemoryObjectStore,
    ObjectMetadata,
    ObjectStore,
    S3CompatibleObjectStore,
    experiment_prefix,
    run_prefix,
)
from .redaction import REDACTED, SecretRedactor, redact, register_secret
from .runner import CloudRunner, CloudRunResult, ResourceGuards
from .service import ComputeBackend, Phase2ControlService, ServiceConfig
from .testgate import (
    CloudTestOpeningRecord,
    is_cloud_test_partition_opened,
    open_cloud_test_partition,
)

__all__ = [
    "REDACTED",
    "ArtifactManifestResponse",
    "AuthContext",
    "CancelRunRequest",
    "CloudJournal",
    "CloudJournalEntry",
    "CloudRunIdentity",
    "CloudRunResult",
    "CloudRunner",
    "CloudTestOpeningRecord",
    "ComputeBackend",
    "CreateRunRequest",
    "ExecutionMode",
    "InMemoryObjectStore",
    "LogsResponse",
    "ObjectMetadata",
    "ObjectStore",
    "Phase2ControlService",
    "ResourceGuards",
    "ResumeRunRequest",
    "RunCreatedResponse",
    "RunLease",
    "RunStatusResponse",
    "S3CompatibleObjectStore",
    "SecretRedactor",
    "ServiceConfig",
    "TokenAuthenticator",
    "acquire_lease",
    "assert_idempotent_match",
    "current_lease",
    "experiment_prefix",
    "is_cloud_test_partition_opened",
    "log_admin_action",
    "open_cloud_test_partition",
    "redact",
    "register_secret",
    "release_lease",
    "run_prefix",
]
