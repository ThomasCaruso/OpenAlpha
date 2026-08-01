"""Phase 2 execution pipeline.

Importing this package must never pull in Torch, a provider client, or any
Kronos asset. Heavy dependencies are resolved lazily inside the stages that need
them; see :mod:`openalpha_bridge.phase2.optional`.
"""

from __future__ import annotations

from .bootstrap import (
    BootstrapResult,
    ConfirmationDecision,
    PairedObservation,
    ResamplingUnit,
    paired_bootstrap,
)
from .cache import CachedExample, FeatureCache, ShardRef
from .gates import GateOutcome, GateResult, GateTable, TerminalConclusion, evaluate_conclusion
from .identity import RunIdentity, verify_locked_hashes
from .kronos import DeterministicFakeKronosBackend, KronosMode, OfficialKronosBackend
from .metrics import ReconstructionMethod, ReconstructionMetrics, compute_reconstruction_metrics
from .optional import BridgeExtra, missing_optional_dependency, require_module
from .pipeline import Phase2Config, Phase2Pipeline
from .preflight import PreflightReport, run_preflight
from .provider import DeterministicFakeProvider, ProviderMode, YahooDailyProvider
from .states import EvidenceClass, Phase2State, StateJournal, assert_transition_allowed
from .testgate import TestOpeningPreconditions, TestOpeningRecord, open_test_partition
from .training import (
    LOCKED_ARCHITECTURE,
    LOCKED_PARAMETER_COUNT,
    NumpyTrainingBackend,
    TrainingConfig,
    assert_locked_architecture,
)

__all__ = [
    "LOCKED_ARCHITECTURE",
    "LOCKED_PARAMETER_COUNT",
    "BootstrapResult",
    "BridgeExtra",
    "CachedExample",
    "ConfirmationDecision",
    "DeterministicFakeKronosBackend",
    "DeterministicFakeProvider",
    "EvidenceClass",
    "FeatureCache",
    "GateOutcome",
    "GateResult",
    "GateTable",
    "KronosMode",
    "NumpyTrainingBackend",
    "OfficialKronosBackend",
    "PairedObservation",
    "Phase2Config",
    "Phase2Pipeline",
    "Phase2State",
    "PreflightReport",
    "ProviderMode",
    "ReconstructionMethod",
    "ReconstructionMetrics",
    "ResamplingUnit",
    "RunIdentity",
    "ShardRef",
    "StateJournal",
    "TerminalConclusion",
    "TestOpeningPreconditions",
    "TestOpeningRecord",
    "TrainingConfig",
    "YahooDailyProvider",
    "assert_locked_architecture",
    "assert_transition_allowed",
    "compute_reconstruction_metrics",
    "evaluate_conclusion",
    "missing_optional_dependency",
    "open_test_partition",
    "paired_bootstrap",
    "require_module",
    "run_preflight",
    "verify_locked_hashes",
]
