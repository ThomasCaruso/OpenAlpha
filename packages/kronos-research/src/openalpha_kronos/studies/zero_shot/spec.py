"""Pinned identity for the Kronos-base zero-shot forecasting benchmark.

Everything that makes this study addressable -- experiment id, specification
digest, artifact root, run-id pattern, schema versions -- is declared here and
shares nothing with either completed structural-validity study.

The frozen model pair is reused rather than re-declared. The already verified
``KRONOS_BASE_SPEC`` and ``KRONOS_BASE_TOKENIZER_SPEC`` are imported from the
base study so there is exactly one place in the tree that says what
Kronos-base is; a second copy could drift. The digests this benchmark expects
are ALSO written out here as literals and checked against the imported pins by
``verify_pinned_pair``, so a silent edit to the shared pin fails this study
loudly instead of quietly changing what it loaded.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final, Literal

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from pydantic import BaseModel, ConfigDict, Field

from openalpha_kronos.studies.structural_validity.base.spec import (
    KRONOS_BASE_SPEC,
    KRONOS_BASE_TOKENIZER_SPEC,
)

__all__ = [
    "ASSET_PANEL",
    "BOOTSTRAP_CONFIDENCE_LEVEL",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONTEXT_CANDLES",
    "ENSEMBLE_SEEDS",
    "HORIZON_CANDLES",
    "MAXIMUM_CONTEXT",
    "ORIGINS_PER_ASSET",
    "ORIGIN_FIRST_INDEX",
    "ORIGIN_STRIDE",
    "PRIMARY_METRIC",
    "REQUIRED_SESSIONS",
    "RETRIEVAL_END_EXCLUSIVE",
    "RETRIEVAL_MAXIMUM_CANDLES",
    "RETRIEVAL_START_INCLUSIVE",
    "SAMPLING_CONFIGURATIONS",
    "SECONDARY_METRICS",
    "WINDOW_CANDLES",
    "ZERO_SHOT_ARTIFACT_OBJECT_NAME",
    "ZERO_SHOT_ARTIFACT_ROOT",
    "ZERO_SHOT_CLAIM_BOUNDARY",
    "ZERO_SHOT_EXPERIMENT_ID",
    "ZERO_SHOT_FAILURE_SCHEMA_VERSION",
    "ZERO_SHOT_MODEL_SPEC",
    "ZERO_SHOT_PROBE_SCHEMA_VERSION",
    "ZERO_SHOT_RUN_ID_PATTERN",
    "ZERO_SHOT_SPECIFICATION_NAME",
    "ZERO_SHOT_SPECIFICATION_SHA256",
    "ZERO_SHOT_SUCCESS_SCHEMA_VERSION",
    "ZERO_SHOT_THRESHOLDS",
    "ZERO_SHOT_TOKENIZER_SPEC",
    "SamplingConfiguration",
    "ZeroShotThresholds",
    "prove_zero_shot_budget",
    "verify_pinned_pair",
    "verify_zero_shot_specification",
    "zero_shot_artifact_key",
]

ZERO_SHOT_EXPERIMENT_ID: Final = "openalpha-kronos-zero-shot-benchmark-v1"

ZERO_SHOT_SPECIFICATION_NAME: Final[str] = "kronos-zero-shot-benchmark-v1.yaml"
ZERO_SHOT_SPECIFICATION_SHA256: Final[str] = (
    "6832c0f7befc54cd7ccec382db8cac1e6314f3fb356eb5eed8e9e9eca9a6fd07"
)

#: A third root. Shares no prefix with the mini diagnostic's
#: ``openalpha-compatibility/bridge-phase2`` subtree or the base study's
#: ``openalpha-compatibility/kronos-base-diagnostic`` subtree.
ZERO_SHOT_ARTIFACT_ROOT: Final[str] = "openalpha-compatibility/kronos-zero-shot-benchmark"
ZERO_SHOT_ARTIFACT_OBJECT_NAME: Final[str] = "kronos_zero_shot_benchmark_terminal.json"

#: ``zsb_`` followed by 8 to 32 lowercase hex characters. Disjoint from both
#: ``canary_`` and ``base_`` by construction: no string matches two of the three.
ZERO_SHOT_RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^zsb_[0-9a-f]{8,32}$")

ZERO_SHOT_SUCCESS_SCHEMA_VERSION: Final = "openalpha.bridge.zero_shot.kronos_zero_shot_benchmark.v1"
ZERO_SHOT_FAILURE_SCHEMA_VERSION: Final = "openalpha.bridge.zero_shot.kronos_zero_shot_failure.v1"
ZERO_SHOT_PROBE_SCHEMA_VERSION: Final = "openalpha.bridge.zero_shot.runtime_probe.v1"

#: Deliberately not the diagnostic boundary string. A benchmark and a diagnostic
#: are different claims and must not share a label.
ZERO_SHOT_CLAIM_BOUNDARY: Final = "DEVELOPMENT BENCHMARK - NOT HOLDOUT OR TRADING EVIDENCE"

#: The frozen pair, imported so there is one source of truth for the pin.
ZERO_SHOT_MODEL_SPEC: Final = KRONOS_BASE_SPEC
ZERO_SHOT_TOKENIZER_SPEC: Final = KRONOS_BASE_TOKENIZER_SPEC

#: What this benchmark expects that pin to be. Written independently of the
#: import so a change to the shared pin cannot pass silently.
EXPECTED_MODEL_REPOSITORY: Final[str] = "NeoQuasar/Kronos-base"
EXPECTED_MODEL_REVISION: Final[str] = "2b554741eca47781b64468546e77fef3e85130e6"
EXPECTED_MODEL_CONFIG_SHA256: Final[str] = (
    "77ebc3038b647709b92be002f801d72e1a385f4c8c2c5aa1cc6cf21fcfe44eb2"
)
EXPECTED_MODEL_WEIGHTS_SHA256: Final[str] = (
    "abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83"
)
EXPECTED_MODEL_WEIGHTS_SIZE_BYTES: Final[int] = 409_264_008
EXPECTED_TOKENIZER_REPOSITORY: Final[str] = "NeoQuasar/Kronos-Tokenizer-base"
EXPECTED_TOKENIZER_REVISION: Final[str] = "0e0117387f39004a9016484a186a908917e22426"
EXPECTED_TOKENIZER_CONFIG_SHA256: Final[str] = (
    "2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2"
)
EXPECTED_TOKENIZER_WEIGHTS_SHA256: Final[str] = (
    "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
)

#: Reused, never duplicated. The volume already holds exactly these two
#: repositories and 425106905 primary bytes.
ZERO_SHOT_CACHE_VOLUME: Final[str] = "openalpha-kronos-base-cache"

ASSET_PANEL: Final[tuple[str, ...]] = ("SPY", "QQQ", "IWM", "DIA")

FREQUENCY: Final[str] = "1d"
CALENDAR: Final[str] = "XNYS"

CONTEXT_CANDLES: Final[int] = 40
HORIZON_CANDLES: Final[int] = 12
WINDOW_CANDLES: Final[int] = CONTEXT_CANDLES + HORIZON_CANDLES
MAXIMUM_CONTEXT: Final[int] = 512

RETRIEVAL_START_INCLUSIVE: Final[str] = "2025-01-02"
RETRIEVAL_END_EXCLUSIVE: Final[str] = "2026-07-01"
RETRIEVAL_MAXIMUM_CANDLES: Final[int] = 512

ORIGINS_PER_ASSET: Final[int] = 25
ORIGIN_FIRST_INDEX: Final[int] = CONTEXT_CANDLES
ORIGIN_STRIDE: Final[int] = HORIZON_CANDLES

#: The last target row of the last origin, plus one. Fixed arithmetic, but
#: written as a constant so the retrieval requirement is a declared number
#: rather than something a reader has to recompute.
REQUIRED_SESSIONS: Final[int] = (
    ORIGIN_FIRST_INDEX + (ORIGINS_PER_ASSET - 1) * ORIGIN_STRIDE + HORIZON_CANDLES
)

PRIMARY_METRIC: Final[str] = "close_return_mae"
SECONDARY_METRICS: Final[tuple[str, ...]] = (
    "close_mae",
    "directional_accuracy",
    "close_return_rmse",
    "median_absolute_return_error",
)

#: Eight paths, the same eight seeds everywhere. Member zero is also the
#: deterministic single path, so it is generated once and read twice.
ENSEMBLE_SEEDS: Final[tuple[int, ...]] = (
    20250102,
    20250103,
    20250104,
    20250105,
    20250106,
    20250107,
    20250108,
    20250109,
)

BOOTSTRAP_RESAMPLES: Final[int] = 2000
BOOTSTRAP_CONFIDENCE_LEVEL: Final[float] = 0.95
BOOTSTRAP_SEED: Final[int] = 20260803


class SamplingConfiguration(BaseModel):
    """One of exactly two predeclared sampling configurations."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    label: Literal["A", "B"]
    temperature: float
    top_k: int = 0
    top_p: float = 0.9
    sample_count: Literal[1] = 1
    prediction_length: int = HORIZON_CANDLES
    context_length_used: int = CONTEXT_CANDLES
    max_context: int = MAXIMUM_CONTEXT
    clip: float = 5.0
    verbose: Literal[False] = False
    parameters_frozen: Literal[True] = True
    gradient_mode: Literal["inference_mode"] = "inference_mode"


#: Exactly two. Adding a third temperature after observing results is forbidden
#: by the specification, and the runner refuses a configuration tuple whose
#: labels or temperatures are not these.
SAMPLING_CONFIGURATIONS: Final[tuple[SamplingConfiguration, ...]] = (
    SamplingConfiguration(label="A", temperature=0.6),
    SamplingConfiguration(label="B", temperature=1.0),
)


class ZeroShotThresholds(BaseModel):
    """Every threshold the decision rules use. Fixed before execution."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    minimum_median_relative_skill: float = 0.0
    minimum_fraction_of_origins_beating_persistence: float = 0.60
    minimum_supporting_assets: int = 3
    total_assets: int = len(ASSET_PANEL)
    bootstrap_confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL


ZERO_SHOT_THRESHOLDS: Final[ZeroShotThresholds] = ZeroShotThresholds()


def zero_shot_artifact_key(run_id: str) -> str:
    """The only key this study writes."""
    return f"{ZERO_SHOT_ARTIFACT_ROOT}/runs/{run_id}/{ZERO_SHOT_ARTIFACT_OBJECT_NAME}"


def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def prove_zero_shot_budget(
    *,
    context_candles: int = CONTEXT_CANDLES,
    horizon_candles: int = HORIZON_CANDLES,
    maximum_context: int = MAXIMUM_CONTEXT,
) -> dict[str, int | bool]:
    """Show the short window fits the budget with nothing truncated.

    Unlike the completed 448 + 64 diagnostics this window does not fill the
    budget, and it is not supposed to. What matters is only that the official
    final-``max_context`` selection is the identity, which it is whenever the
    sum does not exceed the budget: no context row is discarded, so the model
    sees exactly the 40 rows the specification names.
    """
    total = context_candles + horizon_candles
    if context_candles <= 0 or horizon_candles <= 0:
        raise _fail(
            "ZERO_SHOT_BUDGET_INVALID",
            f"context and horizon must both be positive, got {context_candles} and "
            f"{horizon_candles}",
        )
    if total > maximum_context:
        raise _fail(
            "ZERO_SHOT_BUDGET_EXCEEDED",
            (
                f"{context_candles} context + {horizon_candles} generated = {total}, "
                f"which exceeds the maximum context {maximum_context}; the oldest rows "
                "would be truncated and the window would not be the one specified"
            ),
        )
    return {
        "context_candles": context_candles,
        "horizon_candles": horizon_candles,
        "maximum_context": maximum_context,
        "sum": total,
        "fills_budget_exactly": total == maximum_context,
        "truncation_occurs": False,
    }


def verify_pinned_pair() -> dict[str, str]:
    """Check the imported pin still is the pair this benchmark preregistered.

    The pin lives in the base study so there is one source of truth. That is
    only safe if a change to it cannot silently change what this benchmark
    loads, so every field is compared against a literal written here.
    """
    checks: tuple[tuple[str, object, object], ...] = (
        ("model_repository", ZERO_SHOT_MODEL_SPEC.repository, EXPECTED_MODEL_REPOSITORY),
        ("model_revision", ZERO_SHOT_MODEL_SPEC.revision, EXPECTED_MODEL_REVISION),
        ("model_config_sha256", ZERO_SHOT_MODEL_SPEC.config_sha256, EXPECTED_MODEL_CONFIG_SHA256),
        (
            "model_weights_sha256",
            ZERO_SHOT_MODEL_SPEC.weights_sha256,
            EXPECTED_MODEL_WEIGHTS_SHA256,
        ),
        (
            "model_weights_size_bytes",
            ZERO_SHOT_MODEL_SPEC.weights_size_bytes,
            EXPECTED_MODEL_WEIGHTS_SIZE_BYTES,
        ),
        (
            "tokenizer_repository",
            ZERO_SHOT_TOKENIZER_SPEC.repository,
            EXPECTED_TOKENIZER_REPOSITORY,
        ),
        ("tokenizer_revision", ZERO_SHOT_TOKENIZER_SPEC.revision, EXPECTED_TOKENIZER_REVISION),
        (
            "tokenizer_config_sha256",
            ZERO_SHOT_TOKENIZER_SPEC.config_sha256,
            EXPECTED_TOKENIZER_CONFIG_SHA256,
        ),
        (
            "tokenizer_weights_sha256",
            ZERO_SHOT_TOKENIZER_SPEC.weights_sha256,
            EXPECTED_TOKENIZER_WEIGHTS_SHA256,
        ),
    )
    for field, observed, expected in checks:
        if observed != expected:
            raise _fail(
                "ZERO_SHOT_PINNED_PAIR_MISMATCH",
                f"{field} is {observed!r}, but this benchmark preregistered {expected!r}",
                field=field,
            )
    if ZERO_SHOT_MODEL_SPEC.paired_tokenizer_repository != EXPECTED_TOKENIZER_REPOSITORY:
        raise _fail(
            "ZERO_SHOT_PINNED_PAIR_MISMATCH",
            "the model records a different paired tokenizer than this benchmark expects",
            field="paired_tokenizer_repository",
        )
    return {
        "model_repository": EXPECTED_MODEL_REPOSITORY,
        "model_revision": EXPECTED_MODEL_REVISION,
        "tokenizer_repository": EXPECTED_TOKENIZER_REPOSITORY,
        "tokenizer_revision": EXPECTED_TOKENIZER_REVISION,
    }


def verify_zero_shot_specification(research_root: Path | str) -> dict[str, str]:
    """Verify this benchmark's preregistration document, or fail closed.

    Deliberately does not verify the mini chain or the base document: this study
    is governed by its own, and conflating them would let a drift in one be
    reported as a failure of another.
    """
    path = Path(research_root) / ZERO_SHOT_SPECIFICATION_NAME
    if not path.is_file():
        raise _fail(
            "MISSING_ZERO_SHOT_SPECIFICATION",
            f"zero-shot benchmark specification not found: {path}",
            field=ZERO_SHOT_SPECIFICATION_NAME,
        )
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != ZERO_SHOT_SPECIFICATION_SHA256:
        raise _fail(
            "ZERO_SHOT_SPECIFICATION_HASH_MISMATCH",
            f"{ZERO_SHOT_SPECIFICATION_NAME} hashes to {observed}, expected "
            f"{ZERO_SHOT_SPECIFICATION_SHA256}",
            field=ZERO_SHOT_SPECIFICATION_NAME,
        )
    return {ZERO_SHOT_SPECIFICATION_NAME: observed}


class ZeroShotWindow(BaseModel):
    """The declared retrieval window, as a checkable object."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    assets: tuple[str, ...] = ASSET_PANEL
    frequency: Literal["1d"] = "1d"
    calendar: Literal["XNYS"] = "XNYS"
    start_inclusive: str = RETRIEVAL_START_INCLUSIVE
    end_exclusive: str = RETRIEVAL_END_EXCLUSIVE
    required_sessions: int = Field(default=REQUIRED_SESSIONS, gt=0)
    context_candles: int = CONTEXT_CANDLES
    horizon_candles: int = HORIZON_CANDLES
    origins_per_asset: int = ORIGINS_PER_ASSET
    provider_retrievals_permitted_per_asset: Literal[1] = 1


ZERO_SHOT_WINDOW: Final[ZeroShotWindow] = ZeroShotWindow()

__all__ += ["ZERO_SHOT_CACHE_VOLUME", "ZERO_SHOT_WINDOW", "ZeroShotWindow"]
