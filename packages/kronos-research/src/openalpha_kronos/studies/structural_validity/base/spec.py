"""Pinned identity for the Kronos-base replication.

Every value here was resolved independently from the Hugging Face Hub API and
checked against the identities stated in the study brief, rather than copied.
Nothing in this module is a default for anything else: the mini study keeps its
own ``KRONOS_MINI_SPEC`` and ``TOKENIZER_SPEC`` untouched, and no code path
falls back from one family to the other.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError

from ....model.assets import (
    KRONOS_BASE_SPEC,
    KRONOS_BASE_TOKENIZER_SPEC,
    BaseForecastModelSpec,
)

__all__ = [
    "BASE_ARTIFACT_OBJECT_NAME",
    "BASE_ARTIFACT_ROOT",
    "BASE_CLAIM_BOUNDARY",
    "BASE_EXPERIMENT_ID",
    "BASE_FAILURE_SCHEMA_VERSION",
    "BASE_PROBE_SCHEMA_VERSION",
    "BASE_RUN_ID_PATTERN",
    "BASE_SPECIFICATION_NAME",
    "BASE_SPECIFICATION_SHA256",
    "BASE_SUCCESS_SCHEMA_VERSION",
    "KRONOS_BASE_SPEC",
    "KRONOS_BASE_TOKENIZER_SPEC",
    "MAXIMUM_CONTEXT",
    "BaseForecastModelSpec",
    "base_artifact_key",
    "prove_context_budget",
    "verify_base_specification",
]

BASE_EXPERIMENT_ID: Final = "openalpha-kronos-base-replication-v1"

BASE_SPECIFICATION_NAME: Final[str] = "kronos-base-replication-v1.yaml"
BASE_SPECIFICATION_SHA256: Final[str] = (
    "47ad7956adce368d5c6c69e54aea939ab6497bb4d411c96f894a4e30b7be58dd"
)

#: Its own root. Shares no prefix with the mini diagnostic's
#: ``openalpha-compatibility/bridge-phase2/runs/...`` subtree, so no key of one
#: study can be produced by the other.
BASE_ARTIFACT_ROOT: Final[str] = "openalpha-compatibility/kronos-base-diagnostic"
BASE_ARTIFACT_OBJECT_NAME: Final[str] = "kronos_base_diagnostic_terminal.json"

#: ``base_`` followed by 8 to 32 lowercase hex characters. Disjoint from the
#: mini ``canary_`` pattern by construction: no string matches both.
BASE_RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^base_[0-9a-f]{8,32}$")

BASE_SUCCESS_SCHEMA_VERSION: Final = "openalpha.bridge.base_study.kronos_base_diagnostic.v1"
BASE_FAILURE_SCHEMA_VERSION: Final = "openalpha.bridge.base_study.kronos_base_failure.v1"
BASE_PROBE_SCHEMA_VERSION: Final = "openalpha.bridge.base_study.runtime_probe.v1"

BASE_CLAIM_BOUNDARY: Final = "DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE"

#: The official predictor's context budget. A property of the call, not of the
#: checkpoint: neither config.json declares it.
MAXIMUM_CONTEXT: Final[int] = 512


def base_artifact_key(run_id: str) -> str:
    """The only key this study writes."""
    return f"{BASE_ARTIFACT_ROOT}/runs/{run_id}/{BASE_ARTIFACT_OBJECT_NAME}"


def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def prove_context_budget(
    *, context_candles: int, target_candles: int, maximum_context: int = MAXIMUM_CONTEXT
) -> dict[str, int | bool]:
    """Show that the replication window fits the budget with nothing dropped.

    The official predictor keeps the final ``max_context`` tokens of the
    concatenated context and generated sequence before decoding. The direct
    replication uses 448 + 64, which is exactly 512, so that selection is the
    identity and no context row is discarded. Anything that did not sum to the
    budget would silently drop the oldest rows, and the comparison against the
    mini study would no longer be like for like.
    """
    total = context_candles + target_candles
    if context_candles <= 0 or target_candles <= 0:
        raise _fail(
            "BASE_CONTEXT_BUDGET_INVALID",
            f"context and target must both be positive, got {context_candles} and {target_candles}",
        )
    if total > maximum_context:
        raise _fail(
            "BASE_CONTEXT_BUDGET_EXCEEDED",
            (
                f"{context_candles} context + {target_candles} generated = {total}, "
                f"which exceeds the maximum context {maximum_context}; the oldest "
                "rows would be truncated and the window would not be the one specified"
            ),
        )
    return {
        "context_candles": context_candles,
        "target_candles": target_candles,
        "maximum_context": maximum_context,
        "sum": total,
        "fills_budget_exactly": total == maximum_context,
        "truncation_occurs": False,
    }


def verify_base_specification(research_root: Path | str) -> dict[str, str]:
    """Verify the base preregistration document, or fail closed.

    Deliberately does not verify the mini chain: this study is governed by its
    own document, and conflating the two would let a mini drift be reported as
    a base failure or the reverse.
    """
    path = Path(research_root) / BASE_SPECIFICATION_NAME
    if not path.is_file():
        raise _fail(
            "MISSING_BASE_SPECIFICATION",
            f"base specification not found: {path}",
            field=BASE_SPECIFICATION_NAME,
        )
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != BASE_SPECIFICATION_SHA256:
        raise _fail(
            "BASE_SPECIFICATION_HASH_MISMATCH",
            f"{BASE_SPECIFICATION_NAME} hashes to {observed}, expected {BASE_SPECIFICATION_SHA256}",
            field=BASE_SPECIFICATION_NAME,
        )
    return {BASE_SPECIFICATION_NAME: observed}
