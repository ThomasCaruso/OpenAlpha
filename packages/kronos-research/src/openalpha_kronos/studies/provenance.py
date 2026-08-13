"""Historical provenance retained unchanged under Kronos study ownership."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Final

from openalpha_research.failures import ResearchFailureError
from openalpha_research.protocols import verify_sha256_files

MINI_RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^canary_[0-9a-f]{8,32}$")
MINI_ARTIFACT_ROOT: Final[str] = "openalpha-compatibility/bridge-phase2"
MINI_EXPERIMENT_SHA256: Final[str] = (
    "d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2"
)
MINI_AMENDMENT_1_SHA256: Final[str] = (
    "4c60846e8cb791d922c29a5e704fe3a473740171dd2bbb7b3390796ceeeef3c8"
)
MINI_AMENDMENT_2_SHA256: Final[str] = (
    "66f3c8171c2805bccf4b924125bd206edad27c957dfff40a0abf3864fbab10c1"
)
MINI_AMENDMENT_3_SHA256: Final[str] = (
    "d9484020a22df42c93edc19941374c628ea891f08c8bee8151b82264c00bb12b"
)


class EvidenceClass(StrEnum):
    DEVELOPMENT_COMPATIBILITY_CANARY = "development_compatibility_canary"


def mini_run_prefix(run_id: str) -> str:
    return f"{MINI_ARTIFACT_ROOT}/runs/{run_id}"


def legacy_canary_artifact_key(run_id: str) -> str:
    return f"{mini_run_prefix(run_id)}/stage_a_canary_terminal.json"


def verify_locked_hashes(research_root: Path) -> dict[str, str]:
    """Verify the historical mini-study experiment and amendment chain."""
    try:
        return verify_sha256_files(
            research_root,
            {
                "experiment.yaml": MINI_EXPERIMENT_SHA256,
                "phase2-preregistration-amendment.yaml": MINI_AMENDMENT_1_SHA256,
                "phase2-amendment-2-context-prefix.yaml": MINI_AMENDMENT_2_SHA256,
                "phase2-amendment-3-scale-features.yaml": MINI_AMENDMENT_3_SHA256,
            },
        )
    except ResearchFailureError as error:
        historical_codes = {
            "MISSING_PROTOCOL_FILE": "MISSING_LOCK_FILE",
            "PROTOCOL_HASH_MISMATCH": "EXPERIMENT_HASH_MISMATCH",
        }
        first = error.failures[0]
        historical_code = historical_codes.get(first.code)
        if historical_code is None:
            raise
        raise ResearchFailureError(first.model_copy(update={"code": historical_code})) from error
