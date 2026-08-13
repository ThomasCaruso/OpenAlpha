from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .failures import FailureCategory, ResearchFailure, ResearchFailureError
from .identity import file_sha256

__all__ = ["verify_sha256_files"]


def verify_sha256_files(root: Path, expected: Mapping[str, str]) -> dict[str, str]:
    """Verify caller-declared files and return their observed SHA-256 digests."""
    resolved_root = root.resolve()
    observed: dict[str, str] = {}
    for name, wanted_digest in expected.items():
        path = (resolved_root / name).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError:
            raise ResearchFailureError(
                ResearchFailure(
                    category=FailureCategory.INVALID_CONFIGURATION,
                    code="PROTOCOL_PATH_OUTSIDE_ROOT",
                    field=name,
                    message=f"protocol path is outside the declared root: {name}",
                )
            ) from None
        if not path.is_file():
            raise ResearchFailureError(
                ResearchFailure(
                    category=FailureCategory.INVALID_CONFIGURATION,
                    code="MISSING_PROTOCOL_FILE",
                    field=name,
                    message=f"protocol file not found: {path}",
                )
            )
        observed_digest = file_sha256(path)
        observed[name] = observed_digest
        if observed_digest != wanted_digest:
            raise ResearchFailureError(
                ResearchFailure(
                    category=FailureCategory.INVALID_CONFIGURATION,
                    code="PROTOCOL_HASH_MISMATCH",
                    field=name,
                    observed_value=observed_digest,
                    message=(
                        f"{name} expected {wanted_digest}, observed {observed_digest}"
                    ),
                )
            )
    return observed
