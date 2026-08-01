"""GPU-host preflight.

Runs before any provider access and stops the pipeline when a hard requirement
fails. On a CPU-only host it returns a typed NO_ACCELERATOR blocker rather than
raising, so the report is machine-readable.
"""

from __future__ import annotations

import platform
import shutil
import sys
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .identity import verify_locked_hashes
from .optional import BridgeExtra

__all__ = [
    "MAXIMUM_VRAM_BYTES",
    "MINIMUM_FREE_DISK_BYTES",
    "CheckOutcome",
    "PreflightCheck",
    "PreflightReport",
    "run_preflight",
]

#: experiment.yaml accelerator_memory_bytes_maximum.
MAXIMUM_VRAM_BYTES = 25_769_803_776
MINIMUM_FREE_DISK_BYTES = 40 * (1 << 30)
_SUPPORTED_SYSTEMS = frozenset({"Linux", "Windows"})


class CheckOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class PreflightCheck(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    check_id: str
    outcome: CheckOutcome
    observed: str | None
    requirement: str
    hard: bool = True


class PreflightReport(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.preflight.v1"] = (
        "openalpha.bridge.phase2.preflight.v1"
    )
    checks: tuple[PreflightCheck, ...]
    passed: bool
    blocker_code: str | None
    may_access_provider: bool = Field(default=False)

    @property
    def failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(c for c in self.checks if c.outcome is CheckOutcome.FAIL)


def _check(
    check_id: str,
    ok: bool,
    observed: str | None,
    requirement: str,
    *,
    hard: bool = True,
) -> PreflightCheck:
    return PreflightCheck(
        check_id=check_id,
        outcome=CheckOutcome.PASS if ok else CheckOutcome.FAIL,
        observed=observed,
        requirement=requirement,
        hard=hard,
    )


def _torch_checks(stage: str) -> tuple[list[PreflightCheck], str | None]:
    checks: list[PreflightCheck] = []
    try:
        # torch is an optional extra and is absent by design on a base install.
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError:
        checks.append(
            PreflightCheck(
                check_id="torch_import",
                outcome=CheckOutcome.FAIL,
                observed="not installed",
                requirement=f"torch importable via extra '{BridgeExtra.BRIDGE_TRAINING.value}'",
            )
        )
        for skipped in ("cuda_available", "accelerator_count", "vram_capacity"):
            checks.append(
                PreflightCheck(
                    check_id=skipped,
                    outcome=CheckOutcome.SKIPPED,
                    observed=None,
                    requirement="requires torch",
                )
            )
        return checks, "MISSING_OPTIONAL_DEPENDENCY"

    checks.append(_check("torch_import", True, torch.__version__, "torch importable"))

    if not torch.cuda.is_available():
        checks.append(
            _check("cuda_available", False, "no CUDA device", "CUDA must be available")
        )
        for skipped in ("accelerator_count", "vram_capacity"):
            checks.append(
                PreflightCheck(
                    check_id=skipped,
                    outcome=CheckOutcome.SKIPPED,
                    observed=None,
                    requirement="requires an available CUDA device",
                )
            )
        return checks, "NO_ACCELERATOR"

    count = torch.cuda.device_count()
    checks.append(_check("cuda_available", True, "available", "CUDA must be available"))
    checks.append(
        _check("accelerator_count", count == 1, str(count), "exactly one accelerator")
    )
    properties = torch.cuda.get_device_properties(0)
    vram = int(properties.total_memory)
    checks.append(
        _check(
            "vram_capacity",
            vram <= MAXIMUM_VRAM_BYTES,
            f"{vram} bytes",
            f"VRAM at most {MAXIMUM_VRAM_BYTES} bytes",
        )
    )
    blocker = None
    if count != 1:
        blocker = "ACCELERATOR_COUNT_MISMATCH"
    elif vram > MAXIMUM_VRAM_BYTES:
        blocker = "VRAM_ABOVE_LOCK"
    return checks, blocker


def run_preflight(
    *,
    research_root: Path,
    cache_directory: Path,
    repository_root: Path,
    stage: str = "preflight",
    require_accelerator: bool = True,
) -> PreflightReport:
    """Verify the host before any provider or asset access."""
    checks: list[PreflightCheck] = []
    blocker: str | None = None

    system = platform.system()
    checks.append(
        _check(
            "operating_system",
            system in _SUPPORTED_SYSTEMS,
            system,
            f"one of {sorted(_SUPPORTED_SYSTEMS)}",
        )
    )

    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks.append(
        _check(
            "python_version",
            sys.version_info[:2] == (3, 13),
            version,
            "Python 3.13",
        )
    )

    if require_accelerator:
        torch_checks, torch_blocker = _torch_checks(stage)
        checks.extend(torch_checks)
        blocker = blocker or torch_blocker
    else:
        checks.append(
            PreflightCheck(
                check_id="torch_import",
                outcome=CheckOutcome.SKIPPED,
                observed=None,
                requirement="accelerator not required for this stage",
                hard=False,
            )
        )

    try:
        usage = shutil.disk_usage(cache_directory if cache_directory.exists() else Path.cwd())
        free_ok = usage.free >= MINIMUM_FREE_DISK_BYTES
        checks.append(
            _check(
                "free_disk",
                free_ok,
                f"{usage.free} bytes",
                f"at least {MINIMUM_FREE_DISK_BYTES} bytes free",
            )
        )
        if not free_ok:
            blocker = blocker or "INSUFFICIENT_DISK"
    except OSError as error:
        checks.append(_check("free_disk", False, type(error).__name__, "disk usage readable"))
        blocker = blocker or "INSUFFICIENT_DISK"

    cache_outside_git = not _inside_repository(cache_directory, repository_root)
    checks.append(
        _check(
            "cache_outside_git",
            cache_outside_git,
            str(cache_directory),
            "cache directory must live outside the Git worktree",
        )
    )
    if not cache_outside_git:
        blocker = blocker or "CACHE_INSIDE_GIT"

    try:
        observed = verify_locked_hashes(research_root)
        checks.append(
            _check(
                "experiment_hashes",
                True,
                observed["experiment.yaml"][:16],
                "experiment and both amendments byte-identical",
            )
        )
    except Exception as error:  # noqa: BLE001 - reported as a typed check
        checks.append(
            _check("experiment_hashes", False, type(error).__name__, "locked hashes must verify")
        )
        blocker = blocker or "EXPERIMENT_HASH_MISMATCH"

    hard_failures = [c for c in checks if c.outcome is CheckOutcome.FAIL and c.hard]
    passed = not hard_failures
    return PreflightReport(
        checks=tuple(checks),
        passed=passed,
        blocker_code=blocker if not passed else None,
        may_access_provider=passed,
    )


def _inside_repository(candidate: Path, repository_root: Path) -> bool:
    try:
        candidate.resolve().relative_to(repository_root.resolve())
    except ValueError:
        return False
    return True
