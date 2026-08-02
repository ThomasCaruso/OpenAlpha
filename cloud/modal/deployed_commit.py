"""Bind the deploying repository state into the image at build time.

A worker that accepts whatever ``source_commit`` a caller supplies cannot
attribute its own result: the report would name code that need not be the code
that ran. The exact 40-hex HEAD of the deploying worktree is therefore baked
into the image as ``OPENALPHA_DEPLOYED_COMMIT``, and the worker refuses to run
without it.

This module is imported by the Modal app during ``modal deploy``, which runs on
a workstation from an isolated uvx environment. It uses only the standard
library so it cannot fail for want of a dependency.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "COMMIT_ENVIRONMENT_VARIABLE",
    "COMMIT_PATTERN",
    "INSPECTION_SENTINEL",
    "INSPECTION_VARIABLE",
    "DeploymentBindingError",
    "bind_for_image",
    "resolve_deploying_commit",
]

#: The image environment variable the worker reads. Named once, here.
COMMIT_ENVIRONMENT_VARIABLE = "OPENALPHA_DEPLOYED_COMMIT"

#: Exactly forty lowercase hex characters. Not a prefix, not a tag, not HEAD.
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

#: Set by tests and tooling that import the app to inspect it rather than to
#: deploy it. It never relaxes anything: it substitutes a value that is
#: deliberately not a commit, so any image built under it is inert.
INSPECTION_VARIABLE = "OPENALPHA_MODAL_INSPECTION"

#: Chosen to fail the worker's 40-hex check loudly and self-describingly. An
#: image carrying this can be imported and examined but can never run a canary.
INSPECTION_SENTINEL = "NOT-A-COMMIT-INSPECTION-ONLY-IMAGE"


class DeploymentBindingError(RuntimeError):
    """Raised during deployment. Never raised inside a running container."""


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise DeploymentBindingError(
            "DEPLOYED_COMMIT_UNRESOLVABLE: git is not available, so the deploying "
            "commit cannot be determined"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise DeploymentBindingError(
            f"DEPLOYED_COMMIT_UNRESOLVABLE: `git {' '.join(arguments)}` timed out"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip() or f"exit status {error.returncode}"
        raise DeploymentBindingError(
            f"DEPLOYED_COMMIT_UNRESOLVABLE: `git {' '.join(arguments)}` failed: {detail}"
        ) from error
    return completed.stdout.strip()


def resolve_deploying_commit(root: Path) -> str:
    """The exact HEAD of a clean worktree, or refuse to deploy.

    A dirty worktree is fatal. Deploying uncommitted work would produce evidence
    attributed to a commit whose content differs from what executed, and that is
    worse than not deploying at all.
    """
    if not (root / ".git").exists():
        raise DeploymentBindingError(f"DEPLOYED_COMMIT_UNRESOLVABLE: {root} is not a git worktree")

    head = _git(root, "rev-parse", "HEAD")
    if not COMMIT_PATTERN.fullmatch(head):
        raise DeploymentBindingError(
            f"DEPLOYED_COMMIT_UNRESOLVABLE: HEAD resolved to {head!r}, which is not "
            "exactly forty lowercase hex characters"
        )

    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        changed = "\n".join(f"    {line}" for line in dirty.splitlines()[:20])
        raise DeploymentBindingError(
            "DEPLOYED_COMMIT_WORKTREE_DIRTY: refusing to deploy uncommitted work, "
            f"because the image would be attributed to {head} while running "
            f"something else. Tracked changes:\n{changed}"
        )
    return head


def bind_for_image(root: Path, *, environment: Mapping[str, str]) -> str:
    """The value to bake into the image.

    Deploying is strict: a dirty or unresolvable worktree raises and the
    deployment does not happen.

    Importing the app to inspect it is not deploying, and a developer's worktree
    is dirty most of the time. Under the inspection flag the sentinel is baked
    instead of a commit. That is not a relaxation: the sentinel fails the
    worker's format check, so an inspection image cannot run a canary at all.
    """
    if environment.get(INSPECTION_VARIABLE):
        return INSPECTION_SENTINEL
    return resolve_deploying_commit(root)
