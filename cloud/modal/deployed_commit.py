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


#: Directories copied into the image by add_local_dir. An untracked file under
#: any of these is shipped and executed while the image claims to be HEAD, so
#: it is as fatal as an uncommitted edit. Must stay in step with
#: bridge_phase2_app.LOCAL_PACKAGES and the research directory it adds; a test
#: asserts they agree.
IMAGE_SOURCE_ROOTS: tuple[str, ...] = (
    "packages/bridge/src/openalpha_bridge",
    "packages/sentinel/src/openalpha_sentinel",
    "packages/research-core/src/openalpha_research",
    "packages/kronos-research/src/openalpha_kronos",
    "research/bridge-v0",
)

#: Ignored by add_local_dir, so an untracked file matching these never reaches
#: the image and must not block a deployment. Mirrors bridge_phase2_app._IGNORE.
IMAGE_IGNORED_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo")
IMAGE_IGNORED_DIRECTORY = "__pycache__"


def _is_shipped(relative_path: str) -> bool:
    """Whether add_local_dir would copy this path into the image."""
    normalised = relative_path.replace("\\", "/").strip('"')
    if not any(
        normalised == root or normalised.startswith(root + "/") for root in IMAGE_SOURCE_ROOTS
    ):
        return False
    if normalised.endswith(IMAGE_IGNORED_SUFFIXES):
        return False
    return IMAGE_IGNORED_DIRECTORY not in normalised.split("/")


def resolve_deploying_commit(root: Path) -> str:
    """The exact HEAD of a clean worktree, or refuse to deploy.

    A dirty worktree is fatal. Deploying uncommitted work would produce evidence
    attributed to a commit whose content differs from what executed, and that is
    worse than not deploying at all.

    Untracked files count. ``--untracked-files=no`` was used here, which was
    wrong: add_local_dir copies directories, not git indexes, so an untracked
    .py file under a shipped package is built into the image and imported at
    runtime while the image still claims to be HEAD. That is precisely the
    attribution failure this function exists to prevent. Untracked files
    outside the shipped roots, and those add_local_dir ignores, do not block a
    deployment because they cannot reach the image.
    """
    if not (root / ".git").exists():
        raise DeploymentBindingError(f"DEPLOYED_COMMIT_UNRESOLVABLE: {root} is not a git worktree")

    head = _git(root, "rev-parse", "HEAD")
    if not COMMIT_PATTERN.fullmatch(head):
        raise DeploymentBindingError(
            f"DEPLOYED_COMMIT_UNRESOLVABLE: HEAD resolved to {head!r}, which is not "
            "exactly forty lowercase hex characters"
        )

    # --untracked-files=all so that every file inside an untracked directory is
    # listed individually; the default collapses them to the directory name,
    # which would hide what is actually being shipped.
    status = _git(root, "status", "--porcelain", "--untracked-files=all")

    tracked_changes: list[str] = []
    untracked_shipped: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        code, _, path = line.partition(" ")[0], None, line[3:]
        if code == "??":
            if _is_shipped(path):
                untracked_shipped.append(path)
        else:
            tracked_changes.append(line)

    if tracked_changes:
        listed = "\n".join(f"    {line}" for line in tracked_changes[:20])
        raise DeploymentBindingError(
            "DEPLOYED_COMMIT_WORKTREE_DIRTY: refusing to deploy uncommitted work, "
            f"because the image would be attributed to {head} while running "
            f"something else. Tracked changes:\n{listed}"
        )

    if untracked_shipped:
        listed = "\n".join(f"    {path}" for path in sorted(untracked_shipped)[:20])
        raise DeploymentBindingError(
            "DEPLOYED_COMMIT_UNTRACKED_IN_IMAGE: refusing to deploy, because "
            "add_local_dir copies directories rather than the git index, so these "
            f"untracked files would be built into an image attributed to {head} "
            f"without being part of it:\n{listed}"
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
