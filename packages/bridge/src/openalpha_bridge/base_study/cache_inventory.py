"""Read-only inventory of the base study's remote cache volume.

Lives here rather than inside the Modal function so the reload-before-inspection
contract can be exercised with a fake volume and a real temporary directory. The
Modal function is a shell that supplies the actual volume's ``reload`` and the
actual mount path.

Why the reload matters. A Modal volume mounted in a warm container keeps the
view it had at mount time; a commit made by a different container is not visible
until the reader reloads. The first version of this inventory walked the mount
immediately, so after the base runtime probe downloaded and committed roughly
425 MB, a reused inventory container still reported an empty volume. Nothing was
wrong with the storage -- the reader was looking at a stale view and reporting
it as fact.

Nothing here writes, commits, deletes, retries, or swallows a reload failure. An
inventory that cannot refresh its view must fail loudly: reporting an empty
cache it did not actually observe is the exact error this module exists to
prevent.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["build_base_cache_inventory", "walk_cache_root"]


def walk_cache_root(root: Path) -> tuple[list[dict[str, Any]], int]:
    """Describe every Hugging Face repository cached under ``root``.

    Pure observation: no mutation, no reload, no commit. The caller is
    responsible for having refreshed the view first, which is why this is not
    exported as the entry point.
    """
    repositories: list[dict[str, Any]] = []
    total = 0

    if not root.is_dir():
        return repositories, total

    for repo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not repo_dir.name.startswith("models--"):
            continue
        snapshots_root = repo_dir / "snapshots"
        snapshots: list[dict[str, Any]] = []
        repo_bytes = 0
        for item in repo_dir.rglob("*"):
            if item.is_file() and not item.is_symlink():
                repo_bytes += item.stat().st_size
        if snapshots_root.is_dir():
            for snapshot in sorted(p for p in snapshots_root.iterdir() if p.is_dir()):
                resolved_bytes = 0
                files: list[str] = []
                for item in sorted(snapshot.rglob("*")):
                    if item.is_file() or item.is_symlink():
                        files.append(item.relative_to(snapshot).as_posix())
                        try:
                            resolved_bytes += item.resolve().stat().st_size
                        except OSError:
                            # A dangling link in a partially fetched snapshot
                            # is worth reporting as a file with unknown size,
                            # not worth failing the whole inventory over.
                            pass
                snapshots.append(
                    {
                        "revision": snapshot.name,
                        "path": str(snapshot),
                        "files": files,
                        "resolved_bytes": resolved_bytes,
                    }
                )
        total += repo_bytes
        repositories.append(
            {
                "cache_directory": repo_dir.name,
                "repository": repo_dir.name.removeprefix("models--").replace("--", "/", 1),
                "path": str(repo_dir),
                "bytes_on_volume": repo_bytes,
                "snapshots": snapshots,
            }
        )

    return repositories, total


def build_base_cache_inventory(
    *,
    reload: Callable[[], Any],
    mount: str,
    volume_name: str,
    deployed_commit: str,
    inspected_at: datetime,
) -> dict[str, Any]:
    """Refresh the mounted view, then report it. Read-only throughout.

    ``reload`` is called exactly once and before anything touches the
    filesystem -- before ``is_dir``, ``iterdir``, ``rglob`` or any ``stat``. If
    it raises, the exception propagates unchanged: an inventory that could not
    refresh its view has no observation to report, and returning ``exists:
    False`` would be a fabrication indistinguishable from a genuinely empty
    volume.
    """
    # First statement in the function body, deliberately. Every path
    # construction below is inert until it is used, and nothing is used until
    # this has returned.
    reload()

    root = Path(mount) / "huggingface"
    repositories, total = walk_cache_root(root)

    return {
        "volume": volume_name,
        "mount": mount,
        "root": str(root),
        "exists": root.is_dir(),
        "repositories": repositories,
        "total_bytes": total,
        "read_only": True,
        "deletion_supported": False,
        #: Operational metadata only. Never enters a scientific artifact or any
        #: preregistration identity.
        "reloaded_before_inspection": True,
        "deployed_commit": deployed_commit,
        "inspected_at": inspected_at.isoformat(),
    }
