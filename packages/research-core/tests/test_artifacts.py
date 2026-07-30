import hashlib
import importlib.resources
import os
import subprocess
import tomllib
from pathlib import Path

import pytest
from openalpha_research.artifacts import ArtifactRef, LocalArtifactStore
from openalpha_research.errors import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactPathError,
)
from pydantic import ValidationError


def test_workspace_installs_typed_research_core_package() -> None:
    root = Path(__file__).parents[3]
    workspace = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert "openalpha-research-core" in workspace["project"]["dependencies"]
    assert workspace["tool"]["uv"]["sources"]["openalpha-research-core"] == {"workspace": True}
    assert importlib.resources.files("openalpha_research").joinpath("py.typed").is_file()


def test_put_bytes_uses_content_address_and_is_idempotent(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    value = b"auditable research output"

    first = store.put_bytes(value, media_type="application/octet-stream")
    second = store.put_bytes(value, media_type="application/octet-stream")

    digest = hashlib.sha256(value).hexdigest()
    assert first == second
    assert first == ArtifactRef(
        sha256=digest,
        size_bytes=len(value),
        media_type="application/octet-stream",
        relative_path=f"sha256/{digest[:2]}/{digest}",
    )
    assert store.read_bytes(first) == value
    assert store.verify(first)


def test_invalid_media_type_is_rejected_before_bytes_are_published(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValidationError, match="media_type"):
        store.put_bytes(b"value", media_type="not-a-media-type")

    assert list(tmp_path.iterdir()) == []


def test_published_artifact_cannot_be_overwritten(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = store.put_bytes(b"first", media_type="application/octet-stream")

    with pytest.raises(ArtifactConflictError, match="different content"):
        store.publish_at(ref.relative_path, b"second")

    assert store.read_bytes(ref) == b"first"


def test_new_publication_must_match_its_content_address(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    wrong_digest = "0" * 64

    with pytest.raises(ArtifactIntegrityError, match="content address"):
        store.publish_at(f"sha256/00/{wrong_digest}", b"value")


@pytest.mark.parametrize(
    "relative_path",
    (
        "../escape",
        "sha256/aa/../../escape",
        "/absolute/path",
        "C:/absolute/path",
        "sha256/NUL/file",
        "sha256/com1.txt/file",
        "sha256/aa//file",
        "sha256\\aa\\file",
    ),
)
def test_untrusted_artifact_paths_are_rejected(tmp_path: Path, relative_path: str) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ArtifactPathError):
        store.publish_at(relative_path, b"unsafe")


def test_existing_symlink_cannot_escape_store(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    outside = tmp_path / "outside"
    store_root.mkdir()
    outside.mkdir()
    link = store_root / "sha256"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            raise
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (error, completed.stderr)
    store = LocalArtifactStore(store_root)

    with pytest.raises(ArtifactPathError, match="symlink|escapes"):
        store.publish_at("sha256/aa/value", b"unsafe")

    assert not (outside / "aa" / "value").exists()


def test_verify_detects_size_and_hash_tampering(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = store.put_bytes(b"original", media_type="application/octet-stream")
    store.path_for(ref).write_bytes(b"tampered")

    assert not store.verify(ref)
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        store.read_bytes(ref)


def test_reference_path_must_match_its_digest(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    digest = hashlib.sha256(b"value").hexdigest()
    ref = ArtifactRef(
        sha256=digest,
        size_bytes=5,
        media_type="application/octet-stream",
        relative_path=f"sha256/00/{digest}",
    )

    with pytest.raises(ArtifactIntegrityError, match="content address"):
        store.verify(ref)


def test_artifact_reference_cannot_be_mutated_or_unsafely_copied(tmp_path: Path) -> None:
    ref = LocalArtifactStore(tmp_path).put_bytes(b"value", media_type="application/octet-stream")

    with pytest.raises(ValidationError, match="frozen"):
        ref.sha256 = "0" * 64
    with pytest.raises(TypeError, match="bypass validation"):
        ref.model_copy(update={"sha256": "0" * 64})
