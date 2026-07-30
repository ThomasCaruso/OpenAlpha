from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .errors import ArtifactConflictError, ArtifactIntegrityError, ArtifactPathError

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_MEDIA_TYPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_WINDOWS_INVALID = frozenset('<>:"|?*')


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sha256: Sha256
    size_bytes: int = Field(ge=0, strict=True)
    media_type: str = Field(min_length=3, max_length=255, strict=True)
    relative_path: str = Field(min_length=1, max_length=1_024, strict=True)

    def model_post_init(self, __context: object, /) -> None:
        if not _MEDIA_TYPE.fullmatch(self.media_type):
            raise ValueError("media_type must be a concrete type/subtype")

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            raise TypeError(
                "model_copy updates bypass validation; validate a new artifact reference"
            )
        return super().model_copy(update=None, deep=deep)


class LocalArtifactStore:
    """A local immutable store with portable, content-addressed references."""

    def __init__(self, root: Path | str) -> None:
        configured_root = Path(root)
        if configured_root.exists() and configured_root.is_symlink():
            raise ArtifactPathError("artifact store root cannot be a symlink")
        configured_root.mkdir(parents=True, exist_ok=True)
        self.root = configured_root.resolve(strict=True)

    def put_bytes(self, value: bytes, media_type: str) -> ArtifactRef:
        if not isinstance(value, bytes):
            raise TypeError("artifact value must be bytes")
        digest = hashlib.sha256(value).hexdigest()
        relative_path = f"sha256/{digest[:2]}/{digest}"
        ref = ArtifactRef(
            sha256=digest,
            size_bytes=len(value),
            media_type=media_type,
            relative_path=relative_path,
        )
        self.publish_at(relative_path, value)
        return ref

    def publish_at(self, relative_path: str, value: bytes) -> None:
        if not isinstance(value, bytes):
            raise TypeError("artifact value must be bytes")
        target = self._resolve(relative_path)
        self._ensure_safe_parents(target.parent)

        if target.exists():
            self._accept_identical_or_raise(target, value)
            return
        digest = hashlib.sha256(value).hexdigest()
        expected = f"sha256/{digest[:2]}/{digest}"
        if relative_path != expected:
            raise ArtifactIntegrityError(
                "new artifact publication path does not match its content address"
            )

        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            if hashlib.sha256(temporary.read_bytes()).digest() != hashlib.sha256(value).digest():
                raise ArtifactIntegrityError("temporary artifact hash mismatch")

            self._assert_no_symlink_components(target)
            try:
                os.link(temporary, target)
            except FileExistsError:
                self._accept_identical_or_raise(target, value)
            except OSError as error:
                raise ArtifactIntegrityError(
                    f"atomic artifact publication failed for {relative_path}"
                ) from error
            self._sync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def path_for(self, ref: ArtifactRef) -> Path:
        expected = f"sha256/{ref.sha256[:2]}/{ref.sha256}"
        if ref.relative_path != expected:
            raise ArtifactIntegrityError(
                "artifact reference path does not match its content address"
            )
        return self._resolve(ref.relative_path)

    def verify(self, ref: ArtifactRef) -> bool:
        path = self.path_for(ref)
        if not path.is_file() or path.is_symlink():
            return False
        stat = path.stat()
        if stat.st_size != ref.size_bytes:
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == ref.sha256

    def read_bytes(self, ref: ArtifactRef) -> bytes:
        path = self.path_for(ref)
        if not path.is_file() or path.is_symlink():
            raise ArtifactIntegrityError(f"missing artifact: {ref.relative_path}")
        value = path.read_bytes()
        if len(value) != ref.size_bytes:
            raise ArtifactIntegrityError(
                f"artifact size mismatch for {ref.relative_path}: "
                f"expected {ref.size_bytes}, observed {len(value)}"
            )
        observed = hashlib.sha256(value).hexdigest()
        if observed != ref.sha256:
            raise ArtifactIntegrityError(
                f"artifact hash mismatch for {ref.relative_path}: "
                f"expected {ref.sha256}, observed {observed}"
            )
        return value

    def _resolve(self, relative_path: str) -> Path:
        parts = self._validate_relative_path(relative_path)
        target = self.root.joinpath(*parts)
        self._assert_no_symlink_components(target)
        resolved = target.resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise ArtifactPathError("artifact path escapes the configured store root")
        return target

    def _validate_relative_path(self, relative_path: str) -> tuple[str, ...]:
        if not isinstance(relative_path, str) or not relative_path:
            raise ArtifactPathError("artifact path must be a non-empty relative string")
        if "\\" in relative_path or "//" in relative_path:
            raise ArtifactPathError("artifact paths must use canonical forward slashes")
        path = PurePosixPath(relative_path)
        if path.is_absolute() or re.match(r"^[A-Za-z]:", relative_path):
            raise ArtifactPathError("absolute artifact paths are forbidden")
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ArtifactPathError("artifact path traversal is forbidden")
        for part in path.parts:
            if part.endswith((" ", ".")) or any(char in _WINDOWS_INVALID for char in part):
                raise ArtifactPathError("artifact path is not portable")
            device_name = part.rstrip(" .").split(".", maxsplit=1)[0].upper()
            if device_name in _WINDOWS_RESERVED:
                raise ArtifactPathError("reserved device name in artifact path")
            if any(ord(char) < 32 for char in part):
                raise ArtifactPathError("control character in artifact path")
        return path.parts

    def _ensure_safe_parents(self, parent: Path) -> None:
        self._assert_no_symlink_components(parent)
        parent.mkdir(parents=True, exist_ok=True)
        self._assert_no_symlink_components(parent)

    def _assert_no_symlink_components(self, target: Path) -> None:
        current = self.root
        try:
            relative_parts = target.relative_to(self.root).parts
        except ValueError as error:
            raise ArtifactPathError("artifact path escapes the configured store root") from error
        for part in relative_parts:
            current = current / part
            if current.is_symlink():
                raise ArtifactPathError("symlink components are forbidden in artifact paths")

    @staticmethod
    def _accept_identical_or_raise(target: Path, value: bytes) -> None:
        if target.is_symlink() or not target.is_file():
            raise ArtifactConflictError(f"artifact path is not a regular file: {target.name}")
        if target.read_bytes() != value:
            raise ArtifactConflictError(
                f"immutable artifact path already contains different content: {target.name}"
            )

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
