from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["canonical_json", "canonical_sha256", "file_sha256"]


def canonical_json(value: Any) -> bytes:
    """Encode JSON deterministically for content addressing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
