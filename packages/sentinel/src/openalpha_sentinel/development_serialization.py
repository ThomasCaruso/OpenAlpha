from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def canonical_json_bytes(payload: object) -> bytes:
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        )
    except (TypeError, ValueError) as error:
        raise ValueError('payload is not finite canonical JSON') from error
    return serialized.encode('utf-8')


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b''.join(canonical_json_bytes(dict(row)) + b'\n' for row in rows)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)
