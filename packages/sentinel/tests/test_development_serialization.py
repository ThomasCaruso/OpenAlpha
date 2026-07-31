import math
from pathlib import Path

import pytest
from openalpha_sentinel.development_serialization import (
    atomic_write_bytes,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


def test_canonical_json_is_order_independent_and_finite() -> None:
    expected = b'{\x22a\x22:1,\x22b\x22:2}'
    assert canonical_json_bytes({'b': 2, 'a': 1}) == expected
    assert canonical_json_bytes({'a': 1, 'b': 2}) == expected
    with pytest.raises(ValueError):
        canonical_json_bytes({'bad': math.nan})


def test_jsonl_has_declared_order_and_final_newline() -> None:
    assert canonical_jsonl_bytes(({'id': 'a'}, {'id': 'b'})) == (
        b'{\x22id\x22:\x22a\x22}\n{\x22id\x22:\x22b\x22}\n'
    )


def test_hash_and_atomic_write_are_exact(tmp_path: Path) -> None:
    payload = b'phase-3a'
    target = tmp_path / 'nested' / 'artifact.bin'

    atomic_write_bytes(target, payload)

    assert target.read_bytes() == payload
    assert sha256_bytes(payload) == '586a9b64ae3f808034ef7199f9520e181ec176c5a970d822524fc2e8be785e62'
    assert not target.with_suffix('.bin.tmp').exists()
