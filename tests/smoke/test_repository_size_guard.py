"""No weight-sized file may enter the repository.

The Kronos-base weights are 409 MB and the base tokenizer's are 15.8 MB. Both
belong in a remote volume, never in Git, Git LFS, the Modal image, a fixture or
a temporary directory. A committed weight would be effectively permanent, since
removing it later means rewriting history that a completed study's provenance
depends on.

The guard is forward-looking on purpose. Files already tracked at the time it
was written are recorded with their sizes and grandfathered, so it rejects new
mistakes without retroactively condemning legitimate existing research data.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Comfortably above any source file or research table, far below any weight.
MAXIMUM_NEW_FILE_BYTES = 10 * 1024 * 1024

#: Files already tracked when this guard was introduced, with the size each
#: had then. They may shrink or stay the same; growth past the threshold is
#: still refused. Adding an entry here is a deliberate, reviewable act.
ALLOWLIST: dict[str, int] = {
    # Sentinel v1.1 tokenizer round-trip results: preserved research evidence,
    # hash-sealed by its own manifest and not regenerable from the repository.
    "research/sentinel-v1_1/tokenizer_roundtrip/results.json": 6_474_809,
}

#: Suffixes that are model weights whatever their size. A 2 KB stub named
#: model.safetensors is still a weight file being smuggled in as a fixture.
WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".onnx", ".gguf"})

#: Virtualenv path files are legitimately ``.pth`` and are never tracked, but
#: the check runs over tracked files only, so no exception is needed.


def _tracked() -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [name for name in output.split("\0") if name]


def test_no_tracked_file_exceeds_the_threshold_unless_allowlisted() -> None:
    offenders: list[tuple[str, int]] = []
    for name in _tracked():
        path = REPO / name
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size <= MAXIMUM_NEW_FILE_BYTES:
            continue
        allowed = ALLOWLIST.get(name)
        if allowed is not None and size <= allowed:
            continue
        offenders.append((name, size))

    assert offenders == [], (
        f"tracked files exceed {MAXIMUM_NEW_FILE_BYTES} bytes and are not allowlisted: {offenders}"
    )


def test_no_weight_file_is_tracked_at_any_size() -> None:
    offenders = [name for name in _tracked() if Path(name).suffix.lower() in WEIGHT_SUFFIXES]
    assert offenders == [], f"model weight files are tracked: {offenders}"


def test_the_allowlist_only_names_files_that_exist() -> None:
    """A stale entry would silently widen the guard."""
    tracked = set(_tracked())
    for name, size in ALLOWLIST.items():
        assert name in tracked, f"allowlisted file is no longer tracked: {name}"
        assert (REPO / name).stat().st_size <= size, (
            f"{name} grew past its recorded allowance of {size} bytes"
        )


def test_the_threshold_leaves_room_for_source_but_not_for_weights() -> None:
    """A guard set above the base tokenizer would not be a guard."""
    base_tokenizer_weights = 15_842_368
    base_model_weights = 409_264_008
    assert MAXIMUM_NEW_FILE_BYTES < base_tokenizer_weights
    assert MAXIMUM_NEW_FILE_BYTES < base_model_weights

    largest_source = max(
        (REPO / name).stat().st_size
        for name in _tracked()
        if name.endswith(".py") and (REPO / name).is_file()
    )
    assert largest_source < MAXIMUM_NEW_FILE_BYTES


@pytest.mark.parametrize(
    "candidate",
    ["model.safetensors", "weights/model.bin", "fixtures/tiny.ckpt", "cache/kronos.pt"],
)
def test_the_weight_check_would_catch_a_smuggled_fixture(candidate: str) -> None:
    assert Path(candidate).suffix.lower() in WEIGHT_SUFFIXES
