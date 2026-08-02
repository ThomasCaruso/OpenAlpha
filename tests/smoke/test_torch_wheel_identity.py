"""The image installs one exact, verified Torch artifact.

`torch==2.13.0` was resolved against the cu124 index, which stops at 2.6.0. The
requirement therefore fell through to the default index and installed a
different build than the one that was named, with no hash check. These tests
pin the artifact identity so that cannot recur.

Nothing here downloads the wheel. The identity and its published hash are
asserted; pip verifies the bytes against that hash at image build time.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from openalpha_bridge.phase2.runtime_pins import (
    CANARY_RUNTIME_PINS,
    TORCH_WHEEL_FILENAME,
    TORCH_WHEEL_INDEX,
    TORCH_WHEEL_MINIMUM_DRIVER,
    TORCH_WHEEL_MINIMUM_GLIBC,
    TORCH_WHEEL_PLATFORM_TAG,
    TORCH_WHEEL_PYTHON_TAG,
    TORCH_WHEEL_SHA256,
    TORCH_WHEEL_SIZE_BYTES,
    TORCH_WHEEL_SPECIFIER,
    TORCH_WHEEL_URL,
)

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "cloud" / "modal" / "bridge_phase2_app.py"


def _app_constant(name: str) -> str:
    """Read a module-level string constant out of the Modal app by AST."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not defined in {APP.name}")


def test_the_wheel_targets_cpython_313_on_linux_x86_64() -> None:
    assert TORCH_WHEEL_PYTHON_TAG == "cp313"
    assert TORCH_WHEEL_PLATFORM_TAG == "manylinux_2_28_x86_64"
    # abi3 is not used by torch; the ABI tag repeats the python tag.
    assert TORCH_WHEEL_FILENAME.endswith("-cp313-cp313-manylinux_2_28_x86_64.whl")
    assert "aarch64" not in TORCH_WHEEL_FILENAME
    assert "win_amd64" not in TORCH_WHEEL_FILENAME
    assert "cp313t" not in TORCH_WHEEL_FILENAME  # not the free-threaded build


def test_the_wheel_is_a_cuda_build_from_the_official_index() -> None:
    assert TORCH_WHEEL_INDEX == "https://download.pytorch.org/whl/cu126"
    assert TORCH_WHEEL_URL.startswith("https://download.pytorch.org/whl/cu126/")
    # The local version segment is what makes it the CUDA build rather than the
    # default PyPI one, which carries no local version at all.
    assert "%2Bcu126" in TORCH_WHEEL_FILENAME


def test_the_wheel_version_agrees_with_the_lockfile_pin() -> None:
    assert CANARY_RUNTIME_PINS["torch"] == "2.13.0"
    assert TORCH_WHEEL_FILENAME.startswith("torch-2.13.0%2B")


def test_the_specifier_is_a_direct_url_carrying_its_hash() -> None:
    """A direct URL cannot be satisfied by a different artifact, and pip
    verifies the fragment before installing."""
    assert TORCH_WHEEL_SPECIFIER == f"{TORCH_WHEEL_URL}#sha256={TORCH_WHEEL_SHA256}"
    assert re.fullmatch(r"[0-9a-f]{64}", TORCH_WHEEL_SHA256)
    assert TORCH_WHEEL_SIZE_BYTES > 100_000_000  # a real CUDA wheel, not a stub


def test_the_compatibility_argument_is_recorded() -> None:
    assert TORCH_WHEEL_MINIMUM_GLIBC == "2.28"  # Debian bookworm ships 2.36
    assert TORCH_WHEEL_MINIMUM_DRIVER == "525.60.13"  # any CUDA 12.x driver


# ------------------------------------------------------- the image agrees


def test_the_image_installs_that_exact_artifact() -> None:
    assert _app_constant("TORCH_WHEEL_SHA256") == TORCH_WHEEL_SHA256
    assert _app_constant("TORCH_WHEEL_URL").replace("%2B", "+") == TORCH_WHEEL_URL.replace(
        "%2B", "+"
    )


def _executable_lines(path: Path) -> str:
    """Source with comment lines dropped.

    The comments deliberately name the old cu124 mistake so the next reader
    understands why the install is shaped this way. Only executable references
    are being policed here.
    """
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    return "\n".join(kept)


def test_the_image_never_resolves_torch_from_an_index() -> None:
    source = _executable_lines(APP)
    assert "TORCH_WHEEL_SPECIFIER" in source
    # The failure mode being prevented: a bare requirement plus an index that
    # does not carry it, which silently resolves elsewhere.
    assert 'f"torch=={TORCH_VERSION}"' not in source
    assert "extra_index_url" not in source
    assert "cu124" not in source


@pytest.mark.parametrize("stale", ["cu124", "torch==2.5.1"])
def test_no_stale_cuda_reference_is_executable(stale: str) -> None:
    for path in (APP, ROOT / "packages/bridge/src/openalpha_bridge/phase2/runtime_pins.py"):
        assert stale not in _executable_lines(path)
