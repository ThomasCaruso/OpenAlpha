"""verify_deployment must cover both families of document in the image.

The sealed experiment and its three amendments are one family;
the two frozen inference diagnostic specifications are another, separately
hashed and not part of the sealed chain. The check used to cover only the
first, which made it look like it covered the image's research directory when
it covered part of it.

The function body is lifted out and executed against real directories rather
than asserted on as text, so these tests exercise the verifiers rather than
the spelling of the call.
"""

from __future__ import annotations

import ast
import inspect
import shutil
from pathlib import Path
from typing import Any

import pytest
from openalpha_bridge.diagnostic.spec import (
    V1_SPECIFICATION_NAME,
    V1_SPECIFICATION_SHA256,
    V2_SPECIFICATION_NAME,
    V2_SPECIFICATION_SHA256,
    V3_SPECIFICATION_NAME,
    V3_SPECIFICATION_SHA256,
    V4_SPECIFICATION_NAME,
    V4_SPECIFICATION_SHA256,
    verify_diagnostic_specifications,
)
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.identity import (
    AMENDMENT_1_SHA256,
    AMENDMENT_2_SHA256,
    AMENDMENT_3_SHA256,
    EXPERIMENT_SHA256,
    verify_locked_hashes,
)

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "cloud" / "modal" / "bridge_phase2_app.py"
RESEARCH = ROOT / "research" / "bridge-v0"

SEALED = {
    "experiment.yaml": EXPERIMENT_SHA256,
    "phase2-preregistration-amendment.yaml": AMENDMENT_1_SHA256,
    "phase2-amendment-2-context-prefix.yaml": AMENDMENT_2_SHA256,
    "phase2-amendment-3-scale-features.yaml": AMENDMENT_3_SHA256,
}
DIAGNOSTIC = {
    V1_SPECIFICATION_NAME: V1_SPECIFICATION_SHA256,
    V2_SPECIFICATION_NAME: V2_SPECIFICATION_SHA256,
    V3_SPECIFICATION_NAME: V3_SPECIFICATION_SHA256,
    V4_SPECIFICATION_NAME: V4_SPECIFICATION_SHA256,
}


def _function_source(name: str) -> str:
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.get_source_segment(source, function) or ""


def _run_verify_deployment(research_root: Path) -> dict[str, Any]:
    """Execute the real body against ``research_root``.

    The module imports modal at the top level, which the base environment does
    not have, and the body hard-codes the container path. Both are handled by
    lifting the function out and rebinding Path so it resolves to the directory
    under test; everything else, including both verifier calls, is the
    deployed code.
    """
    body = _function_source("verify_deployment")
    stripped = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("@app.function")
    )

    class _RootPath:
        """Resolves the container research path to the directory under test."""

        def __new__(cls, value: str):
            if value == "/root/research/bridge-v0":
                return research_root
            return Path(value)

    namespace: dict[str, Any] = {
        "Any": Any,
        "APP_NAME": "openalpha-bridge-phase2",
        "APP_VERSION": "1.0.0",
        "GPU_CONFIG": "T4",
        "PYTHON_VERSION": "3.13",
        "TORCH_VERSION": "2.13.0",
        "_require_deployed_commit": lambda: "a" * 40,
        "_now": lambda: __import__("datetime").datetime(
            2026, 8, 2, tzinfo=__import__("datetime").UTC
        ),
    }
    exec(compile(stripped, str(APP), "exec"), namespace)  # noqa: S102
    function = namespace["verify_deployment"]
    # The body does `from pathlib import Path`, so patch the module it imports.
    import pathlib

    original = pathlib.Path
    pathlib.Path = _RootPath  # type: ignore[assignment, misc]
    try:
        return function()
    finally:
        pathlib.Path = original  # type: ignore[assignment]


@pytest.fixture
def research_copy(tmp_path: Path) -> Path:
    """A byte-identical copy, so a test can perturb it without touching the real one."""
    destination = tmp_path / "bridge-v0"
    shutil.copytree(RESEARCH, destination)
    return destination


# ============================================ the verifier is actually called


def test_verify_deployment_invokes_the_diagnostic_verifier() -> None:
    body = _function_source("verify_deployment")
    assert "from openalpha_bridge.diagnostic.spec import verify_diagnostic_specifications" in body
    assert "verify_diagnostic_specifications(research_root)" in body
    assert '"diagnostic_specification_hashes": diagnostic_hashes,' in body


def test_the_response_is_not_hard_coded() -> None:
    """No literal digest appears in the body; both mappings come from verifiers."""
    body = _function_source("verify_deployment")
    for digest in (*SEALED.values(), *DIAGNOSTIC.values()):
        assert digest not in body
    assert "verify_locked_hashes(research_root)" in body


def test_both_verifiers_run_against_the_container_research_root() -> None:
    body = _function_source("verify_deployment")
    assert 'research_root = Path("/root/research/bridge-v0")' in body
    assert body.count("research_root") >= 3


# =================================================== the returned contract


def test_both_diagnostic_specifications_are_returned(research_copy: Path) -> None:
    payload = _run_verify_deployment(research_copy)
    observed = payload["diagnostic_specification_hashes"]
    assert set(observed) == {
        "phase2-frozen-inference-diagnostic.yaml",
        "phase2-frozen-inference-diagnostic-v2.yaml",
        "phase2-frozen-inference-diagnostic-v3.yaml",
        "phase2-frozen-inference-diagnostic-v4.yaml",
    }
    assert len(observed) == 4


def test_the_diagnostic_values_match_the_locked_constants(research_copy: Path) -> None:
    payload = _run_verify_deployment(research_copy)
    assert payload["diagnostic_specification_hashes"] == DIAGNOSTIC
    assert payload["diagnostic_specification_hashes"][V1_SPECIFICATION_NAME] == (
        "c909e156a61ac5b5323de9e794aad20521839c0f43f491fb356c1968f730101b"
    )
    assert payload["diagnostic_specification_hashes"][V2_SPECIFICATION_NAME] == (
        "c39fff4541afcc948cd80fcc545398312897efe723deca26554143989e4a175e"
    )
    assert payload["diagnostic_specification_hashes"][V3_SPECIFICATION_NAME] == (
        "f10076b6676a72552b1c9c96720d0087c009fc939e4667509bfcfccf7929bcb6"
    )
    assert payload["diagnostic_specification_hashes"][V4_SPECIFICATION_NAME] == (
        "bd407722adfc3ebf92eb187828d42c2c9cfa57b2fdc0406121f27d39a5c44977"
    )


def test_the_four_sealed_hashes_remain_separately_returned(research_copy: Path) -> None:
    payload = _run_verify_deployment(research_copy)
    assert payload["locked_hashes"] == SEALED
    assert len(payload["locked_hashes"]) == 4
    # Two distinct fields, and neither leaks into the other.
    assert set(payload["locked_hashes"]) & set(payload["diagnostic_specification_hashes"]) == set()


def test_the_rest_of_the_payload_is_unchanged(research_copy: Path) -> None:
    payload = _run_verify_deployment(research_copy)
    assert set(payload) == {
        "app",
        "version",
        "gpu",
        "python",
        "torch",
        "score_mask_sha256",
        "locked_hashes",
        "diagnostic_specification_hashes",
        "deployed_commit",
        "verified_at",
    }
    assert payload["python"] == "3.13"
    assert payload["torch"] == "2.13.0"
    assert payload["deployed_commit"] == "a" * 40


# ======================================================== it fails closed


@pytest.mark.parametrize("name", sorted(DIAGNOSTIC))
def test_a_missing_diagnostic_specification_fails_closed(research_copy: Path, name: str) -> None:
    (research_copy / name).unlink()
    with pytest.raises(BridgeTransformError) as excinfo:
        _run_verify_deployment(research_copy)
    failure = excinfo.value.failures[0]
    assert failure.code == "MISSING_DIAGNOSTIC_SPECIFICATION"
    assert failure.field == name


@pytest.mark.parametrize("name", sorted(DIAGNOSTIC))
def test_a_modified_diagnostic_specification_fails_closed(research_copy: Path, name: str) -> None:
    target = research_copy / name
    target.write_bytes(target.read_bytes() + b"\n# one appended comment\n")
    with pytest.raises(BridgeTransformError) as excinfo:
        _run_verify_deployment(research_copy)
    failure = excinfo.value.failures[0]
    assert failure.code == "DIAGNOSTIC_SPECIFICATION_HASH_MISMATCH"
    assert failure.field == name


@pytest.mark.parametrize("name", sorted(SEALED))
def test_a_modified_sealed_file_still_fails_closed(research_copy: Path, name: str) -> None:
    """The pre-existing behaviour, unchanged by this correction."""
    target = research_copy / name
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(BridgeTransformError) as excinfo:
        _run_verify_deployment(research_copy)
    assert excinfo.value.failures[0].code == "EXPERIMENT_HASH_MISMATCH"


def test_a_sealed_failure_is_raised_before_the_diagnostic_check(research_copy: Path) -> None:
    """Order is deliberate: the sealed chain is checked first."""
    (research_copy / "experiment.yaml").write_bytes(b"tampered\n")
    (research_copy / V2_SPECIFICATION_NAME).write_bytes(b"tampered\n")
    with pytest.raises(BridgeTransformError) as excinfo:
        _run_verify_deployment(research_copy)
    assert excinfo.value.failures[0].code == "EXPERIMENT_HASH_MISMATCH"


# ================================== nothing on disk was touched by any of this


def test_the_real_research_directory_is_untouched() -> None:
    assert verify_locked_hashes(RESEARCH) == SEALED
    assert verify_diagnostic_specifications(RESEARCH) == DIAGNOSTIC


def test_the_verifier_signature_is_what_the_app_calls() -> None:
    parameters = inspect.signature(verify_diagnostic_specifications).parameters
    assert list(parameters) == ["research_root"]
