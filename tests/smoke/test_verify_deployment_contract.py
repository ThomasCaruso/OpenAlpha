"""Historical specification verification independent of a deployment endpoint."""

from __future__ import annotations

import ast
import inspect
import shutil
from pathlib import Path

import pytest
from openalpha_kronos.studies.provenance import (
    MINI_AMENDMENT_1_SHA256 as AMENDMENT_1_SHA256,
)
from openalpha_kronos.studies.provenance import (
    MINI_AMENDMENT_2_SHA256 as AMENDMENT_2_SHA256,
)
from openalpha_kronos.studies.provenance import (
    MINI_AMENDMENT_3_SHA256 as AMENDMENT_3_SHA256,
)
from openalpha_kronos.studies.provenance import (
    MINI_EXPERIMENT_SHA256 as EXPERIMENT_SHA256,
)
from openalpha_kronos.studies.provenance import verify_locked_hashes
from openalpha_kronos.studies.structural_validity.mini.spec import (
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
from openalpha_research.failures import ResearchFailureError

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "cloud" / "modal" / "kronos_research.py"
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
    node = next(
        item
        for item in ast.walk(ast.parse(source))
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def _verify(research_root: Path) -> dict[str, dict[str, str]]:
    return {
        "locked_hashes": verify_locked_hashes(research_root),
        "diagnostic_specification_hashes": verify_diagnostic_specifications(research_root),
    }


@pytest.fixture
def research_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "bridge-v0"
    shutil.copytree(RESEARCH, destination)
    return destination


def test_research_shell_has_no_generic_deployment_endpoint() -> None:
    names = {
        node.name
        for node in ast.walk(ast.parse(APP.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
    }
    assert "verify_deployment" not in names


def test_current_modal_shell_has_no_removed_completed_study_imports() -> None:
    source = APP.read_text(encoding="utf-8")
    for removed_namespace in (
        "openalpha_bridge.diagnostic",
        "openalpha_bridge.base_study",
        "openalpha_bridge.zero_shot",
        "openalpha_bridge.phase2.identity",
    ):
        assert removed_namespace not in source


def test_modal_image_copies_the_new_study_owner() -> None:
    source = APP.read_text(encoding="utf-8")
    assert '("packages/kronos-research/src/openalpha_kronos", "/root/openalpha_kronos")' in source


@pytest.mark.parametrize(
    "function_name",
    (
        "run_mini_structural_validity",
        "run_base_structural_validity",
        "run_zero_shot_benchmark",
        "inventory_zero_shot_artifacts",
    ),
)
def test_completed_studies_use_the_research_object_store(function_name: str) -> None:
    assert "_build_research_store()" in _function_source(function_name)


def test_current_historical_hashes_match_the_locked_constants(research_copy: Path) -> None:
    payload = _verify(research_copy)
    assert payload["locked_hashes"] == SEALED
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


@pytest.mark.parametrize("name", sorted(DIAGNOSTIC))
def test_a_missing_diagnostic_specification_fails_closed(research_copy: Path, name: str) -> None:
    (research_copy / name).unlink()
    with pytest.raises(ResearchFailureError) as excinfo:
        _verify(research_copy)
    failure = excinfo.value.failures[0]
    assert failure.code == "MISSING_DIAGNOSTIC_SPECIFICATION"
    assert failure.field == name


@pytest.mark.parametrize("name", sorted(DIAGNOSTIC))
def test_a_modified_diagnostic_specification_fails_closed(research_copy: Path, name: str) -> None:
    target = research_copy / name
    target.write_bytes(target.read_bytes() + b"\n# one appended comment\n")
    with pytest.raises(ResearchFailureError) as excinfo:
        _verify(research_copy)
    failure = excinfo.value.failures[0]
    assert failure.code == "DIAGNOSTIC_SPECIFICATION_HASH_MISMATCH"
    assert failure.field == name


@pytest.mark.parametrize("name", sorted(SEALED))
def test_a_modified_sealed_file_still_fails_closed(research_copy: Path, name: str) -> None:
    target = research_copy / name
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ResearchFailureError) as excinfo:
        _verify(research_copy)
    assert excinfo.value.failures[0].code == "EXPERIMENT_HASH_MISMATCH"


def test_a_sealed_failure_is_raised_before_the_diagnostic_check(research_copy: Path) -> None:
    (research_copy / "experiment.yaml").write_bytes(b"tampered\n")
    (research_copy / V2_SPECIFICATION_NAME).write_bytes(b"tampered\n")
    with pytest.raises(ResearchFailureError) as excinfo:
        _verify(research_copy)
    assert excinfo.value.failures[0].code == "EXPERIMENT_HASH_MISMATCH"


def test_the_real_research_directory_is_untouched() -> None:
    assert verify_locked_hashes(RESEARCH) == SEALED
    assert verify_diagnostic_specifications(RESEARCH) == DIAGNOSTIC


def test_the_verifier_signature_is_what_callers_use() -> None:
    parameters = inspect.signature(verify_diagnostic_specifications).parameters
    assert list(parameters) == ["research_root"]
