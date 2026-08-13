"""The remote-cache inventory refreshes its view before it reports one.

The failure this exists to prevent: the base runtime probe downloaded and
committed roughly 425 MB into the dedicated volume, and the inventory -- served
by a warm container that had mounted the volume while it was still empty --
reported ``total_bytes: 0``. Nothing was wrong with the storage. The reader
described a stale mount as fact.

These tests are behavioural. A fake volume flips a real temporary directory
from its stale state to its current one when ``reload()`` is called, so
"reloaded before inspection" is observed rather than asserted about source
text. The AST checks at the end supplement that; they do not stand in for it.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from openalpha_kronos.studies.structural_validity.base.cache_inventory import (
    build_base_cache_inventory,
    walk_cache_root,
)
from openalpha_kronos.studies.structural_validity.base.spec import (
    BASE_SPECIFICATION_NAME,
    BASE_SPECIFICATION_SHA256,
    KRONOS_BASE_SPEC,
    KRONOS_BASE_TOKENIZER_SPEC,
)

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "cloud" / "modal" / "kronos_research.py"
RESEARCH = REPO / "research" / "bridge-v0"
NOW = datetime(2026, 8, 3, tzinfo=UTC)
COMMIT = "f" * 40

MODEL_CACHE_DIR = "models--NeoQuasar--Kronos-base"
TOKENIZER_CACHE_DIR = "models--NeoQuasar--Kronos-Tokenizer-base"


class _FakeVolume:
    """A mounted volume whose contents only appear once it is reloaded.

    Stands in for the real Modal behaviour: a warm container's mount is a
    snapshot taken when it started, and another container's commit is invisible
    until the reader refreshes.
    """

    def __init__(self, *, mount: Path, materialise, fail_with: Exception | None = None) -> None:
        self._mount = mount
        self._materialise = materialise
        self._fail_with = fail_with
        self.reload_calls = 0
        #: What the filesystem looked like each time reload() was invoked, so a
        #: test can prove nothing was observed beforehand.
        self.observed_before_each_reload: list[bool] = []

    def reload(self) -> None:
        self.observed_before_each_reload.append((self._mount / "huggingface").is_dir())
        self.reload_calls += 1
        if self._fail_with is not None:
            raise self._fail_with
        self._materialise()

    def commit(self) -> None:  # pragma: no cover - must never be called here
        raise AssertionError("the inventory must never commit")


def _write_snapshot(
    root: Path, *, cache_dir: str, revision: str, weights_bytes: int, config_bytes: int
) -> None:
    """A minimal but realistic Hugging Face snapshot layout.

    Tiny stand-ins for the real files: this suite never materialises a weight,
    and the sizes are chosen only so the totals are checkable.
    """
    blobs = root / cache_dir / "blobs"
    snapshot = root / cache_dir / "snapshots" / revision
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_bytes(b"c" * config_bytes)
    (snapshot / "model.safetensors").write_bytes(b"w" * weights_bytes)
    (root / cache_dir / "refs").mkdir(parents=True, exist_ok=True)
    (root / cache_dir / "refs" / "main").write_bytes(revision.encode())


def _pair_materialiser(mount: Path):
    """Populate the mount with exactly the two pinned base repositories."""

    def materialise() -> None:
        root = mount / "huggingface"
        _write_snapshot(
            root,
            cache_dir=MODEL_CACHE_DIR,
            revision=KRONOS_BASE_SPEC.revision,
            weights_bytes=4096,
            config_bytes=KRONOS_BASE_SPEC.config_size_bytes,
        )
        _write_snapshot(
            root,
            cache_dir=TOKENIZER_CACHE_DIR,
            revision=KRONOS_BASE_TOKENIZER_SPEC.revision,
            weights_bytes=2048,
            config_bytes=301,
        )

    return materialise


def _inventory(mount: Path, volume: _FakeVolume) -> dict[str, Any]:
    return build_base_cache_inventory(
        reload=volume.reload,
        mount=str(mount),
        volume_name="openalpha-kronos-base-cache",
        deployed_commit=COMMIT,
        inspected_at=NOW,
    )


# ============================================ 1-2: reload, once, and first


def test_reload_is_called_exactly_once(tmp_path: Path) -> None:
    volume = _FakeVolume(mount=tmp_path, materialise=_pair_materialiser(tmp_path))
    _inventory(tmp_path, volume)
    assert volume.reload_calls == 1


def test_nothing_is_observed_before_the_reload(tmp_path: Path) -> None:
    """The stale view is empty; if anything looked first, it would see that."""
    volume = _FakeVolume(mount=tmp_path, materialise=_pair_materialiser(tmp_path))
    payload = _inventory(tmp_path, volume)

    # The mount really was empty at the moment reload ran, so every observation
    # in the payload must have come from after it.
    assert volume.observed_before_each_reload == [False]
    assert payload["exists"] is True
    assert payload["total_bytes"] > 0


def test_a_stale_warm_container_view_becomes_current_after_reload(tmp_path: Path) -> None:
    """The exact failure: 425 MB committed elsewhere, reported as zero."""
    stale_root = tmp_path / "huggingface"
    assert not stale_root.exists(), "the container mounted an empty volume"

    # Without a reload, this is what the inventory would have said.
    repositories, total = walk_cache_root(stale_root)
    assert (repositories, total) == ([], 0)

    volume = _FakeVolume(mount=tmp_path, materialise=_pair_materialiser(tmp_path))
    payload = _inventory(tmp_path, volume)

    assert payload["exists"] is True
    assert payload["total_bytes"] > 0
    assert {entry["repository"] for entry in payload["repositories"]} == {
        "NeoQuasar/Kronos-base",
        "NeoQuasar/Kronos-Tokenizer-base",
    }
    revisions = {
        entry["repository"]: [s["revision"] for s in entry["snapshots"]]
        for entry in payload["repositories"]
    }
    assert revisions["NeoQuasar/Kronos-base"] == [KRONOS_BASE_SPEC.revision]
    assert revisions["NeoQuasar/Kronos-Tokenizer-base"] == [KRONOS_BASE_TOKENIZER_SPEC.revision]
    for entry in payload["repositories"]:
        for snapshot in entry["snapshots"]:
            assert sorted(snapshot["files"]) == ["config.json", "model.safetensors"]


def test_the_reported_bytes_come_from_the_reloaded_view(tmp_path: Path) -> None:
    volume = _FakeVolume(mount=tmp_path, materialise=_pair_materialiser(tmp_path))
    payload = _inventory(tmp_path, volume)

    expected = 4096 + KRONOS_BASE_SPEC.config_size_bytes + 2048 + 301
    # Plus the two refs/main files the layout writes.
    expected += len(KRONOS_BASE_SPEC.revision) + len(KRONOS_BASE_TOKENIZER_SPEC.revision)
    assert payload["total_bytes"] == expected


# ================================================ 3: still read-only


def test_the_inventory_never_writes_commits_or_deletes(tmp_path: Path) -> None:
    """A commit from the fake volume would raise; deletion would show up here."""
    volume = _FakeVolume(mount=tmp_path, materialise=_pair_materialiser(tmp_path))
    _inventory(tmp_path, volume)

    root = tmp_path / "huggingface"
    surviving = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert surviving == {
        f"{MODEL_CACHE_DIR}/refs/main",
        f"{MODEL_CACHE_DIR}/snapshots/{KRONOS_BASE_SPEC.revision}/config.json",
        f"{MODEL_CACHE_DIR}/snapshots/{KRONOS_BASE_SPEC.revision}/model.safetensors",
        f"{TOKENIZER_CACHE_DIR}/refs/main",
        f"{TOKENIZER_CACHE_DIR}/snapshots/{KRONOS_BASE_TOKENIZER_SPEC.revision}/config.json",
        f"{TOKENIZER_CACHE_DIR}/snapshots/{KRONOS_BASE_TOKENIZER_SPEC.revision}/model.safetensors",
    }

    # A second inventory sees exactly the same thing.
    second = _FakeVolume(mount=tmp_path, materialise=_pair_materialiser(tmp_path))
    again = _inventory(tmp_path, second)
    assert again["total_bytes"] > 0
    assert again["read_only"] is True
    assert again["deletion_supported"] is False


def test_the_inventory_source_has_no_mutating_call() -> None:
    import openalpha_kronos.studies.structural_validity.base.cache_inventory as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if name:
                called.add(name)
    # ``str.replace`` is deliberately absent from this list: the inventory uses
    # it to turn a cache directory name into a repository name, and it shares a
    # name with the destructive ``Path.replace``.
    for mutating in (
        "commit",
        "unlink",
        "rmtree",
        "remove",
        "rmdir",
        "write_bytes",
        "write_text",
        "mkdir",
        "touch",
        "rename",
        "chmod",
        "symlink_to",
    ):
        assert mutating not in called, f"the inventory calls {mutating}"


# =============================================== 4: failure propagates


def test_a_reload_failure_propagates_instead_of_reporting_empty(tmp_path: Path) -> None:
    class _VolumeUnavailable(RuntimeError):
        pass

    volume = _FakeVolume(
        mount=tmp_path,
        materialise=_pair_materialiser(tmp_path),
        fail_with=_VolumeUnavailable("the volume could not be refreshed"),
    )
    with pytest.raises(_VolumeUnavailable):
        _inventory(tmp_path, volume)

    assert volume.reload_calls == 1, "a failed reload must not be retried"
    # Nothing was materialised, and no payload claiming an empty cache exists.
    assert not (tmp_path / "huggingface").exists()


def test_a_reload_failure_is_not_caught_anywhere_in_the_builder() -> None:
    import openalpha_kronos.studies.structural_validity.base.cache_inventory as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    builder = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "build_base_cache_inventory"
    )
    assert not [n for n in ast.walk(builder) if isinstance(n, ast.Try)], (
        "the builder must not catch anything; a failed reload has to be visible"
    )
    # And the reload really is the first statement executed.
    body = [
        n
        for n in builder.body
        if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)
    ]
    first = body[0]
    assert isinstance(first, ast.Expr)
    assert isinstance(first.value, ast.Call)
    assert isinstance(first.value.func, ast.Name)
    assert first.value.func.id == "reload"


# ==================================== 5-6: payload compatibility


def test_the_payload_keeps_every_previous_field(tmp_path: Path) -> None:
    volume = _FakeVolume(mount=tmp_path, materialise=_pair_materialiser(tmp_path))
    payload = _inventory(tmp_path, volume)

    previous = {
        "volume",
        "mount",
        "root",
        "exists",
        "repositories",
        "total_bytes",
        "read_only",
        "deletion_supported",
        "deployed_commit",
        "inspected_at",
    }
    assert previous <= set(payload), sorted(previous - set(payload))
    # Exactly one new field, and it is operational metadata.
    assert set(payload) - previous == {"reloaded_before_inspection"}
    assert payload["reloaded_before_inspection"] is True

    assert payload["volume"] == "openalpha-kronos-base-cache"
    assert payload["mount"] == str(tmp_path)
    assert payload["root"] == str(tmp_path / "huggingface")
    assert payload["deployed_commit"] == COMMIT
    assert payload["inspected_at"] == NOW.isoformat()
    # Still a plain JSON document.
    assert json.loads(json.dumps(payload)) == payload


def test_an_empty_volume_still_reports_empty_after_a_real_reload(tmp_path: Path) -> None:
    """Empty is a legitimate answer -- once it has actually been observed."""
    volume = _FakeVolume(mount=tmp_path, materialise=lambda: None)
    payload = _inventory(tmp_path, volume)
    assert volume.reload_calls == 1
    assert payload["exists"] is False
    assert payload["repositories"] == []
    assert payload["total_bytes"] == 0
    assert payload["reloaded_before_inspection"] is True


def test_unrelated_directories_are_ignored_not_reported(tmp_path: Path) -> None:
    def materialise() -> None:
        root = tmp_path / "huggingface"
        (root / "xet").mkdir(parents=True, exist_ok=True)
        (root / "xet" / "chunk").write_bytes(b"x" * 16)
        _write_snapshot(
            root,
            cache_dir=MODEL_CACHE_DIR,
            revision=KRONOS_BASE_SPEC.revision,
            weights_bytes=64,
            config_bytes=228,
        )

    volume = _FakeVolume(mount=tmp_path, materialise=materialise)
    payload = _inventory(tmp_path, volume)
    assert [e["repository"] for e in payload["repositories"]] == ["NeoQuasar/Kronos-base"]
    # The non-repository directory contributes nothing to the repository total.
    assert payload["total_bytes"] == 64 + 228 + len(KRONOS_BASE_SPEC.revision)


# ======================= the Modal shell wires the real volume in


def _modal_function(name: str) -> ast.FunctionDef:
    source = APP.read_text(encoding="utf-8")
    return next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def test_the_modal_inventory_passes_the_real_volume_reload_and_observes_nothing() -> None:
    source = APP.read_text(encoding="utf-8")
    node = _modal_function("inventory_base_artifacts")
    body = ast.get_source_segment(source, node) or ""

    assert "reload=base_cache_volume.reload" in body
    without_docstring = ast.FunctionDef(
        name=node.name,
        args=node.args,
        body=[
            n
            for n in node.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
        ],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    code = ast.unparse(ast.fix_missing_locations(without_docstring))
    assert code.count("base_cache_volume.reload") == 1
    assert "base_cache_volume.commit" not in code
    assert "build_base_cache_inventory(" in body
    assert "base_cache_volume.commit()" not in body

    # The shell performs no filesystem observation of its own, so there is no
    # way for it to look before the builder reloads.
    for observation in ("is_dir(", "iterdir(", "rglob(", ".stat(", "exists("):
        assert observation not in body, f"the Modal shell observes the mount: {observation}"


def test_the_modal_inventory_is_still_read_only_and_off_the_control_api() -> None:
    source = APP.read_text(encoding="utf-8")
    body = ast.get_source_segment(source, _modal_function("inventory_base_artifacts")) or ""
    for destructive in (
        "rmtree",
        "unlink(",
        "os.remove",
        "shutil.rm",
        ".delete(",
        "base_cache_volume.commit",
    ):
        assert destructive not in body

    assert "control_api" not in source


def test_the_probe_and_diagnostic_were_not_changed_to_work_around_inventory() -> None:
    source = APP.read_text(encoding="utf-8")
    for name in ("verify_base_runtime", "run_base_structural_validity"):
        body = ast.get_source_segment(source, _modal_function(name)) or ""
        assert body.count("base_cache_volume.commit()") == 1
        assert "base_cache_volume.reload()" not in body
        assert "build_base_cache_inventory" not in body
        assert "_download_base_pair(cache_root)" in body


# ============================ 7-12: nothing scientific moved


def test_no_scientific_schema_or_identity_changed() -> None:
    from openalpha_kronos.studies.structural_validity.base.spec import (
        BASE_ARTIFACT_ROOT,
        BASE_FAILURE_SCHEMA_VERSION,
        BASE_PROBE_SCHEMA_VERSION,
        BASE_RUN_ID_PATTERN,
        BASE_SUCCESS_SCHEMA_VERSION,
    )

    assert BASE_SUCCESS_SCHEMA_VERSION == "openalpha.bridge.base_study.kronos_base_diagnostic.v1"
    assert BASE_FAILURE_SCHEMA_VERSION == "openalpha.bridge.base_study.kronos_base_failure.v1"
    assert BASE_PROBE_SCHEMA_VERSION == "openalpha.bridge.base_study.runtime_probe.v1"
    assert BASE_ARTIFACT_ROOT == "openalpha-compatibility/kronos-base-diagnostic"
    assert BASE_RUN_ID_PATTERN.pattern == r"^base_[0-9a-f]{8,32}$"
    assert KRONOS_BASE_SPEC.weights_sha256 == (
        "abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83"
    )
    assert KRONOS_BASE_TOKENIZER_SPEC.weights_sha256 == (
        "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
    )


def test_the_operational_field_is_absent_from_every_scientific_model() -> None:
    from openalpha_kronos.studies.structural_validity.base.artifact import BaseStudyFailure
    from openalpha_kronos.studies.structural_validity.base.runner import (
        KronosBaseDiagnosticArtifact,
    )
    from openalpha_kronos.studies.structural_validity.base.runtime_probe import (
        BaseRuntimeProbeResult,
    )

    for model in (KronosBaseDiagnosticArtifact, BaseStudyFailure, BaseRuntimeProbeResult):
        assert "reloaded_before_inspection" not in model.model_fields
    assert "reloaded_before_inspection" not in (RESEARCH / BASE_SPECIFICATION_NAME).read_text(
        encoding="utf-8"
    )


def test_every_specification_hash_is_unchanged() -> None:
    expected = {
        BASE_SPECIFICATION_NAME: BASE_SPECIFICATION_SHA256,
        "experiment.yaml": "d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2",
        "phase2-preregistration-amendment.yaml": (
            "4c60846e8cb791d922c29a5e704fe3a473740171dd2bbb7b3390796ceeeef3c8"
        ),
        "phase2-amendment-2-context-prefix.yaml": (
            "66f3c8171c2805bccf4b924125bd206edad27c957dfff40a0abf3864fbab10c1"
        ),
        "phase2-amendment-3-scale-features.yaml": (
            "d9484020a22df42c93edc19941374c628ea891f08c8bee8151b82264c00bb12b"
        ),
        "phase2-frozen-inference-diagnostic.yaml": (
            "c909e156a61ac5b5323de9e794aad20521839c0f43f491fb356c1968f730101b"
        ),
        "phase2-frozen-inference-diagnostic-v2.yaml": (
            "c39fff4541afcc948cd80fcc545398312897efe723deca26554143989e4a175e"
        ),
        "phase2-frozen-inference-diagnostic-v3.yaml": (
            "f10076b6676a72552b1c9c96720d0087c009fc939e4667509bfcfccf7929bcb6"
        ),
        "phase2-frozen-inference-diagnostic-v4.yaml": (
            "bd407722adfc3ebf92eb187828d42c2c9cfa57b2fdc0406121f27d39a5c44977"
        ),
    }
    for name, digest in expected.items():
        observed = hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest()
        assert observed == digest, f"{name} changed: {observed}"


def test_snapshot_download_behaviour_is_untouched() -> None:
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    downloading = {
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and "snapshot_download(" in (ast.get_source_segment(source, n) or "")
    }
    assert downloading == {
        "run_mini_structural_validity",
        "verify_mini_runtime",
        "_download_base_pair",
    }
    helper = ast.get_source_segment(source, _modal_function("_download_base_pair")) or ""
    assert helper.count("snapshot_download(") == 2
    assert helper.count("allow_patterns=list(OFFICIAL_SNAPSHOT_ALLOW_PATTERNS)") == 2


def test_the_inventory_module_cannot_download_anything() -> None:
    import openalpha_kronos.studies.structural_validity.base.cache_inventory as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for verb in ("snapshot_download", "hf_hub_download", "from_pretrained", "urlopen", "requests"):
        assert verb not in source, f"the inventory references {verb}"
