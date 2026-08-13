import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from openalpha_research.artifacts import ArtifactRef, LocalArtifactStore
from openalpha_research.errors import ManifestIntegrityError
from openalpha_research.manifest import (
    REQUIRED_COMPLETED_ARTIFACT_KINDS,
    ArtifactKind,
    EnvironmentMetadata,
    GitMetadata,
    ManifestArtifact,
    ManifestProfile,
    MethodologyStatus,
    RunManifest,
    canonical_manifest_bytes,
    publish_manifest,
)
from openalpha_research.run_state import RunState
from pydantic import ValidationError

NOW = datetime(2026, 7, 29, 6, 0, tzinfo=UTC)
SHA = "a" * 64


def test_completed_manifest_is_canonical_and_content_addressed(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _valid_manifest(store)

    published = publish_manifest(store, manifest)

    assert store.verify(published)
    assert published.media_type == "application/vnd.openalpha.run-manifest+json"
    assert store.read_bytes(published) == canonical_manifest_bytes(manifest)
    assert published.sha256 == hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def test_manifest_canonical_bytes_do_not_depend_on_inventory_order(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _valid_manifest(store)
    reordered = _replace_manifest(manifest, artifacts=tuple(reversed(manifest.artifacts)))

    assert canonical_manifest_bytes(manifest) == canonical_manifest_bytes(reordered)


def test_completed_manifest_rejects_missing_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _valid_manifest(store)
    unknown = ArtifactRef(
        sha256="f" * 64,
        size_bytes=2,
        media_type="application/json",
        relative_path="sha256/ff/" + "f" * 64,
    )
    unknown_forecasts = _artifact(
        ArtifactKind.FORECASTS,
        unknown,
        experiment_id=manifest.experiment_id,
        input_sha256=(manifest.artifacts[0].ref.sha256,),
    )
    artifacts = tuple(
        unknown_forecasts if artifact.kind is ArtifactKind.FORECASTS else artifact
        for artifact in manifest.artifacts
    )

    with pytest.raises(ManifestIntegrityError, match="missing artifact"):
        publish_manifest(store, _replace_manifest(manifest, artifacts=artifacts))


def test_completed_manifest_rejects_mismatched_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _valid_manifest(store)
    target = manifest.artifacts[1].ref
    store.path_for(target).write_bytes(b"x" * target.size_bytes)

    with pytest.raises(ManifestIntegrityError, match="hash mismatch"):
        publish_manifest(store, manifest)


def test_completed_manifest_requires_spec_and_methodology_audit(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _valid_manifest(store)
    without_audit = tuple(
        artifact
        for artifact in manifest.artifacts
        if artifact.kind is not ArtifactKind.METHODOLOGY_AUDIT
    )

    with pytest.raises(ManifestIntegrityError, match="methodology_audit"):
        publish_manifest(store, _replace_manifest(manifest, artifacts=without_audit))


def test_completed_manifest_requires_the_full_core_artifact_inventory(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _valid_manifest(store)
    without_snapshot = tuple(
        artifact
        for artifact in manifest.artifacts
        if artifact.kind is not ArtifactKind.DATA_SNAPSHOT
    )

    with pytest.raises(ManifestIntegrityError, match="data_snapshot"):
        publish_manifest(store, _replace_manifest(manifest, artifacts=without_snapshot))


def test_forecast_evaluation_profile_requires_evidence_not_trading_artifacts(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    legacy = _valid_manifest(store)
    required = {
        ArtifactKind.CANONICAL_SPEC,
        ArtifactKind.DATA_SNAPSHOT,
        ArtifactKind.DATA_QUALITY,
        ArtifactKind.FORECAST_ORIGINS,
        ArtifactKind.FORECASTS,
        ArtifactKind.DIAGNOSTICS,
        ArtifactKind.FORECAST_METRICS,
        ArtifactKind.METHODOLOGY_AUDIT,
    }
    spec = next(
        artifact for artifact in legacy.artifacts if artifact.kind is ArtifactKind.CANONICAL_SPEC
    )
    selected: list[ManifestArtifact] = []
    for artifact in legacy.artifacts:
        if artifact.kind not in required:
            continue
        payload = artifact.model_dump(mode="python")
        payload["input_sha256"] = () if artifact is spec else (spec.ref.sha256,)
        selected.append(ManifestArtifact.model_validate(payload))
    diagnostics_ref = store.put_bytes(
        b'{"kind":"diagnostics","schema_version":"1.0"}',
        media_type="application/json",
    )
    selected.append(
        _artifact(
            ArtifactKind.DIAGNOSTICS,
            diagnostics_ref,
            experiment_id=legacy.experiment_id,
            input_sha256=(spec.ref.sha256,),
        )
    )
    manifest = _replace_manifest(
        legacy,
        profile=ManifestProfile.FORECAST_EVALUATION,
        artifacts=tuple(selected),
    )

    published = publish_manifest(store, manifest)

    assert store.verify(published)
    assert all(
        artifact.kind
        not in {
            ArtifactKind.ORDERS,
            ArtifactKind.FILLS,
            ArtifactKind.ACCOUNTING_LEDGER,
        }
        for artifact in manifest.artifacts
    )


@pytest.mark.parametrize("state", (RunState.RUNNING, RunState.FAILED, RunState.INVALID))
def test_only_completed_runs_can_publish(tmp_path: Path, state: RunState) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _replace_manifest(_valid_manifest(store), state=state)

    with pytest.raises(ManifestIntegrityError, match="completed"):
        publish_manifest(store, manifest)


def test_failed_methodology_cannot_publish_as_completed(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _valid_manifest(store, audit_status=MethodologyStatus.FAILED)

    with pytest.raises(ManifestIntegrityError, match="methodology.*failed"):
        publish_manifest(store, manifest)


def test_manifest_status_must_match_verified_audit_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _valid_manifest(store, audit_status=MethodologyStatus.PASSED_WITH_WARNINGS)
    manifest = _replace_manifest(manifest, methodology_status=MethodologyStatus.PASSED)

    with pytest.raises(ManifestIntegrityError, match="does not match"):
        publish_manifest(store, manifest)


def test_experiment_identity_must_match_canonical_spec(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _replace_manifest(_valid_manifest(store), experiment_id="exp_" + "0" * 64)

    with pytest.raises(ManifestIntegrityError, match="experiment identity"):
        publish_manifest(store, manifest)


def test_artifact_lineage_must_match_manifest_identity(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _valid_manifest(store)
    first = manifest.artifacts[0]
    payload = first.model_dump(mode="python")
    payload["run_id"] = "run_other"
    forged = ManifestArtifact.model_validate(payload)
    artifacts = (forged, *manifest.artifacts[1:])

    with pytest.raises(ManifestIntegrityError, match="lineage.*run"):
        publish_manifest(store, _replace_manifest(manifest, artifacts=artifacts))


def test_artifact_lineage_must_be_acyclic(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _valid_manifest(store)
    target = next(
        artifact for artifact in manifest.artifacts if artifact.kind is ArtifactKind.FORECASTS
    )
    payload = target.model_dump(mode="python")
    payload["input_sha256"] = (target.ref.sha256,)
    cyclic = ManifestArtifact.model_validate(payload)
    artifacts = tuple(cyclic if artifact is target else artifact for artifact in manifest.artifacts)

    with pytest.raises(ManifestIntegrityError, match="cycle"):
        publish_manifest(store, _replace_manifest(manifest, artifacts=artifacts))


def test_spec_and_audit_json_schema_versions_are_verified(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ManifestIntegrityError, match="specification.*schema"):
        publish_manifest(store, _valid_manifest(store, spec_schema_version="2.0"))
    with pytest.raises(ManifestIntegrityError, match="audit.*schema"):
        publish_manifest(store, _valid_manifest(store, audit_schema_version="2.0"))


def test_spec_and_audit_require_json_media_types(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest = _valid_manifest(store)
    spec = next(
        artifact for artifact in manifest.artifacts if artifact.kind is ArtifactKind.CANONICAL_SPEC
    )
    payload = spec.model_dump(mode="python")
    payload["ref"] = ArtifactRef(
        sha256=spec.ref.sha256,
        size_bytes=spec.ref.size_bytes,
        media_type="application/octet-stream",
        relative_path=spec.ref.relative_path,
    )
    wrong_media = ManifestArtifact.model_validate(payload)
    artifacts = tuple(
        wrong_media if artifact is spec else artifact for artifact in manifest.artifacts
    )

    with pytest.raises(ManifestIntegrityError, match="canonical_spec.*media type"):
        publish_manifest(store, _replace_manifest(manifest, artifacts=artifacts))


def test_manifest_models_cannot_be_mutated_or_unsafely_copied(tmp_path: Path) -> None:
    manifest = _valid_manifest(LocalArtifactStore(tmp_path))

    with pytest.raises(ValidationError, match="frozen"):
        manifest.run_id = "run_other"
    with pytest.raises(TypeError, match="bypass validation"):
        manifest.model_copy(update={"state": RunState.RUNNING})
    with pytest.raises(TypeError, match="bypass validation"):
        manifest.artifacts[0].model_copy(update={"producer": ""})


def test_dirty_git_state_requires_diff_hash() -> None:
    with pytest.raises(ValidationError, match="diff_sha256"):
        GitMetadata(commit_sha="b" * 40, dirty=True)


def test_environment_requires_dependency_lock_identity() -> None:
    with pytest.raises(ValidationError):
        EnvironmentMetadata.model_validate(
            {
                "os_name": "Windows",
                "os_version": "11",
                "architecture": "AMD64",
                "python_version": "3.13.5",
                "node_version": "22.17.1",
                "hardware": "CPU-only test host",
            }
        )


def _valid_manifest(
    store: LocalArtifactStore,
    *,
    audit_status: MethodologyStatus = MethodologyStatus.PASSED,
    spec_schema_version: str = "1.0",
    audit_schema_version: str = "1.0",
) -> RunManifest:
    spec_ref = store.put_bytes(
        json.dumps(
            {"schema_version": spec_schema_version},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        media_type="application/json",
    )
    experiment_id = "exp_" + spec_ref.sha256
    artifacts = [
        _artifact(
            ArtifactKind.CANONICAL_SPEC,
            spec_ref,
            experiment_id=experiment_id,
        )
    ]
    previous_ref = spec_ref
    intermediate_kinds = REQUIRED_COMPLETED_ARTIFACT_KINDS - {
        ArtifactKind.CANONICAL_SPEC,
        ArtifactKind.METHODOLOGY_AUDIT,
    }
    for kind in sorted(intermediate_kinds, key=lambda value: value.value):
        ref = store.put_bytes(
            json.dumps(
                {"kind": kind.value, "schema_version": "1.0"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            media_type="application/json",
        )
        artifacts.append(
            _artifact(
                kind,
                ref,
                experiment_id=experiment_id,
                input_sha256=(previous_ref.sha256,),
            )
        )
        previous_ref = ref
    audit_ref = store.put_bytes(
        json.dumps(
            {
                "schema_version": audit_schema_version,
                "methodology_status": audit_status.value,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        media_type="application/json",
    )
    artifacts.append(
        _artifact(
            ArtifactKind.METHODOLOGY_AUDIT,
            audit_ref,
            experiment_id=experiment_id,
            input_sha256=(previous_ref.sha256,),
        )
    )
    return RunManifest(
        schema_version="1.0",
        run_id="run_01",
        attempt_id="attempt_01",
        experiment_id=experiment_id,
        canonical_spec_schema_version="1.0",
        state=RunState.COMPLETED,
        methodology_status=audit_status,
        artifacts=tuple(artifacts),
        git=GitMetadata(commit_sha="b" * 40, dirty=False),
        environment=EnvironmentMetadata(
            os_name="Windows",
            os_version="11",
            architecture="AMD64",
            python_version="3.13.5",
            node_version="22.17.1",
            dependency_lock_sha256=SHA,
            hardware="CPU-only test host",
        ),
        test_evidence=("pytest packages/research-core/tests: passed",),
        created_at=NOW,
    )


def _artifact(
    kind: ArtifactKind,
    ref: ArtifactRef,
    *,
    experiment_id: str,
    input_sha256: tuple[str, ...] = (),
) -> ManifestArtifact:
    return ManifestArtifact(
        kind=kind,
        ref=ref,
        run_id="run_01",
        experiment_id=experiment_id,
        schema_version="1.0",
        producer="openalpha-test",
        producer_version="0.1.0",
        input_sha256=input_sha256,
        created_at=NOW,
        code_commit="b" * 40,
        code_dirty=False,
        dependency_lock_sha256=SHA,
        parameters=(),
        random_seed=7,
    )


def _replace_manifest(manifest: RunManifest, **updates: Any) -> RunManifest:
    payload = manifest.model_dump(mode="python")
    payload.update(updates)
    return RunManifest.model_validate(payload)
