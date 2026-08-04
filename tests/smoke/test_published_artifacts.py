"""The committed terminal artifacts must stay byte-identical to what was published.

These files are content-addressed research records. If any byte changes, the link
between a published result and the code that produced it is broken, so this is a
guard against reformatting, regeneration or accidental rewriting -- not a style
check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "research" / "artifacts"
MANIFEST = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
COMMITTED = MANIFEST["artifacts"]


def artifact_ids() -> list[str]:
    return [entry["run_id"] for entry in COMMITTED]


@pytest.mark.parametrize("entry", COMMITTED, ids=artifact_ids())
def test_the_committed_artifact_matches_its_published_digest(entry: dict) -> None:
    body = (ARTIFACTS / entry["committed_file"]).read_bytes()
    assert hashlib.sha256(body).hexdigest() == entry["artifact_sha256"]
    assert len(body) == entry["size_bytes"]


@pytest.mark.parametrize("entry", COMMITTED, ids=artifact_ids())
def test_the_artifact_is_canonical_json_not_pretty_printed(entry: dict) -> None:
    """Canonical form is what the publisher hashed; pretty-printing breaks it."""
    body = (ARTIFACTS / entry["committed_file"]).read_bytes()
    assert b"\n" not in body, "a newline means the file was reformatted"
    assert b"\r" not in body
    assert not body.startswith(b"{\n")


@pytest.mark.parametrize("entry", COMMITTED, ids=artifact_ids())
def test_the_manifest_describes_the_artifact_it_points_at(entry: dict) -> None:
    """A correct digest under a wrong description would still mislead a reader."""
    payload = json.loads((ARTIFACTS / entry["committed_file"]).read_bytes())
    assert payload["run_id"] == entry["run_id"]
    assert payload["experiment_id"] == entry["experiment_id"]
    assert payload["outcome"] == entry["outcome"]
    assert payload["source_commit"] == entry["source_commit"]
    assert payload["schema_version"] == entry["schema_version"]
    assert payload["specification_sha256"] == entry["specification_sha256"]


@pytest.mark.parametrize("entry", COMMITTED, ids=artifact_ids())
def test_the_artifact_authorizes_nothing_and_leaked_nothing(entry: dict) -> None:
    payload = json.loads((ARTIFACTS / entry["committed_file"]).read_bytes())
    for field, value in payload.items():
        if field.startswith("authorizes_"):
            assert value is False, field
    assert payload["scientific_result_available"] is False

    # Published evidence must carry no credential-shaped material.
    body = (ARTIFACTS / entry["committed_file"]).read_text(encoding="utf-8")
    for token in ("AKIA", "aws_secret", "BEGIN PRIVATE KEY", "Authorization:", "password"):
        assert token not in body, token


def test_the_specification_each_artifact_names_is_present_and_unchanged() -> None:
    """Each committed result must still be traceable to a sealed preregistration."""
    for entry in COMMITTED:
        spec = ROOT / entry["specification"]
        assert spec.is_file(), entry["specification"]
        observed = hashlib.sha256(spec.read_bytes()).hexdigest()
        assert observed == entry["specification_sha256"], entry["specification"]


def test_uncommitted_artifacts_are_declared_rather_than_omitted_silently() -> None:
    """A study whose artifact is absent must say so, with its key and digest."""
    for entry in MANIFEST.get("not_committed", []):
        assert entry["run_id"]
        assert len(entry["artifact_sha256"]) == 64
        assert entry["object_store_key"]
        assert entry["reason_not_committed"]

    coverage = MANIFEST["coverage"]
    assert coverage["artifacts_committed"] == len(COMMITTED)
    assert coverage["artifacts_recorded_only"] == len(MANIFEST.get("not_committed", []))
    assert (
        coverage["studies_completed"]
        == coverage["artifacts_committed"] + coverage["artifacts_recorded_only"]
    )
