"""The v4 corrections: terminal artifacts bound to the operative
specification, and clipping semantics that cannot outrank direct evidence.

Every test executes real code. Nothing here touches a network, an official
asset, Torch, or any partition beyond the already-retrieved window.
"""

from __future__ import annotations

import json

import pytest
from diagnostic_fakes import FakeCodec
from openalpha_bridge.cloud.objectstore import InMemoryObjectStore, put_json
from openalpha_bridge.diagnostic.artifact import (
    _SCHEMA_FOR_OUTCOME,
    DIAGNOSTIC_FAILURE_CODE,
    DIAGNOSTIC_OPERATIONAL_FAILURE_CODE,
    EXPECTED_SPECIFICATIONS,
    FAILURE_SCHEMA_VERSION,
    LEGACY_FAILURE_SCHEMA_VERSION,
    diagnostic_artifact_key,
    specification_identity,
    verify_existing_artifact,
)
from openalpha_bridge.diagnostic.conclusion import (
    PRIMARY_PRIORITY,
    DiagnosticConclusion,
    NextExperiment,
    ReproducibilityCheck,
    decide,
)
from openalpha_bridge.diagnostic.runner import DIAGNOSTIC_SCHEMA_VERSION
from openalpha_bridge.diagnostic.spec import (
    V1_SPECIFICATION_NAME,
    V2_SPECIFICATION_NAME,
    V3_SPECIFICATION_NAME,
    V4_SPECIFICATION_NAME,
    V4_SPECIFICATION_SHA256,
)
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.identity import EXPERIMENT_SHA256
from openalpha_bridge.phase2.invocation import WorkerInvocation
from openalpha_bridge.phase2.states import EvidenceClass
from test_diagnostic_worker import COMMIT, RUN_ID, _Resolver, _run

SPEC_FIELDS = [
    f"{prefix}_{part}" for prefix in EXPECTED_SPECIFICATIONS for part in ("name", "sha256")
]


def _invocation() -> WorkerInvocation:
    return WorkerInvocation.validate_all(
        run_id=RUN_ID, source_commit=COMMIT, deployed_commit=COMMIT
    )


def _store_payload(store: InMemoryObjectStore, payload: dict, *, schema: str) -> None:
    put_json(
        store,
        diagnostic_artifact_key(RUN_ID),
        payload,
        schema_version=schema,
        run_id=RUN_ID,
        experiment_hash=EXPERIMENT_SHA256,
        evidence_class=EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY,
        immutable=True,
    )


# ============================================ the chain the artifact carries


def test_the_identity_helper_returns_the_whole_chain() -> None:
    identity = specification_identity()
    assert set(identity) == {*SPEC_FIELDS, "operative_specification"}
    assert identity["specification_v1_name"] == V1_SPECIFICATION_NAME
    assert identity["specification_v2_name"] == V2_SPECIFICATION_NAME
    assert identity["specification_v3_name"] == V3_SPECIFICATION_NAME
    assert identity["specification_v4_name"] == V4_SPECIFICATION_NAME
    assert identity["specification_v4_sha256"] == V4_SPECIFICATION_SHA256
    assert identity["operative_specification"] == V4_SPECIFICATION_NAME


def test_a_success_artifact_carries_the_whole_chain() -> None:
    result, store, _, _ = _run()
    payload = result.payload
    for field in (*SPEC_FIELDS, "operative_specification"):
        assert field in payload
    assert payload["operative_specification"] == V4_SPECIFICATION_NAME
    assert payload["schema_version"] == DIAGNOSTIC_SCHEMA_VERSION
    assert json.loads(store.get(result.artifact_key).body)["specification_v4_sha256"] == (
        V4_SPECIFICATION_SHA256
    )


def test_a_typed_failure_carries_the_whole_chain() -> None:
    from diagnostic_fakes import fake_assets

    result, _, _, _ = _run(resolver=_Resolver(assets=fake_assets(trainable=1)))
    assert result.outcome == DIAGNOSTIC_FAILURE_CODE
    for field in (*SPEC_FIELDS, "operative_specification"):
        assert field in result.payload
    assert result.payload["operative_specification"] == V4_SPECIFICATION_NAME
    assert result.payload["schema_version"] == FAILURE_SCHEMA_VERSION


def test_an_operational_failure_carries_the_whole_chain() -> None:
    result, _, _, _ = _run(resolver=_Resolver(raises=ConnectionResetError("dropped")))
    assert result.outcome == DIAGNOSTIC_OPERATIONAL_FAILURE_CODE
    for field in (*SPEC_FIELDS, "operative_specification"):
        assert field in result.payload
    assert result.payload["operative_specification"] == V4_SPECIFICATION_NAME
    assert result.payload["schema_version"] == FAILURE_SCHEMA_VERSION


def test_the_legacy_failure_schema_is_never_written() -> None:
    assert LEGACY_FAILURE_SCHEMA_VERSION != FAILURE_SCHEMA_VERSION
    assert LEGACY_FAILURE_SCHEMA_VERSION not in _SCHEMA_FOR_OUTCOME.values()
    assert set(_SCHEMA_FOR_OUTCOME.values()) == {
        DIAGNOSTIC_SCHEMA_VERSION,
        FAILURE_SCHEMA_VERSION,
    }


# ================================================ duplicate invocation


def test_a_v4_success_artifact_is_accepted_on_duplicate_invocation() -> None:
    first, store, _, _ = _run()
    second_resolver = _Resolver()
    from test_diagnostic_worker import _ProviderFactory

    second_factory = _ProviderFactory()
    second, _, _, _ = _run(store=store, resolver=second_resolver, factory=second_factory)

    assert second.already_existed is True
    assert second.content_sha256 == first.content_sha256
    # And it loaded nothing to discover that.
    assert second_resolver.calls == 0
    assert second_factory.calls == 0


@pytest.mark.parametrize("field", SPEC_FIELDS)
def test_altering_any_specification_field_rejects_the_artifact(field: str) -> None:
    _, store, _, _ = _run()
    payload = json.loads(store.get(diagnostic_artifact_key(RUN_ID)).body)
    payload[field] = "tampered" if field.endswith("_name") else "0" * 64

    fresh = InMemoryObjectStore()
    _store_payload(fresh, payload, schema=payload["schema_version"])
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_existing_artifact(fresh, invocation=_invocation())
    assert excinfo.value.failures[0].code == ("DIAGNOSTIC_TERMINAL_ARTIFACT_SPECIFICATION_MISMATCH")


def test_altering_the_operative_specification_rejects_the_artifact() -> None:
    _, store, _, _ = _run()
    payload = json.loads(store.get(diagnostic_artifact_key(RUN_ID)).body)
    payload["operative_specification"] = V3_SPECIFICATION_NAME

    fresh = InMemoryObjectStore()
    _store_payload(fresh, payload, schema=payload["schema_version"])
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_existing_artifact(fresh, invocation=_invocation())
    failure = excinfo.value.failures[0]
    assert failure.code == "DIAGNOSTIC_TERMINAL_ARTIFACT_SPECIFICATION_MISMATCH"
    assert "operative" in failure.message


def test_a_v2_era_failure_artifact_is_not_accepted_as_v4() -> None:
    """The exact defect: a failure recorded under an older schema."""
    store = InMemoryObjectStore()
    legacy = {
        "schema_version": LEGACY_FAILURE_SCHEMA_VERSION,
        "claim_boundary": "DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE",
        "outcome": DIAGNOSTIC_FAILURE_CODE,
        "evidence_class": EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY.value,
        "run_id": RUN_ID,
        "source_commit": COMMIT,
        "deployed_commit": COMMIT,
        "experiment_sha256": EXPERIMENT_SHA256,
        "specification_v2_sha256": "c39fff4541afcc948cd80fcc545398312897efe723deca26554143989e4a175e",
        "failure_stage": "frozen_inference_diagnostic",
        "failure_code": "SOMETHING",
        "message": "legacy",
        "completed_at": "2026-08-02T00:00:00+00:00",
    }
    _store_payload(store, legacy, schema=LEGACY_FAILURE_SCHEMA_VERSION)
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_existing_artifact(store, invocation=_invocation())
    # The outcome and schema no longer pair, so it is refused before anything
    # else is read from it.
    assert excinfo.value.failures[0].code == "DIAGNOSTIC_TERMINAL_ARTIFACT_CORRUPT"


def test_a_v3_success_artifact_is_not_accepted_as_v4() -> None:
    _, store, _, _ = _run()
    payload = json.loads(store.get(diagnostic_artifact_key(RUN_ID)).body)
    payload["schema_version"] = "openalpha.bridge.diagnostic.frozen_inference.v3"

    fresh = InMemoryObjectStore()
    _store_payload(fresh, payload, schema=payload["schema_version"])
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_existing_artifact(fresh, invocation=_invocation())
    assert excinfo.value.failures[0].code == "DIAGNOSTIC_TERMINAL_ARTIFACT_CORRUPT"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("run_id", "canary_deadbeef", "DIAGNOSTIC_TERMINAL_ARTIFACT_RUN_ID_MISMATCH"),
        ("source_commit", "b" * 40, "DIAGNOSTIC_TERMINAL_ARTIFACT_COMMIT_MISMATCH"),
        ("deployed_commit", "b" * 40, "DIAGNOSTIC_TERMINAL_ARTIFACT_COMMIT_MISMATCH"),
        (
            "experiment_sha256",
            "0" * 64,
            "DIAGNOSTIC_TERMINAL_ARTIFACT_EXPERIMENT_MISMATCH",
        ),
        (
            "evidence_class",
            EvidenceClass.REAL_PHASE2.value,
            "DIAGNOSTIC_TERMINAL_ARTIFACT_EVIDENCE_CLASS_MISMATCH",
        ),
        (
            "claim_boundary",
            "SOMETHING ELSE",
            "DIAGNOSTIC_TERMINAL_ARTIFACT_CLAIM_BOUNDARY_MISMATCH",
        ),
    ],
)
def test_every_other_identity_is_verified(field: str, value: str, code: str) -> None:
    _, store, _, _ = _run()
    payload = json.loads(store.get(diagnostic_artifact_key(RUN_ID)).body)
    payload[field] = value

    fresh = InMemoryObjectStore()
    _store_payload(fresh, payload, schema=payload["schema_version"])
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_existing_artifact(fresh, invocation=_invocation())
    assert excinfo.value.failures[0].code == code


def test_tampered_bytes_are_rejected_by_one_of_two_hash_checks() -> None:
    """Two layers verify the content hash, and either is sufficient.

    The object store checks it on read, and verify_existing_artifact
    recomputes it from the bytes independently. The store gets there first, so
    that is the code observed; the second check exists so a store that did not
    verify could not slip a tampered artifact through.
    """
    _, store, _, _ = _run()
    key = diagnostic_artifact_key(RUN_ID)
    stored = store.get(key)
    store._objects[key] = stored.model_copy(  # type: ignore[attr-defined]
        update={"body": stored.body + b" "}
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_existing_artifact(store, invocation=_invocation())
    assert excinfo.value.failures[0].code in {
        "OBJECT_HASH_MISMATCH",
        "DIAGNOSTIC_TERMINAL_ARTIFACT_CORRUPT",
    }


def test_the_artifact_verifier_recomputes_the_hash_itself() -> None:
    """The second layer, asserted structurally since the store shadows it."""
    import inspect

    from openalpha_bridge.diagnostic import artifact as artifact_module

    source = inspect.getsource(artifact_module.verify_existing_artifact)
    assert "hashlib.sha256(stored.body).hexdigest()" in source
    assert "stored.metadata.content_sha256" in source


# ==================================================== clipping semantics


def _decide(method_a, method_b, method_c, method_d):
    return decide(
        method_a=method_a,
        method_b=method_b,
        method_c=method_c,
        method_d=method_d,
        reproducibility=ReproducibilityCheck(performed=True, agrees=True, detail="identical"),
    )


def _artifact_parts(**codec_kwargs):
    result, _, _, _ = _run(resolver=_Resolver())
    return result


def test_material_exposure_alone_is_descriptive_and_never_primary() -> None:
    assert DiagnosticConclusion.MATERIAL_CLIPPING_EXPOSURE_OBSERVED not in PRIMARY_PRIORITY


def test_material_exposure_does_not_control_the_recommendation() -> None:
    """It is never consulted when the recommendation is chosen."""
    import inspect

    from openalpha_bridge.diagnostic import conclusion as conclusion_module

    source = inspect.getsource(conclusion_module._recommend)
    assert "MATERIAL_CLIPPING_EXPOSURE_OBSERVED" in source  # named only to exclude it
    assert "if C.MATERIAL_CLIPPING_EXPOSURE_OBSERVED in matched" not in source


def test_unclipped_invalidity_survives_material_clipping_elsewhere() -> None:
    """The v4 correction, checked end to end.

    Three invalid reconstructions on unclipped inputs, plus heavy clipping
    somewhere else in the sequence. Under v3 clipping would have made the
    reading inconclusive and become primary; under v4 the direct evidence
    stands and the confounded finding does not match.
    """
    from test_frozen_inference_diagnostic import TRUE_TARGET, _shift
    from test_frozen_inference_diagnostic import _run as _diag_run

    codec = FakeCodec(corrupt_round_trip=True, corrupt_count=3)
    artifact, _, _, _ = _diag_run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01), codec=codec)

    a = artifact.method_a
    assert a.invalid_rows_with_unclipped_input > 0
    assert (
        DiagnosticConclusion.ROUNDTRIP_INVALIDITY_ON_UNCLIPPED_INPUTS in artifact.matched_findings
    )
    # Confounding requires zero unclipped invalid rows, so it must not match.
    assert (
        DiagnosticConclusion.ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING
        not in artifact.matched_findings
    )
    assert artifact.conclusion is not (
        DiagnosticConclusion.ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING
    )


def test_the_confounded_predicate_requires_all_four_conditions() -> None:
    """Checked on the rule's recorded inputs rather than by construction."""
    from test_frozen_inference_diagnostic import TRUE_TARGET, _shift
    from test_frozen_inference_diagnostic import _run as _diag_run

    codec = FakeCodec(corrupt_round_trip=True, corrupt_count=3)
    artifact, _, _, _ = _diag_run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01), codec=codec)
    rule = next(e for e in artifact.decision.evaluations if e.rule_id == "R2d")
    assert rule.finding is DiagnosticConclusion.ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING
    assert rule.matched is False
    unclipped = rule.observed["invalid_rows_with_unclipped_input"]
    assert unclipped is not None and unclipped > 0
    assert set(rule.observed) == {
        "invalid_candle_count",
        "row_clipped_fraction",
        "invalid_rows_with_clipped_input",
        "invalid_rows_with_unclipped_input",
    }


def test_material_exposure_is_still_recorded_when_it_occurs() -> None:
    from test_frozen_inference_diagnostic import TRUE_TARGET, _shift
    from test_frozen_inference_diagnostic import _run as _diag_run

    artifact, _, _, _ = _diag_run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    rule = next(e for e in artifact.decision.evaluations if e.rule_id == "R2c")
    assert rule.finding is DiagnosticConclusion.MATERIAL_CLIPPING_EXPOSURE_OBSERVED
    assert "row_clipped_fraction" in rule.observed
    assert "descriptive" in rule.detail


def test_the_retired_clipping_label_is_gone() -> None:
    assert not any(
        c.value == "CLIPPING_EXPOSURE_MAKES_ROUNDTRIP_INCONCLUSIVE" for c in DiagnosticConclusion
    )


def test_confounded_interpretation_outranks_material_invalidity() -> None:
    confounded = PRIMARY_PRIORITY.index(
        DiagnosticConclusion.ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING
    )
    material = PRIMARY_PRIORITY.index(DiagnosticConclusion.ROUNDTRIP_MATERIAL_INVALIDITY)
    reproducibility = PRIMARY_PRIORITY.index(DiagnosticConclusion.REPRODUCIBILITY_FAILURE)
    operational = PRIMARY_PRIORITY.index(DiagnosticConclusion.DIAGNOSTIC_OPERATIONAL_FAILURE)
    assert operational < reproducibility < confounded < material


def test_confounded_interpretation_recommends_investigating_clipping() -> None:
    from openalpha_bridge.diagnostic.conclusion import _recommend

    matched = {
        DiagnosticConclusion.ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING,
        DiagnosticConclusion.MATERIAL_CLIPPING_EXPOSURE_OBSERVED,
    }
    from openalpha_bridge.diagnostic.spec import THRESHOLDS

    assert _recommend(matched, THRESHOLDS) is NextExperiment.INVESTIGATE_CLIPPING_EXPOSURE


def test_unclipped_invalidity_recommends_a_constrained_output_decoder() -> None:
    from openalpha_bridge.diagnostic.conclusion import _recommend
    from openalpha_bridge.diagnostic.spec import THRESHOLDS

    matched = {
        DiagnosticConclusion.ROUNDTRIP_INVALIDITY_ON_UNCLIPPED_INPUTS,
        DiagnosticConclusion.MATERIAL_CLIPPING_EXPOSURE_OBSERVED,
    }
    assert _recommend(matched, THRESHOLDS) is NextExperiment.CONSTRAINED_OUTPUT_DECODER

    # No-skill still takes precedence, because repair cannot create signal.
    with_no_skill = matched | {DiagnosticConclusion.NO_SKILL_AGAINST_PERSISTENCE}
    assert _recommend(with_no_skill, THRESHOLDS) is (
        NextExperiment.ABANDON_STRUCTURAL_VALIDITY_DIRECTION
    )
