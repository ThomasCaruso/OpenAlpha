"""Stage A evidence verification.

Kept separate from production so the check reads as an independent audit of the
report rather than part of the code that produced it.

A fabricated object carrying ``passed=True``, one sequence, and dimension 269
must not survive this. Every field is checked against the run that is actually
executing, and every referenced shard is re-read and re-verified.
"""

from __future__ import annotations

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..windowing import (
    CONTEXT_PREFIX_LENGTH,
    EXAMPLE_LENGTH,
    SCORED_SUFFIX_LENGTH,
    Partition,
    score_mask_sha256,
)
from .cache import CACHE_SCHEMA_VERSION, FeatureCache, ShardRef
from .kronos import BRIDGE_INPUT_DIMENSION, SOURCE_SPEC, ResolvedAssets
from .stage_a import StageAReport

__all__ = ["verify_stage_a_report"]


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


def verify_stage_a_report(
    report: object,
    *,
    run_id: str,
    experiment_sha256: str,
    amendment_sha256: tuple[str, ...],
    evidence_class: str,
    source_commit: str | None,
    provider_identity: str,
    assets: ResolvedAssets,
    expected_sequence_ids: frozenset[str],
    cache: FeatureCache,
    expected_tensor_specification: dict[str, str] | None = None,
) -> StageAReport:
    """Verify Stage A evidence against the active run, or fail closed."""
    if not isinstance(report, StageAReport):
        raise _fail(
            "STAGE_A_REPORT_WRONG_TYPE",
            f"Stage A evidence must be a StageAReport, got {type(report).__name__}",
        )
    if not report.passed:
        raise _fail("STAGE_A_REPORT_NOT_PASSED", "the Stage A report does not report success")

    checks: list[tuple[str, object, object]] = [
        ("run_id", report.run_id, run_id),
        ("experiment_sha256", report.experiment_sha256, experiment_sha256),
        ("amendment_sha256", tuple(report.amendment_sha256), tuple(amendment_sha256)),
        ("evidence_class", report.evidence_class, evidence_class),
        ("provider", report.provider, provider_identity),
        ("score_mask_sha256", report.score_mask_sha256, score_mask_sha256()),
        ("bridge_input_dimension", report.bridge_input_dimension, BRIDGE_INPUT_DIMENSION),
        ("prefix_length", report.prefix_length, CONTEXT_PREFIX_LENGTH),
        ("suffix_length", report.suffix_length, SCORED_SUFFIX_LENGTH),
        ("cache_schema_version", report.cache_schema_version, CACHE_SCHEMA_VERSION),
        ("official_source_revision", report.official_source_revision, SOURCE_SPEC.revision),
        ("kronos_repository", report.kronos_repository, assets.repository),
        ("kronos_revision", report.kronos_revision, assets.revision),
        ("kronos_config_sha256", report.kronos_config_sha256, assets.observed_config_sha256),
        ("kronos_weights_sha256", report.kronos_weights_sha256, assets.observed_weights_sha256),
        (
            "frozen_parameter_sha256",
            report.frozen_parameter_sha256,
            assets.frozen_parameter_sha256,
        ),
    ]
    if source_commit is not None:
        checks.append(("source_commit", report.source_commit, source_commit))

    mismatched = sorted(name for name, observed, expected in checks if observed != expected)
    if mismatched:
        raise _fail(
            "STAGE_A_REPORT_IDENTITY_MISMATCH",
            f"Stage A report does not describe this run: {', '.join(mismatched)}",
        )

    if report.official_source_file_sha256 != dict(SOURCE_SPEC.files):
        raise _fail(
            "STAGE_A_SOURCE_HASH_MISMATCH",
            "Stage A report does not carry the locked official source-file hashes",
        )

    observed_ids = {record.sequence_id for record in report.sequences}
    if observed_ids != set(expected_sequence_ids):
        raise _fail(
            "STAGE_A_UNEXPECTED_SEQUENCES",
            (
                f"Stage A must contain exactly {sorted(expected_sequence_ids)}, "
                f"observed {sorted(observed_ids)}"
            ),
        )

    for record in report.sequences:
        if record.partition != Partition.TRAIN.value:
            raise _fail(
                "STAGE_A_NON_TRAINING_SEQUENCE",
                (
                    f"{record.sequence_id} is a {record.partition} sequence; Stage A may "
                    "not touch validation, reconstruction-test, or external data"
                ),
            )
        if tuple(record.bridge_input_shape) != (EXAMPLE_LENGTH, BRIDGE_INPUT_DIMENSION):
            raise _fail(
                "STAGE_A_INVALID_SHAPE",
                f"{record.sequence_id} reports shape {record.bridge_input_shape}",
            )
        if record.bridge_input_dtype != "float32":
            raise _fail(
                "STAGE_A_INVALID_DTYPE",
                f"{record.sequence_id} reports dtype {record.bridge_input_dtype}",
            )
        if not record.deterministic_replay_matched:
            raise _fail(
                "STAGE_A_NONDETERMINISTIC",
                f"{record.sequence_id} did not reproduce byte-identical features",
            )

        # Re-read the shard so its existence and integrity are observed, not
        # taken from the report that is itself under audit.
        restored = cache.read(
            ShardRef(
                partition=Partition(record.partition),
                sequence_id=record.sequence_id,
                relative_path=record.shard_relative_path,
                content_sha256=record.shard_content_sha256,
                size_bytes=record.shard_size_bytes,
            )
        )
        identity = restored.identity
        shard_checks: list[tuple[str, object, object]] = [
            ("run_id", identity.run_id, run_id),
            ("experiment_sha256", identity.experiment_sha256, experiment_sha256),
            (
                "amendment_sha256",
                tuple(identity.amendment_sha256),
                tuple(amendment_sha256),
            ),
            ("score_mask_sha256", identity.score_mask_sha256, score_mask_sha256()),
            ("evidence_class", identity.evidence_class, evidence_class),
            ("provider", identity.provider, provider_identity),
            (
                "provider_client_version",
                identity.provider_client_version,
                report.provider_client_version,
            ),
            ("symbol", identity.symbol, record.symbol),
            ("interval", identity.interval, record.interval),
            ("partition", identity.partition.value, record.partition),
            ("prefix_start", identity.prefix_start, record.prefix_start),
            ("prefix_end", identity.prefix_end, record.prefix_end),
            ("target_start", identity.target_start, record.target_start),
            ("target_end", identity.target_end, record.target_end),
            (
                "official_source_repository",
                identity.official_source_repository,
                SOURCE_SPEC.repository,
            ),
            (
                "official_source_revision",
                identity.official_source_revision,
                SOURCE_SPEC.revision,
            ),
            (
                "official_source_file_sha256",
                identity.official_source_file_sha256,
                dict(SOURCE_SPEC.files),
            ),
            ("tokenizer_repository", identity.tokenizer_repository, assets.repository),
            ("tokenizer_revision", identity.tokenizer_revision, assets.revision),
            (
                "tokenizer_config_sha256",
                identity.tokenizer_config_sha256,
                assets.observed_config_sha256,
            ),
            (
                "tokenizer_weights_sha256",
                identity.tokenizer_weights_sha256,
                assets.observed_weights_sha256,
            ),
            (
                "frozen_parameter_sha256",
                identity.frozen_parameter_sha256,
                assets.frozen_parameter_sha256,
            ),
            (
                "feature_schema_version",
                identity.feature_schema_version,
                CACHE_SCHEMA_VERSION,
            ),
            (
                "representation_version",
                identity.representation_version,
                report.representation_version,
            ),
            ("candle_data_sha256", identity.candle_data_sha256, restored.source_data_sha256),
        ]
        if source_commit is not None:
            shard_checks.append(("source_commit", identity.source_commit, source_commit))
        if expected_tensor_specification is not None:
            shard_checks.append(
                (
                    "tensor_specification",
                    identity.tensor_specification,
                    expected_tensor_specification,
                )
            )

        drifted = sorted(name for name, got, want in shard_checks if got != want)
        if drifted:
            raise _fail(
                "STAGE_A_SHARD_IDENTITY_MISMATCH",
                f"{record.sequence_id} shard identity drifted: {', '.join(drifted)}",
            )

    return report
