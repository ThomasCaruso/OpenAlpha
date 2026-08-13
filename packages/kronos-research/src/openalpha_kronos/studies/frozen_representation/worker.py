"""The two probe worker bodies, independent of any compute backend.

``run_probe_fit_worker`` extracts representations on the train/validation window,
selects hyperparameters on validation only, refits once, and publishes the fit
artifact. It never touches a test session.

``run_probe_test_worker`` verifies the sealed specification and the exact fit
artifact, checks every eligibility gate, and only then retrieves and scores test
data. If any gate fails it publishes a typed failure and stops.

There is no import of and no call into any completed study's worker, runner or
artifact namespace, no optimizer over Kronos parameters, no candle decoding, no
sampling, and no structural-validity quantity anywhere near a decision.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from openalpha_research.objectstore import ObjectStore
from openalpha_research.providers import MarketDataProvider, RetrievalRequest, validate_series
from openalpha_research.runtime import gpu_snapshot, reset_gpu_statistics
from pydantic import BaseModel, ConfigDict

from openalpha_kronos.model.input import OfficialRow, official_stamp
from openalpha_kronos.model.normalization import fit_context_state
from openalpha_kronos.studies.provenance import EvidenceClass
from openalpha_kronos.studies.structural_validity.mini.safe_logging import (
    StageTracker,
    log_operational_failure,
)

from .artifact import (
    ProbeTerminalArtifact,
    load_verified_fit_artifact,
    publish_probe_fit,
    publish_probe_test,
)
from .controls import CONTROL_BUILDERS, feature_names
from .extraction import HiddenStateExtractor
from .features import (
    OriginWindow,
    build_sample,
    resolve_windows,
    verify_cross_asset_alignment,
)
from .fit import (
    FeatureSetFit,
    evaluate_regression,
    fit_feature_set,
    matrix_from_rows,
)
from .fit import (
    directional_accuracy as scored_directional_accuracy,
)
from .fit import (
    directional_auc as scored_directional_auc,
)
from .invocation import ProbeInvocation
from .spec import (
    ASSET_PANEL,
    CALENDAR,
    CANDIDATE_FEATURE_SET,
    CONTEXT_CANDLES,
    CONTROL_FEATURE_SETS,
    EMBARGO_ORDINALS,
    EXPECTED_TOTAL_PARAMETERS,
    FEATURE_SET_DIMENSIONS,
    FEATURE_SET_IDS,
    FREQUENCY,
    HORIZON_CANDLES,
    MINIMUM_TEST_ORIGINS_PER_ASSET,
    PROBE_CLAIM_BOUNDARY,
    PROBE_EXPERIMENT_ID,
    PROBE_SPECIFICATION_NAME,
    PROBE_SPECIFICATION_SHA256,
    REQUIRED_TRAINABLE_PARAMETERS,
    SEALED_AT_UTC,
    SEALING_COMMIT,
    STRIDE,
    TEST_START_INCLUSIVE,
    TRAIN_ORDINALS,
    TRAIN_VALIDATION_END_EXCLUSIVE,
    TRAIN_VALIDATION_SESSIONS,
    TRAIN_VALIDATION_START_INCLUSIVE,
    VALIDATION_ORDINALS,
    prove_probe_budget,
    verify_partition_geometry,
    verify_pinned_pair,
    verify_probe_specification,
)
from .test import (
    OriginCluster,
    build_comparison,
    decide,
    verify_test_eligibility,
)

__all__ = ["ProbeWorkerResult", "run_probe_fit_worker", "run_probe_test_worker"]


def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class ProbeWorkerResult(BaseModel):
    """What a worker returns. The evidence is the stored artifact."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    artifact_key: str
    content_sha256: str
    outcome: str
    phase: Literal["fit", "test"]
    decision_outcome: str | None = None
    experiment_id: Literal["openalpha-kronos-frozen-representation-probe-v1"] = (
        PROBE_EXPERIMENT_ID
    )
    study_type: Literal["frozen_representation_probe"] = "frozen_representation_probe"
    evidence_class: Literal[EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY] = (
        EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    )
    already_existed: bool
    deployed_commit: str
    authorizes_training: Literal[False] = False
    authorizes_fine_tuning: Literal[False] = False
    authorizes_supervised_adaptation: Literal[False] = False
    authorizes_production_inference: Literal[False] = False
    authorizes_trading_claims: Literal[False] = False
    scientific_result_available: Literal[False] = False
    payload: dict[str, Any]


class _CountingProvider:
    """Wraps the provider so the request count is measured, not asserted."""

    __slots__ = ("_inner", "requests")

    def __init__(self, inner: MarketDataProvider) -> None:
        self._inner = inner
        self.requests: list[RetrievalRequest] = []

    def fetch(self, request: RetrievalRequest) -> Any:
        self.requests.append(request)
        return self._inner.fetch(request)


def _rows_from(series: Any) -> tuple[OfficialRow, ...]:
    return tuple(
        OfficialRow(
            session=c.session,
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            volume=c.volume,
            amount=c.amount,
        )
        for c in series.candles
    )


def _require_pinned_assets(assets: Any) -> None:
    """The loaded pair and its frozen state, checked before anything is read."""
    verify_pinned_pair()
    if assets.model_repository != "NeoQuasar/Kronos-base":
        raise _fail("PROBE_WRONG_MODEL_REPOSITORY", f"loaded {assets.model_repository}")
    if assets.tokenizer_repository != "NeoQuasar/Kronos-Tokenizer-base":
        raise _fail("PROBE_WRONG_TOKENIZER_REPOSITORY", f"loaded {assets.tokenizer_repository}")
    if assets.model_revision != "2b554741eca47781b64468546e77fef3e85130e6":
        raise _fail("PROBE_WRONG_MODEL_REVISION", f"loaded {assets.model_revision}")
    if assets.tokenizer_revision != "0e0117387f39004a9016484a186a908917e22426":
        raise _fail("PROBE_WRONG_TOKENIZER_REVISION", f"loaded {assets.tokenizer_revision}")
    if assets.trainable_parameter_count != REQUIRED_TRAINABLE_PARAMETERS:
        raise _fail(
            "PROBE_PARAMETERS_NOT_FROZEN",
            f"{assets.trainable_parameter_count} trainable parameters, expected 0",
        )
    if assets.total_parameter_count != EXPECTED_TOTAL_PARAMETERS:
        raise _fail(
            "PROBE_UNEXPECTED_PARAMETER_COUNT",
            f"{assets.total_parameter_count} parameters, expected {EXPECTED_TOTAL_PARAMETERS}",
        )


def _feature_rows(
    *,
    windows: tuple[OriginWindow, ...],
    extractor: HiddenStateExtractor,
) -> dict[str, list[tuple[float, ...]]]:
    """Build every feature set from context rows only, one row per origin."""
    rows: dict[str, list[tuple[float, ...]]] = {name: [] for name in FEATURE_SET_IDS}
    for window in windows:
        context = window.context
        # Refit per origin, from context rows only.
        state = fit_context_state(context)
        if state.fitted_candle_count != CONTEXT_CANDLES:
            raise _fail(
                "PROBE_NORMALIZATION_FITTED_ON_WRONG_ROWS",
                f"state fitted from {state.fitted_candle_count} rows, expected {CONTEXT_CANDLES}",
            )
        stamps = tuple(official_stamp(row.session) for row in context)
        extracted = extractor.hidden_state(context, context_stamps=stamps, state=state)
        rows[CANDIDATE_FEATURE_SET].append(extracted.vector)
        for control, builder in CONTROL_BUILDERS.items():
            rows[control].append(builder(context))
    return rows


def _retrieve(
    *,
    provider: _CountingProvider,
    start: date,
    end: date,
    maximum_candles: int,
) -> dict[str, tuple[OfficialRow, ...]]:
    out: dict[str, tuple[OfficialRow, ...]] = {}
    for asset in ASSET_PANEL:
        series = provider.fetch(
            RetrievalRequest(
                symbol=asset, start=start, end=end, maximum_candles=maximum_candles
            )
        )
        validate_series(series)
        out[asset] = _rows_from(series)
    return out


# ------------------------------------------------------------------- fit


def run_probe_fit_worker(
    *,
    store: ObjectStore,
    resolve_runtime: Any,
    provider_factory: Any,
    research_root: Path | str,
    source_commit: str,
    deployed_commit: str,
    run_id: str,
    now: datetime | None = None,
    logger: logging.Logger | None = None,
) -> ProbeWorkerResult:
    """Fit on the train/validation window and publish the fit artifact.

    Touches no test session. The window it retrieves ends strictly before the
    sealed test boundary, so the retrieval itself cannot reach test data.
    """
    stage = StageTracker("validate_invocation")
    invocation = ProbeInvocation.validate_all(
        run_id=run_id, source_commit=source_commit, deployed_commit=deployed_commit
    )

    def execute() -> dict[str, Any]:
        started = time.perf_counter()
        reset_gpu_statistics()
        stage.enter("verify_specification")
        verify_probe_specification(research_root)
        geometry = verify_partition_geometry()
        budget = prove_probe_budget()

        stage.enter("enter_official_runtime")
        with resolve_runtime() as runtime:
            _require_pinned_assets(runtime.assets)
            parameter_before = str(runtime.parameter_digest())

            stage.enter("construct_provider")
            counting = _CountingProvider(provider_factory())

            stage.enter("retrieve_series")
            rows_by_asset = _retrieve(
                provider=counting,
                start=date.fromisoformat(TRAIN_VALIDATION_START_INCLUSIVE),
                end=date.fromisoformat(TRAIN_VALIDATION_END_EXCLUSIVE),
                maximum_candles=512,
            )
            retrievals = []
            for asset in ASSET_PANEL:
                rows = rows_by_asset[asset]
                if len(rows) < TRAIN_VALIDATION_SESSIONS:
                    raise _fail(
                        "PROBE_INSUFFICIENT_FIT_SESSIONS",
                        f"{asset} supplied {len(rows)} sessions, need "
                        f"{TRAIN_VALIDATION_SESSIONS}",
                        field=asset,
                    )
                rows_by_asset[asset] = rows[:TRAIN_VALIDATION_SESSIONS]
                retrievals.append(
                    {
                        "asset": asset,
                        "sessions_retrieved": len(rows),
                        "sessions_used": TRAIN_VALIDATION_SESSIONS,
                        "first_session_used": rows_by_asset[asset][0].session.isoformat(),
                        "last_session_used": rows_by_asset[asset][-1].session.isoformat(),
                    }
                )
            # A fit-phase retrieval must never reach past the sealed boundary.
            boundary = date.fromisoformat(TEST_START_INCLUSIVE)
            for asset in ASSET_PANEL:
                if rows_by_asset[asset][-1].session >= boundary:
                    raise _fail(
                        "PROBE_FIT_TOUCHED_TEST_WINDOW",
                        (
                            f"{asset} fit data reaches {rows_by_asset[asset][-1].session}, "
                            f"on or after the sealed test boundary {boundary}"
                        ),
                        field=asset,
                    )

            stage.enter("resolve_origins")
            windows_by_asset = {
                asset: resolve_windows(asset=asset, rows=rows_by_asset[asset])
                for asset in ASSET_PANEL
            }
            alignment = verify_cross_asset_alignment(windows_by_asset)

            stage.enter("extract_representations")
            train_rows: dict[str, list[tuple[float, ...]]] = {n: [] for n in FEATURE_SET_IDS}
            val_rows: dict[str, list[tuple[float, ...]]] = {n: [] for n in FEATURE_SET_IDS}
            train_samples, val_samples = [], []
            for asset in ASSET_PANEL:
                windows = windows_by_asset[asset]
                selected = {
                    "train": tuple(windows[o] for o in TRAIN_ORDINALS),
                    "validation": tuple(windows[o] for o in VALIDATION_ORDINALS),
                }
                for split, chosen in selected.items():
                    built = _feature_rows(windows=chosen, extractor=runtime.extractor)
                    target = train_rows if split == "train" else val_rows
                    for name in FEATURE_SET_IDS:
                        target[name].extend(built[name])
                    bucket = train_samples if split == "train" else val_samples
                    bucket.extend(build_sample(w) for w in chosen)

            stage.enter("verify_parameters")
            parameter_after = str(runtime.parameter_digest())
            if parameter_before != parameter_after:
                raise _fail(
                    "PROBE_PARAMETERS_MODIFIED",
                    "the parameter digest changed during extraction; the model was not frozen",
                )

        stage.enter("fit_estimators")
        train_target = np.asarray([s.target_return for s in train_samples], dtype=np.float64)
        train_labels = np.asarray([s.target_direction for s in train_samples], dtype=np.float64)
        val_target = np.asarray([s.target_return for s in val_samples], dtype=np.float64)
        val_labels = np.asarray([s.target_direction for s in val_samples], dtype=np.float64)

        fits: dict[str, FeatureSetFit] = {}
        for name in FEATURE_SET_IDS:
            dimension = FEATURE_SET_DIMENSIONS[name]
            fits[name] = fit_feature_set(
                feature_set=name,
                names=feature_names(name),
                train_matrix=matrix_from_rows(train_rows[name], dimension=dimension),
                train_target=train_target,
                train_labels=train_labels,
                validation_matrix=matrix_from_rows(val_rows[name], dimension=dimension),
                validation_target=val_target,
                validation_labels=val_labels,
            )

        return {
            "schema_version": "openalpha.bridge.representation_probe.kronos_frozen_fit.v1",
            "phase": "fit",
            "claim_boundary": PROBE_CLAIM_BOUNDARY,
            "evidence_class": EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY.value,
            "experiment_id": PROBE_EXPERIMENT_ID,
            "study_type": "frozen_representation_probe",
            "specification_name": PROBE_SPECIFICATION_NAME,
            "specification_sha256": PROBE_SPECIFICATION_SHA256,
            "sealing_commit": SEALING_COMMIT,
            "sealed_at_utc": SEALED_AT_UTC,
            "run_id": invocation.run_id,
            "source_commit": invocation.source_commit,
            "deployed_commit": invocation.deployed_commit,
            "assets": list(ASSET_PANEL),
            "frequency": FREQUENCY,
            "calendar": CALENDAR,
            "train_validation_start_inclusive": TRAIN_VALIDATION_START_INCLUSIVE,
            "train_validation_end_exclusive": TRAIN_VALIDATION_END_EXCLUSIVE,
            "test_start_inclusive": TEST_START_INCLUSIVE,
            "retrievals": retrievals,
            "provider_request_count": len(counting.requests),
            "cross_asset_alignment": alignment,
            "partition_geometry": geometry,
            "context_budget": budget,
            "train_ordinals": list(TRAIN_ORDINALS),
            "embargo_ordinals": list(EMBARGO_ORDINALS),
            "validation_ordinals": list(VALIDATION_ORDINALS),
            "stride": STRIDE,
            "context_candles": CONTEXT_CANDLES,
            "horizon_candles": HORIZON_CANDLES,
            "train_samples": len(train_samples),
            "validation_samples": len(val_samples),
            "feature_sets": {
                name: {
                    "dimension": fits[name].dimension,
                    "declared_dimension": FEATURE_SET_DIMENSIONS[name],
                    "is_candidate": name == CANDIDATE_FEATURE_SET,
                    "selected_alpha": fits[name].selected_alpha,
                    "selected_inverse_regularization": (
                        fits[name].selected_inverse_regularization
                    ),
                    "validation_metrics_by_alpha": fits[name].validation_metrics_by_alpha,
                    "validation_directional_by_c": fits[name].validation_directional_by_c,
                    "selected_validation_metrics": fits[name].selected_validation_metrics,
                    "preprocessing": fits[name].standardizer.model_dump(mode="json"),
                    "ridge": fits[name].ridge.model_dump(mode="json"),
                    "logistic": fits[name].logistic.model_dump(mode="json"),
                    "feature_names": list(fits[name].feature_names),
                    "refit_rows": fits[name].refit_rows,
                }
                for name in FEATURE_SET_IDS
            },
            "parameter_sha256_before": parameter_before,
            "parameter_sha256_after": parameter_after,
            "parameters_unmodified": parameter_before == parameter_after,
            "assets_resolved": runtime.assets.model_dump(mode="json"),
            "gpu_measurement": gpu_snapshot().model_dump(mode="json"),
            "total_seconds": round(time.perf_counter() - started, 6),
            "completed_at": (now or datetime.now(UTC)).isoformat(),
            "test_partition_opened": False,
            "training_performed": False,
            "kronos_parameters_updated": False,
            "authorizes_training": False,
            "authorizes_fine_tuning": False,
            "authorizes_supervised_adaptation": False,
            "authorizes_production_inference": False,
            "authorizes_trading_claims": False,
            "scientific_result_available": False,
        }

    def execute_and_log() -> dict[str, Any]:
        try:
            return execute()
        except ResearchFailureError:
            raise
        except Exception as error:
            log_operational_failure(
                error,
                run_id=invocation.run_id,
                deployed_commit=invocation.deployed_commit,
                stage=stage.stage,
                logger=logger,
            )
            raise

    artifact: ProbeTerminalArtifact = publish_probe_fit(
        store, invocation=invocation, execute=execute_and_log, now=now, stage=stage
    )
    return ProbeWorkerResult(
        artifact_key=artifact.key,
        content_sha256=artifact.content_sha256,
        outcome=artifact.outcome,
        phase="fit",
        already_existed=artifact.already_existed,
        deployed_commit=invocation.deployed_commit,
        payload=artifact.payload,
    )


# ------------------------------------------------------------------- test


def run_probe_test_worker(
    *,
    store: ObjectStore,
    resolve_runtime: Any,
    provider_factory: Any,
    research_root: Path | str,
    source_commit: str,
    deployed_commit: str,
    run_id: str,
    expected_fit_digest: str | None = None,
    now: datetime | None = None,
    logger: logging.Logger | None = None,
) -> ProbeWorkerResult:
    """Score the untouched test partition, or refuse and say why.

    Every gate is checked before any test observation is retrieved. A run that
    is not yet eligible publishes a typed failure and reads nothing.
    """
    stage = StageTracker("validate_invocation")
    invocation = ProbeInvocation.validate_all(
        run_id=run_id, source_commit=source_commit, deployed_commit=deployed_commit
    )

    def execute() -> dict[str, Any]:
        started = time.perf_counter()
        reset_gpu_statistics()

        # ---- gates, all before a single test observation is fetched --------
        stage.enter("verify_specification")
        verify_probe_specification(research_root)
        verify_partition_geometry()

        stage.enter("verify_fit_artifact")
        fit_artifact = load_verified_fit_artifact(
            store, invocation=invocation, expected_digest=expected_fit_digest
        )
        fit_payload = fit_artifact.payload
        if fit_payload.get("specification_sha256") != PROBE_SPECIFICATION_SHA256:
            raise _fail(
                "PROBE_FIT_SPECIFICATION_MISMATCH",
                "the fit artifact was produced under a different specification digest",
            )
        fit_feature_sets = fit_payload.get("feature_sets")
        if not isinstance(fit_feature_sets, dict) or set(fit_feature_sets) != set(
            FEATURE_SET_IDS
        ):
            raise _fail(
                "PROBE_FIT_FEATURE_SETS_MISMATCH",
                "the fit artifact does not carry exactly the declared feature sets",
            )

        stage.enter("enter_official_runtime")
        with resolve_runtime() as runtime:
            _require_pinned_assets(runtime.assets)
            parameter_before = str(runtime.parameter_digest())

            stage.enter("construct_provider")
            counting = _CountingProvider(provider_factory())

            stage.enter("retrieve_test_series")
            boundary = date.fromisoformat(TEST_START_INCLUSIVE)
            rows_by_asset = _retrieve(
                provider=counting,
                start=boundary,
                end=(now or datetime.now(UTC)).date(),
                maximum_candles=1024,
            )

            stage.enter("verify_test_eligibility")
            sessions = {a: len(rows_by_asset[a]) for a in ASSET_PANEL}
            firsts = {a: rows_by_asset[a][0].session for a in ASSET_PANEL if rows_by_asset[a]}
            origins = {
                a: len(resolve_windows(asset=a, rows=rows_by_asset[a])) for a in ASSET_PANEL
            }
            eligibility = verify_test_eligibility(
                sessions_by_asset=sessions,
                first_session_by_asset=firsts,
                origins_by_asset=origins,
            )

            stage.enter("resolve_origins")
            windows_by_asset = {
                a: resolve_windows(
                    asset=a, rows=rows_by_asset[a], limit=MINIMUM_TEST_ORIGINS_PER_ASSET
                )
                for a in ASSET_PANEL
            }
            alignment = verify_cross_asset_alignment(windows_by_asset)

            stage.enter("extract_representations")
            per_asset_rows = {
                a: _feature_rows(windows=windows_by_asset[a], extractor=runtime.extractor)
                for a in ASSET_PANEL
            }
            per_asset_samples = {
                a: [build_sample(w) for w in windows_by_asset[a]] for a in ASSET_PANEL
            }

            stage.enter("verify_parameters")
            parameter_after = str(runtime.parameter_digest())
            if parameter_before != parameter_after:
                raise _fail(
                    "PROBE_PARAMETERS_MODIFIED",
                    "the parameter digest changed during extraction",
                )

        # ---- scoring, using ONLY the published fit state -------------------
        stage.enter("score")
        errors_by_set: dict[str, dict[str, np.ndarray]] = {}
        metrics_by_set: dict[str, dict[str, Any]] = {}
        for name in FEATURE_SET_IDS:
            published = fit_feature_sets[name]
            standardizer_state = published["preprocessing"]
            ridge_state = published["ridge"]
            logistic_state = published["logistic"]
            mean = np.asarray(standardizer_state["mean"], dtype=np.float64)
            scale = np.asarray(standardizer_state["scale"], dtype=np.float64)
            coefficients = np.asarray(ridge_state["coefficients"], dtype=np.float64)
            dimension = FEATURE_SET_DIMENSIONS[name]

            per_asset_errors: dict[str, np.ndarray] = {}
            pooled_predicted, pooled_actual, pooled_scores, pooled_labels = [], [], [], []
            for asset in ASSET_PANEL:
                matrix = matrix_from_rows(per_asset_rows[asset][name], dimension=dimension)
                scaled = (
                    (matrix - mean) / scale if dimension else np.zeros((matrix.shape[0], 0))
                )
                predicted = (
                    scaled @ coefficients + ridge_state["intercept"]
                    if dimension
                    else np.full(matrix.shape[0], ridge_state["intercept"], dtype=np.float64)
                )
                actual = np.asarray(
                    [s.target_return for s in per_asset_samples[asset]], dtype=np.float64
                )
                labels = np.asarray(
                    [s.target_direction for s in per_asset_samples[asset]], dtype=np.float64
                )
                scores = (
                    scaled @ np.asarray(logistic_state["coefficients"], dtype=np.float64)
                    + logistic_state["intercept"]
                    if dimension
                    else np.full(
                        matrix.shape[0], logistic_state["intercept"], dtype=np.float64
                    )
                )
                per_asset_errors[asset] = np.abs(predicted - actual)
                pooled_predicted.append(predicted)
                pooled_actual.append(actual)
                pooled_scores.append(scores)
                pooled_labels.append(labels)

            predicted_all = np.concatenate(pooled_predicted)
            actual_all = np.concatenate(pooled_actual)
            scores_all = np.concatenate(pooled_scores)
            labels_all = np.concatenate(pooled_labels)
            errors_by_set[name] = per_asset_errors
            metrics_by_set[name] = {
                **evaluate_regression(predicted_all, actual_all),
                "directional_accuracy": scored_directional_accuracy(scores_all, labels_all),
                "directional_auc": scored_directional_auc(scores_all, labels_all),
                "per_asset_primary_error": {
                    a: float(np.mean(per_asset_errors[a])) for a in ASSET_PANEL
                },
                "samples": int(actual_all.size),
            }

        stage.enter("compute_decision")
        candidate_errors = errors_by_set[CANDIDATE_FEATURE_SET]
        comparisons = []
        for control in CONTROL_FEATURE_SETS:
            control_errors = errors_by_set[control]
            clusters = tuple(
                OriginCluster(
                    ordinal=ordinal,
                    assets=ASSET_PANEL,
                    paired_differences=tuple(
                        float(control_errors[a][ordinal] - candidate_errors[a][ordinal])
                        for a in ASSET_PANEL
                    ),
                )
                for ordinal in range(MINIMUM_TEST_ORIGINS_PER_ASSET)
            )
            comparisons.append(
                build_comparison(
                    control=control,
                    candidate_errors=np.concatenate(
                        [candidate_errors[a] for a in ASSET_PANEL]
                    ),
                    control_errors=np.concatenate([control_errors[a] for a in ASSET_PANEL]),
                    clusters=clusters,
                )
            )

        supporting = tuple(
            asset
            for asset in ASSET_PANEL
            if all(
                float(np.mean(candidate_errors[asset]))
                <= (1.0 - 0.05) * float(np.mean(errors_by_set[control][asset]))
                for control in CONTROL_FEATURE_SETS
            )
        )
        decision = decide(comparisons=tuple(comparisons), supporting_assets=supporting)

        return {
            "schema_version": "openalpha.bridge.representation_probe.kronos_frozen_probe.v1",
            "phase": "test",
            "claim_boundary": PROBE_CLAIM_BOUNDARY,
            "evidence_class": EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY.value,
            "experiment_id": PROBE_EXPERIMENT_ID,
            "study_type": "frozen_representation_probe",
            "specification_name": PROBE_SPECIFICATION_NAME,
            "specification_sha256": PROBE_SPECIFICATION_SHA256,
            "run_id": invocation.run_id,
            "source_commit": invocation.source_commit,
            "deployed_commit": invocation.deployed_commit,
            "fit_artifact_key": fit_artifact.key,
            "fit_artifact_sha256": fit_artifact.content_sha256,
            "test_start_inclusive": TEST_START_INCLUSIVE,
            "eligibility": eligibility.model_dump(mode="json"),
            "cross_asset_alignment": alignment,
            "provider_request_count": len(counting.requests),
            "metrics": metrics_by_set,
            "decision": decision.model_dump(mode="json"),
            "parameter_sha256_before": parameter_before,
            "parameter_sha256_after": parameter_after,
            "parameters_unmodified": parameter_before == parameter_after,
            "gpu_measurement": gpu_snapshot().model_dump(mode="json"),
            "total_seconds": round(time.perf_counter() - started, 6),
            "completed_at": (now or datetime.now(UTC)).isoformat(),
            "test_partition_opened": True,
            "training_performed": False,
            "kronos_parameters_updated": False,
            "authorizes_training": False,
            "authorizes_fine_tuning": False,
            "authorizes_supervised_adaptation": False,
            "authorizes_production_inference": False,
            "authorizes_trading_claims": False,
            "scientific_result_available": False,
        }

    def execute_and_log() -> dict[str, Any]:
        try:
            return execute()
        except ResearchFailureError:
            raise
        except Exception as error:
            log_operational_failure(
                error,
                run_id=invocation.run_id,
                deployed_commit=invocation.deployed_commit,
                stage=stage.stage,
                logger=logger,
            )
            raise

    artifact: ProbeTerminalArtifact = publish_probe_test(
        store, invocation=invocation, execute=execute_and_log, now=now, stage=stage
    )
    return ProbeWorkerResult(
        artifact_key=artifact.key,
        content_sha256=artifact.content_sha256,
        outcome=artifact.outcome,
        phase="test",
        decision_outcome=artifact.decision_outcome,
        already_existed=artifact.already_existed,
        deployed_commit=invocation.deployed_commit,
        payload=artifact.payload,
    )
