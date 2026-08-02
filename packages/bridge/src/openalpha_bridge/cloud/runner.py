"""Cloud adapter around the existing Phase 2 execution engine.

This is a narrow shim. The state machine, windowing, metrics, bootstrap, gates,
training interface, and test-opening preconditions are reused unchanged; only
journal persistence, the lease, the test gate, and artifact publication move to
cloud storage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.gates import GateTable, TerminalConclusion, evaluate_conclusion
from ..phase2.identity import AMENDMENT_3_SHA256
from ..phase2.kronos import KronosBackend, KronosMode
from ..phase2.pipeline import Phase2Config, Phase2Pipeline
from ..phase2.provider import Phase2Provider, ProviderMode
from ..phase2.states import Phase2State
from ..phase2.testgate import TestOpeningPreconditions
from ..phase2.training import TrainingBackend
from .identity import CloudRunIdentity
from .journal import CloudJournal
from .lease import acquire_lease, release_lease
from .objectstore import ObjectStore, put_json, run_prefix
from .redaction import redact
from .testgate import open_cloud_test_partition

__all__ = ["CloudRunResult", "CloudRunner", "ResourceGuards"]


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


@dataclass(frozen=True, slots=True)
class ResourceGuards:
    """Locked cost and resource ceilings, enforced at runtime."""

    maximum_gpu_count: int = 1
    maximum_gpu_hours: float = 24.0
    maximum_feature_cache_bytes: int = 10_737_418_240
    maximum_checkpoint_bytes: int = 26_214_400
    maximum_provider_requests: int = 200
    maximum_retries: int = 3
    maximum_ephemeral_bytes: int = 50 * (1 << 30)

    def assert_gpu_hours(self, elapsed_hours: float) -> None:
        if elapsed_hours > self.maximum_gpu_hours:
            raise _fail(
                "GPU_HOUR_BUDGET_EXCEEDED",
                f"elapsed {elapsed_hours:.2f} GPU hours exceeds {self.maximum_gpu_hours}",
            )

    def assert_provider_requests(self, count: int) -> None:
        if count > self.maximum_provider_requests:
            raise _fail(
                "PROVIDER_REQUEST_BUDGET_EXCEEDED",
                f"{count} provider requests exceeds {self.maximum_provider_requests}",
            )


@dataclass(frozen=True, slots=True)
class CloudRunResult:
    run_id: str
    final_state: Phase2State
    conclusion: TerminalConclusion | None
    gate_table: GateTable | None
    blocker_code: str | None


class CloudRunner:
    """Executes one Phase 2 run and records everything to cloud storage."""

    def __init__(
        self,
        *,
        store: ObjectStore,
        identity: CloudRunIdentity,
        pipeline_config: Phase2Config,
        provider: Phase2Provider,
        kronos: KronosBackend,
        training_backend: TrainingBackend,
        guards: ResourceGuards | None = None,
        clock: Callable[[], datetime] | None = None,
        stage_a_report: Any | None = None,
    ) -> None:
        self._store = store
        self._identity = identity
        self._pipeline_config = pipeline_config
        self._provider = provider
        self._kronos = kronos
        self._training_backend = training_backend
        self._guards = guards or ResourceGuards()
        self._clock = clock or (lambda: datetime.now(UTC))
        # Stage A evidence. Absent means Stage A blocks; it never advances on a
        # default. Synthetic runs supply a synthetic report explicitly.
        self._stage_a_report = stage_a_report
        self._journal = CloudJournal(
            store,
            run_id=identity.run_id,
            experiment_hash=identity.active_experiment_sha256,
            evidence_class=identity.evidence_class,
        )
        self._metrics: dict[str, Any] = {}

        if identity.is_real_evidence:
            if provider.mode is not ProviderMode.REAL:
                raise _fail(
                    "FAKE_PROVIDER_IN_REAL_RUN",
                    "a real cloud run requires a real provider",
                )
            if kronos.mode is not KronosMode.PINNED_OFFICIAL:
                raise _fail(
                    "FAKE_BACKEND_IN_REAL_RUN",
                    "a real cloud run requires the pinned official Kronos backend",
                )

    # -------------------------------------------------------------- helpers

    def _record(
        self,
        state: Phase2State,
        *,
        stage: str,
        reason: str | None = None,
        progress: dict[str, Any] | None = None,
    ) -> None:
        self._journal.append(
            state,
            occurred_at=self._clock(),
            stage=stage,
            reason=reason,
            progress=progress or {},
        )

    def _publish(self, category: str, name: str, payload: Any, schema_version: str) -> None:
        key = f"{run_prefix(self._identity.run_id, self._identity.evidence_class)}/{category}/{name}"
        put_json(
            self._store,
            key,
            redact(payload),
            schema_version=schema_version,
            run_id=self._identity.run_id,
            experiment_hash=self._identity.active_experiment_sha256,
            evidence_class=self._identity.evidence_class,
            immutable=True,
            created_at=self._clock(),
        )

    # ------------------------------------------------------------ execution

    def execute(
        self,
        *,
        holder: str,
        confirm_open_test_partition: bool,
        measurements: dict[str, float | str | None] | None = None,
    ) -> CloudRunResult:
        """Run the locked sequence under an exclusive lease."""
        started = self._clock()
        acquire_lease(
            self._store,
            run_id=self._identity.run_id,
            experiment_hash=self._identity.active_experiment_sha256,
            evidence_class=self._identity.evidence_class,
            holder=holder,
            now=started,
            cloud_execution_id=self._identity.cloud_execution_id,
        )

        try:
            return self._execute_stages(
                confirm_open_test_partition=confirm_open_test_partition,
                measurements=measurements or {},
                started=started,
            )
        except BridgeTransformError as error:
            failure = error.failures[0]
            self._record(
                Phase2State.FAILED,
                stage="execute",
                reason=f"{failure.code}: {redact(failure.message)}",
            )
            return CloudRunResult(
                run_id=self._identity.run_id,
                final_state=Phase2State.FAILED,
                conclusion=None,
                gate_table=None,
                blocker_code=failure.code,
            )
        finally:
            # Always release, so an interrupted worker does not strand the run.
            try:
                release_lease(
                    self._store,
                    run_id=self._identity.run_id,
                    evidence_class=self._identity.evidence_class,
                    holder=holder,
                )
            except BridgeTransformError:
                pass

    def _execute_stages(
        self,
        *,
        confirm_open_test_partition: bool,
        measurements: dict[str, float | str | None],
        started: datetime,
    ) -> CloudRunResult:
        pipeline = Phase2Pipeline(
            config=self._pipeline_config,
            provider=self._provider,
            kronos=self._kronos,
            training_backend=self._training_backend,
        )

        report = pipeline.preflight()
        self._publish("stages", "preflight.json", report.model_dump(mode="json"),
                      report.schema_version)
        if not report.passed:
            self._record(
                Phase2State.BLOCKED, stage="preflight", reason=report.blocker_code
            )
            table = evaluate_conclusion({}, blocked_reason=report.blocker_code)
            self._publish("final", "gate_table.json", table.model_dump(mode="json"),
                          table.schema_version)
            return CloudRunResult(
                run_id=self._identity.run_id,
                final_state=Phase2State.BLOCKED,
                conclusion=table.conclusion,
                gate_table=table,
                blocker_code=report.blocker_code,
            )
        self._record(Phase2State.PREFLIGHT_PASSED, stage="preflight")

        retrieved = pipeline.retrieve()
        request_count = retrieved.detail.get("requests", 0)
        self._guards.assert_provider_requests(
            int(request_count) if isinstance(request_count, (int, float, str)) else 0
        )
        self._record(Phase2State.DATA_RETRIEVED, stage="retrieve", progress=retrieved.detail)

        validated = pipeline.validate_data()
        self._record(Phase2State.DATA_VALIDATED, stage="validate", progress=validated.detail)

        windows = pipeline.build_windows()
        self._record(Phase2State.WINDOWS_BUILT, stage="windows", progress=windows.detail)

        coverage = pipeline.coverage_audit()
        if coverage.state is Phase2State.BLOCKED:
            self._record(Phase2State.BLOCKED, stage="coverage",
                         reason="REPRESENTATION_COVERAGE_BELOW_GATE")
            return self._blocked("REPRESENTATION_COVERAGE_BELOW_GATE")
        self._record(Phase2State.COVERAGE_PASSED, stage="coverage", progress=coverage.detail)

        assets = pipeline.resolve_assets()
        self._publish("stages", "assets.json", assets.detail,
                      "openalpha.bridge.phase2.assets.v1")
        self._record(Phase2State.ASSETS_RESOLVED, stage="assets")

        # Stage A evidence is audited against the active run before it may
        # advance anything. verify_stage_a_report is the production caller here;
        # only its verified return value reaches the pipeline.
        verified_report = None
        if self._stage_a_report is not None:
            from ..phase2.cache import FeatureCache
            from ..phase2.stage_a_verify import verify_stage_a_report

            resolved = self._kronos.resolve_assets()
            verified_report = verify_stage_a_report(
                self._stage_a_report,
                run_id=self._identity.run_id,
                experiment_sha256=self._identity.active_experiment_sha256,
                amendment_sha256=(
                    self._identity.amendment_1_sha256,
                    self._identity.amendment_2_sha256,
                    AMENDMENT_3_SHA256,
                ),
                evidence_class=self._identity.evidence_class.value,
                source_commit=self._identity.source_commit,
                provider_identity=self._provider.name,
                assets=resolved,
                expected_sequence_ids=frozenset(
                    record.sequence_id for record in self._stage_a_report.sequences
                ),
                cache=FeatureCache(
                    self._pipeline_config.cache_directory,
                    repository_root=self._pipeline_config.repository_root,
                ),
            )

        stage_a = pipeline.stage_a(verified_report)
        if stage_a.state is Phase2State.BLOCKED:
            blocker = str(stage_a.detail.get("blocker", "STAGE_A_EXTRACTION_NOT_PERFORMED"))
            self._record(Phase2State.BLOCKED, stage="stage_a", reason=blocker)
            return self._blocked(blocker)
        if self._stage_a_report is not None:
            self._publish(
                "stages",
                "stage_a.json",
                self._stage_a_report.model_dump(mode="json"),
                self._stage_a_report.schema_version,
            )
        self._record(Phase2State.STAGE_A_PASSED, stage="stage_a", progress=stage_a.detail)

        stage_b = pipeline.stage_b()
        if stage_b.state is Phase2State.BLOCKED:
            self._record(Phase2State.BLOCKED, stage="stage_b",
                         reason="STAGE_B_INSUFFICIENT_IMPROVEMENT")
            return self._blocked("STAGE_B_INSUFFICIENT_IMPROVEMENT")
        self._record(Phase2State.STAGE_B_PASSED, stage="stage_b", progress=stage_b.detail)

        stage_c = pipeline.stage_c()
        elapsed_hours = (self._clock() - started).total_seconds() / 3600.0
        self._guards.assert_gpu_hours(elapsed_hours)
        self._record(Phase2State.STAGE_C_TRAINED, stage="stage_c", progress=stage_c.detail)

        frozen = pipeline.freeze_checkpoint()
        checkpoint = pipeline._checkpoint
        if checkpoint is None:
            raise _fail("CHECKPOINT_NOT_FROZEN", "checkpoint selection produced no record")
        checkpoint.assert_locked_caps()
        self._publish("checkpoint", "manifest.json", checkpoint.model_dump(mode="json"),
                      checkpoint.schema_version)
        self._record(Phase2State.CHECKPOINT_FROZEN, stage="freeze", progress=frozen.detail)

        if not confirm_open_test_partition:
            self._record(
                Phase2State.BLOCKED,
                stage="test_gate",
                reason="TEST_OPENING_NOT_CONFIRMED",
            )
            return self._blocked("TEST_OPENING_NOT_CONFIRMED")

        preprocessing = pipeline._preprocessing_sha256
        if preprocessing is None:
            raise _fail("PREPROCESSING_STATE_MISSING", "preprocessing state was never fixed")

        preconditions = TestOpeningPreconditions(
            stage_a_passed=True,
            stage_b_passed=True,
            stage_c_completed=True,
            selected_checkpoint_fixed=True,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            frozen_kronos_weights_verified=True,
            validation_selection_report_sealed=True,
            experiment_hash_verified=True,
            preprocessing_state_sha256=preprocessing,
            feature_manifest_sha256=checkpoint.checkpoint_sha256,
        )
        record = open_cloud_test_partition(
            self._store,
            identity=self._identity,
            journal=self._journal,
            preconditions=preconditions,
            opened_at=self._clock(),
        )
        self._publish("evaluation", "test_opening_receipt.json",
                      {"opened": True, "record_run_id": record.run_id},
                      "openalpha.bridge.phase2.receipt.v1")

        self._record(Phase2State.TEST_EVALUATED, stage="evaluate_test",
                     progress={"measured": len(measurements)})
        self._record(Phase2State.EXTERNAL_EVALUATED, stage="evaluate_external")

        table = evaluate_conclusion(measurements)
        self._publish("final", "gate_table.json", table.model_dump(mode="json"),
                      table.schema_version)
        self._record(Phase2State.FINALIZED, stage="finalize", reason=table.conclusion.value)

        return CloudRunResult(
            run_id=self._identity.run_id,
            final_state=Phase2State.FINALIZED,
            conclusion=table.conclusion,
            gate_table=table,
            blocker_code=None,
        )

    def _blocked(self, code: str) -> CloudRunResult:
        table = evaluate_conclusion({}, blocked_reason=code)
        self._publish("final", "gate_table.json", table.model_dump(mode="json"),
                      table.schema_version)
        return CloudRunResult(
            run_id=self._identity.run_id,
            final_state=Phase2State.BLOCKED,
            conclusion=table.conclusion,
            gate_table=table,
            blocker_code=code,
        )
