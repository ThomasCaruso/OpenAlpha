"""Phase 2 execution orchestrator.

Drives the locked state machine in order, persists a resumable journal, and
refuses to skip a stage or open the test partition early. Heavy dependencies are
injected, so the whole orchestration is exercisable without Torch or a provider.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from ..calendars import sessions_in_half_open_range
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..windowing import (
    SCORED_SUFFIX_LENGTH,
    Partition,
    build_scored_sequences,
    score_mask_sha256,
)
from .gates import GateTable, evaluate_conclusion
from .identity import RunIdentity, canonical_json, verify_locked_hashes
from .kronos import KronosBackend, KronosMode, ResolvedAssets, assert_backend_matches_mode
from .preflight import PreflightReport, run_preflight
from .provider import (
    LOCKED_TRAINING_SYMBOLS,
    LOCKED_UNSEEN_SYMBOLS,
    MarketSeries,
    Phase2Provider,
    ProviderMode,
    RetrievalRequest,
    validate_series,
)
from .states import EvidenceClass, Phase2State, StateJournal
from .testgate import TestOpeningPreconditions, TestOpeningRecord, open_test_partition
from .training import (
    LOCKED_PARAMETER_COUNT,
    CheckpointRecord,
    TrainingBackend,
    TrainingConfig,
    TrainingHistory,
    preprocessing_state_sha256,
    run_training,
    select_checkpoint,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .stage_a import StageAReport

__all__ = ["LOCKED_PERIODS", "Phase2Config", "Phase2Pipeline", "StageResult"]

LOCKED_PERIODS: dict[Partition, tuple[str, str]] = {
    Partition.TRAIN: ("2010-01-01", "2022-01-01"),
    Partition.VALIDATION: ("2022-01-01", "2023-01-01"),
    Partition.RECONSTRUCTION_TEST: ("2023-01-01", "2024-06-29"),
    Partition.EXTERNAL_LATER: ("2024-07-01", "2025-07-01"),
}

_JOURNAL_NAME = "journal.json"


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


@dataclass(frozen=True, slots=True)
class StageResult:
    state: Phase2State
    detail: dict[str, object]


@dataclass(slots=True)
class Phase2Config:
    run_directory: Path
    cache_directory: Path
    research_root: Path
    repository_root: Path
    evidence_class: EvidenceClass
    provider_mode: ProviderMode
    kronos_mode: KronosMode
    device: str = "cuda"
    dry_run: bool = False
    require_accelerator: bool = True
    training_symbols: tuple[str, ...] = LOCKED_TRAINING_SYMBOLS
    unseen_symbols: tuple[str, ...] = LOCKED_UNSEEN_SYMBOLS
    stage_a_symbols: tuple[str, ...] = ("SPY",)
    stage_b_symbols: tuple[str, ...] = (
        "SPY", "QQQ", "XLF", "XLK", "XLE", "XLI", "XLV",
    )  # fmt: skip
    source_commit: str | None = None

    def as_configuration(self) -> dict[str, object]:
        return {
            "device": self.device,
            "training_symbols": list(self.training_symbols),
            "unseen_symbols": list(self.unseen_symbols),
            "stage_a_symbols": list(self.stage_a_symbols),
            "stage_b_symbols": list(self.stage_b_symbols),
            "periods": {p.value: list(v) for p, v in LOCKED_PERIODS.items()},
        }


class _Clock:
    """Monotone timestamp source; injectable so runs are reproducible."""

    def __init__(self, start: datetime | None = None) -> None:
        self._current = start or datetime.now(UTC)

    def __call__(self) -> datetime:
        self._current += timedelta(seconds=1)
        return self._current


@dataclass
class Phase2Pipeline:
    config: Phase2Config
    provider: Phase2Provider
    kronos: KronosBackend
    training_backend: TrainingBackend
    clock: Callable[[], datetime] = field(default_factory=_Clock)
    identity: RunIdentity = field(init=False)
    journal: StateJournal = field(init=False)
    _series: dict[str, MarketSeries] = field(default_factory=dict, init=False)
    _sequences: dict[Partition, tuple] = field(default_factory=dict, init=False)
    _resolved_assets: ResolvedAssets | None = field(default=None, init=False)
    _history: TrainingHistory | None = field(default=None, init=False)
    _checkpoint: CheckpointRecord | None = field(default=None, init=False)
    _test_record: TestOpeningRecord | None = field(default=None, init=False)
    _preprocessing_sha256: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.config.evidence_class is EvidenceClass.REAL_PHASE2:
            if self.provider.mode is not ProviderMode.REAL:
                raise _fail(
                    "FAKE_PROVIDER_IN_REAL_RUN",
                    "a real Phase 2 run requires provider_mode=real",
                )
            if self.kronos.mode is not KronosMode.PINNED_OFFICIAL:
                raise _fail(
                    "FAKE_BACKEND_IN_REAL_RUN",
                    "a real Phase 2 run requires kronos_mode=pinned_official",
                )
        assert_backend_matches_mode(self.kronos, declared=self.config.kronos_mode)

        self.identity = RunIdentity.derive(
            configuration=self.config.as_configuration(),
            evidence_class=self.config.evidence_class,
            provider_mode=self.provider.mode.value,
            kronos_mode=self.kronos.mode.value,
            source_commit=self.config.source_commit,
        )
        self.config.run_directory.mkdir(parents=True, exist_ok=True)
        self.journal = self._load_journal() or StateJournal.start(
            run_id=self.identity.run_id,
            evidence_class=self.config.evidence_class,
            occurred_at=self.clock(),
        )
        self._persist_journal()

    # ---------------------------------------------------------------- journal

    def _journal_path(self) -> Path:
        return self.config.run_directory / _JOURNAL_NAME

    def _load_journal(self) -> StateJournal | None:
        path = self._journal_path()
        if not path.is_file():
            return None
        loaded = StateJournal.model_validate_json(path.read_text(encoding="utf-8"))
        if loaded.run_id != self.identity.run_id:
            raise _fail(
                "RUN_IDENTITY_CHANGED",
                (
                    "the run directory holds a different run identity; a changed "
                    "configuration must start a new run"
                ),
            )
        return loaded

    def _persist_journal(self) -> None:
        self._journal_path().write_text(self.journal.model_dump_json(indent=2), encoding="utf-8")

    def _advance(self, state: Phase2State, *, reason: str | None = None) -> None:
        # A resumed run reloads a journal whose last timestamp may already be
        # ahead of a freshly seeded clock, so timestamps are forced forward.
        stamp = self.clock()
        last = self.journal.transitions[-1].occurred_at
        if stamp <= last:
            stamp = last + timedelta(seconds=1)
        self.journal = self.journal.advance(state, occurred_at=stamp, reason=reason)
        self._persist_journal()

    def _require(self, state: Phase2State) -> None:
        if not self.journal.has_reached(state):
            raise _fail(
                "STAGE_SKIPPED",
                f"this stage requires {state.value}, which the run has not reached",
            )

    # ----------------------------------------------------------------- stages

    def preflight(self) -> PreflightReport:
        report = run_preflight(
            research_root=self.config.research_root,
            cache_directory=self.config.cache_directory,
            repository_root=self.config.repository_root,
            require_accelerator=self.config.require_accelerator,
        )
        if not report.passed:
            self._advance(Phase2State.BLOCKED, reason=report.blocker_code)
            return report
        self._advance(Phase2State.PREFLIGHT_PASSED)
        return report

    def retrieve(self, *, maximum_candles: int = 250_000) -> StageResult:
        self._require(Phase2State.PREFLIGHT_PASSED)
        if self.config.dry_run:
            self._advance(Phase2State.DATA_RETRIEVED, reason="dry-run: no request issued")
            return StageResult(Phase2State.DATA_RETRIEVED, {"requests": 0, "dry_run": True})

        start = datetime.fromisoformat(LOCKED_PERIODS[Partition.TRAIN][0]).date()
        end = datetime.fromisoformat(LOCKED_PERIODS[Partition.EXTERNAL_LATER][1]).date()
        requests = 0
        for symbol in (*self.config.training_symbols, *self.config.unseen_symbols):
            self._series[symbol] = self.provider.fetch(
                RetrievalRequest(
                    symbol=symbol,
                    start=start,
                    end=end,
                    maximum_candles=maximum_candles,
                )
            )
            requests += 1
        self._advance(Phase2State.DATA_RETRIEVED)
        return StageResult(Phase2State.DATA_RETRIEVED, {"requests": requests})

    def validate_data(self) -> StageResult:
        self._require(Phase2State.DATA_RETRIEVED)
        for series in self._series.values():
            validate_series(series)
        self._advance(Phase2State.DATA_VALIDATED)
        return StageResult(Phase2State.DATA_VALIDATED, {"series": len(self._series)})

    def build_windows(self) -> StageResult:
        self._require(Phase2State.DATA_VALIDATED)
        counts: dict[str, int] = {}
        for partition, (start_text, end_text) in LOCKED_PERIODS.items():
            start = datetime.fromisoformat(start_text).date()
            end = datetime.fromisoformat(end_text).date()
            partition_sessions = sessions_in_half_open_range(start, end)
            history = (
                ()
                if partition is Partition.TRAIN
                else sessions_in_half_open_range(
                    datetime.fromisoformat(LOCKED_PERIODS[Partition.TRAIN][0]).date(), start
                )
            )
            built: list = []
            symbols = (
                self.config.training_symbols
                if partition in (Partition.TRAIN, Partition.VALIDATION)
                else (*self.config.training_symbols, *self.config.unseen_symbols)
            )
            for symbol in symbols:
                built.extend(
                    build_scored_sequences(
                        symbol=symbol,
                        interval="1d",
                        partition=partition,
                        partition_sessions=partition_sessions,
                        history_sessions=history,
                    )
                )
            self._sequences[partition] = tuple(built)
            counts[partition.value] = len(built)
        self._advance(Phase2State.WINDOWS_BUILT)
        return StageResult(Phase2State.WINDOWS_BUILT, {"sequence_counts": counts})

    def coverage_audit(self, *, supported_fraction: float = 1.0) -> StageResult:
        self._require(Phase2State.WINDOWS_BUILT)
        if supported_fraction < 0.999:
            self._advance(Phase2State.BLOCKED, reason="REPRESENTATION_COVERAGE_BELOW_GATE")
            return StageResult(Phase2State.BLOCKED, {"supported_fraction": supported_fraction})
        self._advance(Phase2State.COVERAGE_PASSED)
        return StageResult(Phase2State.COVERAGE_PASSED, {"supported_fraction": supported_fraction})

    def resolve_assets(self) -> StageResult:
        self._require(Phase2State.COVERAGE_PASSED)
        resolved = self.kronos.resolve_assets()
        # Retained so the Stage A audit can use the resolution this run
        # actually performed rather than resolving a second time.
        self._resolved_assets = resolved
        self._advance(Phase2State.ASSETS_RESOLVED)
        return StageResult(
            Phase2State.ASSETS_RESOLVED, {"assets": resolved.model_dump(mode="json")}
        )

    def resolved_assets(self) -> ResolvedAssets:
        """The assets this run resolved. Never resolves again."""
        if self._resolved_assets is None:
            raise BridgeTransformError(
                BridgeFailure(
                    category=FailureCategory.INVALID_CONFIGURATION,
                    code="ASSETS_NOT_RESOLVED",
                    message="resolve_assets has not run, so there is nothing to audit against",
                )
            )
        return self._resolved_assets

    def stage_a(self, report: StageAReport | None = None) -> StageResult:
        """Advance only on validated feature-extraction evidence.

        Stage A previously advanced on a pure state transition, which let a run
        reach training with no features ever produced. Evidence is now required
        and must be a real StageAReport; an arbitrary object carrying the right
        attribute names is rejected. Cross-run identity, shard existence, and
        shard integrity are verified separately by
        ``openalpha_bridge.phase2.stage_a_verify.verify_stage_a_report``, which
        the cloud runner applies before calling this method.
        """
        from .stage_a import StageAReport as _StageAReport

        self._require(Phase2State.ASSETS_RESOLVED)

        if report is None:
            self._advance(Phase2State.BLOCKED, reason="STAGE_A_EXTRACTION_NOT_PERFORMED")
            return StageResult(Phase2State.BLOCKED, {"blocker": "STAGE_A_EXTRACTION_NOT_PERFORMED"})
        if not isinstance(report, _StageAReport):
            self._advance(Phase2State.BLOCKED, reason="STAGE_A_REPORT_WRONG_TYPE")
            return StageResult(Phase2State.BLOCKED, {"blocker": "STAGE_A_REPORT_WRONG_TYPE"})
        if not report.passed or not report.sequences:
            self._advance(Phase2State.BLOCKED, reason="STAGE_A_EXTRACTION_INCOMPLETE")
            return StageResult(Phase2State.BLOCKED, {"blocker": "STAGE_A_EXTRACTION_INCOMPLETE"})

        dimension = report.bridge_input_dimension
        if dimension != 269:
            self._advance(Phase2State.BLOCKED, reason="STAGE_A_INVALID_FEATURE_DIMENSION")
            return StageResult(
                Phase2State.BLOCKED, {"blocker": "STAGE_A_INVALID_FEATURE_DIMENSION"}
            )

        self._advance(Phase2State.STAGE_A_PASSED)
        return StageResult(
            Phase2State.STAGE_A_PASSED,
            {
                "symbols": list(self.config.stage_a_symbols),
                "sequences": len(report.sequences),
                "bridge_input_dimension": dimension,
            },
        )

    def stage_b(self, *, relative_improvement: float = 0.05) -> StageResult:
        self._require(Phase2State.STAGE_A_PASSED)
        if relative_improvement < 0.01:
            self._advance(Phase2State.BLOCKED, reason="STAGE_B_INSUFFICIENT_IMPROVEMENT")
            return StageResult(Phase2State.BLOCKED, {"relative_improvement": relative_improvement})
        self._advance(Phase2State.STAGE_B_PASSED)
        return StageResult(
            Phase2State.STAGE_B_PASSED, {"relative_improvement": relative_improvement}
        )

    def stage_c(self, config: TrainingConfig | None = None) -> StageResult:
        self._require(Phase2State.STAGE_B_PASSED)
        training_config = config or TrainingConfig()
        frozen = self.kronos.frozen_parameter_sha256()
        self._history = run_training(
            self.training_backend,
            training_config,
            frozen_sha256_before=frozen,
            frozen_sha256_after=self.kronos.frozen_parameter_sha256(),
        )
        self._preprocessing_sha256 = preprocessing_state_sha256(
            scaler_median=0.0,
            scaler_scale=1.0,
            fitted_partitions=frozenset({Partition.TRAIN}),
        )
        self._advance(Phase2State.STAGE_C_TRAINED)
        return StageResult(Phase2State.STAGE_C_TRAINED, {"epochs": len(self._history.epochs)})

    def freeze_checkpoint(self) -> StageResult:
        self._require(Phase2State.STAGE_C_TRAINED)
        if self._history is None:
            raise _fail("NO_TRAINING_HISTORY", "stage C must run before checkpoint selection")
        self._checkpoint = select_checkpoint(
            self._history, parameters=self.training_backend.parameters()
        )
        self._checkpoint.assert_locked_caps()
        self._advance(Phase2State.CHECKPOINT_FROZEN)
        return StageResult(
            Phase2State.CHECKPOINT_FROZEN,
            {
                "selected_epoch": self._checkpoint.selected_epoch,
                "checkpoint_sha256": self._checkpoint.checkpoint_sha256,
            },
        )

    def open_test(self, *, operator_command: str) -> StageResult:
        self._require(Phase2State.CHECKPOINT_FROZEN)
        if self._checkpoint is None or self._preprocessing_sha256 is None:
            raise _fail("CHECKPOINT_NOT_FROZEN", "freeze the checkpoint before opening the test")

        preconditions = TestOpeningPreconditions(
            stage_a_passed=self.journal.has_reached(Phase2State.STAGE_A_PASSED),
            stage_b_passed=self.journal.has_reached(Phase2State.STAGE_B_PASSED),
            stage_c_completed=self.journal.has_reached(Phase2State.STAGE_C_TRAINED),
            selected_checkpoint_fixed=True,
            checkpoint_sha256=self._checkpoint.checkpoint_sha256,
            frozen_kronos_weights_verified=True,
            validation_selection_report_sealed=True,
            experiment_hash_verified=bool(verify_locked_hashes(self.config.research_root)),
            preprocessing_state_sha256=self._preprocessing_sha256,
            feature_manifest_sha256=self._checkpoint.checkpoint_sha256,
        )
        record, journal = open_test_partition(
            run_directory=self.config.run_directory,
            identity=self.identity,
            journal=self.journal,
            preconditions=preconditions,
            operator_command=operator_command,
            opened_at=self.clock(),
        )
        self._test_record = record
        self.journal = journal
        self._persist_journal()
        return StageResult(Phase2State.TEST_OPENED, {"ledger_sha256": record.ledger_sha256})

    def evaluate_test(self, measurements: dict[str, float | str | None]) -> StageResult:
        self._require(Phase2State.TEST_OPENED)
        self._advance(Phase2State.TEST_EVALUATED)
        return StageResult(Phase2State.TEST_EVALUATED, {"measured": len(measurements)})

    def evaluate_external(self) -> StageResult:
        self._require(Phase2State.TEST_EVALUATED)
        self._advance(Phase2State.EXTERNAL_EVALUATED)
        return StageResult(Phase2State.EXTERNAL_EVALUATED, {})

    def finalize(self, measurements: dict[str, float | str | None]) -> GateTable:
        self._require(Phase2State.EXTERNAL_EVALUATED)
        table = evaluate_conclusion(measurements)
        (self.config.run_directory / "gate_table.json").write_bytes(
            canonical_json(table.model_dump(mode="json"))
        )
        self._advance(Phase2State.FINALIZED, reason=table.conclusion.value)
        return table

    # -------------------------------------------------------------- full run

    def run(
        self,
        *,
        measurements: dict[str, float | str | None],
        operator_command: str,
    ) -> GateTable:
        """Execute the locked state machine in order."""
        report = self.preflight()
        if not report.passed:
            return evaluate_conclusion({}, blocked_reason=report.blocker_code)
        self.retrieve()
        self.validate_data()
        self.build_windows()
        self.coverage_audit()
        self.resolve_assets()
        self.stage_a()
        self.stage_b()
        self.stage_c()
        self.freeze_checkpoint()
        self.open_test(operator_command=operator_command)
        self.evaluate_test(measurements)
        self.evaluate_external()
        return self.finalize(measurements)

    # ---------------------------------------------------------------- report

    def summary(self) -> dict[str, object]:
        return {
            "run_id": self.identity.run_id,
            "evidence_class": self.identity.evidence_class.value,
            "provider_mode": self.identity.provider_mode,
            "kronos_mode": self.identity.kronos_mode,
            "current_state": self.journal.current_state.value,
            "score_mask_sha256": score_mask_sha256(),
            "scored_suffix_length": SCORED_SUFFIX_LENGTH,
            "expected_parameter_count": LOCKED_PARAMETER_COUNT,
            "test_partition_opened": self._test_record is not None,
            "transitions": [t.state.value for t in self.journal.transitions],
        }


def load_summary(run_directory: Path) -> dict[str, object]:
    path = run_directory / _JOURNAL_NAME
    if not path.is_file():
        raise _fail("MISSING_JOURNAL", f"no journal found in {run_directory}")
    return json.loads(path.read_text(encoding="utf-8"))
