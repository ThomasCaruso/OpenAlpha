"""Machine-readable Phase 2 success and failure gates.

Every locked threshold is encoded here. The terminal conclusion is derived from
the gate table, so a free-form narrative can never override a failed gate.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "GateOutcome",
    "GateResult",
    "GateTable",
    "TerminalConclusion",
    "evaluate_conclusion",
    "locked_gate_specifications",
]


class TerminalConclusion(StrEnum):
    BRIDGE_2K_FEASIBLE = "BRIDGE_2K_FEASIBLE"
    BRIDGE_2K_PARTIALLY_FEASIBLE = "BRIDGE_2K_PARTIALLY_FEASIBLE"
    TOKENS_INSUFFICIENT = "TOKENS_INSUFFICIENT_FOR_COMPETITIVE_RECONSTRUCTION"
    OPERATIONALLY_BLOCKED = "OPERATIONALLY_BLOCKED"


class GateOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"


class GateScope(StrEnum):
    STRUCTURAL = "structural"
    QUALITY = "quality"
    OPERATIONAL = "operational"
    INTEGRITY = "integrity"


class GateSpecification(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    gate_id: str
    scope: GateScope
    threshold: float | str
    comparison: Literal["<=", ">=", "==", "<", ">"]
    explanation: str
    blocking: bool = True


class GateResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.gate.v1"] = "openalpha.bridge.phase2.gate.v1"
    gate_id: str
    scope: GateScope
    threshold: float | str
    comparison: str
    measured_value: float | str | None
    outcome: GateOutcome
    evidence_artifact: str | None
    explanation: str
    blocking: bool


class GateTable(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.gate_table.v1"] = (
        "openalpha.bridge.phase2.gate_table.v1"
    )
    results: tuple[GateResult, ...]
    conclusion: TerminalConclusion
    scientific_result_available: bool

    @property
    def failed(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if r.outcome is GateOutcome.FAIL)

    @property
    def not_evaluated(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if r.outcome is GateOutcome.NOT_EVALUATED)


def locked_gate_specifications() -> tuple[GateSpecification, ...]:
    """Every Phase 2 gate from experiment.yaml, in evaluation order."""
    return (
        GateSpecification(
            gate_id="structurally_invalid_candle_fraction",
            scope=GateScope.STRUCTURAL,
            threshold=0.0,
            comparison="<=",
            explanation="Bridge output must contain zero structurally invalid candles.",
        ),
        GateSpecification(
            gate_id="token_identifier_parity_fraction",
            scope=GateScope.INTEGRITY,
            threshold=1.0,
            comparison=">=",
            explanation="Official token identifiers must be preserved exactly.",
        ),
        GateSpecification(
            gate_id="frozen_weight_hash_parity_fraction",
            scope=GateScope.INTEGRITY,
            threshold=1.0,
            comparison=">=",
            explanation="Frozen Kronos weights must be unchanged across the run.",
        ),
        GateSpecification(
            gate_id="post_output_projection",
            scope=GateScope.INTEGRITY,
            threshold="false",
            comparison="==",
            explanation="No post-output projection may repair Bridge output.",
        ),
        GateSpecification(
            gate_id="high_low_range_mae_projection_ratio",
            scope=GateScope.QUALITY,
            threshold=0.90,
            comparison="<=",
            explanation="Range MAE must be at most 0.90 of terminal projection.",
        ),
        GateSpecification(
            gate_id="high_low_range_paired_bootstrap_upper",
            scope=GateScope.QUALITY,
            threshold=0.0,
            comparison="<",
            explanation="The paired-bootstrap difference upper bound must be below zero.",
        ),
        GateSpecification(
            gate_id="full_ohlc_mae_official_multiplier",
            scope=GateScope.QUALITY,
            threshold=1.05,
            comparison="<=",
            explanation="Full OHLC MAE must be within 1.05x the official decoder.",
        ),
        GateSpecification(
            gate_id="close_mae_official_multiplier",
            scope=GateScope.QUALITY,
            threshold=1.05,
            comparison="<=",
            explanation="Close MAE must be within 1.05x the official decoder.",
        ),
        GateSpecification(
            gate_id="close_return_mae_official_multiplier",
            scope=GateScope.QUALITY,
            threshold=1.05,
            comparison="<=",
            explanation="Close-return MAE must be within 1.05x the official decoder.",
        ),
        GateSpecification(
            gate_id="trainable_parameter_count",
            scope=GateScope.OPERATIONAL,
            threshold=1_000_000,
            comparison="<=",
            explanation="Trainable parameters must not exceed the locked cap.",
        ),
        GateSpecification(
            gate_id="checkpoint_size_bytes",
            scope=GateScope.OPERATIONAL,
            threshold=26_214_400,
            comparison="<=",
            explanation="The checkpoint must fit the locked 25 MiB cap.",
        ),
        GateSpecification(
            gate_id="median_latency_official_multiplier",
            scope=GateScope.OPERATIONAL,
            threshold=2.0,
            comparison="<=",
            explanation="Median latency must be within 2x the official decoder.",
        ),
        GateSpecification(
            gate_id="p95_incremental_latency_ms",
            scope=GateScope.OPERATIONAL,
            threshold=50.0,
            comparison="<=",
            explanation="p95 incremental latency must stay under 50 ms.",
        ),
        GateSpecification(
            gate_id="incremental_peak_memory_bytes",
            scope=GateScope.OPERATIONAL,
            threshold=536_870_912,
            comparison="<=",
            explanation="Incremental peak memory must stay under 512 MiB.",
        ),
        GateSpecification(
            gate_id="canonical_hash_match_fraction",
            scope=GateScope.INTEGRITY,
            threshold=1.0,
            comparison=">=",
            explanation="Deterministic replay must reproduce identical canonical hashes.",
        ),
    )


def _compare(measured: float, threshold: float, comparison: str) -> bool:
    match comparison:
        case "<=":
            return measured <= threshold
        case ">=":
            return measured >= threshold
        case "<":
            return measured < threshold
        case ">":
            return measured > threshold
        case "==":
            return measured == threshold
        case _:  # pragma: no cover - specifications are closed
            raise ValueError(f"unsupported comparison {comparison}")


def evaluate_conclusion(
    measurements: dict[str, float | str | None],
    *,
    evidence: dict[str, str] | None = None,
    blocked_reason: str | None = None,
) -> GateTable:
    """Derive the terminal conclusion from the locked gate table.

    ``blocked_reason`` short-circuits to OPERATIONALLY_BLOCKED without inventing
    quality results.
    """
    evidence = evidence or {}
    results: list[GateResult] = []

    for spec in locked_gate_specifications():
        measured = measurements.get(spec.gate_id)
        if measured is None:
            outcome = GateOutcome.NOT_EVALUATED
        elif isinstance(spec.threshold, str) or isinstance(measured, str):
            outcome = (
                GateOutcome.PASS if str(measured) == str(spec.threshold) else GateOutcome.FAIL
            )
        else:
            outcome = (
                GateOutcome.PASS
                if _compare(float(measured), float(spec.threshold), spec.comparison)
                else GateOutcome.FAIL
            )
        results.append(
            GateResult(
                gate_id=spec.gate_id,
                scope=spec.scope,
                threshold=spec.threshold,
                comparison=spec.comparison,
                measured_value=measured,
                outcome=outcome,
                evidence_artifact=evidence.get(spec.gate_id),
                explanation=spec.explanation,
                blocking=spec.blocking,
            )
        )

    table = tuple(results)
    if blocked_reason is not None:
        return GateTable(
            results=table,
            conclusion=TerminalConclusion.OPERATIONALLY_BLOCKED,
            scientific_result_available=False,
        )

    evaluated = [r for r in table if r.outcome is not GateOutcome.NOT_EVALUATED]
    if not evaluated:
        return GateTable(
            results=table,
            conclusion=TerminalConclusion.OPERATIONALLY_BLOCKED,
            scientific_result_available=False,
        )

    failures = [r for r in table if r.outcome is GateOutcome.FAIL]
    structural_failures = [r for r in failures if r.scope is GateScope.STRUCTURAL]
    integrity_failures = [r for r in failures if r.scope is GateScope.INTEGRITY]
    quality_failures = [r for r in failures if r.scope is GateScope.QUALITY]

    if integrity_failures or structural_failures:
        conclusion = TerminalConclusion.OPERATIONALLY_BLOCKED
    elif not failures:
        conclusion = TerminalConclusion.BRIDGE_2K_FEASIBLE
    elif len(quality_failures) == len([r for r in table if r.scope is GateScope.QUALITY]):
        conclusion = TerminalConclusion.TOKENS_INSUFFICIENT
    else:
        conclusion = TerminalConclusion.BRIDGE_2K_PARTIALLY_FEASIBLE

    return GateTable(
        results=table,
        conclusion=conclusion,
        scientific_result_available=conclusion
        is not TerminalConclusion.OPERATIONALLY_BLOCKED,
    )
