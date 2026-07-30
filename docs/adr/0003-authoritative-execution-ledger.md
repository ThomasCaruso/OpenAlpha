# ADR 0003: OpenAlpha Owns the Authoritative Execution Ledger

- Status: Superseded by ADR 0010 for Sentinel v0
- Date: 2026-07-28

## Context

VectorBT is fast and useful, but its public execution abstractions use simplified percentage slippage, bar-level ordering assumptions, and limited order lifecycles. Its current Apache 2.0 with Commons Clause license also creates redistribution and commercial-service constraints for a flagship open-source core.

## Decision

The original generalized-platform plan was to implement OpenAlpha's deterministic event-sourced order, fill, cash, holdings, and equity accounting as the authoritative simulation path. That work was never implemented.

ADR 0010 removes trading and portfolio simulation from Sentinel v0. Existing manifest vocabulary is preserved for compatibility, but Sentinel must not create fake order, fill, or accounting artifacts. Economic evaluation may receive a new ADR only after the reliability holdout establishes a reason to build it.

## Consequences

- No execution-ledger implementation is on the Sentinel roadmap.
- The historical rationale remains available if a later evidence-backed experiment needs economic evaluation.
- Sentinel manifests need an additive non-trading completion profile rather than repurposed trading artifact meanings.

## Sources

- [VectorBT portfolio documentation](https://vectorbt.dev/api/portfolio/base/)
- [VectorBT license](https://vectorbt.dev/terms/license/)
