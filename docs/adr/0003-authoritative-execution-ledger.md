# ADR 0003: OpenAlpha Owns the Authoritative Execution Ledger

- Status: Accepted
- Date: 2026-07-28

## Context

VectorBT is fast and useful, but its public execution abstractions use simplified percentage slippage, bar-level ordering assumptions, and limited order lifecycles. Its current Apache 2.0 with Commons Clause license also creates redistribution and commercial-service constraints for a flagship open-source core.

## Decision

Implement OpenAlpha's deterministic event-sourced order, fill, cash, holdings, and equity accounting as the authoritative simulation path. VectorBT is an optional adapter for research sweeps, analytics, and parity checks; no valid run depends exclusively on it.

## Consequences

- Financial assumptions are explicit per event and can evolve deliberately.
- Accounting invariants can be tested without hidden framework defaults.
- More implementation and verification work is required.
- Optional VectorBT use requires license disclosure and pinned engine/version settings.

## Sources

- [VectorBT portfolio documentation](https://vectorbt.dev/api/portfolio/base/)
- [VectorBT license](https://vectorbt.dev/terms/license/)
