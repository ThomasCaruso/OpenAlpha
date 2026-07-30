# ADR 0006: Research Runs Execute in a Durable Worker

- Status: Superseded by ADR 0009 for the initial release
- Date: 2026-07-28

## Context

Data downloads, model loading, autoregressive inference, backtests, and reports can exceed request timeouts and memory budgets. Failed jobs must preserve diagnostics without presenting partial output as valid.

## Decision

The original service-first decision is deferred. The initial release executes one bounded proof slice through a CLI application service while preserving immutable run events, explicit failures, resource diagnostics, and atomic artifact publication. A durable worker may be introduced only after measured scheduling or isolation requirements justify it and must reuse the same typed evidence contracts.

No expensive research work executes on an API request thread.

## Consequences

- The proof slice tests terminal-state publication, idempotency, explicit timeout/resource failure, and reproducibility without a queue.
- A future worker must not change forecast, ledger, protocol, or manifest semantics.

