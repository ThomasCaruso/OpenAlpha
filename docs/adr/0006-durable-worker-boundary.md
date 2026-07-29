# ADR 0006: Research Runs Execute in a Durable Worker

- Status: Accepted
- Date: 2026-07-28

## Context

Data downloads, model loading, autoregressive inference, backtests, and reports can exceed request timeouts and memory budgets. Failed jobs must preserve diagnostics without presenting partial output as valid.

## Decision

FastAPI validates and enqueues immutable run commands. A separate worker leases jobs from a durable database-backed queue abstraction, emits append-only state events and heartbeats, checks cancellation between stages, and atomically publishes artifacts. Laptop mode uses SQLite with one worker; full mode uses PostgreSQL and supports concurrent workers.

No expensive research work executes on an API request thread.

## Consequences

- Restart, timeout, cancellation, retry, and idempotency semantics are first-class.
- The first implementation must test leases and terminal-state publication.
- SQLite laptop mode is deliberately single-worker.
- A specialized queue product may replace the adapter later if operational evidence justifies it.

