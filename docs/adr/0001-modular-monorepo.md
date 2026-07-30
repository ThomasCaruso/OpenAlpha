# ADR 0001: Modular Monorepo with Process Isolation

- Status: Amended by ADR 0009
- Date: 2026-07-28

## Context

OpenAlpha needs full-stack product quality, expensive research jobs, and strict financial contracts, but the first objective is one working vertical slice. Early microservices would distribute undeveloped contracts; a notebook/dashboard architecture would collapse product and research concerns.

## Decision

Use one repository with focused Python packages. Domain packages do not import presentation, queue, or vendor infrastructure. A Next.js application, FastAPI API, and separate worker remain permissible future adapters, but ADR 0009 removes them from the initial acceptance boundary until the CLI forecast-to-outcome slice is verified.

## Consequences

- The initial proof slice has no request thread; resource limits and explicit diagnostics apply in the CLI process.
- Cross-package changes remain atomic and testable.
- Deployment can split services later without rewriting financial logic.
- The repository needs disciplined dependency direction and package-level contract tests.

