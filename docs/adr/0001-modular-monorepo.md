# ADR 0001: Modular Monorepo with Process Isolation

- Status: Accepted
- Date: 2026-07-28

## Context

OpenAlpha needs full-stack product quality, expensive research jobs, and strict financial contracts, but the first objective is one working vertical slice. Early microservices would distribute undeveloped contracts; a notebook/dashboard architecture would collapse product and research concerns.

## Decision

Use one repository with focused Python packages, a Next.js web application, a FastAPI API, and a separate worker process. Domain packages do not import web, API, queue, or vendor infrastructure. API and worker share typed application contracts.

## Consequences

- Expensive jobs never run on request threads.
- Cross-package changes remain atomic and testable.
- Deployment can split services later without rewriting financial logic.
- The repository needs disciplined dependency direction and package-level contract tests.

