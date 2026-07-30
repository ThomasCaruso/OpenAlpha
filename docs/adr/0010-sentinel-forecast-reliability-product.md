# ADR 0010: Sentinel Is a Forecast-Reliability Product

- Status: Accepted
- Date: 2026-07-30
- Amends: ADR 0002, ADR 0005, ADR 0007, ADR 0008, and ADR 0009
- Supersedes: ADR 0003 for Sentinel v0

## Context

Kronos Reality Check established the need for causal benchmarks and immutable evidence, but a passive benchmark does not help a developer decide whether to use a forecast before its outcome. Financial models usually emit a forecast even when sampling, context, regime, analogue, or data-quality evidence suggests elevated risk.

## Decision

OpenAlpha's product is Sentinel: a forecast-time reliability, failure-detection, and intervention layer. The first development proof asks whether pre-outcome Kronos diagnostics predict later five-session return error for SPY and QQQ.

Benchmarking remains an evaluation dependency. It is not the product surface. Protocol v1, a public CLI/SDK/API, and broad model support are deferred until Sentinel v0 produces a go/no-go result.

Phase 1 defines contracts in documentation and one experiment configuration only. Phase 2 may add one narrow `packages/sentinel` package for a real one-origin chain while reusing the existing artifact store, journals, manifests, and path confinement.

## Consequences

- The previous Reality Check plan and design remain archived history.
- Kronos is the first adapter, not a permanent product identity.
- USE, 50/50 BLEND, and ABSTAIN are the only v0 interventions.
- No holdout result may change diagnostics, labels, risk configuration, thresholds, or continuation criteria.
- Negative or inconclusive results can end the project.
