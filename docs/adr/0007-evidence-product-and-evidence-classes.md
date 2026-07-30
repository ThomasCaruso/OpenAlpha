# ADR 0007: Evidence Product with Non-Equivalent Evidence Classes

- Status: Amended by ADR 0010
- Date: 2026-07-30

## Context

The original generalized research-platform scope would delay the empirical question and risk substituting architecture or presentation for evidence. Retrospective, sealed, and truly forward forecasts also carry materially different credibility.

## Decision

The evidence-class decision remains valid, but ADR 0010 changes the flagship from a passive Kronos benchmark to Sentinel forecast reliability. All forecasts and results remain classified as `historical_replay`, `sealed_historical_test`, or `live_precommitted_forecast`. The class is immutable provenance and survives aggregation.

Work is prioritized by whether it predicts, explains, or reduces forecast failure. Benchmark evidence remains a required input. General platform capabilities remain non-goals.

## Consequences

- Negative or inconclusive findings are valid deliverables.
- Historical replay cannot be marketed as live prediction.
- Aggregates cannot silently mix evidence classes.
- Sentinel v0 remains historical development/holdout evidence, not a live claim.
