# ADR 0007: Evidence Product with Non-Equivalent Evidence Classes

- Status: Accepted
- Date: 2026-07-30

## Context

The original generalized research-platform scope would delay the empirical question and risk substituting architecture or presentation for evidence. Retrospective, sealed, and truly forward forecasts also carry materially different credibility.

## Decision

OpenAlpha’s flagship is Kronos Reality Check. All forecasts and results are classified as `historical_replay`, `sealed_historical_test`, or `live_precommitted_forecast`. The class is immutable provenance, survives aggregation, and is visible in every scoreboard, audit page, report, and export.

Work is prioritized by whether it helps determine where Kronos adds forecast or economic value beyond declared baselines. General platform capabilities are non-goals until the CLI evidence slice succeeds.

## Consequences

- Negative or inconclusive findings are valid deliverables.
- Historical replay cannot be marketed as live prediction.
- Aggregates cannot silently mix evidence classes.
- Product breadth is intentionally reduced.
