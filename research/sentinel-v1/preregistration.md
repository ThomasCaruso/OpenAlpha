# Sentinel v1 Phase 3B Preregistration

**Label:** DEVELOPMENT FEASIBILITY - NOT HOLDOUT EVIDENCE

This study compares raw generation, terminal projection, stepwise
project-and-re-encode, and bounded valid-candidate resampling on 12 fixed SPY/QQQ
development origins. Its objective is to determine whether one impossible candle can
be prevented from contaminating later autoregressive steps without retraining.

The exact source pins, origins, seeds, method behavior, metrics, barriers, hard-fail
rules, and numerical continuation gates are normative in `experiment.yaml`. That
file is content-hashed before any v1 outcome comparison. Any semantic change requires
a new version and hash; an unattractive result is not a reason to amend it.

## Analysis discipline

- Every applicable method is run for every origin and seed.
- Method pairs share the same input, timestamps, and initial RNG seed.
- Failed paths remain failures; no replacement origin or seed is selected.
- Raw and projected paths are distinct immutable artifacts.
- An invalid constrained output is a contract breach and is converted to an explicit
  hard failure, never silently returned.
- All declared metrics and thresholds are reported, not only favorable ones.
- The v0 holdout and v0 risk model remain untouched.

## Decision rule

Only an in-loop method that passes all eight continuation gates can support
`CONSTRAINED_DECODING_SUCCEEDS`. Terminal projection alone can support structural
safety but cannot establish the central autoregressive-feedback hypothesis. If the
tokenizer round trip prevents a practical in-loop guarantee, the report must select
`TECHNICALLY_INFEASIBLE_WITH_CURRENT_TOKENIZER` or
`VALIDITY_SUCCEEDS_QUALITY_DEGRADES`, as supported by the measurements.

The Kronos-small/base canary is conditional and must not run unless an in-loop mini
method passes all gates.
