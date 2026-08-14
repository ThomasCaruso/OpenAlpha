# OpenAlpha Master Plan

## Historical Bridge decision

This section preserves the retired Bridge product decision for historical context.
The current research architecture is documented in
[ARCHITECTURE.md](ARCHITECTURE.md), and the current research direction is in
[STATUS.md](STATUS.md).

OpenAlpha remained one project. Its proposed first public integration combined
Sentinel assurance, a learned Bridge compatibility decoder, and the evidence record.

The separately versioned Bridge v0 direction begins only with compatibility design
and the Phase 1 mathematical contract. It freezes the official encoder, implicit
codebook, Tokenizer-2k causal decoder trunk, Kronos-mini forecasting transformer,
token vocabulary, and every original generated token sequence. Only a 17,605-
parameter constrained sequence reconstruction head may later be trained. The raw
official decoder output and the separately labeled Bridge output must both remain
available.

This direction builds on rather than reverses Sentinel v1.1. The v1.1 locked rule
stopped inference-time token filtering and support-conditioned decoding after the
primary classification `TOKENIZER_CONSTRAINT_DEFECT`. Its documentation explicitly
left a separately trained constraint-preserving reconstruction direction outside
that stopped track. Bridge intervenes only at the continuous reconstruction
boundary and does not change token generation.

Sentinel v0 reliability prediction remains retired after a negative chronological
development result. Its frozen policy will not be run on the untouched holdout.
Sentinel v1 remains complete with **VALIDITY SUCCEEDS, QUALITY DEGRADES**: terminal
projection guaranteed returned validity, stepwise re-encoding hard-failed
excessively, and candidate resampling failed the range gate. No prior outcome,
threshold, result, artifact, or claim boundary is changed.

Under that retired direction, Bridge-2K reconstruction feasibility had to pass
before forecast integration. Bridge-base was prohibited unless Bridge-2K passed all
reconstruction, fixed-forecast, and external-generalization gates. See
`OPENALPHA_PIVOT.md`,
`ARCHITECTURE.md`, `STATUS.md`, `KRONOS_COMPATIBILITY_BOUNDARY.md`, and
`research/bridge-v0/experiment.yaml`.

## Preserved v0 mission and result

Sentinel v0 attempted to predict when a financial forecasting model was likely to
fail before its outcome and to select USE, BLEND, or ABSTAIN. Development evidence
did not support that proposition. No further v0 feature mining, refitting, threshold
tuning, or holdout evaluation is authorized.

Kronos remains the first model. Sentinel v1 prevents structurally impossible K-lines
from entering an autoregressive rollout and measures the fidelity, diversity, and
cost consequences.

## Sentinel v1 definition of success

Phase 3B succeeds only if an in-loop method can truthfully state:

> We enforced the versioned financial grammar before an invalid generated state
> conditioned later tokens, returned only valid paths or explicit hard failures, and
> measured the paired effects on path quality, range fidelity, barriers, diversity,
> latency, and memory without retraining or changing weights.

The decoder may fail its gates. A verified negative conclusion is valid.

## Completed Stage 3B - bounded decoder feasibility

1. Close v0 and preserve its negative evidence.
2. Trace and pin the official token-generation boundary.
3. Lock 12 development origins and numerical gates.
4. Compare raw autoregression, terminal projection, stepwise project/re-encode, and
   bounded valid-candidate resampling.
5. Run the model-size canary only after an in-loop mini method passes all gates.
6. Stop before a larger experiment or any untouched-holdout access.

All six steps are complete. Across 36 paired seed paths, raw Kronos output was
structurally valid 36.11% of the time. Terminal projection and both in-loop methods
returned only valid paths when they returned a path. Stepwise project/re-encode
hard-failed 21 paths, while candidate resampling returned all 36 paths but exceeded
the preregistered high-low range-error limit. The model-size canary was therefore
not run.

## Current next direction

Official-protocol replication is the next research direction. Its protocol must be
designed and its preregistration cryptographically sealed before code is written.
No official-protocol implementation exists.

## Historical next justified scope

The retired Bridge plan would have implemented Phase 1 only: the typed constrained
representation, stable forward and inverse transforms, causal previous-close
chaining, normalization-state contract, optional-volume behavior, independent
structural validation, deterministic serialization, property tests, and numerical
edge-case tests. It prohibited retrieving the Bridge corpus, fitting a head, running
reconstruction metrics, generating a checkpoint, running Kronos forecasts, or
accessing the untouched holdout until the complete Phase 1 gate had passed and its
artifacts had been committed.

The former v1.1 instruction to stop inference-only decoder research remained
binding within that retired plan. Bridge was not another candidate-selection method
and could not reuse v0 outcomes for feature mining or make an improved-forecasting
claim.

## Stage 0 — Preserved evidence infrastructure

Status: complete in `c037bd5` and `27f0c69`.

- content-addressed, path-confined artifacts;
- immutable run journals;
- verified completed-run manifests;
- experiment identity prototype;
- provider-independent market-data boundary;
- existing tests, Ruff, and Pyright;
- no hand-written dependency on an undocumented Yahoo endpoint.

## Stage 1 — Direction and experiment lock

Scope:

- Sentinel direction, failure taxonomy, and methodology;
- SPY/QQQ daily, five-session experiment;
- development 2024-07-01–2025-06-30;
- untouched holdout 2025-07-01–2026-06-30;
- weekly cutoff rule;
- fixed Kronos ensemble, baseline, diagnostics, targets, risk/action policies, evaluation outputs, and continuation criteria;
- minimal internal market-data, forecast-provider, diagnostic-vector, and later decision field contracts;
- exact pinned source/model/tokenizer revisions;
- archived unimplemented Reality Check plan/spec.

Acceptance:

- protocol v1 remains unfrozen;
- no experiment code or real forecast runs;
- no empirical Sentinel claim is made;
- the full preserved quality gates pass.

## Stage 2 — Smallest real diagnostic

For one historical SPY cutoff:

1. fetch one causal context from Yahoo Finance through pinned yfinance with every request option explicit;
2. run the seed-reproducibility probe and nine declared real Kronos requests;
3. generate the last-value baseline;
4. average only the three 512-context close paths and publish every path before outcome access;
5. compute and publish the diagnostic vector without inventing an unfitted Sentinel action;
6. resolve the five-session outcome;
7. append errors and a human-readable audit record;
8. verify artifacts, lifecycle, and manifest.

Stop if real Kronos cannot operate within the locked feasibility boundary.

Alpaca remains a later independent verification provider. No provider-independent empirical claim is permitted until a representative sample has been compared across separately sourced providers.

## Stage 3 — Development sample

- Materialize and hash the exact weekly origin list.
- Run chronological SPY/QQQ development cutoffs.
- Inspect only development diagnostic validity, missingness, and redundancy under fixed rules.
- Fit logistic failure probability and secondary ridge error regression.
- Freeze preprocessing, coefficients, failure/reason thresholds, and USE/BLEND/ABSTAIN policy.

## Stage 4 — Untouched holdout

- Run the frozen pipeline once on the later year.
- Report all five coverage levels, risk quintiles, calibration, baseline comparison, asset/time stability, and runtime/cost.
- Do not alter diagnostics, labels, models, thresholds, or actions.
- Apply the locked project-decision mapping.

## Stage 5 — Decision

Record exactly one:

- PROCEED;
- PROCEED WITH A NARROWER DIAGNOSTIC SET;
- CHANGE THE FAILURE-DETECTION APPROACH;
- STOP THE PROJECT.

Only then decide whether public interfaces, other models, exogenous-event data, complex repair, or protocol v1 are justified.

## Explicit non-goals

No frontend, accounts, database, trading, portfolio optimization, public SDK/API, generalized copilot, marketplace, MLOps platform, event ingestion, technical-indicator library, fine-tuning, complex repair, universal protocol schema, broad universe/horizons, large dataset, or committed checkpoint is part of Sentinel v0.

## Dependency principles

- Preserve and reuse `openalpha-research-core`; change it only for a proven incompatibility.
- Add at most one narrow `packages/sentinel` package when Phase 2 begins.
- Provider, diagnostics, labels, risk, action, evaluation, and report modules remain small internal boundaries.
- Real claims require real Kronos; fakes are test-only and explicitly synthetic.
- Raw provider data and model caches remain outside Git.
