# Sentinel v1.1 Token-Manifold Compatibility Preregistration

**DEVELOPMENT COMPATIBILITY RESEARCH - NOT HOLDOUT EVIDENCE**

## Question and stopping boundary

This bounded study distinguishes tokenizer constraint defects, off-manifold token
pairs, low valid conditional probability mass, candidate-search failure, and
model/tokenizer-size specificity. It reuses the twelve Sentinel v1 development
origins. The untouched holdout begins 2025-07-01 and may not be accessed.

This is the final decoder investigation. If no authorized support-aware method
passes every locked continuation gate, constrained-decoder research stops.

## Corpus and round-trip audit

The support corpus is SPY, QQQ, IWM, DIA, TLT, HYG, GLD, EFA, EEM, and XLF. Raw
daily yfinance observations end on 2024-06-28. Each instrument contributes exactly
the final 1,536 complete XNYS sessions, split into three non-overlapping 512-session
windows. There are no replacement instruments.

Tokenizer-2k and Tokenizer-base use the pinned predictor preprocessing, including
the in-memory derived amount field. The audit records structural validity,
reconstruction errors, token-pair counts, and results by instrument and causal
volatility regime. The Wilson thresholds in experiment.yaml decide material defect,
overwhelming validity, or ambiguity.

## Support and probability

Empirical support means an exact tokenizer-specific hierarchical pair occurs at
least twice in the fixed corpus. Jeffreys smoothing uses alpha 0.5. Generated
support analysis uses the exact twelve v1 origins and three seeds.

Probability mass uses the actual temperature-1.0, top-p-0.9 hierarchical sampling
distribution. Nested 64-, 256-, and 1,024-pair grids yield lower bounds and upper
bounds that add all uncovered mass. Truncated estimates are never described as
exact. Zero lower mass has an explicit status and no nonfinite JSON value.

## Model-size canary and classification

The canary uses only 2024-07-05 and 2025-04-11 for SPY and QQQ with three seeds.
Mini/Tokenizer-2k runs first, then small/Tokenizer-base, then base/Tokenizer-base
when every operational limit remains satisfied. Skips are immutable results.

Root cause is selected mechanically in the precedence recorded in experiment.yaml.
Outcomes do not enter classification.

## Conditional method and outcome firewall

Support-conditioned decoding is authorized only for OFF_MANIFOLD_TOKEN_COMBINATIONS,
LOW_VALID_PROBABILITY_MASS, or CANDIDATE_SEARCH_FAILURE. It filters the bounded
candidate grid for mathematical validity and exact-pair support, renormalizes
original model probability, and samples deterministically. An empty set is a hard
failure. No projection or unconstrained fallback is allowed.

The optional draft-conditioned variant has one fixed distance weight and may run
only when its forecast-time eligibility gate passes. Forecasts are sealed before
the separate outcome resolver reads the already declared development outcomes.

## Decision

A method justifies one separately preregistered larger experiment only if all
continuation gates in experiment.yaml pass. Otherwise Sentinel becomes a structural
validator, immutable raw-output audit layer, deterministic terminal-projection
gateway, token-manifold compatibility profiler, and model-selection safety check.

Raw provider data, model weights, tokenizer weights, and caches remain outside Git.
Sentinel v0 and v1 files remain unchanged.
