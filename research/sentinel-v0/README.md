# Sentinel v0 Research Workspace

This directory contains the locked development-only configuration for testing whether pre-outcome Kronos instability and regime diagnostics predict later five-session forecast error.

## Boundaries

- `experiment.yaml` is the Phase 1 experiment lock.
- `results/` is for compact derived evaluation artifacts and their provenance.
- `reports/` is for human-readable artifact-grounded reports.
- Raw Alpaca responses, normalized restricted datasets, model weights, and Hugging Face caches remain outside Git.
- Protocol v1 is not defined or frozen here.

## Phase gates

1. Phase 2 proves one real SPY cutoff with nine Kronos paths, one baseline, a persisted diagnostic vector, a resolved outcome, an error, and an audit record. No Sentinel action exists before the Phase 3 risk-model freeze.
2. Phase 3 runs development cutoffs and freezes preprocessing, the risk model, reason thresholds, and action policy.
3. Phase 4 runs the untouched holdout once and applies the continuation rules in `experiment.yaml`.
4. Phase 5 records PROCEED, PROCEED WITH A NARROWER DIAGNOSTIC SET, CHANGE THE FAILURE-DETECTION APPROACH, or STOP THE PROJECT.

No result is valid unless its manifest identifies the exact experiment bytes, data request, forecast-provider pins, diagnostic configuration, risk configuration, code commit, dependency lock, and artifact hashes.
