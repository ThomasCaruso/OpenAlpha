# Sentinel v0 Amendment: Structural-Validity Diagnostics

- Amendment date: 2026-07-30
- Previous experiment SHA-256: 587bc36ac1dd941c66d9e94c92d7b26a33faeb8e7e7a38883762f0cb54897950
- Amended experiment SHA-256: fd50c208f465ce99750b11d4de5d5bc8c391bd73cb9ff450037c596c3e37d8bc
- Governance state: discovered during development, before risk-model fitting or holdout inspection
- Audit conclusion: OFFICIAL_RAW_PATH_STRUCTURAL_INVALIDITY_CONFIRMED

## Exact changed fields

- schema_version: sentinel-v0.3 to sentinel-v0.4
- status: changed to phase_2_5_structural_validity_amended_before_development_sample
- amendment: now supersedes the prior yfinance configuration hash and records this audit
- diagnostics: added the eleven structural diagnostics listed below
- expected_error_correlation_signs: added locked development expectations
- structural_validity: added the validity contract, severity denominator, raw-output policy, projection policy, and runtime gate

## Added diagnostics

1. INVALID_PATH_FRACTION
2. INVALID_CANDLE_FRACTION
3. TOTAL_CONSTRAINT_VIOLATIONS
4. MAX_CONSTRAINT_VIOLATION_SEVERITY
5. MEAN_CONSTRAINT_VIOLATION_SEVERITY
6. EARLIEST_INVALID_HORIZON_STEP
7. HIGH_LOW_INVERSION_COUNT
8. HIGH_BELOW_BODY_COUNT
9. LOW_ABOVE_BODY_COUNT
10. NONFINITE_OUTPUT_COUNT
11. NONPOSITIVE_PRICE_COUNT

Price-related severity is the absolute constraint gap divided by the final observed raw close at the cutoff. This denominator is causal, positive for an accepted input, and common to every path at an origin.

## Locked expectation

Larger invalid-path and invalid-candle fractions, violation counts, and violation severities are expected to associate with larger future Kronos error. A lower EARLIEST_INVALID_HORIZON_STEP is expected to associate with larger error because invalidity appears sooner. NONFINITE_OUTPUT_COUNT and NONPOSITIVE_PRICE_COUNT are runtime gates rather than eligible statistical predictors because their forecasts may not yield a defensible scoreable return.

These are development hypotheses, not established empirical relationships. They will not be redefined after holdout inspection.

## Runtime and preservation policy

- Validate every raw path before downstream use.
- Persist original model output and structured violations without mutation.
- Block use for nonfinite values, nonpositive prices, missing or duplicate timestamps, or the wrong horizon.
- Mark finite OHLC-ordering failures structurally invalid but retain them for development error analysis.
- Never silently substitute the baseline or a projected path.
- CONSTRAINT_PROJECTION_V0 may be emitted only as a separate labeled artifact; it preserves open and close and is not an accuracy claim.

The sealed Phase 2 forecast and outcome continue to reference the prior experiment hash. They were not rewritten. This amendment governs later Sentinel v0 development origins and is locked before any holdout evaluation.
