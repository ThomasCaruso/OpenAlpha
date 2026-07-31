# Sentinel Phase 2.5 Structural-Validity and Integration Audit

**DEVELOPMENT AUDIT - NOT EMPIRICAL EVIDENCE**

Conclusion category: **OFFICIAL_RAW_PATH_STRUCTURAL_INVALIDITY_CONFIRMED**

## Integration finding

- Direct official output invalid: True.
- OpenAlpha provider output invalid: True.
- Direct official and OpenAlpha named OHLCV values match: True.
- Exact sealed comparison permitted: True.
- Direct official and sealed Phase 2 path match: True.
- Golden official fixture output valid: False.

The pinned official predictor performs named-column selection, derives amount when absent, normalizes each feature, tokenizes and decodes, averages internally in normalized space, then applies the inverse feature transform. It does not enforce or repair OHLC ordering.

## Phase 2 structural measurements

- Invalid path fraction: 0.7777777777777778 (nine sealed paths).
- Invalid candle fraction: 0.5333333333333333.
- Total constraint violations: 52.
- Maximum normalized severity: 0.014086593240486476 of the cutoff close.
- Same-input, same-seed repeatability across all three 512-context seeds: True.

## Averaging and projection

- Sealed 512 offline average matches the unchanged Phase 2 canonical close path: True.
- Official sample_count=3 and sample_count=5 outputs were validated separately. No equality with offline independent-seed averaging is claimed.
- Projection return invariant held for every projected path: True.
- CONSTRAINT_PROJECTION_V0 preserves raw output separately and changes only high and low. It is not evidence of improved accuracy.

## Canary recurrence

- Invalid paths: 12/18 (fraction 0.6666666666666666).
- The canary contains exactly six origins and eighteen official sample_count=1 paths. It cannot establish correlation, prevalence, or predictive value.

## Integrity

- Sealed Phase 2 descriptor/artifact fingerprint unchanged: True.
- Pinned official source inventory unchanged: True.
- The sealed Phase 2 forecast and outcome records were read and verified, never rewritten.

## Limitations

- Yahoo/yfinance is an unofficial, non-point-in-time development source.
- No independent provider comparison was performed.
- Raw close returns omit dividends.
- The official golden fixture is intraday while Sentinel Phase 2 uses daily bars.
- Structural invalidity has not yet been shown to predict forecast error.

## Next justified task

Lock structural-validity diagnostics and a mandatory raw-output validity gate in the development configuration before any holdout work. Then run the already declared chronological development sample without changing these definitions.
