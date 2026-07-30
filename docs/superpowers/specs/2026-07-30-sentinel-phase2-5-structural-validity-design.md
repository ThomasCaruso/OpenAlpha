# Sentinel Phase 2.5 Structural-Validity and Integration Audit Design

## Status and claim boundary

This design governs a development-only forensic audit of the structural OHLC violations observed during the sealed Sentinel Phase 2 SPY origin. It does not classify those violations as a Kronos failure, does not change the sealed Phase 2 forecast or outcome, and does not authorize the full Sentinel development sample.

The audit must end with exactly one classification:

- `OPENALPHA_INTEGRATION_BUG`
- `OFFICIAL_RAW_PATH_STRUCTURAL_INVALIDITY_CONFIRMED`
- `AMBIGUOUS`
- `NONRECURRING_EDGE_CASE`

## Preserved evidence boundary

The existing Phase 2 private state at `~/.cache/openalpha-sentinel/phase2/run-spy-20240705` is read-only. Before and after Phase 2.5, the audit records hashes for `creation.json`, `resolved.json`, and the inventory of content-addressed artifacts. Phase 2.5 writes only to `~/.cache/openalpha-sentinel/phase2_5` and to the compact repository outputs declared by the user.

No raw Yahoo response, reusable market-data history, model weight, tokenizer weight, or cache file enters Git. Compact traces contain schemas, hashes, dimensions, dtypes, date bounds, and safe numerical summaries only.

## Investigative approach

The audit combines runtime boundary instrumentation with black-box comparison. It does not modify the pinned official Kronos source.

Runtime wrappers observe the actual objects passed through these boundaries and immediately delegate to the original pinned functions:

1. normalized Yahoo OHLCV snapshot;
2. 512-row context slice;
3. official predictor DataFrame preparation;
4. normalized `float32` feature tensor;
5. tokenizer encoded token representation;
6. autoregressive model generation;
7. tokenizer decoded normalized tensor;
8. official predictor inverse transformation;
9. official predictor output DataFrame;
10. OpenAlpha named-column canonicalization;
11. stored Phase 2 forecast artifact.

Each trace boundary records column names and order, shape, dtype, units, timestamp bounds, minimum and maximum by feature, canonical hash, normalization or inverse-transformation status, and any copy, reorder, clip, cast, or rounding operation. Token traces record shapes, dtypes, ranges, and hashes rather than token payloads.

## Pinned-source facts to verify in the report

The audit cites exact paths and functions from pinned source revision `67b630e67f6a18c9e9be918d9b4337c960db1e9a`, including:

- `model/kronos.py:KronosPredictor.predict`
- `model/kronos.py:KronosPredictor.generate`
- `model/kronos.py:auto_regressive_inference`
- `model/kronos.py:KronosTokenizer.encode`
- `model/kronos.py:KronosTokenizer.decode`
- `model/kronos.py:calc_time_stamps`
- `README.md` prediction example and input contract
- `tests/test_kronos_regression.py`

The report verifies named-column selection, optional volume and amount behavior, the derived amount formula, timestamp semantics, per-feature normalization, clipping, tokenization, autoregressive sampling, decoded-path averaging, denormalization, output schema, and the absence or presence of official structural repair.

## Golden-path and three-boundary comparison

The direct golden path uses the official regression input fixture from the pinned source clone with the pinned `NeoQuasar/Kronos-mini` model revision `f4e68697d9d5aed55cef5c96aabc3376bcad9f81` and `NeoQuasar/Kronos-Tokenizer-2k` revision `26966d0035065a0cae0ebad7af8ece35bc1fb51c`. The upstream fixture remains outside OpenAlpha Git.

For the Phase 2 origin, the audit re-fetches raw SPY data with the already pinned yfinance request. Exact comparison with the sealed path is permitted only if the normalized input hash equals `f09446b7f7d541907401ca133580eac205c99acb5e27d84bf922943cc4d57bbe`. If the hash differs, the report preserves the difference and does not claim an exact stored-output reproduction.

With an exact input match, the same context-512 request and seed are executed through:

- A: direct pinned `KronosPredictor.predict` outside OpenAlpha artifact code;
- B: the OpenAlpha typed subprocess provider;
- C: the immutable Phase 2 forecast artifact.

Named-column values, timestamps, canonical hashes, and structural violations are compared. The first unequal boundary determines whether OpenAlpha introduced a transformation or interpretation error.

## Structural-validity contract

A single focused module, `openalpha_sentinel.structural_validity`, validates model outputs without changing them. It returns immutable structured violations with path ID, one-based horizon step, timestamp, code, observed gap, normalized severity, and a flag indicating that the violation existed before outcome access.

The contract checks:

- finite and positive open, high, low, and close;
- `high >= open`, `high >= close`, and `high >= low`;
- `low <= open` and `low <= close`;
- nonnegative predicted volume when present;
- expected timestamp presence and exact equality;
- no duplicate timestamps;
- exact horizon length.

Price gaps use the final causally observed cutoff close as the fixed severity denominator. Negative-volume severity uses the final causally observed volume, floored at one. Nonfinite violations have no numeric severity and are counted separately. No unavailable or nonfinite severity is replaced with zero.

The controlled reason codes are:

- `NONFINITE_OPEN`, `NONFINITE_HIGH`, `NONFINITE_LOW`, `NONFINITE_CLOSE`, `NONFINITE_VOLUME`
- `NONPOSITIVE_OPEN`, `NONPOSITIVE_HIGH`, `NONPOSITIVE_LOW`, `NONPOSITIVE_CLOSE`
- `HIGH_BELOW_OPEN`, `HIGH_BELOW_CLOSE`, `HIGH_BELOW_LOW`
- `LOW_ABOVE_OPEN`, `LOW_ABOVE_CLOSE`
- `NEGATIVE_VOLUME`
- `MISSING_TIMESTAMP`, `UNEXPECTED_TIMESTAMP`, `DUPLICATE_TIMESTAMP`, `HORIZON_LENGTH_MISMATCH`

The module calculates the declared structural diagnostics independently rather than collapsing them into a reliability score.

The fixed Phase 2.5 diagnostic names are `INVALID_PATH_FRACTION`, `INVALID_CANDLE_FRACTION`, `TOTAL_CONSTRAINT_VIOLATIONS`, `MAX_CONSTRAINT_VIOLATION_SEVERITY`, `MEAN_CONSTRAINT_VIOLATION_SEVERITY`, `EARLIEST_INVALID_HORIZON_STEP`, `HIGH_LOW_INVERSION_COUNT`, `HIGH_BELOW_BODY_COUNT`, `LOW_ABOVE_BODY_COUNT`, `NONFINITE_OUTPUT_COUNT`, and `NONPOSITIVE_PRICE_COUNT`.

## Column and timestamp defenses

Tests deliberately permute physical DataFrame column order and confirm that named selection is invariant. Tests swap column labels or values and confirm that invalid input is rejected or output violations are detected. Provider extraction must compare the official output index exactly with the declared five XNYS sessions before mapping values.

The first output maps to `2024-07-08`; the fifth maps to `2024-07-12`. Both Kronos and baseline returns use the same raw `2024-07-05` cutoff close, and outcome resolution remains raw close-to-close log return through `2024-07-12`.

## Individual and averaged forecasts

The sealed Phase 2 canonical forecast remains unchanged. Phase 2.5 compares:

1. nine sealed individual `sample_count=1` paths;
2. the offline average of the three context-512 paths;
3. an offline average of all nine paths for diagnosis only;
4. a direct official `sample_count=3` result;
5. a direct official `sample_count=5` result if its execution remains inside the established CPU runtime boundary.

Official `sample_count=N` repeats one context across an internal sampling batch, decodes the samples, and averages them in normalized feature space before denormalization. The audit does not claim equality with independently seeded offline averages unless the internal random sequence is reproduced and verified.

## Repeatability

The context-512 Phase 2 requests for seeds `1729`, `2027`, and `7919` are repeated with Python, NumPy, PyTorch CPU, and available accelerator random generators reset immediately before each inference. Output hashes and exact violation locations, codes, gaps, and severities are compared with the sealed paths. Different-seed patterns are compared without assuming they should match.

## Constraint projection experiment

`CONSTRAINT_PROJECTION_V0` is an explicitly separate derived artifact. It preserves raw candles and keeps open and close unchanged. For each candle:

- `repaired_high = max(high, open, close, low)`
- `repaired_low = min(low, open, close, repaired_high)`

The projection records changed fields, original and projected values, absolute adjustment, and normalized adjustment. Tests require structural validity after projection and exact invariance of every open, close, and implied close return. The report measures range and path-volatility changes but makes no accuracy claim.

## Bounded canary

Only after direct integration equivalence is established, the audit runs three context-512, `sample_count=1` seeds for both SPY and QQQ at these mechanically predeclared development cutoffs:

- `2024-09-27`
- `2025-01-31`
- `2025-05-30`

The dates are spaced across the already declared development period and are not selected using forecast performance. Each cutoff records the exact five following XNYS sessions, input hash, validity diagnostics, canonical three-path return, realized raw return, Kronos error, baseline error, runtime, and failures. The canary runs no shorter contexts, fits no model, estimates no correlation, and makes no population claim.

For each canary cutoff, all three forecasts and their validity records are completed before the logically separate outcome loader accesses any future session. A canary failure remains in the results and is not replaced with another cutoff.

## Evidence and outputs

Phase 2.5 produces:

- `docs/SENTINEL_STRUCTURAL_VALIDITY.md`
- `research/sentinel-v0/phase2_5/integration_trace.json`
- `research/sentinel-v0/phase2_5/official_comparison.json`
- `research/sentinel-v0/phase2_5/structural_validity.json`
- `research/sentinel-v0/phase2_5/averaging_comparison.json`
- `research/sentinel-v0/phase2_5/repair_experiment.json`
- `research/sentinel-v0/phase2_5/canary_results.json`
- `research/sentinel-v0/phase2_5/report.md`

Generated execution receipts and private normalized rows remain outside Git. Repository JSON outputs are compact, sorted, finite, provenance-rich derived artifacts.

## Governance decision

If official raw-path invalidity is confirmed, the audit adds a pre-holdout Sentinel v0 amendment that lists every structural diagnostic, its fixed definition, and its expected error relationship, then regenerates the experiment hash while preserving the prior configuration and amendment history. No structural diagnostic is added after holdout inspection.

If OpenAlpha introduced the violations, the integration fix is test-first, the sealed original remains untouched, and a new explicitly versioned development record reruns Phase 2. If the source is ambiguous, the full sample remains blocked. If the canary does not reproduce official invalidity, validity checks remain defensive but structural invalidity is not promoted as a central hypothesis.

## Verification and stop condition

The final commit requires targeted tests, the complete test suite, Ruff, Pyright, offline verification of the original Phase 2 artifact chain, before/after Phase 2 descriptor and inventory hash equality, `git diff --check`, and a scan proving no raw datasets, generated caches, or model weights are tracked.

Phase 2.5 stops after classification and reporting. It does not start the full development sample or fit a Sentinel risk model.
