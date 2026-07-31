# Sentinel Phase 3A Development Sample Design

Date: 2026-07-30
Status: approved design awaiting implementation plan
Claim boundary: DEVELOPMENT ANALYSIS - NOT HOLDOUT EVIDENCE

## Objective

Phase 3A runs the complete chronological Sentinel v0 development sample and freezes the reliability system before any holdout origin is accessed.

It must answer five questions:

1. How often do official Kronos paths violate the locked structural contract?
2. Do forecast-time diagnostics predict later Kronos return error?
3. Does an interpretable Sentinel risk model work under chronological development validation?
4. Do abstention, blending, or valid-path aggregation reduce development error?
5. What exact system, if any, should be frozen for the untouched holdout?

Phase 3A does not produce holdout evidence, a publication claim, a frontend, a new forecasting model, or a generalized platform.

## Preserved state

The implementation preserves all Git history and does not modify:

- sealed Phase 2 forecast or outcome records;
- Phase 2.5 private receipts or committed compact evidence;
- model and tokenizer revisions;
- official Kronos source revision;
- the Sentinel v0.4 experiment YAML or its SHA-256;
- existing content-addressed artifacts, journals, manifests, and path confinement.

The governing Sentinel v0.4 SHA-256 remains:

fd50c208f465ce99750b11d4de5d5bc8c391bd73cb9ff450037c596c3e37d8bc

The YAML field target_usable_per_asset.minimum equals 100 and conflicts with the explicit one-year weekly design. The YAML remains byte-for-byte immutable. The Phase 3A manifest records that the explicit instruction to use every eligible weekly development origin is authoritative for this run.

## Structural and reliability assurance

Sentinel has two independent assurance layers.

Structural assurance validates whether generated paths satisfy finite-value, positive-price, OHLC-ordering, volume, timestamp, duplicate, and horizon constraints.

Reliability assurance estimates future return error from forecast-time features.

A structural failure never becomes a structurally valid raw path. It does not automatically label the raw close-return forecast inaccurate. Phase 3A measures that relationship.

Every completed origin retains distinct fields for:

- structural_status;
- structural_diagnostics;
- raw_close_return_forecast;
- projected_path;
- projected_path_status;
- reliability_features;
- realized_outcome;
- forecast_error.

Raw generated paths remain immutable and private. CONSTRAINT_PROJECTION_V0 is stored separately and never overwrites a raw forecast.

## Development population

The population is defined by cutoff origins, not outcome dates.

- development cutoff start: 2024-07-01;
- development cutoff end: 2025-06-30;
- assets: SPY and QQQ;
- origin rule: final XNYS session of each complete ISO week;
- asset order within a cutoff: SPY, then QQQ;
- exact cutoff count: 52;
- exact origin count: 104;
- first cutoff: 2024-07-05;
- final cutoff: 2025-06-27.

The exact ordered cutoff and origin lists are materialized, canonically serialized, content-addressed, and sealed before the pilot.

The final June cutoff has five legitimate outcome sessions that extend into July. Those rows may be retrieved only to resolve that eligible development origin. No cutoff on or after 2025-07-01 is accepted, created, retrieved, scored, inspected, or summarized as an origin.

A central development-boundary guard validates every CLI operation, provider request purpose, state descriptor, analysis row, and manifest record. There is no Phase 3A command that enumerates holdout origins.

## Ensemble and canonical forecast

Every origin uses the locked nine requests:

- contexts 128, 256, and 512;
- seeds 1729, 2027, and 7919;
- sample_count 1;
- temperature 1.0;
- top_p 0.9;
- five XNYS forecast sessions.

The locked canonical forecast is unchanged:

1. retain all three individual 512-context paths;
2. average their close values timestamp by timestamp;
3. calculate the five-session raw log return from the final averaged close and the final observed raw close at the cutoff.

The 128- and 256-context paths remain diagnostic stress paths only.

## Architecture

Phase 3A adds focused modules rather than a framework:

1. Development manifest
   - computes and validates the exact population;
   - enforces cutoff and outcome-purpose boundaries;
   - creates the immutable ordered manifest.

2. Inference cache
   - identifies requests by model, tokenizer, source revisions, normalized causal input hash, ordered sessions, context, seed, temperature, top-p, sample count, and horizon;
   - verifies path hash, shape, timestamps, and provenance before reuse;
   - imports prior Phase 2 or Phase 2.5 paths only when this complete inference identity matches;
   - stores all cache content outside Git.

3. Origin lifecycle
   - creates and seals forecast-time evidence;
   - resolves outcomes only after the forecast seal verifies;
   - appends outcome evidence without mutating the forecast;
   - creates one compact deterministic row after complete-chain verification;
   - persists explicit terminal failure records.

4. Development runner
   - supports preflight, pilot, run, analyze, freeze, report, and verify operations;
   - checkpoints after every origin;
   - skips only origins with a verified terminal Phase 3A descriptor;
   - resumes without repeating verified inference or resolution.

5. Analysis
   - reads verified compact rows and structural path summaries only;
   - never reopens provider histories;
   - produces prevalence, diagnostic, model, intervention, and coverage artifacts.

6. Freeze
   - serializes final feature, preprocessing, model, threshold, reason, policy, structural-gate, software, and provenance configuration;
   - content-addresses and verifies the freeze before any future holdout command can exist.

A pinned scikit-learn dependency supplies auditable preprocessing, logistic regression, ridge regression, and deterministic coefficient extraction. The numerical model implementation is not recreated locally.

## Origin lifecycle

Each origin executes these operations in order:

1. Validate that the origin is in the sealed development manifest.
2. Retrieve causal raw OHLCV through the locked yfinance request policy.
3. Reject all input rows after the cutoff and require the cutoff as the final session.
4. Compute the normalized-input hash.
5. Reuse or run each exact inference request.
6. Persist all nine individual raw paths privately.
7. Validate individual paths and compute structural diagnostics.
8. Create and validate the three-path canonical 512 average.
9. Produce CONSTRAINT_PROJECTION_V0 separately for every raw path and the canonical path.
10. Compute every causal nonstructural diagnostic.
11. Seal and independently verify forecast-time evidence.
12. Only after step 11, retrieve the five declared outcome sessions.
13. Calculate realized return, errors, direction, labels inputs, and individual-path errors.
14. Append the outcome and verify the complete origin chain.
15. Write one compact deterministic analysis row and terminal descriptor.
16. Update the run checkpoint and manifest status.

A completed origin is never reopened for inference unless verification fails. Verification failure creates a new attempt while preserving the failed attempt.

## Causal diagnostics

All twenty-five locked diagnostics are computed.

Historical analogue candidates are eligible only when their input window and five-session outcome resolve no later than the current cutoff and use causal per-window normalization. Overlap exclusions and minimum support remain locked.

RECENT_MODEL_ERROR uses the eight most recent scheduled forecasts for the same asset whose outcome end is no later than the current cutoff and whose outcome record was sealed before the current diagnostic vector. It is unavailable before eight such records exist.

Unavailable diagnostics use a structured null status and reason. Zero is used only when a fully computed diagnostic is mathematically zero, such as a verified absence of nonfinite outputs. Missing values are never replaced by zero.

Every vector records its ordered diagnostic version, configuration hash, causal-through timestamp, input hashes, and creation time.

## Structural outputs

Each origin records:

- validity of all nine individual paths;
- validity of the canonical 512 average;
- validity of each projected path;
- invalid-path and invalid-candle fractions;
- violation count and category counts;
- maximum and mean normalized severity;
- earliest invalid step;
- context, seed, asset, and horizon-step breakdowns;
- all projection modifications and normalized adjustment magnitudes.

For every projection, verification proves:

- every locked structural constraint passes;
- open and close are unchanged;
- implied close return is unchanged;
- all changed high and low values are recorded.

Individual path return error is retained so valid and invalid path errors can be compared without substituting the canonical error for a path-level measurement.

## Private and committed data

Private confined state contains normalized causal rows when needed, raw model paths, projection details, receipts, and content-addressed lifecycle artifacts.

Git may contain only compact derived outputs allowed by policy. It never contains:

- Yahoo response objects;
- reusable raw histories;
- model or tokenizer weights;
- Hugging Face caches;
- yfinance caches;
- CSV exports;
- temporary worker files;
- credentials.

Committed origin records contain hashes, safe provenance, structural summaries, diagnostic values and missingness, return forecasts, outcomes, errors, labels, and runtime metadata.

## Operational pilot

Pilot membership is the first ten ordered origins without a verified terminal Phase 3A record at pilot start. Prior raw inference may still be reused when its complete cache key verifies.

The pilot runs the full lifecycle but inspects only:

- request success rate;
- inference and ensemble latency;
- model-cache growth;
- private artifact size;
- deterministic replay behavior;
- process stability;
- provider failures;
- validation failures;
- projected full-run duration and storage.

It does not inspect diagnostic-to-outcome relationships to change methodology.

Automatic continuation requires the locked operational criteria:

- at least 95 percent real-request success;
- median nine-request ensemble latency no more than 600 seconds;
- model and tokenizer cache no more than 2 GiB;
- hosted-equivalent cost no more than USD 0.50 per cutoff.

A local run has zero hosted inference charge. Projected private artifact storage is reported but creates no new stop threshold. An unhandled repeated crash, unverifiable cache entry, forecast-before-outcome violation, or inability to resume is operational instability and stops the run.

## Development table

The completed deterministic JSONL table has one row per manifest origin, including terminal failures.

Completed rows contain:

- origin and asset identity;
- cutoff and horizon end;
- data and configuration hashes;
- exact model revisions;
- canonical and baseline returns;
- all diagnostics and missingness indicators;
- structural and projection status;
- realized return;
- canonical, baseline, and individual-path errors;
- direction correctness;
- deployability label;
- failure label after threshold materialization;
- runtime, retry, cache, and verification metadata.

Failure rows contain the same identity fields, terminal failure code, stage, attempt hashes, and unavailable-analysis status.

Serialization uses UTF-8, one canonical JSON object per line, sorted keys, compact separators, finite numbers only, and a final newline. Ordered rows follow manifest order. The manifest stores the byte SHA-256.

## Failure threshold

After every eligible origin has a terminal status, the primary failure threshold is the pooled 75th percentile of completed canonical absolute return errors, calculated with NumPy quantile method linear. The method and NumPy version are stored.

The fixed development label is error greater than or equal to this threshold. The same numeric threshold is frozen for future holdout use.

This threshold uses the full development outcome distribution as required by v0.4. That fact is disclosed when interpreting development cross-validation. No holdout observation contributes to it.

## Chronological validation

Validation groups rows by unique cutoff so both assets remain together.

The locked three-split expanding TimeSeriesSplit over unique cutoff dates uses a one-week gap. Only validation rows receive out-of-fold predictions; early training-only rows are excluded from risk-coverage calculations and their count is reported.

For every feature family and hyperparameter:

- missingness filtering is fitted on training rows only;
- redundancy filtering is fitted on training rows only;
- medians and missing indicators are fitted on training rows only;
- means and standard deviations are fitted on training rows only;
- the model is fitted on training rows only;
- predictions are generated only for the later validation fold.

Feature removal follows v0.4:

- invalid definition;
- more than 20 percent training missingness;
- zero train variance;
- absolute training Spearman redundancy above 0.95, retaining the more directly interpretable diagnostic.

Logistic C values are 0.01, 0.1, 1.0, and 10.0. Ridge alpha values are 0.1, 1.0, 10.0, and 100.0.

Mean fold log loss selects logistic hyperparameters. Mean fold MAE selects ridge hyperparameters. These are development-selected cross-validated predictions, not untouched evidence.

If a training fold has only one failure-label class, that fold and candidate are recorded as invalid rather than fitted. A feature-family candidate is eligible only when all three folds produce finite predictions. If no family is eligible, reliability-model freezing stops with recommendation C or D while structural reporting continues.

## Feature-family selection

Three families use identical rows and folds:

A. Structural only: the eleven locked structural diagnostics.

B. Nonstructural only: sampling, context, baseline, regime, analogue, recent-error, and horizon diagnostics.

C. Combined: every locked diagnostic.

The primary family is chosen by the one-standard-error rule:

1. find the family and C with lowest mean fold log loss;
2. calculate its standard error across the three fold losses;
3. retain every family whose mean loss is no more than one standard error above the minimum;
4. select the eligible family with the fewest declared candidate diagnostics before fold-specific preprocessing;
5. break equal feature-count ties in order structural only, nonstructural only, combined.

The final ridge model uses the logistic-selected feature family and chooses alpha by its own mean fold MAE. All three family results remain reported.

## Confidence intervals and stability

Descriptive confidence intervals use a deterministic moving-block bootstrap:

- resampling unit: unique cutoff week with both assets together;
- block length: four consecutive cutoffs;
- resamples: 1,000;
- random seed: 314159;
- interval: percentile 95 percent.

The analysis reports pooled, per-asset, and chronological-quarter summaries. A confidence interval is omitted with an explicit reason when support or variation is inadequate.

## Structural analysis

Prevalence outputs include:

- invalid individual paths;
- invalid candles;
- invalid canonical forecasts;
- category and severity distributions;
- asset, quarter, context, seed, and horizon-step breakdowns.

Error analysis includes:

- individual valid versus invalid path return error;
- origin error by invalid-path and invalid-candle fractions;
- error by severity and earliest invalid step;
- canonical-valid versus canonical-invalid error;
- violation-category error.

Incremental structural value is evaluated through structural-only, nonstructural-only, and combined out-of-fold models and a development regression comparison that controls for recent volatility, canonical predicted-return magnitude, baseline disagreement, and context disagreement.

The result classifies structural invalidity as a return-error warning, a downstream schema and safety issue only, or both. Prevalence alone never establishes causation.

## Diagnostic analysis

Every diagnostic is reported against:

- future absolute canonical return error;
- failure label;
- deployability label;
- direction correctness.

Outputs include Spearman correlation for continuous relationships, transparent grouped summaries, missingness, expected-sign agreement, per-asset sign consistency, quarterly stability, and bootstrap intervals where supported.

The report keeps separate sections for:

- development descriptive statistics;
- out-of-fold predictive performance;
- final full-development coefficients.

Unfavorable, null, unstable, and removed diagnostics remain visible.

## Intervention policies

P0 raw acceptance uses the locked canonical Kronos forecast at 100 percent coverage.

P1 ranks completed out-of-fold rows by predicted failure risk and reports accepted canonical performance at 100, 90, 80, 70, and 50 percent coverage.

P2 applies USE to low risk, a fixed 50/50 Kronos and zero-return-baseline blend to intermediate risk, and ABSTAIN to highest risk. Development evaluation uses the 50th and 80th percentiles of pooled out-of-fold risk. The final frozen numeric thresholds use risk scores from the final full-development model.

P3 is exploratory. When at least two individual 512-context paths are structurally valid, it averages only those valid close paths. Otherwise it retains the locked canonical forecast and records the flag. It never changes the Phase 3A canonical field.

P3 is retained for holdout only when all conditions hold:

- at least ten completed origins use valid-path aggregation;
- pooled paired mean error improvement over the canonical forecast is positive;
- the lower bound of the 95 percent moving-block-bootstrap interval for paired improvement is above zero;
- mean error does not worsen for either SPY or QQQ.

Retention requires an explicit hashed amendment before holdout. Otherwise P3 remains a rejected development candidate.

## Risk coverage

Risk coverage uses only out-of-fold predictions from the selected family.

For pooled and each asset, it reports:

- risk versus future-error Spearman;
- error by risk quintile;
- canonical MAE at 100, 90, 80, 70, and 50 percent coverage;
- directional accuracy at every coverage;
- sample counts;
- baseline and 50/50 blend comparisons;
- quarterly stability.

Ties are resolved deterministically by risk, cutoff, then asset. Every declared coverage is reported.

## Final model and action freeze

After analysis, the selected feature family is refitted on all completed development rows.

The freeze stores:

- completed and failed origin identities;
- fixed failure threshold;
- retained and removed diagnostics with reasons;
- feature order;
- missingness indicators;
- imputation medians;
- standardization means and scales;
- logistic C, coefficients, and intercept;
- ridge alpha, coefficients, and intercept;
- reliability conversion round(100 times one minus failure probability), clamped to 0 through 100;
- final development risk distribution;
- USE threshold at the 50th risk percentile;
- BLEND upper threshold at the 80th risk percentile;
- blend weight 0.50;
- ABSTAIN rule above the 80th percentile;
- structural-gate behavior;
- P3 decision and amendment status;
- reason thresholds;
- software versions;
- experiment, code, table, analysis, and model artifact hashes.

Failure-reason thresholds use pooled development 20th percentiles for low-agreement triggers and 80th percentiles for high-risk triggers. STRUCTURALLY_INVALID_MODEL_OUTPUT is emitted deterministically from the validity record and does not require an error-model contribution.

The freeze is canonically serialized, content-addressed, verified, and created before any holdout-origin operation.

## Runtime structural contract

For any structurally invalid raw path:

- structural_status is FAILED;
- the immutable raw path remains auditable;
- it is never labeled valid;
- projected output may be exposed separately;
- a path-dependent downstream consumer must use projected output or abstain;
- the raw close-return forecast remains separately evaluable when finite and aligned;
- reliability action follows the frozen Sentinel risk policy.

Nonfinite prices, nonpositive prices, duplicate or missing timestamps, wrong horizon, or unaligned output block deployable raw return use. Finite OHLC-ordering failure alone does not force ABSTAIN unless the development-frozen reliability policy independently does so.

## Recommendation mapping

Operational feasibility uses the locked S6 criteria.

Development reliability support mirrors the locked holdout criteria on selected out-of-fold predictions:

- R1: pooled risk-error Spearman at least 0.20;
- R2: canonical MAE at 70 percent coverage no more than 90 percent of full-coverage MAE;
- R3: highest-risk-quintile error exceeds lowest and at least three of four adjacent changes are nondecreasing;
- R4: each asset has positive Spearman and 70-percent MAE no greater than its own full-coverage MAE;
- R5: at least one diagnostic with a preregistered sign has the expected sign and absolute Spearman at least 0.15.

Structural assurance is material when invalid-path prevalence or invalid-canonical prevalence is at least 0.05.

Apply the Phase 3A recommendation in order:

A. PROCEED TO LOCKED HOLDOUT when S6 and R1 through R5 pass and the complete freeze verifies.

B. PROCEED AS STRUCTURAL CONTRACT ONLY when S6 passes, structural assurance is material, pooled Spearman is nonpositive, and 70-percent MAE is not lower than full-coverage MAE.

C. CHANGE THE RELIABILITY APPROACH when S6 passes, A does not apply, and either pooled Spearman is positive or 70-percent MAE is lower than full-coverage MAE.

D. STOP when S6 fails, or when none of A through C applies and structural assurance is not material.

The report shows every rule and input regardless of recommendation.

## Repository outputs

Phase 3A creates:

research/sentinel-v0/development/
    origins/
    development_table.jsonl
    development_manifest.json
    structural_prevalence.json
    diagnostic_analysis.json
    out_of_fold_predictions.jsonl
    model_comparison.json
    intervention_analysis.json
    risk_coverage.json
    freeze_candidate.json
    report.md

Origin files are compact terminal summaries, not raw forecasts or market histories.

Phase 3A updates:

- docs/SENTINEL_STRUCTURAL_VALIDITY.md;
- docs/SENTINEL_METHODOLOGY.md;
- docs/SENTINEL_FAILURE_TAXONOMY.md;
- docs/STATUS.md.

## CLI

The narrow CLI operations are:

- preflight: materialize and seal the development population;
- pilot: process the first ten Phase 3A-nonterminal origins and apply operational gates;
- run: resume remaining origins chronologically;
- analyze: build the deterministic table and development analyses;
- freeze: fit final models and seal holdout configuration;
- report: render verified development findings;
- verify: verify manifest, origin chains, table, analyses, and freeze without network access.

No Phase 3A command accepts a holdout cutoff or constructs a holdout list.

## Testing

Automated tests use synthetic OHLCV or explicit fixtures and deterministic fake inference only.

Required tests cover:

- exact 52-cutoff and 104-origin manifest;
- boundary rejection for every holdout cutoff;
- permitted final-development outcome sessions;
- input causality;
- exact inference-cache identity and corruption rejection;
- resume without repeated completed inference;
- forecast seal before outcome loader execution;
- terminal failure preservation;
- all structural fields and projection invariants;
- causal recent-error and analogue availability;
- explicit diagnostic missingness;
- deterministic JSONL and manifest hashing;
- chronological fold grouping and gap;
- training-only preprocessing;
- logistic and ridge hyperparameter selection;
- one-standard-error feature-family selection;
- structural ablation;
- coverage calculations and counts;
- P0 through P3 behavior;
- P3 retention rule;
- freeze serialization and verification;
- report provenance;
- policy scan.

## Completion gates

Phase 3A can be committed only when:

- all 104 manifest origins have one terminal status;
- every completed chain verifies;
- every failure is accounted for;
- the development table reproduces byte-for-byte;
- all out-of-fold predictions and analyses verify;
- the holdout freeze verifies;
- no holdout origin was accessed;
- targeted and complete tests pass;
- Ruff and Pyright pass;
- artifact and ledger verification pass;
- experiment-hash and policy scans pass;
- git diff --check passes;
- the worktree contains no restricted artifact.

The report is labeled DEVELOPMENT ANALYSIS - NOT HOLDOUT EVIDENCE and ends before any holdout execution.

## Exact next task after Phase 3A

If recommendation A is produced, the next task is a separately authorized, one-time holdout execution using only the verified freeze hash.

If recommendation B, C, or D is produced, the next task follows that recommendation. No holdout origin is accessed automatically.
