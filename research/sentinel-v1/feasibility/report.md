# Sentinel v1 Phase 3B Bounded Feasibility Report

**DEVELOPMENT FEASIBILITY - NOT HOLDOUT EVIDENCE**

## Research question

Can hard financial-domain constraints be enforced inside the Kronos
autoregressive generation loop without retraining while preserving forecast
fidelity, diversity, and practical runtime?

## Scope

- Eligible origins: 12
- Completed origins: 12
- Failed origins: 0
- Assets: SPY and QQQ; daily; five XNYS sessions; context 512.
- Seeds: 1729, 2027, and 7919.
- The untouched holdout was not accessed.

## Method results

| Method | Success | Hard failures | Valid returned paths | OHLC MAE | Range MAE | Close MAE | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAW_AUTOREGRESSIVE | 36/36 | 0 | 0.36111111 | 0.02760710 | 0.01030152 | 0.02675216 | 416.70030000 |
| TERMINAL_PROJECTION | 36/36 | 0 | 1.00000000 | 0.02734030 | 0.00959934 | 0.02675216 | 417.03230000 |
| STEPWISE_PROJECT_REENCODE | 15/36 | 21 | 1.00000000 | 0.02333886 | 0.00844207 | 0.02483735 | 546.66135000 |
| VALID_CANDIDATE_RESAMPLING | 36/36 | 0 | 1.00000000 | 0.02547039 | 0.01045756 | 0.02442360 | 1539.20420000 |

The methods are paired by origin, context, forecast timestamps, and
initial seed. Candidate rejection is the declared point at which a
constrained trajectory may diverge from its raw trajectory.

## Full path and range metrics

| Method | Open MAE | High MAE | Low MAE | Close MAE | Range MAE | Body MAE | Upper-wick MAE | Lower-wick MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW_AUTOREGRESSIVE | 0.02615138 | 0.02894471 | 0.02858014 | 0.02675216 | 0.01030152 | 0.00974600 | 0.00583153 | 0.00452859 |
| TERMINAL_PROJECTION | 0.02615138 | 0.02775833 | 0.02869933 | 0.02675216 | 0.00959934 | 0.00974600 | 0.00445466 | 0.00437339 |
| STEPWISE_PROJECT_REENCODE | 0.02160112 | 0.02267426 | 0.02424269 | 0.02483735 | 0.00844207 | 0.01044457 | 0.00366310 | 0.00432812 |
| VALID_CANDIDATE_RESAMPLING | 0.02423239 | 0.02620714 | 0.02701845 | 0.02442360 | 0.01045756 | 0.00818142 | 0.00534510 | 0.00488186 |

| Method | Range-direction accuracy | Path-shape distance | Parkinson-volatility error | Cumulative-range error | Five-session return MAE | Direction accuracy |
|---|---:|---:|---:|---:|---:|---:|
| RAW_AUTOREGRESSIVE | 0.65000000 | 0.03120043 | 0.00608199 | 0.03410522 | 0.04430859 | 0.27777778 |
| TERMINAL_PROJECTION | 0.70000000 | 0.03096771 | 0.00607541 | 0.03215818 | 0.04430859 | 0.27777778 |
| STEPWISE_PROJECT_REENCODE | 0.61333333 | 0.02752200 | 0.00366550 | 0.02277440 | 0.04752906 | 0.26666667 |
| VALID_CANDIDATE_RESAMPLING | 0.65000000 | 0.02869408 | 0.00716637 | 0.03873913 | 0.03795021 | 0.30555556 |

The zero-return baseline five-session MAE was 0.01943683.

### Error by horizon step

| Method | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|---|---:|---:|---:|---:|---:|
| RAW_AUTOREGRESSIVE | 0.01618563 | 0.01972994 | 0.02655998 | 0.03334551 | 0.04221443 |
| TERMINAL_PROJECTION | 0.01583241 | 0.01952379 | 0.02629495 | 0.03303481 | 0.04201554 |
| STEPWISE_PROJECT_REENCODE | 0.01051012 | 0.01590341 | 0.02191384 | 0.02609017 | 0.04227674 |
| VALID_CANDIDATE_RESAMPLING | 0.01603872 | 0.01886607 | 0.02409679 | 0.03087003 | 0.03748035 |

## Barrier-event accuracy

| Method | Barrier | Positive touch | Negative touch | Either touch |
|---|---:|---:|---:|---:|
| RAW_AUTOREGRESSIVE | 0.5% | 0.55555556 | 0.63888889 | 1.00000000 |
| RAW_AUTOREGRESSIVE | 1.0% | 0.52777778 | 0.55555556 | 1.00000000 |
| RAW_AUTOREGRESSIVE | 2.0% | 0.52777778 | 0.27777778 | 0.50000000 |
| TERMINAL_PROJECTION | 0.5% | 0.55555556 | 0.63888889 | 1.00000000 |
| TERMINAL_PROJECTION | 1.0% | 0.52777778 | 0.55555556 | 1.00000000 |
| TERMINAL_PROJECTION | 2.0% | 0.52777778 | 0.27777778 | 0.50000000 |
| STEPWISE_PROJECT_REENCODE | 0.5% | 0.80000000 | 0.73333333 | 1.00000000 |
| STEPWISE_PROJECT_REENCODE | 1.0% | 0.80000000 | 0.66666667 | 1.00000000 |
| STEPWISE_PROJECT_REENCODE | 2.0% | 0.33333333 | 0.60000000 | 0.80000000 |
| VALID_CANDIDATE_RESAMPLING | 0.5% | 0.66666667 | 0.63888889 | 1.00000000 |
| VALID_CANDIDATE_RESAMPLING | 1.0% | 0.63888889 | 0.50000000 | 0.97222222 |
| VALID_CANDIDATE_RESAMPLING | 2.0% | 0.47222222 | 0.30555556 | 0.50000000 |

## Diversity and operational metrics

| Method | Pairwise diversity | Return variance | Repeated-path rate | Raw-path distance | Median total ms | Peak bytes |
|---|---:|---:|---:|---:|---:|---:|
| RAW_AUTOREGRESSIVE | 0.01800002 | 0.00014938 | 0.00000000 | 0.00000000 | 416.70030000 | 667443200 |
| TERMINAL_PROJECTION | 0.01775931 | 0.00014938 | 0.00000000 | 0.00097610 | 417.03230000 | 667443200 |
| STEPWISE_PROJECT_REENCODE | 0.01558582 | 0.00014361 | 0.00000000 | 0.00134518 | 546.66135000 | 433975296 |
| VALID_CANDIDATE_RESAMPLING | 0.01559381 | 0.00012937 | 0.00000000 | 0.00894290 | 1539.20420000 | 667443200 |

Terminal projection's median projection-only overhead was 0.31200000 ms; its total includes the paired raw rollout.
Peak memory is the Windows process-wide cumulative working-set peak
observed after each method. It is a real operational upper bound but
cannot isolate incremental memory by method within the shared worker.

## Intervention audit

| Method | Steps | Projection changes | Mean projection | Max projection | Re-encoded token changes | Candidate rejection | Mean validation ms | Search expansions | Fallback | Mean selected rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW_AUTOREGRESSIVE | {} | 0 | not computable | not computable | 0/0 | not computable | not computable | 0 | not computable | not computable |
| TERMINAL_PROJECTION | {"1": 16, "2": 15, "3": 20, "4": 12, "5": 17} | 80 | 0.00344716 | 0.01645425 | 0/0 | not computable | not computable | 0 | not computable | not computable |
| STEPWISE_PROJECT_REENCODE | {"1": 13, "2": 2, "3": 2, "4": 1, "5": 5} | 27 | 0.00392961 | 0.01234894 | 2/23 | not computable | not computable | 0 | not computable | not computable |
| VALID_CANDIDATE_RESAMPLING | {"1": 13, "2": 11, "3": 13, "4": 11, "5": 18} | 0 | not computable | not computable | 0/0 | 0.67012987 | 651.30718485 | 3 | 0.00000000 | 9.31818182 |

## Continuation gates

### STEPWISE_PROJECT_REENCODE

- close_mae: PASS
- determinism: PASS
- diversity: PASS
- full_ohlc_mae: PASS
- hard_failure_rate: FAIL
- high_low_range_mae: PASS
- no_retraining_or_weight_modification: PASS
- returned_path_structural_validity: PASS
- runtime_and_memory: PASS

### VALID_CANDIDATE_RESAMPLING

- close_mae: PASS
- determinism: PASS
- diversity: PASS
- full_ohlc_mae: PASS
- hard_failure_rate: PASS
- high_low_range_mae: FAIL
- no_retraining_or_weight_modification: PASS
- returned_path_structural_validity: PASS
- runtime_and_memory: PASS

## Generation behavior

- Raw structural validity rate: 0.36111111.
- Stepwise project/re-encode hard failures: 21/36.
- Candidate-resampling interventions: 66.
- Candidate rejection rate: 0.67012987.
- Candidate fallback rate: 0.00000000.

Terminal projection guarantees final OHLC ordering when it succeeds but
does not prevent invalid intermediate states from conditioning later tokens.
Projection is not presented as improved predictive accuracy.

## Model-size canary

Status: **not_run_prerequisite_not_met**.

Kronos-small and Kronos-base are not run unless an in-loop mini method
passes every locked continuation gate.

## Conclusion

**VALIDITY_SUCCEEDS_QUALITY_DEGRADES**

Stepwise project/re-encode returned only valid paths but hard-failed
21 of 36 paths because the tokenizer round trip could reintroduce an
invalid candle. Bounded valid-candidate resampling returned 36 of 36
valid paths with no fallback, but its high-low range MAE exceeded the
preregistered non-degradation threshold. No in-loop method passed every
continuation gate, so the model-size canary and larger study were not run.

This conclusion is bounded to the tested checkpoints, assets, origins,
frequency, seeds, and five-session horizon. It is not evidence that Kronos
is broadly broken, nor that constrained decoding improves accuracy.
