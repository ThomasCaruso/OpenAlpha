# Sentinel Failure Taxonomy

## Rule

A failure reason is a deterministic interpretation of recorded diagnostics available before the outcome. No LLM generates reason codes or explanations. Each emitted reason contains:

- code;
- diagnostic name and value;
- development-frozen threshold or comparison;
- fixed human-readable explanation;
- `available_before_outcome: true`;
- severity in `[0, 1]`.

Thresholds are fitted on development diagnostics only and frozen before holdout scoring. Unless stated otherwise, a high-risk threshold is the pooled development 80th percentile and a low-agreement threshold is the 20th percentile. Severity is the diagnostic's clipped percentile distance beyond its threshold.

## Phase 3A frozen taxonomy state

The one-standard-error simplicity rule selected the structural-only feature
family. Consequently, the frozen v0 model-contributing reasons are limited to
the structural diagnostics below. The other candidate reason codes remain in
the controlled taxonomy for future development research, but this freeze does
not emit them as model-contributing explanations.

| Diagnostic | Frozen trigger |
|---|---:|
| `INVALID_PATH_FRACTION` | greater than or equal to 0.6666666667 |
| `INVALID_CANDLE_FRACTION` | greater than or equal to 0.3111111111 |
| `MAX_CONSTRAINT_VIOLATION_SEVERITY` | greater than or equal to 0.01256164184 |
| `MEAN_CONSTRAINT_VIOLATION_SEVERITY` | greater than or equal to 0.003678265020 |
| `EARLIEST_INVALID_HORIZON_STEP` | less than or equal to 1 |
| `HIGH_LOW_INVERSION_COUNT` | greater than or equal to 2 |
| `LOW_ABOVE_BODY_COUNT` | greater than or equal to 3 |
| `STRUCTURALLY_INVALID_MODEL_OUTPUT` | canonical structural status is `FAILED` |

The final development logistic failure threshold is
`0.046428259296972106`; the USE/BLEND and BLEND/ABSTAIN risk thresholds are
`0.24972087273272664` and `0.2684525799345617`, derived from the final
full-development refit's risk distribution. The exact feature ordering,
coefficients, preprocessing statistics, and reason records are bound by freeze
SHA-256
`c073d0e8d6cc4760211fc07f20c896444cd31d72a02d31a048e8c8d1ae6038a9`.
No holdout reason has been emitted.

## V0 reason codes

| Code | V0 trigger | Fixed explanation |
|---|---|---|
| `HIGH_SAMPLING_DISAGREEMENT` | directional agreement below its 20th percentile or return/path dispersion above its 80th percentile | Kronos sampling runs disagree materially about the five-session outcome. |
| `CONTEXT_WINDOW_INSTABILITY` | context direction agreement below its 20th percentile or 128-versus-512 return spread above its 80th percentile | The conclusion changes when the amount of historical context changes. |
| `VOLATILITY_REGIME_SHIFT` | absolute short/long volatility log-ratio above its 80th percentile | Recent volatility differs materially from the longer causal window. |
| `WEAK_HISTORICAL_ANALOGUE_SUPPORT` | nearest-analogue distance above its 80th percentile | Similar eligible historical contexts are unusually distant. |
| `HIGH_ANALOGUE_OUTCOME_DISPERSION` | analogue outcome dispersion above its 80th percentile | Similar historical contexts led to inconsistent five-session outcomes. |
| `STRONG_BASELINE_DISAGREEMENT` | absolute Kronos-minus-baseline return difference above its 80th percentile | Kronos disagrees sharply with the causal last-value baseline. |
| `RECENT_MODEL_DEGRADATION` | trailing resolved Kronos error above its 80th percentile | Recently resolved Kronos forecasts have been unusually inaccurate. |
| `INPUT_DATA_ANOMALY` | gap/outlier score above its 80th percentile or the data-quality record is warned | Recent input bars contain an unusual gap, return, or quality warning. |
| `STRUCTURALLY_INVALID_MODEL_OUTPUT` | any locked structural constraint fails; severity and exact codes come from the pre-outcome validity record | The raw model forecast violates one or more declared numerical, candle-ordering, or alignment constraints. |
| `UNCERTAINTY_MISCALIBRATION` | inactive in v0 | Reserved until a causal rolling calibration diagnostic is validated; it must not be emitted in v0. |
| `UNKNOWN_FAILURE_MODE` | predicted failure probability exceeds the ABSTAIN threshold but no active reason trigger fires | Sentinel predicts elevated failure risk without a mapped diagnostic explanation. |

## Contribution ordering

For the logistic risk model, each standardized diagnostic contribution is `coefficient × standardized_value`. Only positive risk contributions are eligible as top reasons. Emitted active reasons are ordered by positive contribution, then severity, then code. At most three reasons appear in a v0 decision.

A reason outside the selected structural family may be emitted only by a later
explicitly versioned development freeze that defines its threshold and role.
No inactive candidate is silently promoted to a supporting reason.

## Post-outcome classification

After resolution, Sentinel appends observed error, failure label, deployability label, direction correctness, and path error. It does not rewrite pre-outcome reasons. A postmortem may mark a reason as supported, unsupported, or indeterminate by the outcome, but this is a new immutable record.

## Taxonomy discipline

- Reason definitions do not change after holdout outcomes are inspected.
- Failed inference and missing diagnostics remain visible; they are not converted to favorable defaults.
- Structural reasons are emitted from deterministic pre-outcome validation, never inferred from realized error or projected output.
- `UNKNOWN_FAILURE_MODE` is evidence of taxonomy incompleteness, not permission to invent a narrative.
- Exogenous-event and uncertainty-miscalibration mechanisms remain research candidates but are not claimed as measured by Sentinel v0.
