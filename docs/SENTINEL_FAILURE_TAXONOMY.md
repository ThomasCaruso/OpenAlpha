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

The baseline, volatility, analogue, and data-anomaly reasons can be emitted even when their logistic coefficient is zero only if their frozen threshold fires; such a reason is labeled `supporting`, not `model_contributing`.

## Post-outcome classification

After resolution, Sentinel appends observed error, failure label, deployability label, direction correctness, and path error. It does not rewrite pre-outcome reasons. A postmortem may mark a reason as supported, unsupported, or indeterminate by the outcome, but this is a new immutable record.

## Taxonomy discipline

- Reason definitions do not change after holdout outcomes are inspected.
- Failed inference and missing diagnostics remain visible; they are not converted to favorable defaults.
- Structural reasons are emitted from deterministic pre-outcome validation, never inferred from realized error or projected output.
- `UNKNOWN_FAILURE_MODE` is evidence of taxonomy incompleteness, not permission to invent a narrative.
- Exogenous-event and uncertainty-miscalibration mechanisms remain research candidates but are not claimed as measured by Sentinel v0.
