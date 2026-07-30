# Sentinel Research Question and Evidentiary Standard

## Primary question

> Can information available at forecast time predict the future error of a financial forecasting model, and can selective use, blending, or abstention outperform blindly accepting every forecast?

Kronos is the first forecast provider, not the assumed winner and not the permanent product boundary.

## V0 hypotheses

- **H1 — predictable error:** higher pre-outcome Sentinel risk is associated with larger future Kronos five-session return error.
- **H2 — monotonic risk:** realized error generally increases across ordered risk buckets.
- **H3 — selective improvement:** rejecting the highest-risk forecasts reduces MAE among accepted forecasts.
- **H4 — directional usefulness:** selective acceptance does not sacrifice and may improve direction accuracy.
- **H5 — simple repair:** a 50/50 baseline blend improves medium-risk forecasts.
- **H6 — stability:** useful signal is not entirely confined to SPY, QQQ, or one short period.
- **H7 — operational value:** any error reduction is large enough to justify nine inference requests per cutoff.

Failure to meet the locked continuation criteria is preserved as evidence against the current approach.

## Claim boundary

Sentinel v0 is historical development and holdout feasibility. It does not establish live performance, trading profitability, causal model failure mechanisms, or generalization to other assets, horizons, providers, or forecast models.

The holdout is untouched by risk-model selection, but it is still a historical evaluation using modern provider data. No result may be described as live or precommitted.

## Fixed estimands

- Spearman association between predicted risk and future absolute Kronos error.
- Error across risk quintiles.
- MAE and directional accuracy at 100%, 90%, 80%, 70%, and 50% coverage.
- Error of USE, 50/50 BLEND, and ABSTAIN policy regions.
- Failure-probability calibration and confusion matrix.
- Per-asset, pooled, quarterly, diagnostic, latency, cost, and failure-rate results.

Benchmark results are always shown beside Sentinel interventions. Benchmarking supplies evidence; reliability assessment is the product.

## Interpretation

- Computed values come from verified artifacts.
- Failure reasons derive from diagnostics and frozen thresholds.
- An explanation is not proof of a causal failure mechanism.
- Negative, mixed, and inconclusive results are valid.
- No holdout observation may change labels, features, preprocessing, thresholds, actions, or continuation rules.
