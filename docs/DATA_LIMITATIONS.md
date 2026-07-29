# Data Limitations

## Purpose

Market data is an empirical input with legal, temporal, and quality limitations—not neutral truth. OpenAlpha records these limitations per snapshot and carries them into reports and the user interface.

## Free-provider limitations

The Phase 1 public demonstration will use a no-paid-key equity adapter only after its terms and observable adjustment behavior are documented. Freely accessible data must not be described as institutional-grade, guaranteed complete, or point-in-time.

Common limitations include:

- undocumented vendor corrections and backfills;
- rate limits, outages, and schema changes;
- ambiguous split/dividend adjustment;
- rounded prices or volumes;
- incomplete delisting and symbol-history coverage;
- redistribution restrictions;
- missing official exchange sequence/session metadata;
- absence of bid/ask, auction, and trade-condition data.

The normalized snapshot preserves provider identity and retrieval time so later corrections cannot silently rewrite a completed run.

## Adjustment and corporate actions

Adjusted OHLC is provider-defined. Some vendors adjust close only; others back-adjust all price fields. Volume treatment may differ. An adjusted series is unsuitable for reconstructing historical executable prices unless the simulation policy accounts for how adjustments transformed the bars.

Every snapshot therefore records:

- raw versus adjusted field status;
- split and dividend handling;
- whether all OHLC fields share one adjustment factor;
- whether volume was inversely adjusted;
- provider evidence or an explicit `unknown` classification.

Unknown adjustment semantics produce a methodology warning.

## Survivorship and universe bias

A current ticker list evaluated historically excludes delisted, merged, renamed, or failed securities. One current ETF avoids cross-sectional constituent selection in Phase 1 but does not eliminate fund survival or launch-selection bias.

Multi-asset phases require point-in-time universe membership where available and must label approximations.

## Timestamp and calendar risk

- Provider dates may be timezone-naive.
- Daily-bar labels can denote session date, open time, or close time.
- Holidays, early closes, halts, and unscheduled closures may be absent or revised.
- Constructing future timestamps from realized returned rows can leak knowledge of future missing sessions.

OpenAlpha normalizes to a named exchange calendar and stores both provider timestamp semantics and normalized session identity.

## Missing and stale observations

Missing bars are not automatically zeros and are not silently forward-filled. The quality report distinguishes:

- expected non-trading sessions;
- provider gaps;
- exchange halts;
- unavailable fields;
- zero-volume/stale-price runs;
- newly listed or insufficient-history assets.

Policies are feature-specific. A causal carry-forward may be valid for some as-of macro fields but not for missing market bars.

## Outliers and bad ticks

Large moves may be genuine, corporate-action artifacts, currency/unit errors, or bad data. OpenAlpha emits warnings using causal and cross-field checks but does not delete observations merely because they harm a result. Any repair creates a transformed snapshot with a new hash and an auditable rule.

## Volume and amount

Kronos accepts optional `volume` and `amount`. Its official predictor fills both with zero when volume is absent and synthesizes amount as volume times mean OHLC when amount alone is absent.

OpenAlpha records whether amount is:

- provider supplied;
- causally derived and by what formula;
- unavailable and represented by a declared sentinel policy.

Synthetic amount must not be mislabeled as provider-reported turnover.

## Foundation-model contamination

The Kronos paper reports pretraining through June 2024 on roughly 12.11 billion bars from 45 global exchanges, but the released checkpoint does not include per-record provenance or dataset hashes. Evaluating after June 2024 reduces direct temporal overlap; it cannot establish that an asset or related market pattern was absent from pretraining.

Any result is therefore an evaluation of a fixed released model under known cutoff information, not a proof of uncontaminated zero-shot generalization.

## Revised macro and fundamental data

Current FRED/SEC values may differ from what was known historically. These sources remain outside admissible model features until adapters preserve release/vintage/as-of timestamps and the evaluator performs as-of joins. Publication dates are not interchangeable with period dates.

## Execution limits of OHLCV

Daily OHLCV cannot identify:

- intrabar event order;
- queue position;
- spread throughout the session;
- market impact;
- hidden liquidity;
- exact partial-fill path;
- whether a stop and target were hit in which order.

Phase 1 therefore uses a conservative, explicit next-bar convention and sensitivity analysis. It does not claim exchange-level execution fidelity.

## Legal and redistribution constraints

Provider access does not imply redistribution rights. Raw data and demo snapshots require a documented license/terms review before inclusion in a public repository or hosted artifact store. When redistribution is not permitted, reproducibility uses a fetch recipe, snapshot hash, and user-local storage; reports may publish derived aggregates only where allowed.

## Reviewer checklist

A valid data-quality artifact answers:

1. Who supplied the data and when?
2. What exact bytes and normalized table were used?
3. What calendar and timezone semantics apply?
4. What was adjusted, repaired, derived, dropped, or unavailable?
5. Which gaps, duplicates, stale runs, and outliers were detected?
6. What point-in-time and survivorship guarantees are absent?
7. May the snapshot be redistributed?

