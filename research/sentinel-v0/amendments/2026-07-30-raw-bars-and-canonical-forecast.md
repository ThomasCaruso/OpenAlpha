# Sentinel v0 Pre-Execution Amendment: Raw Bars and Canonical 512-Context Forecast

- Amendment date: 2026-07-30
- Prior experiment SHA-256: `c6459409e6b750217b4e81d09d461dc4b8cf303746d3efdcc8de80762e5a763c`
- Amended experiment SHA-256: `261e4bac51b9b3d6b68b6cf0128d406ca63cbf4bcd9735ccbda1aa04d1fa761a`
- Execution state at amendment: no Alpaca request, model download, or Kronos inference performed

## Exact changed fields

- `schema_version`: `sentinel-v0.1` → `sentinel-v0.2`
- `status`: `phase_1_locked` → `phase_2_amended_before_execution`
- `market_data.request.adjustment`: `all` → `raw`
- `market_data.representation`: added `raw_ohlcv`
- `market_data.return_target`: added `raw_five_session_close_to_close_log_return`
- `market_data.corporate_action_policy`: added the no-engine/dividend-limitation rule
- `ensemble.primary_path_aggregation`: `pointwise_median` → arithmetic mean of the three 512-context close paths
- `ensemble.primary_return_aggregation`: `median` → log of final averaged 512 close divided by raw cutoff close
- `ensemble.canonical_forecast`: added the 512-context, three-seed canonical definition
- `ensemble.stress_test_paths`: added the non-canonical role of 128/256 contexts
- `ensemble.reproducibility_probe`: added A/B/C seeded replay requests and mismatch policy
- `baseline.path_rule`: adjusted cutoff close → raw cutoff close
- `targets.continuous_primary`: changed to absolute raw five-session close-to-close log-return error
- `targets.secondary`: adjusted-close path error → raw-close path error
- `action_policy.use.output`: median forecast → mean 512-context forecast
- `phase_2_origin`: added exact SPY cutoff, XNYS sessions, OHLCV fields, no-amount rule, no-action rule, and report label
- `analogue_policy`: added causal normalization, non-overlap treatment, and minimum support
- `phase_2_diagnostic_availability`: added structured first-origin unavailability for recent error and uncertainty calibration

## Methodological reason

The development proof must not assume that a currently requested dividend-, split-, and spin-off-adjusted series exactly represents the data available at the historical forecast cutoff. Phase 2 therefore uses raw OHLCV and predicts raw five-session close-to-close log return. This avoids one form of historical revision assumption while deliberately accepting that dividend return is omitted and corporate actions may create discontinuities. No corporate-action engine is introduced.

The nine paths have separate evidentiary roles. Only the three 512-context paths define the canonical Kronos point forecast; their close paths are averaged timestamp by timestamp. The six shorter-context paths measure context sensitivity and instability and cannot influence the canonical forecast.

This amendment precedes all real data access and inference. It does not create an empirical claim or alter the development/holdout dates, assets, horizon, failure targets, risk model, action thresholds, or continuation criteria.
