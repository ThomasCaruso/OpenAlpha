DEVELOPMENT ANALYSIS - NOT HOLDOUT EVIDENCE

Recommendation: CHANGE_THE_RELIABILITY_APPROACH
Freeze SHA-256: c073d0e8d6cc4760211fc07f20c896444cd31d72a02d31a048e8c8d1ae6038a9
Eligible origins: 104
Completed origins: 104
Failed origins: 0
Generated paths: 936

## Q1. How often did official Kronos paths violate OHLC constraints?
Invalid-path prevalence: 0.5053418803418803. Invalid-candle prevalence: 0.20534188034188033. Invalid canonical-forecast prevalence: 0.5480769230769231.

## Q2. Which violations were most common?
Violation counts: {'HIGH_BELOW_CLOSE': 568, 'HIGH_BELOW_LOW': 159, 'HIGH_BELOW_OPEN': 545, 'LOW_ABOVE_CLOSE': 41, 'LOW_ABOVE_OPEN': 146}.

## Q3. Did violations become more common later in the forecast horizon?
Violation counts by step 1-5: [376, 262, 274, 328, 219].
Monotonic increase across horizon steps: no.

## Q4. Did structural violations predict larger return errors?
Canonical-invalid mean absolute error: 0.033611654686856104; canonical-valid mean absolute error: 0.03677708475964355. Invalid was worse: no.
Individual-invalid mean absolute error: 0.031170218064670175; individual-valid mean absolute error: 0.029007764601431018.
Controlled structural coefficients are retained in diagnostic_analysis.json.

## Q5. Did instability diagnostics predict error?
Pooled chronological OOF risk/error Spearman: 0.08147548653877766. Diagnostic analysis includes 25 declared diagnostics.

## Q6. Which feature family performed best out of sample within development?
Lowest mean logistic log loss family: nonstructural (0.5287820394613587).
Selected chronological OOF family: structural.

## Q7. Did abstention reduce accepted-forecast error?
100%: n=78, MAE=0.03495780695289239; 90%: n=71, MAE=0.03606158086779489; 80%: n=63, MAE=0.036876007730205586; 70%: n=55, MAE=0.03567348817334279; 50%: n=39, MAE=0.0345707248748697

## Q8. Did blending help?
P2 accepted MAE: 0.03299876798908981; accepted count: 62; fixed blend weight: 0.5.

## Q9. Did valid-path aggregation help?
P3 retained for holdout: False; application count: 22; pooled mean improvement: -0.0013143109004098987.

## Q10. Were effects consistent across SPY and QQQ?
Per-asset risk/error Spearman: {'QQQ': -0.026720647773279354, 'SPY': 0.11578947368421054}.

## Q11. Which diagnostics failed?
NONFINITE_OUTPUT_COUNT: CONSTANT_DIAGNOSTIC; NONPOSITIVE_PRICE_COUNT: CONSTANT_DIAGNOSTIC

## Q12. What exactly has been frozen for holdout?
Verified freeze c073d0e8d6cc4760211fc07f20c896444cd31d72a02d31a048e8c8d1ae6038a9; selected family structural.

## Q13. Is executing the holdout justified?
CHANGE_THE_RELIABILITY_APPROACH. No holdout origin was accessed by this report.

## Artifact provenance
- asset_risk_error_spearman: f8ae21778a21b744bf786b673a630735aace59cb3033987c470537237248ae12
- diagnostic_analysis: bfe13701e2167af5a4c71befbe949709d7b83166befc4164b03251f39ed801e0
- failed_origin_count: 5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9
- freeze_verified: b5bea41b6c623f7c09f1bf24dcae58ebab3c0cdd90ad966bc43a45b44867e12b
- intervention_analysis: 40d0ac9aa7fcff082898178b050a75e79deb973f63473c3c12f75cd879f50889
- model_comparison: e91affa7a49a9be3f9c73a542e6f9749842eed60c1a9428f0217bfb4d0b52c87
- operational_feasibility: 8267976b17a1aec14ae640f8901f9698d4b9ab4c2af6581a5a2df3d64d7d78d1
- origin_summary: 3d6ec36050e4415fe7b4b35bd642f4a3d63ac6a1c31bf003c5ca92c74ecbba9d
- risk_coverage: 5523bb6130f8acf933f3e0622efdf18f09128a4d80a32cf8a83cef66d5048b7e
- structural_error: 31aede587fe9b0dd26b4c5b4256479dec8ff22a517ff2d5c5d4e9a4021af01e4
- structural_prevalence: facb8b3a93953399fb616c073e997a2703703163d4ba992f759af67d0293396c
