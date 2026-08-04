# openalpha-sentinel

Structural validation and audit for financial forecast output, plus the earlier
Sentinel investigations (v0, v1, v1.1) that preceded the Kronos studies.

## What it does

Validates timestamps, shapes, values, financial structure and provenance;
preserves raw model output unchanged; and produces immutable audit records so an
invalid path cannot be silently consumed downstream.

Structural validity here means the OHLC inequalities — `high >= max(open, close)`,
`low <= min(open, close)`, `high >= low` — plus finiteness and positivity.

## Notable modules

| Module | Purpose |
| --- | --- |
| `structural_validity.py` | The candle validity rules |
| `constrained_decoding.py`, `constraint_compatibility.py` | Constrained-decoding investigation (v1) |
| `token_manifold.py`, `sentinel_v1_1.py` | Token-manifold audit (v1.1) |
| `risk_model.py`, `diagnostics.py`, `interventions.py` | Forecast-time risk and abstention study |
| `development_*.py` | The v0 development pipeline: origins, resolution, reporting, freezing |
| `providers/` | Market-data provider adapters |

## Status

These investigations are complete and their evidence is preserved under
`research/sentinel-v0/`, `research/sentinel-v1/` and `research/sentinel-v1_1/`.
They include negative results that were kept rather than discarded — forecast-time
diagnostics did not usefully rank later error, and abstention did not reduce
accepted-forecast error.

```bash
uv run pytest packages/sentinel/tests -q
```
