# OpenAlpha: Kronos Reality Check Design

## Outcome

Build an evidence-first research product that can preregister an evaluation, generate and permanently record Kronos and baseline forecasts, append realized outcomes later, measure statistical and economic value, and state exactly where Kronos did and did not help.

## Selected approach

Use a CLI-first typed Python evidence kernel. Preserve the existing experiment identity, content-addressed artifact store, immutable run journals, and completed manifests. Add narrow ports for authenticated market data and model inference, a versioned research protocol, an append-only forecast/outcome ledger, rolling-origin evaluation, and artifact-grounded reporting.

Rejected alternatives:

- continuing the generalized platform, because services and UI do not answer the research question;
- building the full five-asset benchmark before one real forecast resolves, because it multiplies unverified assumptions;
- starting with a live dashboard, because presentation would precede reproducible evidence.

## Evidence model

`EvidenceClass` is one of `historical_replay`, `sealed_historical_test`, or `live_precommitted_forecast`. It is present on protocols, forecasts, outcomes, metrics, aggregate comparisons, and report statements. A live forecast is immutable before its outcome window; an outcome is a new record referencing it. Historical and live aggregates remain separate unless a table explicitly displays them side by side.

## Protocol boundary

`research/protocol/v1.yaml` is validated into a frozen typed model and serialized canonically. `v1.sha256` identifies the exact bytes. It locks hypotheses, universe, provider requests, representations, periods, horizons, models and parameters, targets, metrics, tests, costed strategy, failure/success criteria, exclusions, missing/corporate-action policies, and seeds. A changed field produces a new protocol version and hash.

The initial target universe is SPY, QQQ, IWM, TLT, and GLD. Development is 2017-01-01 through 2022-12-31; validation is 2023-01-01 through 2023-12-31. Because Kronos reports pretraining through June 2024, 2024-01-01 through 2024-06-30 is replay-only quarantine, not clean sealed evidence. The candidate sealed window is 2024-07-01 through 2025-12-31 and is frozen only after provider availability and protocol validation succeed. Live evaluation begins with the first successfully persisted production forecast.

## Data boundary

`MarketDataProvider` accepts a provider-independent request containing symbols, feed, timeframe, start, end, adjustment, as-of behavior, and pagination bounds. `AlpacaHistoricalBarsProvider` uses the fixed official host and environment credentials. It returns raw bytes plus response metadata; normalization and quality logic remain provider-neutral.

The protocol stores paired views: fully adjusted daily SIP bars for forecast targets and raw daily SIP bars for hypothetical fills. Both are content-addressed. Missing credentials, entitlement mismatch, rate limits, malformed pages, page cycles, or provider errors fail explicitly. No fallback exists.

## Forecast and ledger flow

```text
locked protocol
  -> authenticated paired data snapshots
  -> causal forecast origin
  -> real Kronos + declared baseline forecasts
  -> immutable forecast records
  -> later immutable outcome records
  -> return/direction/price metrics
  -> fixed next-bar decision and costs
  -> verified ledger chain and completed manifest
  -> CLI audit and reproducibility output
```

Each forecast records protocol hash, evidence class, asset/session/cutoff, snapshot and context hashes, provider/feed, model/checkpoint/configuration, horizon, predictions, expected evaluation time, runtime/hardware/seed, and optional uncertainty. Hash chaining links records. Outcome resolution rejects mutation, early scoring, duplicate outcomes, or mismatched snapshots.

## Evaluation design

Rolling origins use only data at or before each cutoff. Forecasts are persisted before outcomes are read by the scorer. Fixed horizons are 1, 5, and 20 XNYS sessions. Primary targets are future log return, direction, and defensible absolute-return magnitude; price and OHLC errors are secondary.

The full benchmark compares Kronos with last value, random walk/drift, moving average, exponential smoothing or ARIMA, ridge, and gradient-boosted trees through one adapter contract. The first proof slice uses real Kronos and last value only, without weakening the locked full-protocol model list.

Primary inference is limited to block-bootstrap confidence intervals, defensible Diebold-Mariano comparisons, directional-accuracy intervals, Holm correction across declared families, and stability by asset/time. The fixed long/cash strategy executes no earlier than the next bar and reports all declared costs beside buy-and-hold, cash, and baseline signals.

## Product surfaces

The CLI commands are `protocol validate`, `snapshot fetch`, `forecast create`, `outcome resolve`, `ledger verify`, `run proof-slice`, and `reproduce`. Only after the proof slice succeeds may a read-only application add Study Overview, Live Forecasts, Model Scoreboard, Forecast Audit, Historical Results, Methodology, Research Report, and Reproducibility.

## Error and validity rules

Provider, credential, data-quality, causal, model, resource, ledger-integrity, outcome-timing, accounting, and manifest failures are typed and preserved. A failed Kronos forecast is visible and never replaced. Synthetic fixtures are labeled and inadmissible as research evidence. Report generation reads verified artifacts only.

## Testing

Use TDD for protocol hashing, provider request formation and secret redaction, deterministic synthetic normalization, causal origins, adapter contracts, forecast immutability, append-only outcomes, chain verification, metrics, costs, and the CLI. Network and real-Kronos tests are separately marked and require explicit credentials/resources. The proof slice is accepted only with real Alpaca data, real Kronos inference, a persisted baseline, a later or explicitly historical outcome, metrics, ledger verification, and a completed manifest.
