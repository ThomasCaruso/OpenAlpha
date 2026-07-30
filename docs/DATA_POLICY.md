# OpenAlpha Sentinel Data and Model-Artifact Policy

## Scope

This policy governs market inputs and forecast-provider artifacts for Sentinel v0. Hashes prove byte identity, not data correctness, historical vintage, model validity, or redistribution rights.

## Market-data provider

Alpaca remains the first US equity/ETF provider because its historical-bars endpoint is authenticated and documented. The provider-independent boundary allows a future lawful provider without changing diagnostics or evaluation.

The fixed endpoint is `https://data.alpaca.markets/v2/stocks/bars`. Credentials come only from `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`. Missing credentials, SIP entitlement, invalid responses, and rate limits fail explicitly. There is no Yahoo adapter and no provider fallback.

Every request records provider/adapter identity, symbols, SIP feed, 1Day timeframe, inclusive start/end, adjustment, cutoff-date `asof`, ascending sort, bounded pagination, retrieval time, and request/response hashes.

## V0 representation

Sentinel v0 uses `adjustment=raw` daily OHLCV bars for Kronos context, the last-value baseline, diagnostics, and return/path outcomes. The primary realized return is `log(raw close after five XNYS sessions / raw cutoff close)`.

This choice avoids assuming that a currently requested split-, dividend-, and spin-off-adjusted history is identical to what was available at the historical cutoff. It also excludes dividend return and may expose split or other corporate-action discontinuities. V0 records those limitations and does not build a corporate-action engine.

V0 has no trading simulation. A later economic experiment would require explicit dividend and corporate-action accounting under a separately amended design.

Daily timestamps are normalized to XNYS sessions while preserving provider UTC timestamps. `asof` controls symbol mapping; it is not a guarantee that today's historical response matches the vintage available at the historical cutoff.

## Storage

- Raw Alpaca response bytes are validated, hashed, and discarded.
- Normalized input is retained only when legally permitted and necessary, inside the user-local confined content-addressed store.
- Git contains request recipes, provenance, hashes, configurations, forecasts, diagnostics, outcomes, and derived aggregate metrics—not restricted raw bars.
- Tests use purpose-built fixtures marked synthetic and inadmissible as empirical evidence.

## Model inference and weights

The official Kronos-mini page currently reports no hosted Hugging Face Inference Provider. Sentinel v0 therefore uses a temporary local Hugging Face cache outside Git unless a documented ephemeral option proves simpler during the Phase 2 feasibility check.

The source, model, and tokenizer revisions are pinned in `research/sentinel-v0/experiment.yaml`. Downloaded file hashes, cache class, device, runtime, and failure information are recorded. Model weights, tokenizer weights, cache directories, and generated temporary deployments are never committed.

No adapter may require arbitrary remote code, untrusted pickle/joblib loading, or a permanent paid deployment for v0. Real inference failures cannot be replaced with fake Kronos output.

## Redistribution

Alpaca states its API market data cannot be redistributed. Public reproducibility uses user-owned credentials, request metadata, hashes, code/configuration identity, and permitted derived aggregates. Any raw-data publication requires documented permission and a policy revision.

Kronos source and released model/tokenizer pages state MIT licensing, but downstream publication still records exact source/model identities and license metadata.

## Reproducibility limitations

A third party may reproduce code and request parameters yet receive corrected provider history, a changed access entitlement, or different hardware-dependent stochastic output. Sentinel records these limitations and never describes historical replay as a literal reconstruction of the information service available at the cutoff.

## Primary sources

- [Alpaca historical stock bars](https://docs.alpaca.markets/us/reference/stockbars)
- [Alpaca Market Data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq)
- [Alpaca redistribution policy](https://alpaca.markets/support/redistribute-alpaca-api)
- [Official Kronos repository](https://github.com/shiyu-coder/Kronos)
- [Kronos-mini model card](https://huggingface.co/NeoQuasar/Kronos-mini)
- [Kronos-Tokenizer-2k model card](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k)
