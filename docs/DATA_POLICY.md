# OpenAlpha Data Policy

## Scope

This policy governs market data used by Kronos Reality Check. Provider access is a research input with licensing, revision, feed, timestamp, and corporate-action limitations. A content hash proves identity, not correctness or redistribution rights.

## Initial provider

Alpaca is the first supported US equity and ETF provider because its historical stock-bars endpoint is authenticated and officially documented. The provider-independent contract must permit another lawful provider without changing rolling-origin evaluation or model adapters.

The fixed endpoint is `https://data.alpaca.markets/v2/stocks/bars`. Credentials are read only from `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` and sent in the documented authentication headers. Missing, invalid, or insufficiently entitled credentials are explicit provider failures. There is no Yahoo adapter and no fallback provider.

Every request declares and records:

- provider and adapter version;
- symbols;
- `feed`;
- `timeframe`;
- inclusive `start` and `end`;
- `adjustment`;
- `asof`, including the difference between a date and `-`;
- pagination tokens and retrieval timestamp;
- response/request hashes without exposing credentials.

The first protocol requests `feed=sip` and `timeframe=1Day`. Alpaca documents that SIP combines all US exchanges while IEX is a single-exchange feed. If the account cannot access the declared historical SIP request, the run fails; it must not silently use IEX.

## Representations and corporate actions

The study stores two separately hashed views from separately declared requests:

1. **Forecast-target view — `adjustment=all`.** Used for model inputs and return targets to reduce mechanical discontinuities from splits, cash dividends, and spin-offs. It is a revised, provider-adjusted representation and is not an executable price series.
2. **Execution view — `adjustment=raw`.** Used for hypothetical next-bar reference prices and cost accounting. Corporate actions during a held position require explicit events or the experiment is invalid; adjusted forecast prices are never used as fills.

The protocol records both hashes and prohibits joining rows across views without matching provider, feed, symbol, session, retrieval batch, and declared mapping. Volume semantics follow the requested adjustment and remain provider-defined.

## Timestamp and as-of semantics

Daily provider timestamps are normalized to named XNYS sessions while preserving the original UTC timestamp. `asof` controls symbol-entity mapping, not a full vintage snapshot of all historical corrections. Historical replay therefore cannot claim that a modern response is bit-for-bit identical to what the provider would have returned at the forecast date.

This unresolved point-in-time limitation is a methodology warning in replay and sealed-test reports. Live forecasts preserve the actual local snapshot obtained before prediction.

## Storage and credentials

- API keys and secrets exist only in environment variables or an external secret store.
- Secrets never enter URLs, logs, exceptions, manifests, fixtures, or Git.
- Raw provider responses and normalized market snapshots are stored only in the user-local content-addressed artifact root, which is ignored by Git.
- Tests use deterministic synthetic bars or purpose-built fixtures labeled `synthetic`.
- Synthetic fixtures are never admissible research evidence.
- Public reports may contain derived aggregate metrics and small non-reconstructive illustrations, not raw provider datasets.

## Redistribution and access restrictions

Alpaca states that its API data may not be redistributed. OpenAlpha therefore does not commit or publish raw responses, Parquet snapshots, or a public data download. Reproduction requires the user’s own Alpaca credentials and compares locally computed snapshot hashes and declared query metadata. Any future redistribution requires documented written permission and a policy revision.

## Limitations shown in evidence

Every data manifest and report discloses feed coverage, access plan, adjustment, retrieval time, missing sessions, corrections, stale/zero volume, timestamp normalization, symbol mapping, and the absence of a guaranteed point-in-time vintage. Provider data is supplied as-is and may be incomplete, delayed, corrected, or unavailable.

## Official references

- [Alpaca historical stock bars](https://docs.alpaca.markets/us/v1.4.2/reference/stockbars)
- [Alpaca Market Data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq)
- [Alpaca feed description](https://alpaca.markets/support/data-provider-alpaca)
- [Alpaca redistribution policy](https://alpaca.markets/support/redistribute-alpaca-api)
