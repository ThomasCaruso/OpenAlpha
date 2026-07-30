# ADR 0011: yfinance for Sentinel Phase 2 Development Data

- Status: Accepted
- Date: 2026-07-30
- Supersedes for Phase 2 only: ADR 0008

## Context

Sentinel Phase 2 needs one small daily SPY history to test the real context-to-Kronos-to-diagnostics-to-outcome chain. Waiting for authenticated SIP access would test an institutional integration before testing the reliability concept. A hand-written client for an undocumented Yahoo endpoint remains rejected.

yfinance is an open-source client that obtains data from Yahoo Finance. It is not an official Yahoo integration, its availability and response behavior may change, and it is not an institutional point-in-time source.

## Decision

Use `yfinance==1.5.2` as the default local development adapter for the single Phase 2 origin. The adapter supplies the existing provider-independent port and explicitly sets daily interval, inclusive start, exclusive end, no automatic/back adjustment, no repair, corporate actions enabled for audit metadata, progress and threads disabled, and a 30-second timeout.

Use only raw Open, High, Low, Close, and Volume as model input. Exclude Adj Close. Treat dividends and splits as warnings, enforce the XNYS cutoff locally, and retrieve the outcome only after the forecast is sealed.

Do not commit or redistribute raw responses, reusable histories, CSV exports, caches, or private normalized snapshots. Store only declared request/provenance, hashes, quality results, compact forecasts/diagnostics/outcomes, warnings, and derived metrics in Git.

Retain Alpaca as an optional independent verification adapter. Before a serious publication, rerun a representative sample with at least one separate provider and report row, return, forecast, diagnostic, and conclusion differences.

## Consequences

- Phase 2 requires no market-data subscription or credentials.
- The one-origin proof remains development evidence, not institutional or provider-independent evidence.
- Provider instability and historical revision risk are explicit limitations.
- There is no silent provider fallback and no cross-provider request in Phase 2.

## Sources

- [yfinance package and legal notice](https://pypi.org/project/yfinance/)
- [yfinance download parameters](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars)
