# ADR 0008: Authenticated Alpaca Data Boundary without Fallback

- Status: Amended by ADR 0010
- Date: 2026-07-30

## Context

Yahoo’s no-key chart endpoint is undocumented and unsuitable for an auditable provider boundary. The study needs explicit feed, adjustment, timestamp, symbol-mapping, access, and redistribution semantics.

## Decision

Use Alpaca’s documented historical stock-bars API through a provider-independent port. Credentials come only from `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`. Requests explicitly set feed, timeframe, start, end, adjustment, and as-of behavior. Sentinel v0 pins SIP daily raw bars. Authentication, entitlement, rate-limit, and provider failures are explicit; no adapter silently falls back to IEX, Yahoo, or another provider.

For Sentinel v0, validate and hash raw response bytes in memory, then discard them. Retain compact normalized inputs only in user-local content-addressed storage when legally permitted and necessary. Do not commit or redistribute raw Alpaca market data. Automated tests use data labeled as synthetic.

Raw close-to-close return is not total return: dividends are omitted and corporate actions can create discontinuities. This is a declared feasibility limitation, not a request to build a corporate-action engine.

## Consequences

- Real experiments require a user-owned Alpaca account and credentials.
- Reproduction may verify procedures and locally generated hashes without shipping the raw dataset.
- Feed and adjustment choices become protocol fields rather than adapter defaults.
- Another provider can implement the same port without changing the research engine.

## Sources

- [Historical bars request fields](https://docs.alpaca.markets/us/reference/stockbars)
- [Authentication and feed behavior](https://docs.alpaca.markets/us/docs/market-data-faq)
- [Redistribution restriction](https://alpaca.markets/support/redistribute-alpaca-api)
