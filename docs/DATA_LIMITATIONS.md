# Data Limitations

`docs/DATA_POLICY.md` is the governing access and storage policy. This document records limitations that must accompany every result.

## Provider and entitlement limits

Alpaca is the first supported US equity and ETF provider. Access requires user-supplied environment credentials and the requested feed entitlement. SIP and IEX do not have equivalent venue coverage; the initial protocol pins SIP and fails explicitly if it is unavailable. Rate limits, outages, schema changes, corrections, and entitlement changes can prevent exact refetching.

Provider identity, feed, timeframe, start, end, adjustment, as-of behavior, retrieval timestamp, request identity, and response hash are recorded. A successful HTTP response is not accepted until schema, pagination, bounds, ordering, and quality checks pass.

## Modern-data and point-in-time limits

Historical bars retrieved today may include corrections, corporate-action adjustments, or symbol mappings that were not available in identical form at a historical forecast cutoff. Alpaca's `asof` parameter controls symbol mapping behavior; it does not prove complete historical-vintage reconstruction. Historical replay and sealed results disclose this residual limitation.

## Paired representations

The initial study requests two content-addressed views over the same sessions:

- `adjustment=all` for model inputs and return targets, so splits, dividends, and supported spin-offs do not masquerade as forecastable returns;
- `adjustment=raw` for hypothetical next-bar execution prices, so fills are expressed in observed price units.

This does not make daily OHLCV an execution tape. Corporate actions crossing a simulated holding period require an explicit accounting adjustment; ambiguous cases fail or are excluded under the locked protocol rather than silently repaired.

## Timestamp and calendar limits

Provider timestamps, XNYS session labels, market timezone, early closes, holidays, halts, and missing sessions are distinct concepts. Future returned rows are never used to infer the schedule available at a forecast cutoff. Normalization stores provider timestamps and canonical session identities.

## Missing, stale, and revised observations

Missing bars are not zeros and are not silently forward-filled. Quality artifacts distinguish expected closures, provider gaps, halts, unavailable fields, duplicates, zero-volume observations, stale runs, and insufficient context. Any repair creates a new derived snapshot and records its causal rule and lineage.

## Universe and contamination limits

The five current ETFs avoid historical constituent selection but do not remove fund-survival or asset-selection bias. Kronos checkpoint provenance does not expose row-level pretraining data; evaluation after June 2024 reduces direct temporal overlap but cannot prove the absence of related information or learned market structure.

## Execution limits

Daily bars do not reveal spread paths, auction mechanics, queue position, market impact, partial fills, or intrabar event order. Economic results are hypothetical, use a locked next-bar convention and fixed costs, and are not live-brokerage performance.

## Storage and redistribution

Raw provider requests and responses remain local, outside Git and public artifacts. Alpaca states that its API market data may not be redistributed. Public reproducibility therefore uses request recipes, normalized-schema descriptions, content hashes, code and protocol hashes, and permitted derived aggregates; another researcher must fetch their own licensed copy.

## Reviewer checklist

A valid data-quality artifact answers:

1. Which provider, endpoint, feed, entitlement, and retrieval time produced the data?
2. Which explicit request and raw response hashes identify the source bytes?
3. Which timestamp, calendar, adjustment, as-of, and corporate-action semantics apply?
4. Which rows or fields were missing, stale, duplicated, transformed, repaired, or excluded?
5. Which point-in-time, survivorship, contamination, and execution guarantees remain absent?
6. May the raw or derived output be redistributed?
