# Sentinel v0 Amendment: yfinance Development Provider

- Amendment date: 2026-07-30
- Previous experiment SHA-256: `261e4bac51b9b3d6b68b6cf0128d406ca63cbf4bcd9735ccbda1aa04d1fa761a`
- Amended experiment SHA-256: `587bc36ac1dd941c66d9e94c92d7b26a33faeb8e7e7a38883762f0cb54897950`
- Execution state at amendment: no Yahoo Finance or Alpaca market-data request, model download, or Kronos inference performed

## Exact changed fields

- `schema_version`: `sentinel-v0.2` to `sentinel-v0.3`
- `status`: `phase_2_amended_before_execution` to `phase_2_yfinance_amended_before_execution`
- `amendment.supersedes_sha256` and the amendment change list/reason
- `market_data.provider`: `alpaca` to `yahoo_finance`
- `market_data.client`: added `yfinance`
- `market_data.client_version`: added `1.5.2`
- `market_data.underlying_data_source`: added `Yahoo Finance`
- `market_data.access_class`: added `unofficial_public_interface`
- `market_data.intended_use`: added `local_research_and_education`
- `market_data.redistribution`: set to `prohibited_by_project_policy`
- `market_data.credentials_required`: set to `false`
- `market_data.official_yahoo_integration`: set to `false`
- `market_data.institutional_point_in_time_source`: set to `false`
- `market_data.request`: replaced Alpaca endpoint/feed/pagination/as-of fields with explicit yfinance daily start/end, raw adjustment, repair/action, threading, progress, timeout, and related arguments
- `market_data.cutoff_policy`: added exclusive-end semantics, exact 2024-07-05 local cutoff, post-cutoff rejection, and XNYS validation
- `market_data.outcome_request`: added the separate post-seal request and exact 2024-07-08 through 2024-07-12 outcome sessions
- `market_data.raw_response_storage`, `normalized_storage`, and `cache_storage`: made the ephemeral/private/outside-Git policy explicit
- `market_data.adj_close_as_model_input`: set to `false`
- `market_data.action_columns_use`: restricted to audit metadata and warnings
- `market_data.limitations`: added unofficial-interface, availability, point-in-time, redistribution, and provider-independence warnings
- `market_data.optional_verification_provider`: retained Alpaca but disabled it for Phase 2
- `market_data.cross_provider_verification`: disabled for Phase 2 and required before serious publication

The raw OHLCV representation, raw five-session close-to-close log-return target, canonical three-path 512-context forecast, six shorter-context stress paths, and seed-reproducibility probe remain unchanged from the preceding amendment.

## Methodological reason

The single-origin daily SPY feasibility proof needs a small causal history but does not justify waiting for an authenticated institutional market-data integration. The pinned open-source yfinance client provides credential-free local research access while keeping adjustment, repair, action, timeout, and cutoff behavior explicit.

This is a scope decision, not an upgrade in data quality. yfinance is not an official Yahoo integration or institutional point-in-time source. Raw responses and reusable histories cannot enter Git or be redistributed. Before a serious Sentinel result is published, a representative sample must be checked through an independently sourced provider such as Alpaca, and any bar, return, forecast, diagnostic, or conclusion differences must be reported.

## Dependency lock

- yfinance version: `1.5.2`
- PyPI source archive SHA-256: `5935d457fc62cf2f7e9bf1b2d019a8fec8fb0072f58a095eb53740b70a6a06ed`
- PyPI wheel SHA-256: `197fc03485c246547a5a9184956c60150ea33b6f740d877e02a97f123d5cd2b9`
- Dependency resolution: `uv.lock`

## Execution boundary

This amendment authorizes no provider fallback and no cross-provider check in Phase 2. The next external market-data operation may occur only after this amendment is committed. Outcome rows remain inaccessible until the forecast artifact is sealed.
