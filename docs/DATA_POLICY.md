# OpenAlpha Sentinel Data and Model-Artifact Policy

## Scope

This policy governs market inputs and forecast-provider artifacts for Sentinel v0. Hashes prove byte identity, not data correctness, historical vintage, model validity, or redistribution rights.

## Phase 2 development provider

Phase 2 uses Yahoo Finance data through the pinned open-source `yfinance==1.5.2` client. This is an unofficial public interface for local research and education, not an official Yahoo integration, authenticated institutional feed, or point-in-time market-data service. Availability, schemas, and response behavior may change without project control.

The provider-independent market-data port remains. Alpaca is retained as an optional, separately sourced verification adapter for a later phase; it is not called during the single-origin Phase 2 proof. No adapter silently falls back to another provider.

Every Phase 2 download explicitly sets:

- `interval="1d"`;
- `auto_adjust=False`;
- `back_adjust=False`;
- `repair=False`;
- `actions=True`;
- `progress=False`;
- `threads=False`;
- `timeout=30.0`.

The request also explicitly sets start, exclusive end, `prepost=False`, `rounding=False`, `keepna=False`, and `multi_level_index=False`. The persisted provenance records provider, client and version, all request arguments, retrieval timestamp, normalized schema, row count, first/last session, normalized-input hash, quality checks, and corporate-action warnings.

## Causal cutoff and representation

`yfinance` treats start as inclusive and end as exclusive. For the SPY 2024-07-05 forecast, creation requests history through an exclusive end of 2024-07-06, rejects any row after 2024-07-05, and requires the final XNYS input session to be exactly 2024-07-05. The logically separate resolver requests the five expected sessions through 2024-07-12 only after the forecast record is sealed.

Sentinel v0 uses raw Open, High, Low, Close, and Volume. `Adj Close` is never a Kronos input. Price repair is disabled because it can alter observations using surrounding information. Dividend and split columns are audit metadata and warnings only.

The primary realized return is `log(raw close after five XNYS sessions / raw cutoff close)`. This avoids treating a modern adjusted history as exact point-in-time truth, but it omits dividend return and may expose split or other corporate-action discontinuities. V0 records those limitations and does not build a corporate-action engine or make trading-return claims.

## Storage and redistribution

- Raw Yahoo response payloads, complete reusable historical datasets, yfinance caches, and CSV bar exports are never committed.
- Project policy prohibits redistribution of raw downloaded data.
- A compact normalized snapshot may exist only as a private local artifact outside Git when required for reproduction and permitted.
- Git may contain request recipes, provenance, hashes, configurations, compact forecasts, diagnostics, outcomes, warnings, and derived metrics.
- Tests use purpose-built fixtures marked synthetic and inadmissible as market evidence.

## Cross-provider verification

No cross-provider comparison is part of the one-origin Phase 2 proof. Before any serious Sentinel publication, a representative sample must be rerun through at least one independently sourced provider such as Alpaca. The report must compare row values, returns, forecasts, diagnostics, and whether the Sentinel conclusion changes. Until then, the project cannot claim provider-independent evidence.

## Model inference and weights

The official Kronos-mini page currently reports no hosted Hugging Face Inference Provider. Sentinel v0 therefore uses a temporary local Hugging Face cache outside Git unless a documented ephemeral option proves simpler during the Phase 2 feasibility check.

The source, model, and tokenizer revisions are pinned in `research/sentinel-v0/experiment.yaml`. Downloaded file hashes, cache class, device, runtime, and failure information are recorded. Model weights, tokenizer weights, cache directories, and generated temporary deployments are never committed.

No adapter may require arbitrary remote code, untrusted pickle/joblib loading, or a permanent paid deployment for v0. Real inference failures cannot be replaced with fake Kronos output.

## Reproducibility limitations

A third party may reproduce code and request parameters yet receive corrected Yahoo history or different hardware-dependent stochastic output. yfinance is not an institutional point-in-time source. Sentinel records these limitations and never describes historical replay as a literal reconstruction of the data service available at the cutoff.

## Primary sources

- [yfinance package and legal notice](https://pypi.org/project/yfinance/)
- [yfinance download parameters](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [Alpaca historical stock bars](https://docs.alpaca.markets/us/reference/stockbars)
- [Official Kronos repository](https://github.com/shiyu-coder/Kronos)
- [Kronos-mini model card](https://huggingface.co/NeoQuasar/Kronos-mini)
- [Kronos-Tokenizer-2k model card](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k)

## Historical Bridge Phase 2 execution policy

The retired Bridge-2K Phase 2 design reused this policy unchanged. Its locked
provider was Yahoo Finance through pinned `yfinance==1.5.2`; Binance was reserved
for a later phase behind continuation and data-policy gates.

At the pipeline-preparation commit, zero provider requests had been issued and
zero candles retrieved. The retrieval client had been implemented but not invoked.

The planned pipeline was governed by these rules:

- Provider integrations were resolved lazily, so importing the research package
  loaded no provider client, Torch, or Kronos asset.
- Provider exception text was never propagated into logs or artifacts, because it
  could carry request URLs and query parameters. Failures surfaced as the typed
  `PROVIDER_REQUEST_FAILED` code with the exception class name only.
- No credential was required. The locked provider was a public interface and the
  pinned Tokenizer-2k repository was public. The environment template contained
  paths and execution knobs only, and no filled copy was committed.
- The feature cache lived outside the Git worktree, was content-addressed, and was
  capped in code at 10,737,418,240 bytes. Both preflight and the cache
  constructor refused a cache path inside the repository.
- Feature-cache shards, checkpoints, and raw candles were never exported. The
  export script copied manifests, hashes, aggregates, and reports only, and
  failed with `FORBIDDEN_ARTIFACT_IN_EXPORT` otherwise.
- Reconstruction-test shards were not loadable before the explicit, now-retired
  test-opening transition.
- Fake-provider fixtures were stamped `provider_mode: fake`, carried no retrieval
  timestamp, and were rejected by any run declaring `evidence_class: real_phase2`.

## Historical cloud execution and the provider decision

The historical Phase 2 design moved execution to managed cloud infrastructure.
Its locked provider remained Yahoo Finance through pinned `yfinance==1.5.2`, to
be called from inside the Modal worker rather than from a workstation.

An official-API swap to Alpaca was evaluated and deliberately not taken. Alpaca's
documented historical stock coverage begins 2016-01-01, while the locked training
period begins 2010-01-01. Adopting it would have cost roughly 60 percent of the
training corpus (40 to 16 scored suffixes per symbol, 35,840 to 14,336 pooled
scored candles) and would have required a second methodological change beyond the
provider swap. Validation, reconstruction test, and external periods were
unaffected, and both locked sample minima still passed, but shortening the
training period would have been a scientific cost that had to be chosen explicitly rather than
absorbed inside a provider amendment.

Keeping `yfinance` therefore preserved the entire hash chain and required no
pre-data amendment at all. The provider abstraction was unchanged, so adopting
Alpaca, Polygon, or Tiingo later would have required one implementation plus an explicit pre-data
amendment recording old provider, new provider, reason, feed, adjustment
behaviour, OHLCV fields, corporate-action handling, pagination, missing-session
behaviour, timestamp semantics, endpoint class, credentials requirement, and a
revised experiment hash.

The planned cloud design added these rules:

- Provider credentials, if a future provider had needed them, would have existed only in Modal
  Secrets. They were never accepted in a run-creation request or a GitHub
  workflow input, and were never written to journals, logs, artifacts, or exceptions.
- Provider exception text was still never propagated; failures surfaced as the
  typed `PROVIDER_REQUEST_FAILED` code with the exception class name only.
- Raw provider responses were stored in neither Git nor the immutable artifact
  store. Normalized derived features lived only in the cache volume, under the
  10 GiB cap, and were cleaned up after manifests and hashes were sealed.
- The planned `GET /artifacts` endpoint filtered model weights, feature shards,
  raw data, and the lease object out of every manifest.
- Fake-provider fixtures remained stamped `provider_mode: fake` with no retrieval
  timestamp, wrote to a separate object-store root, and were rejected by any run
  declaring `evidence_class: real_phase2`.

At the cloud-conversion commit, zero provider requests had been issued and zero
candles retrieved.
