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

## Bridge Phase 2 execution pipeline

The Bridge-2K Phase 2 pipeline reuses this policy unchanged. The locked provider
remains Yahoo Finance through pinned `yfinance==1.5.2`; Binance remains reserved
for Phase 4 after the continuation and data-policy gates.

As of the pipeline-preparation commit, zero provider requests have been issued
and zero candles retrieved. The retrieval client is implemented but has never
been invoked.

Additional Phase 2 rules:

- The provider client resolves `yfinance` lazily. Importing `openalpha_bridge`
  pulls in no provider client, no Torch, and no Kronos asset.
- Provider exception text is never propagated into logs or artifacts, because it
  can carry request URLs and query parameters. Failures surface as the typed
  `PROVIDER_REQUEST_FAILED` code with the exception class name only.
- No credential is required. The locked provider is a public interface and the
  pinned Tokenizer-2k repository is public. The environment template contains
  paths and execution knobs only, and no filled copy is ever committed.
- The feature cache lives outside the Git worktree, is content-addressed, and is
  capped in code at 10,737,418,240 bytes. Both preflight and the cache
  constructor refuse a cache path inside the repository.
- Feature-cache shards, checkpoints, and raw candles are never exported. The
  export script copies manifests, hashes, aggregates, and reports only, and
  fails with `FORBIDDEN_ARTIFACT_IN_EXPORT` otherwise.
- Reconstruction-test shards cannot be loaded before the explicit test-opening
  transition; see [BRIDGE_TEST_OPENING_POLICY.md](BRIDGE_TEST_OPENING_POLICY.md).
- Fake-provider fixtures are stamped `provider_mode: fake`, carry no retrieval
  timestamp, and are rejected by any run declaring `evidence_class: real_phase2`.
