# OpenAlpha Architecture

## Position

OpenAlpha is a CLI-first evidence kernel organized as a modular Python monorepo. Its primary object is an immutable forecast linked to a locked protocol, causal data snapshot, model configuration, later outcome, comparison metrics, and reproducibility manifest.

The initial release has no required API, worker, MLflow service, database server, or web application. Those transports may be added only after the real forecast-to-outcome path works through the CLI.

## Preserved and planned packages

```text
packages/
  experiment-spec/      # preserved v1 execution-spec contracts
  research-core/        # preserved artifacts, manifests, run journals
  reality-check/        # protocol, provider/model ports, ledger, evaluation, CLI
research/
  protocol/             # human and machine v1 protocol plus SHA-256 lock
  reports/              # generated local reports; raw provider data excluded
```

The existing `experiment-spec` package is preserved as an execution-contract prototype. The new protocol schema is broader and evidence-focused; any reuse or migration is explicit. Modules inside `reality-check` keep protocol, data, models, ledger, evaluation, orchestration, and CLI boundaries distinct without multiplying packages before those boundaries need independent release cycles.

## Evidence flow

```text
protocol YAML
  -> validate + canonical hash
  -> authenticated provider requests
  -> adjusted target snapshot + raw execution snapshot
  -> causal origin and context
  -> real Kronos + baseline forecast artifacts
  -> append forecast ledger records
  -> later append outcome records
  -> forecast/statistical/economic metrics
  -> chain verification + completed-run manifest
  -> CLI audit/reproduction
  -> later scoreboard, audit page, and paper
```

The scorer accepts forecast IDs and outcome snapshots, not an unrestricted full dataset. Historical runners persist each forecast before obtaining the target slice. Live runners cannot resolve an outcome before its expected evaluation time.

## Core contracts

### Research protocol

A frozen, extra-forbid model locks hypotheses, universe, data requests and representations, periods, evidence classes, horizons, targets, models, metrics, statistical family, strategy/costs, success/failure criteria, exclusions, missing/corporate-action policies, and seeds. Canonical UTF-8 JSON produces the protocol hash. A published version is never updated in place.

### Market data provider

`MarketDataProvider.fetch_historical_bars(request)` accepts a provider-independent request with symbols, feed, timeframe, start, end, adjustment, as-of behavior, page limit, and sort. `AlpacaHistoricalBarsProvider` alone knows headers and the fixed Alpaca URL. It obtains credentials from the environment, paginates with bounds and cycle detection, redacts failures, and records feed/adjustment/retrieval metadata. No provider fallback exists.

### Data snapshots

Raw bytes, normalized tables, quality findings, and manifests are separate content-addressed artifacts stored outside Git. Forecast-target and execution views never share an ambiguous hash. Sessions use XNYS labels while retaining original provider timestamps.

### Model adapters

One typed adapter contract records model/checkpoint/version/configuration, causal fit/context window, cutoff, horizon, runtime, hardware, seed, prediction, uncertainty capabilities, and artifact inputs. A failed model emits a failure record and cannot be silently replaced. Test-only adapters mark outputs as synthetic and inadmissible.

### Forecast ledger

Forecast and outcome are distinct immutable record types. Each record has a canonical payload hash and previous-record hash. An outcome references exactly one forecast; duplicate or premature resolution fails. Chain verification checks every payload, link, evidence class, protocol hash, and referenced artifact.

### Evaluation and economic accounting

Rolling origins come from the declared XNYS calendar and enforce input cutoffs. Metrics distinguish returns, direction, magnitude/volatility, price, path, and interval behavior. Results always place Kronos beside baselines. The fixed strategy uses the declared threshold, next eligible raw bar, commissions, and slippage. Cash, buy-and-hold, and equivalent baseline signals use matching periods and costs.

## Evidence classes

`historical_replay`, `sealed_historical_test`, and `live_precommitted_forecast` are non-interchangeable provenance. Aggregation keys include evidence class. A UI or report may compare classes but cannot pool them without an explicitly labeled analysis.

## Error and validity model

Typed failures cover protocol, credentials, provider entitlement/rate limits, data quality, causality, model capability/resource, forecast persistence, outcome timing, ledger integrity, statistics, accounting, artifact integrity, and infrastructure. Failures remain visible. A manifest publishes only after required artifacts and methodology status verify.

## Security and storage

- Fixed HTTPS provider hosts and bounded pagination prevent SSRF and unbounded fetches.
- Secrets are environment-only and redacted from logs, artifacts, and exceptions.
- Existing path confinement protects local datasets and reports.
- Raw Alpaca data stays in ignored user-local artifact storage and is never redistributed.
- Model files use pinned revisions/hashes and reviewed safe formats.

## Deferred topology

Live scheduling will eventually need durable execution, and the evidence application will need read APIs. Those systems call the same protocol, provider, ledger, evaluation, and manifest services used by the CLI; they do not redefine research semantics.
