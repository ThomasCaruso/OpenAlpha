# OpenAlpha Sentinel Architecture

## Position

OpenAlpha is a local, evidence-first Python research repository. Sentinel v0 adds no service topology. Its primary object is a pre-outcome reliability decision linked to causal data, an ensemble forecast, diagnostics, a later outcome, and verified provenance.

No frontend, API server, worker, database, MLflow service, or public SDK is required.

## Repository shape

```text
packages/
  experiment-spec/      # preserved execution-spec prototype
  research-core/        # preserved artifact, journal, and manifest infrastructure
  sentinel/             # Phase 2 only: one narrow internal package
research/
  sentinel-v0/
    experiment.yaml     # fixed v0 experiment choices
    results/            # compact derived artifacts only
    reports/            # artifact-grounded reports only
```

Phase 1 creates no `sentinel` package. When Phase 2 begins, the package contains only focused modules for forecast-provider contracts, diagnostics, labels, risk scoring, actions, evaluation, reporting, and one orchestration service.

## Evidence flow

```text
experiment.yaml
  -> exact weekly origin list
  -> development: causal raw context -> nine paths -> 512-path mean -> diagnostics -> later outcome
  -> frozen development risk model and action policy
  -> holdout: causal context -> forecasts -> diagnostics -> decision
  -> later holdout outcome -> immutable error and postmortem
  -> untouched holdout report
  -> verified manifests and go/no-go decision
```

Outcome data is inaccessible to the forecast/diagnostic operation. A separate resolver receives forecast identities after their artifacts are durably published. Development origins have no Sentinel action because no frozen risk model exists yet. Holdout decisions use only the frozen development configuration and are published before resolution.

## Minimal internal boundaries

### Market data

The provider-independent historical-bars port remains unchanged in principle. Phase 2's first adapter is a pinned yfinance client over Yahoo Finance's unofficial public interface. Interval, inclusive start, exclusive end, raw adjustment behavior, repair, actions, threading, timeout, and local XNYS cutoff enforcement are explicit; no fallback exists. Alpaca remains an optional later adapter for independent verification.

### Forecast provider

The v0 port accepts model/checkpoint/source identity, ordered OHLCV, context length, horizon, seed, temperature, top-p, path count, and request metadata. It returns paths, summary, duration, provider/checkpoint/request identity, and typed failure.

Kronos uses a pinned temporary cache outside Git. A deterministic provider is test-only.

### Diagnostics

Pure functions consume causal context, the nine forecast paths, the baseline, eligible prior contexts, and only previously resolved model errors. They emit fourteen named values and missingness evidence.

### Labels and outcomes

Outcome resolution computes five-session return error, development-frozen failure label, baseline-relative label, direction correctness, and path error. It appends; it never mutates the forecast or decision.

### Risk and action

Development-only preprocessing feeds logistic and ridge models. A frozen config converts failure probability to 0–100 reliability and USE/BLEND/ABSTAIN. Reasons are deterministic diagnostic mappings, never generated text.

### Evaluation and reporting

The evaluator reports risk/error correlation, quintiles, coverage, direction, baseline, calibration, stability, diagnostic contribution, and runtime/cost. Reports read verified artifacts only.

## Preserved infrastructure mapping

- `LocalArtifactStore` publishes immutable request, forecast, diagnostic, decision, outcome, and report bytes.
- `RunStateJournal` records the experiment attempt lifecycle.
- Sentinel forecast/decision/outcome/postmortem events are new append-only payload artifacts; the existing journal is not rewritten into a forecast database.
- `RunManifest` binds all compatible artifact kinds and provenance. Existing required artifact semantics remain until a failing compatibility test justifies an amendment.
- Existing path confinement governs every artifact path.

## Error model

Typed failures distinguish provider availability or response, data quality, insufficient history, model loading, inference resource, seed control, ensemble completeness, diagnostic availability, outcome timing, risk configuration, holdout mutation, artifact integrity, and report provenance.

Failures remain in counts. No fake output silently replaces a real model failure.

## Security and storage

- Pinned client versions, bounded requests, explicit timeouts, and no silent provider fallback.
- Secrets are not required by the Phase 2 adapter; future provider credentials remain excluded from URLs, logs, artifacts, exceptions, and Git.
- Raw market responses are hashed then discarded in v0.
- Restricted normalized contexts, if retained, remain in the user-local confined store.
- Checkpoints and Hugging Face caches remain outside Git.
- No arbitrary remote code or pickle/joblib loading.

## Deferred architecture

A public `sentinel.forecast(...)` interface is a product hypothesis, not a v0 deliverable. Scheduling, services, databases, dashboards, additional models, and complex repair are considered only after the holdout decision.
