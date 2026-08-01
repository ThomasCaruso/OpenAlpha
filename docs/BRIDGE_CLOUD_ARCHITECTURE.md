# Bridge Phase 2 Cloud Architecture

**Status: cloud conversion complete and synthetically validated. No empirical
Phase 2 run has been executed.**

OpenAlpha is an API-operated service. There is no GPU host to rent, administer,
or SSH into, and nothing heavy is installed on a workstation.

## Production path

```
GitHub deployment
  -> authenticated OpenAlpha control API   (Modal web endpoints, CPU)
  -> Modal background GPU worker           (one T4)
  -> market-data API                       (yfinance, inside the worker)
  -> Hugging Face Hub API                  (pinned Kronos assets)
  -> cloud storage                         (Modal Volume + Cloudflare R2)
  -> status and artifact APIs
```

No Kubernetes. No Terraform. No AWS account: the artifact store is
S3-compatible and defaults to Cloudflare R2.

## Layering

The scientific engine stays cloud-agnostic. The cloud package is a narrow
adapter around it, not a rewrite.

| Layer | Package | Cloud-aware? |
|---|---|---|
| State machine, run identity, windowing, metrics, bootstrap, gates, training interface, test-opening preconditions | `openalpha_bridge.phase2` | No |
| Object store, journal, lease, cloud test gate, redaction, control service, runner | `openalpha_bridge.cloud` | Yes |
| Modal app, image, secrets, GPU worker | `cloud/modal/bridge_phase2_app.py` | Modal-specific |

Only the third layer names Modal. `ComputeBackend` is a five-method protocol, so
another managed backend could be added later without touching the engine or the
control service. Exactly one backend is implemented today, deliberately.

## Components

**Control API.** Six operations, all authenticated, exposed as Modal CPU web
endpoints so there is no separate service to host. Implemented in
`cloud/service.py` as plain Python, which is why it is testable without a web
framework. There is no remote-shell or arbitrary-command operation.

**GPU worker.** One background function running the complete pipeline in one
image: preflight, retrieval, validation, windowing, coverage, asset resolution,
Stage A, Stage B, Stage C, checkpoint selection and freeze, the one-time test
opening, test evaluation, external evaluation, bootstrap, gate decision, and
artifact export. Feature extraction and training are never split across
environments.

**Synthetic worker.** A separate CPU function using fake provider and fake
Kronos components, writing to a separate object-store namespace under a
different evidence class. It can never create the real test-opening record.

## Run identity

A run identity binds source commit, original experiment hash, both amendment
hashes, active experiment hash, score-mask hash, provider identity, Kronos
revision, cloud image digest, Modal app version, dependency-lock hash,
object-store namespace, operator, and evidence class.

Change any of those and the `run_id` changes. A rebuilt cloud image is a
different run. Idempotency is keyed on `(idempotency_key, binding)`: the same
key with identical inputs returns the same run; the same key with conflicting
inputs fails with `IDEMPOTENCY_KEY_CONFLICT`.

## Concurrency

Run creation spawns the worker and returns immediately. The caller polls the
status endpoint; no HTTP connection is held open for the experiment.

Before starting or resuming, a worker acquires an exclusive lease. Modal retries
are disabled on the GPU worker so a platform retry cannot duplicate a scientific
run. Journal entries are immutable per-sequence keys, so a second writer at the
same sequence fails with `CONCURRENT_JOURNAL_WRITE` even if the lease were
somehow bypassed.

When a worker is interrupted the last verified state and the latest checkpoint
survive in cloud storage, the lease expires, a verified resume is permitted, and
the test-opening record is never recreated.

## Data provider

The locked provider is unchanged: Yahoo Finance through pinned
`yfinance==1.5.2`, now called from inside the cloud worker. Keeping it avoids a
methodological change; see [DATA_POLICY.md](DATA_POLICY.md) for why an
official-API swap was evaluated and deferred rather than taken silently.

The provider abstraction is unchanged, so swapping in Alpaca, Polygon, or Tiingo
later is a single implementation plus a pre-data amendment.

## Cost

The Bridge head is 17,605 parameters. The GPU exists only for frozen Kronos
forward passes over roughly 2,900 sequences, which is minutes of T4 time. A full
Phase 2 run fits comfortably inside Modal's monthly free credits, and R2 storage
for compact artifacts is well inside its free tier.

Resource guards are enforced in code: one GPU, 24 GPU-hour ceiling, 10 GiB
feature-cache cap, 25 MiB checkpoint cap, bounded provider requests and retries,
and automatic shutdown at a terminal state.

## Local boundary

Local and ordinary CI execution may run only static analysis, unit tests,
state-machine tests, synthetic fake-provider tests, serialization tests, API
client tests, cloud-adapter mocks, and deployment-manifest validation.

Local execution must not install Torch by default, retrieve market data, fetch
Kronos weights, train, open the test partition, store feature caches, or produce
empirical evidence. All empirical work happens in managed cloud execution.
