# Bridge Phase 2 Control API

All endpoints require `Authorization: Bearer <token>`. The token lives in Modal
Secrets and is never accepted as a query parameter, body field, or workflow
input. The public cannot start GPU jobs.

Base URL is the deployed Modal endpoint, referred to below as
`$OPENALPHA_API_BASE_URL`.

## Conventions

- Every mutating operation accepts an optional `idempotency_key`.
- Exit semantics: `200` success, `400` invalid request, `401` unauthorized,
  `403` unapproved commit, `404` unknown run, `409` conflict, `429` rate limited.
- No endpoint returns raw market data, model weights, tokens, or credentials.
- There is no remote-shell or arbitrary-command endpoint.

## POST /v1/bridge/phase2/runs

Creates a run and returns immediately. The experiment continues in the
background worker.

Request accepts only approved inputs. It deliberately has no field for
architecture, metrics, thresholds, partitions, periods, symbols, or training
configuration; those are locked by the experiment and its amendments.

```json
{
  "experiment_hash": "d52a9be7...f47b2",
  "source_commit": "abc1234...",
  "operator": "your-name",
  "execution_mode": "synthetic",
  "idempotency_key": "optional-stable-key",
  "confirm_real_evidence": false,
  "confirm_open_test_partition": false
}
```

`execution_mode` is `synthetic` or `real`. A real run additionally requires
`confirm_real_evidence: true`, and the reconstruction-test partition opens only
when `confirm_open_test_partition: true`.

Response:

```json
{
  "run_id": "run_9f2c...",
  "state": "CREATED",
  "created_at": "2026-08-01T09:00:01Z",
  "experiment_hash": "d52a9be7...f47b2",
  "status_url": "https://.../v1/bridge/phase2/runs/run_9f2c...",
  "artifact_url": "https://.../v1/bridge/phase2/runs/run_9f2c.../artifacts",
  "cloud_execution_id": "fc-...",
  "evidence_class": "real_phase2",
  "idempotent_replay": false
}
```

Errors: `UNKNOWN_EXPERIMENT_HASH`, `UNAPPROVED_SOURCE_COMMIT`,
`REAL_EVIDENCE_NOT_CONFIRMED`, `IDEMPOTENCY_KEY_CONFLICT`, `RATE_LIMIT_EXCEEDED`.

## GET /v1/bridge/phase2/runs/{run_id}

```json
{
  "run_id": "run_9f2c...",
  "state": "STAGE_C_TRAINED",
  "current_stage": "stage_c",
  "latest_journal_sequence": 11,
  "latest_journal_sha256": "…",
  "created_at": "2026-08-01T09:00:01Z",
  "updated_at": "2026-08-01T09:42:10Z",
  "blocker_code": null,
  "failure_code": null,
  "progress": {"epochs": 6},
  "cloud_execution_id": "fc-...",
  "cloud_execution_status": "running",
  "artifacts_available": false,
  "test_partition_opened": false,
  "test_sealed": true,
  "final_conclusion": null,
  "evidence_class": "real_phase2"
}
```

`test_sealed` stays `true` until the one-time gate opens. `final_conclusion`
appears only once the gate table is written.

## POST /v1/bridge/phase2/runs/{run_id}/resume

Permitted only from a verified `BLOCKED` or resumable `FAILED` state, and only
when no live lease is held. Body: `{"operator": "...", "idempotency_key": "..."}`.

Errors: `RUN_NOT_RESUMABLE`, `RUN_ALREADY_FINALIZED`, `RUN_LEASE_HELD`.

A resume never recreates the test-opening record.

## POST /v1/bridge/phase2/runs/{run_id}/cancel

Records the cancellation in the journal **before** terminating the cloud
execution, so an interrupted cancel still leaves the intent durably recorded.
Body: `{"operator": "...", "reason": "...", "idempotency_key": "..."}`.

## GET /v1/bridge/phase2/runs/{run_id}/artifacts

Returns the compact artifact manifest with content hashes and, where the store
supports it, short-lived signed download URLs.

Model weights, feature-cache shards, raw provider responses, and the lease
object are filtered out and never listed.

```json
{
  "run_id": "run_9f2c...",
  "evidence_class": "real_phase2",
  "artifacts": [
    {
      "key": "openalpha/bridge-phase2/runs/run_9f2c.../final/gate_table.json",
      "category": "final",
      "content_sha256": "…",
      "size_bytes": 4096,
      "created_at": "2026-08-01T10:12:00Z",
      "download_url": "https://…"
    }
  ],
  "manifest_sha256": "…"
}
```

## GET /v1/bridge/phase2/runs/{run_id}/logs

Structured stage logs derived from the journal. Every message and progress value
passes through the redaction layer. No secret, raw provider response, or private
model artifact appears.

## Client

`scripts/bridge_phase2_cloud.py` is a thin authenticated client. It installs
nothing, retrieves nothing, and trains nothing.

```bash
export OPENALPHA_API_BASE_URL=https://…
export OPENALPHA_API_TOKEN=…

python scripts/bridge_phase2_cloud.py start \
  --operator your-name --mode synthetic --source-commit "$(git rev-parse HEAD)"

python scripts/bridge_phase2_cloud.py status --run-id syn_…
python scripts/bridge_phase2_cloud.py artifacts --run-id syn_…
```
