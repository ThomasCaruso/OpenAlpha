# Bridge Phase 2 Cloud Storage

Two distinct storage classes. A workstation directory is never the source of
truth.

## A. Reusable cloud cache — Modal Volume

Volume `openalpha-bridge-phase2-cache`, mounted at `/cache` in the worker.

| Path | Contents |
|---|---|
| `/cache/huggingface` | pinned Kronos asset cache (`HF_HOME`) |
| `/cache/features/{run_id}` | derived frozen-feature cache |
| `/cache/runs/{run_id}` | resumable working state |

Properties: outside Git, content-addressed shard names, atomic writes with
interrupted-write recovery, partition-separated directories, explicit cleanup,
and a **10 GiB hard cap enforced in code** (`MAXIMUM_CACHE_BYTES`). Different
experiment hashes never share a namespace.

A cache hit still re-verifies the pinned Kronos config, weights, and codebook
hashes before use.

## B. Immutable run artifact store — S3-compatible

Cloudflare R2 by default: S3 API, 10 GiB free, no egress fees, no AWS account.
Any S3-compatible endpoint works via `OPENALPHA_S3_ENDPOINT_URL`.

### Key layout

```
openalpha/
  bridge-phase2/
    runs/
      {run_id}/
        identity/     run_identity.json, lease.json
        journal/      000001.json, 000002.json, …
        stages/       preflight.json, assets.json, …
        checkpoint/   manifest.json
        evaluation/   test_opening_record.json
        final/        gate_table.json, report.md
    experiments/
      {experiment_hash}/
```

Synthetic runs use a **separate root**, `openalpha-synthetic/bridge-phase2/…`,
so synthetic evidence can never appear under the real namespace. This is
asserted by test.

### Object metadata

Every object records schema version, run ID, experiment hash, content hash,
prior journal hash where applicable, creation timestamp, and evidence class.
Reads verify the content hash and fail with `OBJECT_HASH_MISMATCH` on drift.

### Conditional writes

`put_immutable` uses `IfNoneMatch: "*"`, which R2 and S3 both honour. A key that
already exists fails with `OBJECT_ALREADY_EXISTS`. This single primitive
provides three guarantees:

1. **One-time test gate.** `evaluation/test_opening_record.json` can be created
   exactly once per run. No resume can recreate it.
2. **Append-only journal.** Each entry is its own key `journal/{sequence}.json`
   chained by `prior_entry_sha256`. Two workers at the same sequence cannot both
   win; the loser gets `CONCURRENT_JOURNAL_WRITE`.
3. **Exclusive lease.** `identity/lease.json` is acquired by conditional create.
   A live lease held by another worker fails with `RUN_LEASE_HELD`; an expired
   lease may be taken over so an interrupted worker never strands a run.

### What is never stored in the artifact store

Raw provider responses, complete reusable historical datasets, feature-cache
shards, and model weights. Those live in the cache volume or nowhere.
`GET /artifacts` additionally filters weights, shards, raw data, and the lease
out of every manifest.

### What is never stored in Git

Raw market data, checkpoints, Kronos assets, feature caches, and secrets. Git
holds manifests, hashes, aggregates, configurations, and reports only.

## Cleanup

The locked experiment requires automatic cache cleanup once manifests and hashes
are sealed. The cache exposes `cleanup()` per partition or for the whole run.
The immutable artifact store is never cleaned: it is the durable record.
