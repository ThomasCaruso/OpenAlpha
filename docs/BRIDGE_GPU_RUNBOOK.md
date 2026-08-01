# Bridge Phase 2 Cloud Runbook

> **This document replaced the manually administered GPU-host runbook.**
> There is no GPU host to rent, configure, or SSH into. Phase 2 is an
> API-operated cloud job. If you are looking for `scripts/bridge_phase2_run.sh`
> or `uv sync --extra bridge-gpu` on a workstation, those paths were removed;
> see [BRIDGE_CLOUD_ARCHITECTURE.md](BRIDGE_CLOUD_ARCHITECTURE.md).

Read [BRIDGE_TEST_OPENING_POLICY.md](BRIDGE_TEST_OPENING_POLICY.md) before a
real run. The reconstruction-test partition opens exactly once, ever.

## Prerequisites

One-time setup in [BRIDGE_MODAL_DEPLOYMENT.md](BRIDGE_MODAL_DEPLOYMENT.md):
Modal and Cloudflare R2 accounts, three Modal secrets, four GitHub
secrets/variables, and one deployment.

On your machine you need only `git`, `python`, and a bearer token. No CUDA, no
Torch, no `nvidia-smi`, no Linux host, no local cache, no `/etc` files.

## Step 1 — Deploy

Push, or dispatch **Deploy Bridge Phase 2 Cloud**. The workflow verifies hashes
and tests, deploys the Modal app, and confirms the built image carries the
locked experiment. It never starts an empirical run.

## Step 2 — Synthetic validation

Always run this first. It exercises the real state machine, journal, lease,
cloud test gate, and artifact path against fake components, contacts no provider
API, and loads no Kronos asset.

```bash
python scripts/bridge_phase2_cloud.py start \
  --operator your-name --mode synthetic --source-commit "$(git rev-parse HEAD)"
```

Expect a `syn_…` run ID. Poll until `FINALIZED`:

```bash
python scripts/bridge_phase2_cloud.py status --run-id syn_…
```

Its artifacts live under `openalpha-synthetic/…` and are labelled
**SYNTHETIC CLOUD PIPELINE VALIDATION - NOT EMPIRICAL EVIDENCE**.

## Step 3 — The real run

This retrieves market data, downloads pinned Kronos assets, trains the locked
head, and — if confirmed — opens the reconstruction-test partition once and
seals it forever. Threshold changes after this point are prohibited.

Via GitHub (preferred, because the confirmations are explicit):

dispatch **Start Bridge Phase 2 Run** with

- `operator`: your name
- `experiment_hash`: `d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2`
- `execution_mode`: `real`
- `confirm_real_evidence`: checked
- `confirm_open_test_partition`: checked only when you intend the one-time gate

Via the client:

```bash
python scripts/bridge_phase2_cloud.py start \
  --operator your-name --mode real \
  --source-commit "$(git rev-parse HEAD)" \
  --confirm-real-evidence \
  --confirm-open-test-partition \
  --idempotency-key "phase2-first-real-run"
```

Creation returns immediately with a `run_id`. The experiment continues in the
background; no connection is held open.

To stop before the held-out partition, omit `--confirm-open-test-partition`. The
run trains, freezes the checkpoint, and stops at `BLOCKED` with
`TEST_OPENING_NOT_CONFIRMED`, leaving the test sealed and the run resumable.

## Step 4 — Monitor

```bash
python scripts/bridge_phase2_cloud.py status --run-id run_…
python scripts/bridge_phase2_cloud.py logs   --run-id run_…
```

Status shows current state and stage, completed windows, Stage A and Stage B
results, Stage C epoch and validation progress, checkpoint-selection state,
whether the test is still sealed, evaluation progress, the final gate result,
and typed blocker or failure codes. It never exposes raw market data, weights,
tokens, or credentials.

## Step 5 — Resume after interruption

```bash
python scripts/bridge_phase2_cloud.py resume --run-id run_… --operator your-name
```

Permitted only from a verified `BLOCKED` or resumable `FAILED` state and only
when no live lease is held. Resume continues from the last verified journal
entry and the latest checkpoint. It never recreates the test-opening record.

## Step 6 — Retrieve results

```bash
python scripts/bridge_phase2_cloud.py artifacts --run-id run_…
```

Returns the compact manifest with content hashes and signed download URLs.
Weights, feature shards, and raw provider data are filtered out. Nothing is
copied between machines by hand.

The terminal conclusion comes from `final/gate_table.json`, not from prose.

## Failure modes

| Blocker | Meaning | Action |
|---|---|---|
| `UNAUTHORIZED` | bad or missing bearer token | check `OPENALPHA_API_TOKEN` |
| `UNKNOWN_EXPERIMENT_HASH` | wrong experiment hash | use the locked value |
| `REAL_EVIDENCE_NOT_CONFIRMED` | missing confirmation | pass the flag deliberately |
| `RATE_LIMIT_EXCEEDED` | too many runs this hour | wait |
| `RUN_LEASE_HELD` | another worker owns the run | wait for expiry, then resume |
| `TEST_OPENING_NOT_CONFIRMED` | stopped before the gate | expected; resume when ready |
| `TEST_PARTITION_ALREADY_OPENED` | the gate is one-time | expected; cannot be redone |
| `GPU_HOUR_BUDGET_EXCEEDED` | past the 24-hour lock | investigate before rerunning |
| `NO_ACCELERATOR` | worker lacked a GPU | redeploy; check `gpu=` |

## After the run

Record the conclusion, gate table, checkpoint hash, and test-opening record in
`research/bridge-v0/`, and update `docs/STATUS.md`. Phase 3 forecast integration
requires Phase 2 success and remains unauthorized until then.
