# Bridge Phase 2 Modal Deployment

One-time setup, then everything is API-operated. You never install CUDA or
Torch, never administer a host, and never copy artifacts by hand.

## What you need

| Account | Why | Cost |
|---|---|---|
| [Modal](https://modal.com) | managed compute, GPU worker, control API | free monthly credits cover a full run |
| [Cloudflare R2](https://developers.cloudflare.com/r2/) | S3-compatible artifact store | 10 GiB free, no egress fees |
| GitHub | deployment and manual run trigger | existing |

No AWS account. No Kubernetes. No Terraform.

## 1. Create the R2 bucket

Create a bucket (for example `openalpha-artifacts`) and an API token with
read/write access. Note the account endpoint,
`https://<account-id>.r2.cloudflarestorage.com`.

## 2. Create Modal secrets

Three secrets, by name. Values live only in Modal and never appear in code,
requests, journals, logs, artifacts, or workflow inputs.

```bash
pip install modal
modal token new

modal secret create openalpha-api \
  OPENALPHA_API_TOKEN=<generate-a-long-random-token> \
  OPENALPHA_API_BASE_URL=<filled-in-after-first-deploy>

modal secret create openalpha-storage \
  OPENALPHA_ARTIFACT_BUCKET=openalpha-artifacts \
  OPENALPHA_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com \
  OPENALPHA_S3_REGION=auto \
  AWS_ACCESS_KEY_ID=<r2-access-key-id> \
  AWS_SECRET_ACCESS_KEY=<r2-secret-access-key>

# Tokenizer-2k is public, so the token may be omitted entirely.
modal secret create openalpha-huggingface HUGGING_FACE_HUB_TOKEN=""
```

Every one of these names is registered with the redaction layer, so a value that
somehow reaches a log line is scrubbed before it is written.

## 3. Add GitHub secrets

In repository settings, environment `bridge-phase2-cloud`:

| Kind | Name | Value |
|---|---|---|
| Secret | `MODAL_TOKEN_ID` | from `modal token new` |
| Secret | `MODAL_TOKEN_SECRET` | from `modal token new` |
| Secret | `OPENALPHA_API_TOKEN` | same token as the Modal secret |
| Variable | `OPENALPHA_API_BASE_URL` | deployed endpoint base URL |

Provider and storage credentials are never GitHub secrets and never workflow
inputs; only Modal holds them.

## 4. Deploy

Push to `main` or `feature/openalpha-kronos-bridge`, or dispatch **Deploy Bridge
Phase 2 Cloud** manually.

The workflow verifies the locked experiment and amendment hashes, runs the base
suite, Ruff, and Pyright, records the dependency-lock hash, deploys the Modal
app, and then calls `verify_deployment` to confirm the built image carries the
locked experiment. Deployment fails if the image and the committed locks
disagree.

**Deployment never starts an empirical run.**

To deploy by hand instead:

```bash
modal deploy cloud/modal/bridge_phase2_app.py
modal run cloud/modal/bridge_phase2_app.py::verify_deployment
```

Copy the printed base URL into the `openalpha-api` secret and the GitHub
variable.

## 5. The image

Code-defined and built by Modal. You never build Docker locally.

| Pin | Value |
|---|---|
| Python | 3.13 |
| Torch | 2.5.1 (CUDA 12.4 index) |
| GPU | T4 |
| numpy / pydantic | `>=2.5,<3` / `>=2.11,<3` |
| Provider | `yfinance==1.5.2` |
| Hub / weights | `huggingface-hub>=0.34,<1`, `safetensors>=0.4,<1` |
| Storage / API | `boto3>=1.35,<2`, `fastapi>=0.115,<1` |

`docker/bridge-phase2.Dockerfile` remains only as a reproducibility reference.
It is no longer an execution path.

## 6. Start a run

Synthetic first — it proves deployment, spawning, polling, journals, resume, the
cloud gate, and artifact retrieval, and contacts no provider API.

Via GitHub: dispatch **Start Bridge Phase 2 Run** with `execution_mode:
synthetic`.

Via the client:

```bash
export OPENALPHA_API_BASE_URL=https://…
export OPENALPHA_API_TOKEN=…

python scripts/bridge_phase2_cloud.py start \
  --operator your-name --mode synthetic --source-commit "$(git rev-parse HEAD)"
```

For the real run, see [BRIDGE_GPU_RUNBOOK.md](BRIDGE_GPU_RUNBOOK.md).

## 7. Cost and guards

The Bridge head is 17,605 parameters. The GPU exists only for frozen Kronos
forward passes over roughly 2,900 sequences: minutes of T4 time, a few dollars
at most, inside Modal's free credits.

Enforced in code: one GPU, 24 GPU-hour ceiling, 10 GiB feature-cache cap, 25 MiB
checkpoint cap, bounded provider requests and retries, GPU worker retries
disabled so a platform retry cannot duplicate a scientific run, and automatic
shutdown at a terminal state. No idle GPU worker is kept alive.

## Troubleshooting

| Symptom | Meaning |
|---|---|
| `UNAUTHORIZED` | bearer token missing or wrong |
| `UNKNOWN_EXPERIMENT_HASH` | `experiment_hash` is not the locked value |
| `REAL_EVIDENCE_NOT_CONFIRMED` | real runs need `confirm_real_evidence` |
| `RUN_LEASE_HELD` | another worker holds the run; wait for expiry |
| `TEST_PARTITION_ALREADY_OPENED` | expected — the gate is one-time |
| `MISSING_OPTIONAL_DEPENDENCY` | the image is missing an extra; redeploy |
| `NO_ACCELERATOR` | worker scheduled without a GPU; check `gpu=` |
