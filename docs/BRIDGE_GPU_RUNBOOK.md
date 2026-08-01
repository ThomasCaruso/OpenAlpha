# Bridge Phase 2 GPU Runbook

One operator, one GPU host, one consistent environment, from checkout to the
final gate decision.

Read [BRIDGE_TEST_OPENING_POLICY.md](BRIDGE_TEST_OPENING_POLICY.md) before
running. The reconstruction-test partition opens exactly once.

## 1. Host requirements

| Requirement | Value |
|---|---|
| Operating system | Linux (Ubuntu 22.04 verified) or Windows |
| Python | 3.13 |
| Accelerator | exactly one CUDA device |
| VRAM | at most 25,769,803,776 bytes (24 GiB) |
| Free disk | at least 40 GiB |
| CUDA runtime | 12.4 |
| Network | Yahoo Finance and huggingface.co reachable |

Compute budget from the lock: at most 24 GPU hours for Phase 2, one accelerator.

### Disk breakdown

| Item | Approximate size |
|---|---|
| Torch and CUDA wheels | 3 GiB |
| Pinned Kronos Tokenizer-2k assets | under 1 GiB |
| Feature cache | up to 10 GiB (hard cap, enforced) |
| Run artifacts and journals | under 100 MiB |
| Headroom | remainder |

The feature-cache cap is enforced in code at 10,737,418,240 bytes.

## 2. Choose an environment

### Option A: Docker with a pinned CUDA runtime (recommended)

```bash
git clone <repository> openalpha && cd openalpha
git checkout feature/openalpha-kronos-bridge

docker build -f docker/bridge-phase2.Dockerfile -t openalpha-bridge-phase2:latest .
```

### Option B: locked uv environment on the host

```bash
curl -LsSf https://astral.sh/uv/0.11.19/install.sh | sh
uv python install 3.13
uv sync --locked --group dev          # base: no Torch
uv sync --locked --extra bridge-gpu   # adds Torch, hub client, provider client
```

`bridge-gpu` installs the CUDA build of Torch. On a host whose CUDA runtime is
not 12.4, install the matching Torch wheel explicitly before syncing the extra.

## 3. Configure

```bash
cp docker/bridge-phase2.env.template /etc/openalpha/phase2.env   # outside the repo
$EDITOR /etc/openalpha/phase2.env                                # set OPENALPHA_OPERATOR
set -a && . /etc/openalpha/phase2.env && set +a
```

No secret is required. The locked provider is a public interface and
Tokenizer-2k is a public repository. Never place the run or cache directory
inside the Git worktree; preflight refuses.

## 4. Preflight

```bash
scripts/bridge_phase2_gpu_preflight.sh "$OPENALPHA_RUN_DIR" "$OPENALPHA_CACHE_DIR"
```

Windows hosts:

```powershell
scripts\bridge_phase2_gpu_preflight.ps1 -RunDir $env:OPENALPHA_RUN_DIR -CacheDir $env:OPENALPHA_CACHE_DIR
```

Preflight verifies the operating system, Python version, Torch import, CUDA
availability, exactly one accelerator, VRAM against the lock, free disk, the
cache location, tracked-artifact hygiene, the source commit, the worktree state,
and the experiment and amendment hashes. It stops **before** any provider access
when a hard requirement fails.

Exit 0 means proceed. Exit 2 prints a typed blocker: `NO_ACCELERATOR`,
`MISSING_OPTIONAL_DEPENDENCY`, `INSUFFICIENT_DISK`, `CACHE_INSIDE_GIT`,
`ACCELERATOR_COUNT_MISMATCH`, `VRAM_ABOVE_LOCK`, or `EXPERIMENT_HASH_MISMATCH`.

## 5. Run

```bash
scripts/bridge_phase2_run.sh "$OPENALPHA_RUN_DIR" "$OPENALPHA_CACHE_DIR" "$OPENALPHA_OPERATOR"
```

This executes the locked sequence: preflight, retrieval, validation, windowing,
coverage audit, asset resolution, Stage A, Stage B, Stage C, checkpoint freeze,
the one-time test opening, test evaluation, external evaluation, and the final
gate decision.

To stop before the held-out partition, run the stages individually:

```bash
python -m openalpha_bridge.phase2 stage-a --run-dir "$OPENALPHA_RUN_DIR" --cache-dir "$OPENALPHA_CACHE_DIR" --json
python -m openalpha_bridge.phase2 stage-b --run-dir "$OPENALPHA_RUN_DIR" --cache-dir "$OPENALPHA_CACHE_DIR" --json
python -m openalpha_bridge.phase2 stage-c --run-dir "$OPENALPHA_RUN_DIR" --cache-dir "$OPENALPHA_CACHE_DIR" --json
```

No stage command opens the test partition. Only `run --open-test-partition`
does, and only once.

## 6. Resume after interruption

```bash
scripts/bridge_phase2_resume.sh "$OPENALPHA_RUN_DIR" "$OPENALPHA_CACHE_DIR" "$OPENALPHA_OPERATOR"
```

Resume continues from the last verified state in `journal.json`. Interrupted
cache writes are recovered by deleting partial temporaries; committed shards are
content-verified on read. If the configuration changed, resume fails with
`RUN_IDENTITY_CHANGED` instead of continuing under different settings.

A run that already opened the test partition cannot reopen it. Resuming past
that point reuses the sealed record.

## 7. Verify and export

```bash
python -m openalpha_bridge.phase2 verify --run-dir "$OPENALPHA_RUN_DIR" --cache-dir "$OPENALPHA_CACHE_DIR" --json
scripts/bridge_phase2_export.sh "$OPENALPHA_RUN_DIR" ./phase2-export
```

Export copies manifests, hashes, aggregates, the gate table, the journal, and
the test-opening record. It refuses to export weights, shards, or raw candles
and writes `SHA256SUMS`.

## 8. Cleanup

```bash
python -c "
from pathlib import Path
from openalpha_bridge.phase2.cache import FeatureCache
import os
print(FeatureCache(Path(os.environ['OPENALPHA_CACHE_DIR'])).cleanup(), 'shards removed')
"
```

The lock requires automatic cache cleanup after manifests and hashes are sealed.
Raw provider data and checkpoints are never committed to Git.

## 9. Expected failure modes

| Symptom | Blocker | Action |
|---|---|---|
| `import torch` fails | `MISSING_OPTIONAL_DEPENDENCY` | `uv sync --locked --extra bridge-gpu` |
| No CUDA device | `NO_ACCELERATOR` | Run on a GPU host; Stage C on CPU is prohibited |
| Two or more GPUs visible | `ACCELERATOR_COUNT_MISMATCH` | Set `CUDA_VISIBLE_DEVICES=0` |
| VRAM above the lock | `VRAM_ABOVE_LOCK` | Use a smaller accelerator or amend the lock |
| Cache under the repository | `CACHE_INSIDE_GIT` | Move the cache outside the worktree |
| Lock file drift | `EXPERIMENT_HASH_MISMATCH` | Restore the committed experiment and amendments |
| Second test opening | `TEST_PARTITION_ALREADY_OPENED` | Expected; the record is immutable |

## 10. After the run

The terminal conclusion comes from `gate_table.json`, not from prose. Record the
conclusion, the gate table, the checkpoint hash, and the test-opening record in
`research/bridge-v0/`, and update `docs/STATUS.md`.

Phase 3 forecast integration requires Phase 2 success and remains unauthorized
until then.
