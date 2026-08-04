# Reproduction

How run `zsb_25e0256eefb2b07a` was produced, and what a third party can verify
without access to the private object store.

This study is closed. The run ID is spent and must never be reused. Reproduction
means confirming the recorded result, not re-running for a different answer.

---

## 1. What is pinned

| Input | How it is pinned |
| --- | --- |
| Source code | Exact commit baked into the Modal image as `OPENALPHA_DEPLOYED_COMMIT` and re-verified at execution; `source_commit` must equal `deployed_commit` |
| Specification | SHA-256 of the YAML, verified inside the container before any work |
| Model / tokenizer | Hugging Face repository + revision + SHA-256 of `config.json` and `model.safetensors`, all verified before the weights load |
| Official source | Vendored at a pinned upstream revision with both as-committed and CRLF-normalized digests |
| Data | Provider, symbols, calendar, session range, and a SHA-256 of each retrieved series |
| Origins | Integer index offsets fixed in the specification, not dates chosen at run time |
| Seeds | Eight enumerated seeds, identical for every cell |
| Bootstrap | Method, block length, resample count, confidence level and seed all fixed in the specification |
| Artifact | Immutable object key + content SHA-256; publication refuses to overwrite |

---

## 2. The gate sequence that produced this run

Each gate had to pass before the next was attempted. Every remote function was
invoked exactly once.

**Gate 1 — authoritative checkout.**

```powershell
git rev-parse HEAD                      # 8a438cc5cf64462fcb04fcad995ef12ea4779246
git branch --show-current               # feature/openalpha-kronos-zero-shot-benchmark
git status --porcelain -uall            # empty, including untracked
git rev-list --left-right --count HEAD...origin/feature/openalpha-kronos-zero-shot-benchmark   # 0 0
```

**Gate 2 — deployment.** Binds the commit into the image; invokes nothing.

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
modal deploy cloud/modal/bridge_phase2_app.py
# => Step 1: ENV OPENALPHA_DEPLOYED_COMMIT=8a438cc5cf64462fcb04fcad995ef12ea4779246
```

**Gate 3 — deployment verification.** `verify_zero_shot_benchmark_deployment`
confirms the deployed commit, the benchmark specification digest, the model and
tokenizer identities, the asset panel, origin offsets, temperatures, seeds,
metrics, decision rules, namespaces, the complete moving-block design, **and
every historical mini, sealed and base specification digest**.

**Gate 4 — no prior artifact.** `inventory_zero_shot_artifacts` returned
`run_count: 0`, proving no `zsb_` object existed before this run.

**Gate 5 — remote cache.** `inventory_base_remote_cache` confirmed exactly the
two pinned repositories at the pinned revisions, 425,106,905 primary bytes,
reloaded before inspection, read-only, no deletion support.

**Gate 6 — runtime probe.** `verify_zero_shot_benchmark_runtime` loaded the
frozen weights after verifying all four asset digests, executed one synthetic
encode/generate/decode, and confirmed the parameter hash was unchanged. It
retrieves no market data and authorizes nothing, including this benchmark.

**Gate 7 — the benchmark.** One fresh run ID matching `^zsb_[0-9a-f]{8,32}$`,
then:

```text
run_zero_shot_benchmark(
    source_commit = "8a438cc5cf64462fcb04fcad995ef12ea4779246",
    run_id        = "zsb_25e0256eefb2b07a",
)
```

Invoked exactly once with `retries=0`. It checks for an existing terminal
artifact before loading any weight or issuing any provider request, runs the
benchmark, evaluates the preregistered rules, and publishes one immutable
object. Publication reported `already_existed: false`.

---

## 3. What a third party can verify from this repository alone

No credentials and no GPU required:

```bash
# 1. The specification digest the run was bound to
sha256sum research/bridge-v0/kronos-zero-shot-benchmark-v1.yaml
# 6832c0f7befc54cd7ccec382db8cac1e6314f3fb356eb5eed8e9e9eca9a6fd07

# 2. The vendored upstream source digests
sha256sum vendor/kronos/67b630e6/model/*.py

# 3. The complete analysis, decision and bootstrap code, and its test suite
uv run pytest -q
uv run ruff check .
uv run pyright
```

Every numeric result is mirrored in `results-summary.json`, generated directly
from the artifact payload rather than transcribed. The model and tokenizer
digests can be checked against the public Hugging Face revisions.

**What cannot be verified externally:** the terminal artifact lives in a private
S3-compatible bucket, so byte-level confirmation of
`93688f04ab2d887cacc56fad717d1cd2e018635b70e4a36d4e0cf480b265236a` requires
bucket access. This is stated plainly rather than glossed: the numbers here are
mirrored faithfully, but an outside reader is trusting that mirror.

---

## 4. Recomputing the confidence interval from published data

The moving-block interval can be re-derived from the repository alone, because
`results-summary.json` records all 25 origin clusters per configuration:

```python
import json
from openalpha_bridge.zero_shot.aggregation import (
    OriginCluster, paired_origin_moving_block_bootstrap,
)

summary = json.load(open("research/reports/kronos-zero-shot-benchmark/results-summary.json"))
for label, config in summary["configurations"].items():
    clusters = tuple(
        OriginCluster(
            ordinal=c["ordinal"],
            assets=tuple(c["assets"]),
            paired_differences=tuple(c["paired_differences"]),
        )
        for c in config["origin_clusters"]
    )
    interval = paired_origin_moving_block_bootstrap(
        clusters, block_length=4, seed=20260803, resamples=2000, confidence_level=0.95,
    )
    print(label, interval.lower, interval.upper, interval.excludes_zero_favorably)
```

This reproduces the published bounds exactly. The bootstrap is deterministic:
one generator seeded once from the specification, so the interval cannot be
re-rolled.

---

## 5. Determinism boundary

Forecasts are sampled, not greedy (top-p 0.9, temperatures 0.6 and 1.0).
Determinism comes from the enumerated seeds, not from greedy decoding.
Reproducing the exact token paths requires the same seeds **and** the same
numerical environment: PyTorch `2.13.0+cu126`, CUDA `12.6`, NumPy `2.5.1`, on a
Tesla T4 (capability 7.5). Different hardware or library versions may produce
different sampled paths while leaving the qualitative conclusion intact. This is
an expected boundary, not a defect.

The analysis layer is fully deterministic and hardware-independent: given the
recorded paired differences, the aggregation, bootstrap and decision rules
reproduce bit-for-bit anywhere.

---

## 6. What reproduction does not license

Confirming these numbers does not permit re-running the study under different
settings, reusing the run ID, reinterpreting the conclusion against different
thresholds, or treating the result as holdout or trading evidence. The successor
study named by the decision layer requires its own preregistration.
