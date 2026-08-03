# Reproduction

How the two completed structural-validity results were produced, and how a third party would
re-derive them. Both studies are closed: this document describes reproduction, **not** re-running
for a different answer. Neither study may be amended, retried, reopened, reinterpreted or
optimized, and the recorded run IDs are spent and must never be reused.

---

## 1. What is fixed

Reproduction is meaningful only because every input is pinned:

| Input | How it is pinned |
| --- | --- |
| Source code | Exact git commit, baked into the Modal image as `OPENALPHA_DEPLOYED_COMMIT` at deploy time and re-verified by the deployment gate. |
| Specification | SHA-256 of the specification YAML, verified before execution. |
| Model and tokenizer | Hugging Face repository plus revision, plus SHA-256 of both `config.json` and `model.safetensors`, verified before the weights are loaded. |
| Official model source | Vendored at a pinned upstream revision, with both as-committed and CRLF-normalized SHA-256 digests verified at image build. |
| Data | Provider, symbol, frequency, calendar, session range, and a SHA-256 of the retrieved candle matrix. |
| Seeds | Enumerated in the specification, not drawn at runtime. |
| Thresholds and decision rules | Enumerated in the specification before execution. |
| Artifact | Immutable object key plus content SHA-256; publication refuses to overwrite. |

## 2. Kronos-base study — exact procedure

The Kronos-base run `base_03b08cbc706193d6` was produced by the following gate sequence. Each gate
had to pass before the next was attempted, and every remote function was invoked exactly once.

**Gate 1 — authoritative checkout.**

```powershell
git rev-parse HEAD                       # 0e6193baabd747dc0b2610a0ab80c3320e32fdd0
git status --porcelain -uall             # empty, including untracked
git branch --show-current                # feature/openalpha-kronos-base-diagnostic
git rev-parse origin/feature/openalpha-kronos-base-diagnostic
git rev-list --left-right --count HEAD...origin/feature/openalpha-kronos-base-diagnostic   # 0 0
```

**Gate 2 — deployment.** The image binds the commit; no function is invoked by deploying.

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
modal deploy cloud/modal/bridge_phase2_app.py
# => Step 1: ENV OPENALPHA_DEPLOYED_COMMIT=0e6193baabd747dc0b2610a0ab80c3320e32fdd0
```

**Gate 3 — deployment verification.** `verify_base_deployment`, invoked once. Confirms the deployed
commit, experiment identity, base specification digest, model and tokenizer identities with config
and weight digests, the context budget (448 + 64 = 512, exact fill, no truncation), all eight
historical mini/sealed specification digests, and the remote cache volume name.

**Gate 4 — remote cache inventory.** `inventory_base_remote_cache`, invoked once. Confirms the
volume reloaded before inspection, is read-only, does not support deletion, and contains exactly
the two pinned repositories at the pinned revisions with only `config.json` and `model.safetensors`
in each snapshot.

**Gate 5 — runtime probe.** `verify_base_frozen_inference_runtime`, invoked once. Loads the frozen
weights, verifies all four asset digests first, performs a minimal non-scientific forward pass, and
confirms parameter hashes are unchanged. It authorizes nothing:
`authorizes_the_base_diagnostic = false`.

**Gate 6 — post-probe cache confirmation.** `inventory_base_remote_cache`, invoked once again, and
compared field-by-field against Gate 4 to prove no weight-sized duplicate appeared.

**Gate 7 — the diagnostic.** One fresh run ID matching `^base_[0-9a-f]{8,32}$` is generated, then:

```text
kronos_base_frozen_inference_diagnostic(
    source_commit = "0e6193baabd747dc0b2610a0ab80c3320e32fdd0",
    run_id        = "base_03b08cbc706193d6",
)
```

Invoked exactly once, with `retries = 0`. It checks for an existing terminal artifact before doing
any work, runs Methods A through D, evaluates the preregistered rules, and publishes one immutable
terminal object. Publication reported `already_existed: false`, confirming the run ID was unspent.

## 3. Verifying the recorded results without re-running

A third party with read access to the object store can verify the results without any GPU, any
model weights and any provider call:

1. Fetch the object at the key recorded in `artifact-manifest.json`.
2. Compute its SHA-256 and compare against the recorded `artifact_sha256`.
3. Compare the numeric fields against `results-summary.json`.

The digests are:

```text
openalpha-compatibility/bridge-phase2/runs/canary_0a92fde788bd685c/frozen-inference-diagnostic/frozen_inference_diagnostic_terminal.json
  8e3d8a333c11c1939c5d182db73012fc3e0fc01da42368eb23f71661aadab18c

openalpha-compatibility/kronos-base-diagnostic/runs/base_03b08cbc706193d6/kronos_base_diagnostic_terminal.json
  84ec1b197d4bf7a3b8f35a38d0feb8d5be086b61de23df0924d83ec273c731e6
```

## 4. Internal reproducibility check

The Kronos-base diagnostic contains its own reproducibility test, and it passed. Method B and
Method D rollout zero share a seed (`20150507`) and settings, and are computed through independent
code paths. The artifact records `performed: true`, `agrees: true`, `detail: "identical"`, with
agreement on coarse tokens, fine tokens, the raw decoded suffix, forecast metrics, sampling
log-probabilities and validity.

This was independently confirmed from the returned payload: rollout zero's coarse and fine token ID
sequences are element-wise identical to Method B's, and both report
`close_return_mae = 0.00920398038369002`.

## 5. Determinism boundary

The forecast is sampled, not greedy — temperature `1.0`, top-p `0.9`, top-k `0`. Determinism comes
from the enumerated seeds, not from greedy decoding. Reproduction therefore requires the same seeds
**and** the same numerical environment: PyTorch `2.13.0+cu126`, CUDA `12.6`, NumPy `2.5.1`, on a
Tesla T4 (capability 7.5). Different GPU architectures or library versions may produce different
sampled paths while leaving the qualitative conclusions intact. This is a known and expected
boundary, not a defect.

## 6. What reproduction cannot re-open

Reproduction confirms the recorded numbers. It does not license:

- amending or re-running either study under different settings;
- reusing either run ID;
- reinterpreting the conclusions against different thresholds;
- creating another structural repair experiment.

The successor work is a separately preregistered study with a disjoint identity, described in
`report.md` §7.
