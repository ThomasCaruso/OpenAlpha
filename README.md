# OpenAlpha

**Preregistered empirical research on whether a pretrained financial foundation
model produces usable forecasts.**

Three completed studies against [Kronos](https://github.com/shiyu-coder/Kronos),
a published time-series foundation model. Three negative results — including one
that falsified this project's own original thesis.

| Study | Question | Conclusion |
| --- | --- | --- |
| Kronos-mini frozen-inference diagnostic | Does the tokenizer round trip preserve OHLC structure? | `ROUNDTRIP_MATERIAL_INVALIDITY` |
| Kronos-base replication | Does the larger released model behave differently? | `ROUNDTRIP_MATERIAL_INVALIDITY` |
| Kronos-base zero-shot benchmark | Do frozen forecasts beat persistence at a paper-style horizon? | `NO_ZERO_SHOT_SKILL` |

Every conclusion was reached under decision rules fixed and cryptographically
sealed **before** the data was touched.

📊 **[Read the research reports →](research/reports/)** · 🔒 **[Verify the raw artifacts →](research/artifacts/)**

---

## What this project actually is

OpenAlpha began as a product thesis: pretrained financial models emit candles
that violate elementary structure (`high < open`, `low > close`), so a
constraint-preserving decoder — "Bridge" — would make them safe to consume.

That thesis was tested and **closed by this project's own research**. The
structural defect is real and material. But a deterministic repair that restored
*complete* structural validity changed the primary forecast metric by exactly
`0.0`. Structural validity and forecast skill turned out to be separate
properties, and the former was not the binding constraint. Both structural
studies independently recommended `ABANDON_STRUCTURAL_VALIDITY_DIRECTION`, and it
was abandoned.

A fair objection survived: those studies used one asset, one origin, and a
448 → 64 horizon far longer than the published daily benchmark. Perhaps the model
had never been asked properly. The third study asks properly — four ETFs, 25
chronological origins each, a 40 → 12 horizon, two temperatures, 1,600
generations. It does not beat a zero-return baseline either.

Bridge was never trained. That is the finding, not an omission.

---

## Headline results

### Structural validity is not forecast skill

| | Kronos-mini | Kronos-base |
| --- | --- | --- |
| Tokenizer round-trip invalid fraction | 24.61% | 12.30% |
| Target-suffix invalid fraction | 93.75% | 60.94% |
| Structurally valid rollouts | 0 / 64 | 0 / 64 |
| Forecast error vs persistence | 0.00731 vs 0.00337 | 0.00920 vs 0.00337 |
| **Improvement from full structural repair** | **0.0** | **0.0** |

Deterministic projection restored validity on all 64 candles — 11 repaired, zero
unrepairable — and the primary metric did not move.

### No zero-shot skill at a paper-style horizon

| | T=0.6 | T=1.0 | Required |
| --- | --- | --- | --- |
| Median relative skill vs persistence | −0.0417 | −0.0836 | > 0 |
| Origins beating persistence | 31% | 26% | ≥ 60% |
| Supporting assets | 0 / 4 | 0 / 4 | ≥ 3 / 4 |
| Moving-block 95% CI | [−0.00065, −0.00022] | [−0.00158, −0.00035] | excludes 0 favorably |

All four preregistered conditions failed under both temperatures. Both intervals
exclude zero on the *unfavorable* side. Three non-model baselines also failed to
beat persistence, so the window is hard rather than the model uniquely poor.

---

## Why the method matters more than the result

A negative result is only worth anything if the protocol could have produced a
positive one. The controls here are the substance of the project.

**Preregistration is enforced, not promised.** Each specification's SHA-256 is
verified *inside the execution container* before any work begins. A drifted
document fails the run closed.

**The running code is pinned to a commit.** The source commit is baked into the
container image at deploy time and re-verified at execution; `source_commit` must
equal `deployed_commit` or the run aborts.

**Statistical design was corrected twice, before execution, each time weakening
the claim it could make:**

| Revision | Resampling unit | Defect it fixed |
| --- | --- | --- |
| initial | 100 asset-origin rows | Four ETFs share windows — errors are cross-sectionally dependent |
| second | 25 origin clusters | Origins overlap: 40-session context, 12-session stride → adjacent origins share 28 sessions |
| final | moving blocks of 4 clusters | Block length `ceil(40/12)` spans every overlapping context relationship |

Each correction *widened* the confidence interval. Both superseded estimators
were deleted from the codebase, so no weaker interval can reach the decision
layer — a structural guarantee, not a convention.

**Leakage is prevented structurally.** Normalization is refit per origin from
context rows only. Baselines take `(context, target_sessions)` — a target row
cannot arrive through the signature. No holdout was ever opened.

**Failures are typed and sanitized.** An operational failure records the
exception class and a fixed message, never exception text, which can carry
credentials or provider responses. A test plants a fake credential in an
exception and asserts it appears in neither the artifact nor the logs.

**Results cannot be quietly revised.** Write-once artifacts, `retries=0`, run IDs
spent under every outcome including failure. Duplicate invocation loads no
weights and issues no provider request.

---

## Repository tour

```
packages/bridge/      Research engine: diagnostics, zero-shot benchmark,
                      metrics, bootstrap, decision rules, artifact publication
packages/sentinel/    Structural validation and audit layer
packages/research-core, packages/experiment-spec
cloud/modal/          GPU execution shells — thin wrappers; all logic is importable
research/bridge-v0/   Preregistrations, sealed with .sha256 sidecars
research/reports/     ← the completed studies and their evidence
research/sentinel-*/  Earlier investigations and their immutable artifacts
docs/                 Architecture, ADRs, methodology, data policy
vendor/kronos/        Two upstream files, verbatim, for offline conformance checks
```

| | |
| --- | --- |
| Source | ~44,000 lines |
| Tests | ~24,000 lines · **1,549 passing** |
| Static analysis | `ruff` clean · `pyright` clean |
| Stack | Python 3.13 · PyTorch 2.13 · Modal (T4) · S3-compatible storage · `uv` |

---

## Quick start

No GPU, no credentials, and no model weights are needed to run everything that
does not touch a checkpoint.

```bash
uv sync --locked --group dev
uv run pytest -q          # 1,549 tests
uv run ruff check .
uv run pyright
```

Verify a sealed preregistration for yourself:

```bash
sha256sum research/bridge-v0/kronos-zero-shot-benchmark-v1.yaml
# 6832c0f7befc54cd7ccec382db8cac1e6314f3fb356eb5eed8e9e9eca9a6fd07
```

Executing a study additionally requires a Modal account and object-store
credentials. See [`docs/BRIDGE_MODAL_DEPLOYMENT.md`](docs/BRIDGE_MODAL_DEPLOYMENT.md)
and [`docs/BRIDGE_GPU_RUNBOOK.md`](docs/BRIDGE_GPU_RUNBOOK.md).

---

## Reproducibility and its limits

Pinned for every study:

```
Kronos source        67b630e67f6a18c9e9be918d9b4337c960db1e9a
Kronos-base          2b554741eca47781b64468546e77fef3e85130e6
Kronos-Tokenizer-base 0e0117387f39004a9016484a186a908917e22426
Kronos-mini          f4e68697d9d5aed55cef5c96aabc3376bcad9f81
Kronos-Tokenizer-2k  26966d0035065a0cae0ebad7af8ece35bc1fb51c
```

Two of the three immutable terminal artifacts are committed **byte for byte** in
[`research/artifacts/`](research/artifacts/) — the exact object-store bodies, not
summaries. Verify them yourself:

```bash
python scripts/verify_artifacts.py     # recomputes SHA-256 against the published digests
```

The Kronos-mini artifact is recorded but not committed; its body was not retained
locally and retrieving it needs object-store credentials. Its key and digest are
in [`research/artifacts/manifest.json`](research/artifacts/manifest.json).

Every study additionally mirrors its **complete** numeric results into
`results-summary.json`, generated directly from the artifact payload rather than
transcribed. All three studies are verifiable at the level of reported numbers;
two of three are verifiable at the byte level. That line is drawn explicitly in
every report rather than glossed over.

Raw provider data, caches and checkpoints are never committed.

---

## What this research does not establish

- **Not** that Kronos is universally broken, or useless at other horizons, assets
  or frequencies.
- **Not** a contradiction of published Kronos results — the paper's evaluation
  protocol was not replicated.
- **Not** that Kronos would fail after task-specific fine-tuning.
- **Not** that Kronos internal representations lack usable information — direct
  generation and representation quality are different hypotheses, and only the
  first was tested.
- **Not** a trading, profitability, or economic claim. There is no backtest, no
  transaction-cost model, and no execution assumption anywhere in this
  repository.
- **Not** a held-out result. Every finding is development evidence; no untouched
  partition has been opened.

Every authorization field in every artifact is `false`.

---

## Status and what would come next

The structural direction is **closed**. The zero-shot generation direction is
**stopped** by its own preregistered rule.

One hypothesis survives: a model can generate poorly and still encode useful
internal state. The preregistered decision therefore routes to a
frozen-representation probe. A design note exists at
[`research/bridge-v0/kronos-frozen-representation-probe-design.md`](research/bridge-v0/kronos-frozen-representation-probe-design.md),
including a source audit confirming that `Kronos.decode_s1` already returns the
transformer hidden state publicly — so the probe would need no upstream
modification.

It is **not implemented and not authorized**. It would require its own
preregistration.

---

## Tooling disclosure

Research and implementation were aided by Codex and Claude Code. All
preregistrations, thresholds, decision rules, seeds and origin definitions were
fixed and hashed before execution; every result is bound to a pinned commit and
specification digest recorded inside its own artifact, and the full analysis code
and test suite are in this repository for inspection.

---

## License

[MIT](LICENSE). Third-party attributions are in [NOTICE](NOTICE): the two
vendored Kronos files are MIT, © 2025 ShiYu. Model weights are not redistributed —
they are fetched at pinned revisions from the Hugging Face Hub and remain subject
to their upstream terms.
