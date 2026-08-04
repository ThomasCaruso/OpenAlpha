# Research Reports

Completed, preregistered studies. Each was sealed before execution, executed
exactly once against pinned public checkpoints, and published to an immutable
terminal artifact.

## The three studies

| # | Study | Question | Conclusion | Report |
| --- | --- | --- | --- | --- |
| 1 | Kronos-mini frozen-inference diagnostic | Does the tokenizer round trip preserve OHLC structure, and does repairing it help? | `ROUNDTRIP_MATERIAL_INVALIDITY` | [structural-validity](kronos-structural-validity/) |
| 2 | Kronos-base replication | Does the larger released model behave differently? | `ROUNDTRIP_MATERIAL_INVALIDITY` | [structural-validity](kronos-structural-validity/) |
| 3 | Kronos-base zero-shot benchmark | Do frozen zero-shot forecasts beat persistence at a paper-style horizon? | `NO_ZERO_SHOT_SKILL` | [zero-shot-benchmark](kronos-zero-shot-benchmark/) |

All three are negative results. All three were preregistered before the data was
touched, and all three ran to their preregistered conclusion rather than being
stopped early or reinterpreted.

## How they connect

Studies 1 and 2 tested whether Kronos output is *structurally* usable — whether
decoded candles satisfy the OHLC inequalities. Both found material invalidity.
Critically, study 2 also showed that a deterministic repair restoring **complete**
structural validity changed the primary forecast metric by exactly `0.0`.

That closed the structural direction. Structural validity and forecast skill are
separate properties, and the former was not the binding constraint. Both studies
recommended `ABANDON_STRUCTURAL_VALIDITY_DIRECTION`, and it was abandoned.

But those studies used one asset, one origin, and a 448 → 64 horizon far longer
than the published daily benchmark. That left a fair objection: perhaps the model
was never asked properly. Study 3 asks properly — four assets, 25 chronological
origins each, a 40 → 12 horizon, two temperatures, 1,600 generations. It does not
beat a zero-return baseline.

```
Study 1 (mini)  ─┐
                 ├─→ structural invalidity is real, but repairing it
Study 2 (base)  ─┘   changes forecast error by exactly zero
                         │
                         ▼
                 direction abandoned; new question:
                 was the horizon simply wrong?
                         │
                         ▼
Study 3 (zero-shot benchmark) ─→ no skill vs persistence
                         │
                         ▼
                 generation direction stopped;
                 representation probe permitted (not run)
```

## Shared discipline

Every study in this directory follows the same rules, enforced in code rather
than by convention:

- **Preregistration is sealed.** The specification's SHA-256 is verified inside
  the execution container. A drifted document fails the run closed.
- **The commit is bound.** The source commit is baked into the image and
  re-verified at execution; `source_commit` must equal `deployed_commit`.
- **One run, one artifact.** Write-once objects, `retries=0`, run IDs spent under
  every outcome including failure. Duplicate invocation loads no weights and
  issues no provider request.
- **Failures are typed and preserved.** An operational failure records the
  exception class and a fixed message — never exception text, which can carry
  credentials or provider responses.
- **Namespaces are disjoint.** `canary_`, `base_` and `zsb_` are pairwise
  disjoint by construction; no run ID of one study can name an object of another.
- **Nothing authorizes anything.** Every authorization field in every artifact is
  `false`. No holdout was opened, no optimizer constructed, no backtest run.

## Verifiability

Two of the three terminal artifacts are committed **byte for byte** under
[`../artifacts/`](../artifacts/), as the exact object-store bodies. Recompute
their digests yourself:

```bash
python scripts/verify_artifacts.py
```

The Kronos-mini artifact is recorded but not committed — its body was not
retained locally and retrieving it needs object-store credentials. Its key and
digest are in [`../artifacts/manifest.json`](../artifacts/manifest.json), and its
numeric results are mirrored like every other study's.

Each report also mirrors its complete numeric results into
`results-summary.json`, generated directly from the artifact payload rather than
transcribed by hand.

So: all three studies are verifiable at the level of reported numbers, and two of
three are verifiable at the byte level. That boundary is stated in each report
rather than glossed.

## Boundary

Everything here is **development evidence**. Not holdout evidence, not trading
evidence, not a profitability claim. No untouched partition has been opened.
