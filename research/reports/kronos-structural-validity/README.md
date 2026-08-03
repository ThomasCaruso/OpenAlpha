# Kronos Structural Validity — Research Report

This directory contains the publication-ready technical report for two completed,
preregistered frozen-inference structural-validity diagnostics:

| Study | Model pair | Run ID | Conclusion |
| --- | --- | --- | --- |
| Kronos-mini | `NeoQuasar/Kronos-mini` + `NeoQuasar/Kronos-Tokenizer-2k` | `canary_0a92fde788bd685c` | `ROUNDTRIP_MATERIAL_INVALIDITY` |
| Kronos-base | `NeoQuasar/Kronos-base` + `NeoQuasar/Kronos-Tokenizer-base` | `base_03b08cbc706193d6` | `ROUNDTRIP_MATERIAL_INVALIDITY` |

Both studies recommended `ABANDON_STRUCTURAL_VALIDITY_DIRECTION`. The structural-validity
hypothesis is closed. This directory is a record, not an open line of work.

## Contents

| File | Purpose |
| --- | --- |
| `report.md` | The technical report. Evidence, interpretation, limitations and unsupported claims are separated into distinct sections. |
| `results-summary.json` | Machine-readable numeric summary of both studies, for programmatic comparison and for the tests that pin these values. |
| `artifact-manifest.json` | Immutable object keys, artifact SHA-256 digests, run IDs, experiment identities, specification hashes and source commits. |
| `reproduction.md` | How each result was produced and how a third party would re-derive it. |
| `limitations.md` | The full limitation and non-claim inventory, expanded beyond the summary in `report.md`. |

## What this directory deliberately does not contain

- **No terminal artifacts.** The two terminal objects live in the immutable object store.
  `artifact-manifest.json` records their keys and digests; the bytes are not copied into Git.
- **No model weights**, in any form, in Git, Git LFS, test fixtures or CI artifacts.
- **No amendment** to either completed study. Both are frozen. Their specifications, schemas,
  run IDs, source pins, model pins, thresholds, seeds, conclusions and terminal objects are
  immutable and are pinned by tests.

## Evidence class

Both studies are labelled `development_compatibility_canary`. Neither is holdout evidence,
production evidence or trading evidence, and neither authorizes training, Stage B, Stage C,
test-partition opening, production inference or trading claims. Every authorization field in
both terminal artifacts is `false`.

## Successor work

The structural-validity question is answered and closed. The follow-on study is a separately
preregistered zero-shot forecasting benchmark with a disjoint identity in every namespace:

- Experiment ID `openalpha-kronos-zero-shot-benchmark-v1`
- Specification `research/bridge-v0/kronos-zero-shot-benchmark-v1.yaml`
- Artifact root `openalpha-compatibility/kronos-zero-shot-benchmark`
- Run-ID namespace `zsb_<8-32 lowercase hex>`

That benchmark asks a different question — external validity of zero-shot forecast skill under
paper-style horizons — and is not a structural repair experiment. See
`research/bridge-v0/kronos-zero-shot-benchmark-v1.yaml` and its preregistration.
