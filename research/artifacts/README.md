# Terminal Artifacts

The immutable evidence records produced by the completed studies, committed as
the **exact object-store bytes** so that anyone can verify them independently.

```bash
python scripts/verify_artifacts.py
```

That recomputes each file's SHA-256 and compares it against the digest recorded
at publication. It exits non-zero on any mismatch, and it runs as part of the
test suite.

## What is here

| File | Study | Run | SHA-256 | Size |
| --- | --- | --- | --- | --- |
| `kronos_base_diagnostic_terminal.json` | Kronos-base structural | `base_03b08cbc706193d6` | `84ec1b19…c731e6` | 956 KB |
| `kronos_zero_shot_benchmark_terminal.json` | Zero-shot benchmark | `zsb_25e0256eefb2b07a` | `93688f04…65236a` | 1.34 MB |

Object-store keys, specifications, source commits and schema versions are in
[`manifest.json`](manifest.json).

## What is not here

The Kronos-mini diagnostic artifact (`canary_0a92fde788bd685c`, digest
`8e3d8a33…dab18c`) is **recorded but not committed**. That run predates the
session in which the other two bodies were captured, and retrieving it needs
object-store credentials. Its key and digest are in `manifest.json` so it can be
added later and checked against the same manifest, and its complete numeric
results are already mirrored in
[`../reports/kronos-structural-validity/results-summary.json`](../reports/kronos-structural-validity/results-summary.json).

Two of three studies are therefore independently verifiable at the byte level;
all three are verifiable at the level of reported numbers.

## Why the formatting looks strange

Each file is a single line with no whitespace and no trailing newline. That is
canonical JSON — sorted keys, `,`/`:` separators, ASCII-escaped — which is what
the publisher hashes and stores. It is deliberately not pretty-printed:
reformatting would change the bytes and break the digest, which is the entire
point of a content-addressed record.

`.gitattributes` marks these files `-text` so git never rewrites their line
endings on checkout.

For readable versions, see each study's `results-summary.json`, which is
generated from these payloads.

## Rules

These are published research records.

- **Never edit, reformat, regenerate or replace them.** Any byte change breaks
  the link between a result and the code that produced it.
- Corrections happen by running a **new** study with a **new** run ID and its own
  preregistration — never by amending a published artifact.
- The run IDs `base_03b08cbc706193d6` and `zsb_25e0256eefb2b07a` are permanently
  spent.

## What each artifact contains

Everything the study measured, not a summary: every asset-origin, every generated
path's token stream, every metric, every baseline, all bootstrap clusters, the
complete rule evaluations, the resolved asset digests, parameter hashes before
and after, GPU measurements, and every authorization field.

They contain no model weights, no credentials, no raw provider data, and no
exception text.
