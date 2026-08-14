# OpenAlpha Research Architecture

OpenAlpha is an evidence-first Python research repository. Its active code supports
reproducible empirical studies, verification of completed evidence, and reusable
research contracts. It is not a product service, trading system, or deployment
control plane.

## Repository boundary

```text
packages/
  experiment-spec/    immutable experiment-document contracts
  research-core/      model-independent research infrastructure
  kronos-research/    Kronos integration and model-specific studies
  sentinel/           structural validation and completed Sentinel studies
cloud/modal/
  kronos_research.py  thin shell for eight completed-study functions
research/             immutable specifications, artifacts, results, and reports
```

Current names describe current ownership. Historical identifiers remain unchanged
inside sealed specifications, artifacts, reports, and provenance records. Git
history preserves retired operational material without presenting it as live
architecture.

## Strict dependency direction

`experiment-spec` and `research-core` are foundational boundaries:

```text
experiment-spec       (no local study dependency)
research-core         (no model- or study-specific dependency)
      ^
      |-- kronos-research
      `-- sentinel
```

`kronos-research` depends on `research-core`. Among workspace packages, `sentinel`
depends only on `research-core`. Neither subject package is imported by
`research-core`, and the two subject packages do not depend on each other.
`experiment-spec` remains separate because experiment-document canonicalization is
its own contract rather than a model-specific concern.

The `research/` tree is outside the runtime package graph. It holds immutable
research records and historical prose; package refactors do not rewrite their
identities or conclusions.

## Package responsibilities

### `packages/experiment-spec`

Loads, validates, canonicalizes, and migrates versioned experiment specification
documents. Stable canonical bytes make a sealed specification digest reproducible.

### `packages/research-core`

Provides model-independent infrastructure: content identity, protocols, run
manifests and state, artifact and object-store handling, providers, calendars,
resampling, runtime measurements, typed failures, redaction, and safe logging. It
contains no Kronos, Sentinel, or retired product knowledge.

### `packages/kronos-research`

Owns the pinned Kronos model/tokenizer boundary, evaluation utilities, and the
three completed executable study families:

- Kronos-mini structural-validity diagnostic;
- Kronos-base structural-validity replication;
- Kronos-base zero-shot benchmark.

It also preserves the preregistered frozen-representation probe source and tests.
That probe is inactive, has no CLI, workflow, Modal entrypoint, or authorization,
and has not been executed.

No official-protocol replication package or implementation exists. Its protocol
must be designed and its preregistration sealed before code is written.

### `packages/sentinel`

Owns structural-validity rules, audit records, provider adapters, and the completed
Sentinel investigations. Its immutable evidence remains under the corresponding
`research/sentinel-*` paths.

## Completed-study Modal boundary

`cloud/modal/kronos_research.py` is a thin optional shell. Study computation remains
importable and testable outside Modal. Its exact public surface is:

| Function | Completed-study purpose |
|---|---|
| `run_mini_structural_validity` | reproduce the completed mini diagnostic |
| `verify_mini_runtime` | verify the pinned mini runtime |
| `verify_base_runtime` | verify the pinned base runtime |
| `run_base_structural_validity` | reproduce the completed base study |
| `inventory_base_artifacts` | inventory completed base-study artifacts |
| `verify_zero_shot_runtime` | verify the pinned zero-shot runtime |
| `run_zero_shot_benchmark` | reproduce the completed zero-shot study |
| `inventory_zero_shot_artifacts` | inventory completed zero-shot artifacts |

The shell has no training, generic command, control API, continuation,
test-opening, frozen-representation, or future-study function. The retirement
migration neither deploys nor invokes it.

## Evidence and integrity

Specifications, amendments, terminal artifacts, result summaries, and completed
reports are research records rather than live application state. Historical run
IDs, schema names, artifact keys, digests, and implementation names remain intact
where they establish provenance.

The `Research Integrity` CI workflow verifies this boundary offline. It runs the
offline suite, checks sealed specifications and published artifacts, lints active
Python, type-checks the workspace, and asserts that Torch is absent from the
verification environment. Architecture guards additionally enforce package
direction, inactive-study boundaries, the exact Modal surface, and current-doc
links.
