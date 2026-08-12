# Retire Bridge Phase 2 and Recast OpenAlpha as a Research Codebase

**Date:** 2026-08-12

**Status:** Approved design

## Objective

Retire the abandoned Bridge product, training, and control-plane infrastructure
without altering OpenAlpha's scientific record. Complete the research-oriented
package boundary already begun by `openalpha_research`, move surviving
Kronos-specific code into a package that names its subject, and present the
repository as a serious empirical financial-ML research codebase.

The cleanup must not produce new scientific evidence. The next authorized new
research direction is an official-protocol Kronos replication. The existing
frozen-representation probe is preserved but remains inactive.

## Governing Rules

1. Preserve evidence.
2. Preserve reproducibility.
3. Preserve reusable research infrastructure.
4. Remove abandoned product infrastructure.
5. Do not let historical naming dictate the architecture of the current
   repository.
6. Historical names preserve truth; live names describe the current
   architecture.
7. The migration itself produces no new scientific evidence.

## Scientific State Before and After the Migration

The migration does not change any scientific conclusion, authorization, or
execution state.

- The Kronos-mini structural-validity diagnostic remains completed with
  `ROUNDTRIP_MATERIAL_INVALIDITY`.
- The Kronos-base replication remains completed with
  `ROUNDTRIP_MATERIAL_INVALIDITY`.
- The Kronos-base zero-shot benchmark remains completed with
  `NO_ZERO_SHOT_SKILL`.
- The structural-validity direction remains abandoned.
- The zero-shot generation direction remains stopped by its preregistered rule.
- The frozen-representation probe remains unexecuted and produces no scientific
  conclusion. Its implementation is preserved solely across the namespace
  migration.
- Official-protocol replication becomes the next authorized research direction,
  but no `official_protocol` Python package, implementation, or execution surface
  is created until its protocol is designed and its preregistration is sealed.

No migration verification may invoke a Kronos model, retrieve provider data,
open a partition, deploy Modal, or write a new scientific artifact.

## Target Repository Ownership

The target package structure is conceptual. It fixes ownership and dependency
direction without requiring a directory for every responsibility when one
coherent module is sufficient.

```text
packages/
├── research-core/
│   └── src/openalpha_research/
│       ├── artifacts/
│       ├── provenance/
│       ├── identity/
│       ├── protocols/
│       ├── providers/
│       ├── statistics/
│       ├── runtime/
│       └── errors.py
├── kronos-research/
│   └── src/openalpha_kronos/
│       ├── model/
│       ├── tokenizer/
│       ├── runtime/
│       └── studies/
│           ├── structural_validity/
│           │   ├── mini/
│           │   └── base/
│           ├── zero_shot/
│           └── frozen_representation/
├── sentinel/
└── experiment-spec/
```

`research-core` is an existing boundary being completed, not a new generic
layer. `sentinel` and `experiment-spec` remain separate. This cleanup does not
merge them merely for visual uniformity.

### Dependency direction

```text
openalpha_kronos  ───────▶ openalpha_research
openalpha_sentinel ──────▶ openalpha_research  (only where genuinely shared)

openalpha_research ──X──▶ openalpha_kronos
openalpha_research ──X──▶ openalpha_sentinel
```

`openalpha_research` must contain no knowledge of Kronos, Sentinel, or Bridge.
It imports no study-specific package. An automated repository guard enforces
this rule after migration.

## Module Classification Contract

Every active module under `openalpha_bridge` is classified by its actual
responsibility. There is no wholesale namespace rename.

### Move to `openalpha_research`

Only code that is model-independent, already used by more than one study or
clearly part of the repository's scientific execution contract moves into
research-core:

- artifact publication and immutable object handling;
- canonical serialization and content identity;
- provenance, run identity, and invocation validation;
- sealed-protocol and specification verification;
- provider interfaces and generic validated market-series records;
- statistical estimators and resampling primitives;
- generic runtime measurements;
- safe failure, redaction, and logging primitives.

Moving a module into research-core must not weaken study-specific guards. For
example, a generic moving-block estimator may live in core, while the Kronos
zero-shot study retains its exact asset count, origin count, block length,
seed, and decision-rule validation.

Completed-study-specific behavior stays with its study even if it might be
reusable in theory. Research-core is not an archive for old code.

### Move to `openalpha_kronos`

Kronos-specific executable research code belongs to `kronos-research`:

- pinned Kronos source, checkpoint, and tokenizer identities;
- source and asset conformance checks;
- official model/tokenizer loading and isolation;
- Kronos-specific preprocessing and inference contracts;
- the completed Kronos-mini structural-validity diagnostic;
- the completed Kronos-base structural-validity replication;
- the completed zero-shot benchmark;
- the mechanically preserved inactive frozen-representation probe;
- study-specific artifacts, decision rules, schemas, fixtures, and tests.

The mini and base implementations live under
`studies/structural_validity/mini` and
`studies/structural_validity/base`. Historical `canary_...` run identifiers and
artifact namespaces remain unchanged.

### Delete

Code is deleted when its sole purpose is the abandoned Bridge product or
training direction:

- the Bridge financial transform and decoder implementation;
- the Bridge trainable head and training pipeline;
- feature extraction and caches used solely by Bridge training;
- Stage A planning, compatibility-canary, continuation, Stage B/Stage C,
  state-machine, gate, and test-opening machinery;
- the generic cloud control API, service, lease/journal control plane, and
  remote command surface;
- the Bridge CLI and cloud client;
- the Bridge Docker image and environment template;
- Bridge deployment and run-start workflows;
- Bridge operational runbooks and current architecture documents whose subject
  no longer exists;
- tests whose only subject is deleted infrastructure.

### Canary distinction

The word `canary` does not decide ownership.

- The published `canary_0a92fde788bd685c` result was produced by the
  Kronos-mini frozen-inference diagnostic. Its implementation, verification,
  historical run identity, and evidence path survive under the mini
  structural-validity study.
- The separate `phase2/canary.py` path describes an amended Stage A Bridge
  compatibility check. It depends on Bridge feature caches, Bridge dimensions,
  amendments, and Stage B authorization fields. That path and its worker,
  artifact publisher, and tests are deleted.

## Historical Compatibility Boundary

Immutable and historically identifying material remains at its current path
and retains its original content unless a generated current index needs a new
link.

Historical uses of `Bridge` may remain in:

- experiment and run identifiers;
- artifact keys and serialized schema identifiers;
- hashes, digests, and provenance fields;
- preregistration filenames and sealed amendments;
- immutable terminal artifacts;
- completed reports and reproduction instructions that truthfully describe the
  code used at execution time;
- historical research prose.

The cleanup must not retroactively replace `openalpha_bridge` or `Bridge Phase
2` inside completed reports merely to match the new architecture.

Live Bridge naming is removed from:

- active Python import paths;
- package and distribution names;
- current dependency groups;
- active CI workflow and job names;
- cloud application names;
- current CLIs and operational surfaces;
- current architecture and repository-tour documentation.

Git history is the archive for deleted operational documentation. The current
tree does not retain an executable-looking `docs/archive/bridge` collection.

## Frozen-Representation Preservation Checkpoint

The current uncommitted frozen-representation implementation must be committed
before the namespace migration begins. That preservation commit includes the
existing implementation, its tests, the shared resampling extraction it
currently depends on, and its existing Modal additions only as a faithful
snapshot of the worktree.

The preservation commit message and body must state that the work is:

- inactive;
- governed by the already sealed development-only preregistration, but not
  authorized for execution;
- not executed;
- associated with no scientific conclusion; and
- preserved solely across the namespace migration.

After the snapshot, migration into
`openalpha_kronos.studies.frozen_representation` is mechanical. Scientific
parameters and behavior are not improved during the move.

The migrated dormant probe has:

- no CLI command;
- no Modal entrypoint;
- no workflow;
- no deployment or generic execution hook;
- no authorization field changed to true;
- no new result or artifact.

The probe-related Modal functions present in the preservation snapshot are
removed as part of the migration. They do not appear in the surviving research
cloud shell.

## Surviving Cloud Boundary

`cloud/modal/bridge_phase2_app.py` is replaced by one restrained
`cloud/modal/kronos_research.py` shell. It contains only thin, explicit
reproducibility or runtime functions for completed studies. Study computation
remains importable and testable outside Modal.

The shell contains no:

- Bridge training or feature-cache functions;
- generic control API;
- authenticated remote command surface;
- continuation or “start next phase” semantics;
- test-opening mechanism;
- frozen-representation function;
- official-protocol function;
- automatic scientific run.

No migration workflow deploys or invokes the shell. Static tests verify its
contents offline.

## README and Current Documentation

The supplied image is stored unchanged at:

`docs/assets/openalpha-research-flow.png`

It appears near the top of the root README, after the one-sentence project
description and before the completed-study table. The Markdown includes useful
alt text describing the seven-stage flow from the Bridge thesis through the
three studies to the current official-protocol replication direction.

The current README and current status/architecture documentation tell the
cleaned-up story:

```text
OpenAlpha
├── research-core        reusable scientific infrastructure
├── kronos-research      current model-specific research
├── sentinel             completed historical research
├── experiment-spec      experiment specification contracts
└── research/            immutable evidence and reports
```

They state that the representation probe is frozen and that official-protocol
replication is next. Completed reports are not rewritten to modernize their
historical implementation names.

## Migration Sequence

1. Commit the approved design specification without staging unrelated work.
2. Snapshot the current dormant probe work in its preservation commit.
3. Add failing architecture guards for the target boundaries.
4. Extend research-core with only demonstrated model-independent primitives.
5. Move completed Kronos studies and their tests into `kronos-research`.
6. Move the dormant probe mechanically and remove every probe execution
   surface.
7. Replace the Modal monolith with the completed-study-only Kronos research
   shell.
8. Delete abandoned Bridge code, package metadata, workflows, Docker/client
   infrastructure, current operational documentation, and obsolete tests.
9. Add the supplied README image and rewrite current package/architecture/status
   documentation.
10. Run the offline integrity and architecture verification suite.

The migration may use intermediate compatibility fixes inside a task, but the
final tree exposes no `openalpha_bridge` shim or deprecated package. A shim
would preserve the obsolete architecture and weaken the repository guard.

## Verification Strategy

Verification is offline and non-scientific. It uses unit tests, preserved
fixtures, static source inspection, sealed hashes, and committed artifacts.

### Architecture guards

The repository suite fails if:

- `packages/bridge/` exists;
- active Python source imports `openalpha_bridge`;
- `openalpha_research` imports or contains explicit knowledge of Kronos,
  Sentinel, or Bridge;
- an active workflow name or body contains `Bridge Phase 2`;
- a frozen-representation CLI, Modal function, or workflow exists;
- an `official_protocol` Python package or module exists;
- the surviving Modal shell exposes training, control-plane, generic-command,
  test-opening, probe, or future-study functions.

Historical research files are excluded from live-name guards. Their names and
contents are evidence, not active architecture.

### Evidence integrity

- Recompute every sealed preregistration and amendment digest.
- Confirm every committed terminal artifact is byte-for-byte unchanged.
- Run the artifact verification script and reproduce all published digests.
- Recompute generated `results-summary.json` data from the preserved artifacts
  and require numerical identity.
- Run the completed mini, base, and zero-shot study suites after their namespace
  moves.
- Test historical run-ID, artifact-key, schema, and invocation compatibility.

### Engineering verification

- Run the research-core, Kronos research, Sentinel, experiment-spec, smoke, and
  full offline test suites.
- Run Ruff and Pyright.
- Verify package metadata and locked dependencies install without the deleted
  Bridge distribution or groups.
- Verify current README links, including the diagram asset.
- Inspect the final diff for accidental scientific parameter changes and
  accidental `.superpowers/` files.

Tests marked `network` or `kronos` are not run as migration verification. Modal
is not deployed. No provider or checkpoint client is called.

## Acceptance Criteria

The migration is complete only when all of the following are true:

1. `packages/bridge/` no longer exists.
2. No live Python code imports `openalpha_bridge`.
3. `openalpha_research` contains no Kronos-, Sentinel-, or Bridge-specific
   knowledge and imports no study package.
4. `openalpha_kronos` contains all surviving Kronos-specific executable research
   code.
5. Completed mini, base, and zero-shot results remain numerically identical.
6. Every sealed specification and amendment digest remains unchanged.
7. Every committed terminal artifact remains byte-for-byte unchanged.
8. Historical identifiers remain unchanged wherever they participate in
   provenance.
9. Frozen-representation source and tests survive mechanically, but no CLI,
   Modal entrypoint, workflow, authorization, or new result exists.
10. No `official_protocol` Python package or module exists.
11. `cloud/modal/kronos_research.py` contains no training, control-plane,
    generic-command, test-opening, probe, or future-study functionality.
12. Active CI verifies research integrity and contains no Bridge Phase 2
    terminology.
13. Current package names, dependency groups, architecture documentation, and
    repository tour contain no live Bridge identity.
14. Historical reports and immutable evidence remain historically accurate.
15. The supplied README diagram is tracked at the approved path and displayed
    near the README top.
16. The cleanup performs zero new scientific execution.
17. The final diff contains no visual-companion session files.

## Out of Scope

- Designing, preregistering, or implementing the official-protocol replication.
- Running or improving the frozen-representation probe.
- Reorganizing Sentinel for aesthetic consistency.
- Merging `experiment-spec` into research-core.
- Renaming immutable artifacts, historical keys, schemas, or sealed research
  files.
- Rewriting completed reports to pretend the new package architecture existed
  when the studies ran.
