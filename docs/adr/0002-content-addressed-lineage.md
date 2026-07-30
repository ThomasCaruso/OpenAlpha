# ADR 0002: Canonical Specifications and Content-Addressed Lineage

- Status: Amended by ADR 0010
- Date: 2026-07-28

## Context

Mutable configuration, provider revisions, and untracked artifacts make quantitative results irreproducible. MLflow records only what an application submits and is not itself an immutable experiment protocol.

## Decision

Normalize YAML/JSON into a versioned typed specification, serialize it canonically, and identify it by SHA-256. Persist canonical bytes before queueing. Market snapshots and generated artifacts are content-addressed and carry input hashes, producer version, run ID, code revision, environment-lock hash, and timestamps.

Run attempts append state events. Modifying a specification creates a new experiment hash. A completed manifest is published only after all required hashes verify.

For Sentinel v0, the content-addressed lineage decision remains active, but the universal research-protocol schema is deferred. Phase 1 records the narrow `research/sentinel-v0/experiment.yaml` experiment lock; Phase 2 content-addresses those exact bytes without freezing protocol v1 or expanding the preserved experiment-spec prototype.

## Consequences

- Reproduction can verify identity before computation.
- Equivalent YAML and JSON produce one experiment identity.
- Canonical number/string rules and schema migrations become critical tested code.
- Storage garbage collection must understand references rather than file age alone.

