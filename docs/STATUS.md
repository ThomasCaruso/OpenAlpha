# Project Status

Last updated: 2026-08-13

## At a glance

| Direction | Status |
|---|---|
| structural-validity direction | **closed** |
| zero-shot generation direction | **stopped** |
| frozen-representation probe | **preregistered, preserved, inactive, not authorized or executed** |
| next direction | **official-protocol replication; protocol and sealed preregistration precede code** |

There is no official-protocol implementation or execution surface in the current
repository.

## Completed evidence and conclusions

Three preregistered studies executed against pinned Kronos checkpoints. Each
reached its preregistered conclusion:

| Study | Historical run ID | Conclusion |
|---|---|---|
| Kronos-mini frozen-inference diagnostic | `canary_0a92fde788bd685c` | `ROUNDTRIP_MATERIAL_INVALIDITY` |
| Kronos-base structural-validity replication | `base_03b08cbc706193d6` | `ROUNDTRIP_MATERIAL_INVALIDITY` |
| Kronos-base zero-shot benchmark | `zsb_25e0256eefb2b07a` | `NO_ZERO_SHOT_SKILL` |

The structural-validity direction is closed. Both structural studies found material
OHLC invalidity, but deterministic repair restored complete structural validity
while changing the primary forecast metric by exactly `0.0`. Both therefore
recommended `ABANDON_STRUCTURAL_VALIDITY_DIRECTION`.

The zero-shot generation direction is stopped by its preregistered rule. Across
four ETFs, 25 chronological origins, a 40-to-12 horizon, two temperatures, and
1,600 generations, all four success conditions failed at both temperatures. This
is development evidence, not a claim that Kronos is universally ineffective or a
replication of the published official protocol.

The completed reports and evidence are under
[`../research/reports/`](../research/reports/) and
[`../research/artifacts/`](../research/artifacts/). Historical run IDs, artifact
keys, specification digests, and conclusions remain unchanged.

## Preserved but inactive work

The frozen-representation probe has a sealed development-only preregistration:
[`../research/bridge-v0/kronos-frozen-representation-probe-v1.yaml`](../research/bridge-v0/kronos-frozen-representation-probe-v1.yaml)
with its committed SHA-256 sidecar. Its source and tests are mechanically preserved
under the Kronos research package.

The probe is inactive. It has no CLI command, workflow, Modal function, generic
execution hook, or authorization. It has not been executed and has produced no
scientific conclusion. Its preservation does not make it the next research
direction.

## Next direction

The next direction is an official-protocol Kronos replication. Planning that study
is authorized; execution is not. The evaluation protocol must first be designed,
then preregistered and cryptographically sealed. Only after those steps may an
implementation be created. No package, module, command, or cloud function for that
direction exists today.

This ordering prevents code or observed results from shaping the protocol and keeps
future evidence distinguishable from the three completed studies.

## Current engineering surface

- `packages/experiment-spec` and `packages/research-core` provide foundational
  specification and research contracts.
- `packages/kronos-research` contains pinned Kronos integration, completed-study
  code, and the inactive preserved probe.
- `packages/sentinel` contains structural validation and completed Sentinel work.
- `cloud/modal/kronos_research.py` exposes only eight explicit functions for the
  three completed studies.
- `research/` remains the immutable scientific record, separate from package
  ownership.
- The `Research Integrity` workflow verifies sealed specifications, committed
  artifacts, offline tests, lint, types, and the absence of Torch from its
  verification environment.

The retired Bridge product was never trained. It has no current operational or
deployment surface; Git history is the archive for its former operational design.
The retirement migration creates no new evidence, opens no partition, retrieves no
provider data, loads no model, and deploys no cloud application.
