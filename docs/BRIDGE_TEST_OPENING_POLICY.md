# Bridge Phase 2 Test-Opening Policy

**The reconstruction-test partition is opened exactly once, by an explicit
operator transition, after the checkpoint is frozen.**

This document defines the guard. It is enforced in code by
`openalpha_bridge.phase2.testgate` and the state machine in
`openalpha_bridge.phase2.states`, not by convention.

## Why the guard exists

The Phase 2 preregistration fixes every threshold before any held-out number is
observed. If the test partition could be read during development, threshold
selection would silently become outcome-dependent and the experiment would stop
being a preregistered test.

Validation is the only held-out partition that may influence anything, and only
for checkpoint selection and early stopping.

## Preconditions

`open_test_partition` fails closed unless all of the following hold:

| Precondition | Meaning |
|---|---|
| `stage_a_passed` | The pipeline smoke completed. |
| `stage_b_passed` | The information-sufficiency probe met its minimum improvement. |
| `stage_c_completed` | Locked training finished. |
| `selected_checkpoint_fixed` | A checkpoint was selected from validation only. |
| `checkpoint_sha256` | The selected checkpoint's content hash is recorded. |
| `frozen_kronos_weights_verified` | Frozen weights hash identically before and after training. |
| `validation_selection_report_sealed` | The selection report is written and immutable. |
| `experiment_hash_verified` | The experiment and both amendments are byte-identical. |
| `preprocessing_state_sha256` | Training-only preprocessing state is fixed and hashed. |
| `feature_manifest_sha256` | The feature manifest is fixed and hashed. |

The state machine additionally refuses any `TEST_OPENED` transition whose
predecessor is not `CHECKPOINT_FROZEN`.

## The immutable record

Opening writes `test_opening_record.json` into the run directory containing the
timestamp, run ID, experiment hash, checkpoint hash, source commit,
preprocessing hash, feature-manifest hash, operator command, and prior ledger
hash.

The file is created with an exclusive open. A second attempt under the same run
ID fails with `TEST_PARTITION_ALREADY_OPENED`. The record is never deleted or
recreated; a genuinely new evaluation requires a new run identity, which in turn
requires a changed configuration, which is itself an amendment.

## Feature-cache enforcement

`FeatureCache` refuses to load any `reconstruction_test` shard until
`open_test_gate()` has been called, failing with `TEST_SHARD_LOAD_BEFORE_GATE`.
The guard is at the storage layer, so no evaluation path can bypass it by
reading shards directly.

## Synthetic runs

Synthetic pipeline validation uses `EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION`
and writes `synthetic_test_opening_record.json` in a separate namespace. A
synthetic run can never create the real record, and its run IDs are prefixed
`syn_`.

A real run additionally requires `provider_mode: real` and
`kronos_mode: pinned_official`. Constructing a real pipeline with a fake provider
or fake Kronos backend fails immediately.

## Current status

`test_partition_opened` is **false**. No run has opened the reconstruction-test
partition. No test candle has been retrieved, cached, or scored.
