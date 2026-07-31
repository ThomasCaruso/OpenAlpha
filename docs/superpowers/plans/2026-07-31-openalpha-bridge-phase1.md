# OpenAlpha Bridge Phase 1 Implementation Plan

> Execution scope: mathematical/runtime contract only. Do not retrieve data, load Kronos assets, forecast, train, or alter preserved Sentinel evidence.

**Goal:** Implement a versioned, immutable, causally recursive OHLC(V) representation whose successful outputs are structurally valid by construction and whose unsupported inputs fail with exact typed diagnostics.

**Architecture:** Add a small `openalpha-bridge` workspace package. NumPy provides the framework-neutral Phase 1 tensor boundary. The package owns configuration, typed failures/results, target encoding, versioned head activations, causal inverse reconstruction, deterministic serialization, and a thin audit adapter over the existing Sentinel structural validator. It does not contain a model, checkpoint, tokenizer, or provider integration.

**Locked numerical policy:** Preserve the Phase 0 lock: price domain `[1e-12, 1e12]`, volume domain `[0, 1e15]`, `log(4)` gap/body/wick caps, `log1p(1e15)` volume cap, log-price guard `[-300, 300]`, float64 target computation, float32 runtime output, float64 reference output, no clipping, and typed no-output failures. Phase 2's learned suffix remains at most 64 candles; Phase 1 also exercises longer supported source round trips to measure recursion without changing that experiment limit.

---

## Task 1: Establish the package and configuration contract

**Files:**
- Create: `packages/bridge/pyproject.toml`
- Create: `packages/bridge/src/openalpha_bridge/__init__.py`
- Create: `packages/bridge/src/openalpha_bridge/config.py`
- Create: `packages/bridge/src/openalpha_bridge/errors.py`
- Test: `packages/bridge/tests/test_config.py`
- Modify: `pyproject.toml`
- Modify mechanically: `uv.lock`

1. Write failing tests for the default frozen configuration, exact Phase 0 constants, stable configuration hash, unknown versions/dtypes/policies/transforms, nonpositive/unsafe limits, contradictory volume settings, and rejected `model_copy(update=...)` bypasses.
2. Run `uv run pytest packages/bridge/tests/test_config.py -q` and confirm collection/import failure.
3. Implement enums, frozen strict Pydantic configuration, cross-field exponential safety checks, and typed failure records/exceptions.
4. Add the package to the workspace environment, refresh the lock without upgrading unrelated dependencies, and rerun the test to green.

## Task 2: Validate and encode source targets honestly

**Files:**
- Create: `packages/bridge/src/openalpha_bridge/models.py`
- Create: `packages/bridge/src/openalpha_bridge/transform.py`
- Test: `packages/bridge/tests/test_encode_targets.py`

1. Write failing tests for single `[T,F]` and batch `[B,T,F]` inputs, exact log-difference formulas, recursive declared previous-close use, bullish/bearish/doji/zero-wick cases, required/optional/price-only volume semantics, and read-only deterministic result arrays.
2. Add failures for invalid OHLC, nonpositive/nonfinite prices, negative/present-nonfinite volume, exact inside/outside domain boundaries, and explicit structured sequence/candle/field/value/bound diagnostics.
3. Implement strict rank/feature/mask/anchor checks with no broadcasting; validate the source candle before target construction; calculate targets in float64 using log differences and `log1p`; canonicalize only roundoff-scale negative zero.
4. Run the focused tests to green.

## Task 3: Map head outputs and reconstruct causally

**Files:**
- Create: `packages/bridge/src/openalpha_bridge/activations.py`
- Extend: `packages/bridge/src/openalpha_bridge/transform.py`
- Test: `packages/bridge/tests/test_decode.py`
- Test: `packages/bridge/tests/test_shapes_and_volume.py`

1. Write failing tests for the locked scaled-`tanh` signed map and bounded stable-softplus nonnegative map, transformed-feature exposure, raw nonfinite failures, saturation/cap behavior, and documented inverse target mappings.
2. Write failing reconstruction tests for one candle, full sequences, batches, exact initial-anchor cardinality, no cross-sequence broadcasting, volume modes, and causal future-perturbation independence.
3. Implement log-space guarded reconstruction with per-step finite/underflow/overflow checks and previous reconstructed close chaining. Emit OHLC only for price-only, explicit presence metadata for optional volume, and never convert missing volume to zero.
4. Run focused tests to green.

## Task 4: Reuse Sentinel validation as an independent audit

**Files:**
- Create: `packages/bridge/src/openalpha_bridge/validation.py`
- Test: `packages/bridge/tests/test_validation.py`

1. Write failing tests for finite/positive/OHLC/volume checks, expected feature and sequence length, timestamp alignment, duplicate timestamps, and strictly increasing timestamps.
2. Implement a non-mutating Bridge adapter that delegates candle grammar to `openalpha_sentinel.structural_validity.validate_forecast_path`, adds Bridge-only shape/order diagnostics, and returns structured immutable results. Use deterministic internal sessions only when callers omit timestamps; never expose them as observed timestamps.
3. Verify decoded success is valid even before audit, then audit every fixture independently.

## Task 5: Canonical serialization and numerical audit

**Files:**
- Create: `packages/bridge/src/openalpha_bridge/serialization.py`
- Create: `packages/bridge/src/openalpha_bridge/numerics.py`
- Test: `packages/bridge/tests/test_serialization.py`
- Test: `packages/bridge/tests/test_numerics.py`

1. Write failing tests for schema/version/field order independence, explicit dtype and little-endian IEEE-754 encoding, UTC ISO timestamps, explicit missing-volume masks, negative-zero normalization, SHA-256 replay identity, and rejection of nonfinite canonical values.
2. Implement type-specific canonical payloads and UTF-8 canonical JSON bytes using compact sorted keys. Encode floating tensors as canonical little-endian bytes plus shape/dtype; normalize missing-volume storage under the explicit mask.
3. Implement per-field absolute, relative, and log error plus final/maximum recursive close drift. Lock float64 tolerances at `rtol=1e-12, atol=1e-12`; lock float32 price tolerances at `rtol=2e-5, atol=1e-6`, float32 volume at `rtol=5e-5, atol=1e-3`, and float32 log-space/recursive relative tolerance at `1e-4`.
4. Test lengths 1, 5, 128, 512, and 2,048 using deterministic supported synthetic sequences; explicitly test guard failures and float32 boundary behavior.

## Task 6: Property and edge-case proof

**Files:**
- Create: `packages/bridge/tests/test_properties.py`
- Extend: `packages/bridge/tests/test_numerics.py`
- Extend: `packages/bridge/tests/test_shapes_and_volume.py`

1. Add a deterministic Hypothesis proof with `max_examples=10000`, no example database, covering finite raw outputs, both body directions, doji/zero wicks, cap-adjacent values, small/large supported anchors, zero/large/missing volume, and batch sizes.
2. For every successful decode, assert the grammar directly and through the Sentinel audit. A guarded typed no-output numerical failure is allowed; an invalid emitted candle is not.
3. Add explicit NaN/infinity, zero/negative price, negative volume, every OHLC-order defect, exponential/expm1 guard, just-inside/outside caps, malformed ranks/features/masks/anchors, and boundary-drift tests.
4. Run `uv run pytest packages/bridge/tests -q` and retain the fresh case/count result.

## Task 7: Document the implemented contract

**Files:**
- Modify: `docs/BRIDGE_MATHEMATICAL_REPRESENTATION.md`
- Modify: `docs/OPENALPHA_KRONOS_BRIDGE.md`
- Modify: `docs/KRONOS_COMPATIBILITY_BOUNDARY.md`
- Modify: `docs/STATUS.md`
- Create: `docs/BRIDGE_NUMERICAL_CONTRACT.md`

Document exact formulas, activation/inverse formulas and gradients, supported/unsupported domain, causal recursion, volume semantics, dtype-specific tolerances, failure categories, serialization schema/hash rules, worked synthetic examples, proof, and known limits. State explicitly that no Bridge head was trained and no learned performance was measured.

## Task 8: Verify preservation and seal Phase 1

1. Run targeted Bridge tests and the complete repository suite.
2. Run Ruff and Pyright with the repository's configured commands.
3. Recompute `research/bridge-v0/experiment.yaml` SHA-256 and compare with `experiment.sha256` and the locked value.
4. Compare Sentinel v0/v1/v1.1 tree hashes against `ce507303fe8633e74ae60248ecb7450ea0dcc707`; confirm no Sentinel research artifact or source hash changed.
5. Run `git diff --check`, inspect `git diff --stat`, and confirm there are no data/checkpoint/model files.
6. Commit all Phase 1 work as one logically scoped commit with message `feat(bridge): prove phase 1 mathematical contract`.
7. Report the commit, files, formulas/caps/mappings, numeric results, longest sequence, property count, edge coverage, hash behavior, total tests, unresolved risks, and the exact gated Phase 2 task. Stop before data retrieval or learned decoder training.
