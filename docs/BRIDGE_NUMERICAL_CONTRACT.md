# Bridge Numerical Contract

## Scope

This is the implemented Phase 1 numerical/runtime contract for
`packages/bridge`. It covers target construction, activation mapping, recursive
OHLC(V) reconstruction, validation, numerical audit, and canonical serialization.
It does not load Kronos, retrieve market data, define a trained checkpoint, or
measure learned reconstruction or forecast quality.

The controlling research lock remains
`research/bridge-v0/experiment.yaml` with SHA-256
`d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2`.

## Versioned configuration

`BridgeRepresentationConfig` is a frozen, strict Pydantic model. Its representation
version is `openalpha.bridge.financial.v1`; its canonical configuration schema is
`openalpha.bridge.config.v1`. Equivalent validated configurations serialize to the
same UTF-8 canonical JSON and SHA-256.

The default, Bridge-2K-compatible configuration locks:

| Quantity | Value |
|---|---:|
| Minimum source price and anchor | `1e-12` |
| Maximum source price and anchor | `1e12` |
| Maximum source volume | `1e15` |
| Absolute gap-return cap | `log(4) = 1.3862943611198906` |
| Absolute body-return cap | `log(4) = 1.3862943611198906` |
| Upper-wick log cap | `log(4) = 1.3862943611198906` |
| Lower-wick log cap | `log(4) = 1.3862943611198906` |
| Log-volume cap | `log1p(1e15) = 34.538776394910684` |
| Scientific log-price guard | `[-300, 300]` |
| Near-zero numerical epsilon | `1e-14` |
| Target/reference dtype | `float64` |
| Primary runtime output dtype | `float32` |
| Bridge-2K suffix limit | `64` candles |
| Overflow policy | typed `raise`, no output |
| Out-of-domain policy | typed `raise`, no clipping |

Configurations reject nonpositive or nonfinite caps, unknown representation or
activation versions, unsupported dtypes, clipping policies, volume/feature-order
contradictions, unsafe float64 exponential guards, source bounds outside the guard,
and channel limits that can leave the guard in one supported step.

Phase 1 numerical stress tests may instantiate the same representation version with
a larger `maximum_decode_steps`, up to 4,096, solely to expose recursive drift. This
does not change the locked 64-candle Phase 2 suffix.

## Input classes and failures

Target construction distinguishes three classes before emitting a tensor:

- Supported input: valid finite OHLC(V), source values within the price/volume
  bounds, and every transformed coordinate within its exact cap. Encoding succeeds.
- Out-of-domain input: a valid source candle or transformed coordinate exceeds a
  bound. `BridgeTransformError` carries an immutable `BridgeFailure` with category,
  code, zero-based sequence and candle indices, field, observed value, applicable
  lower/upper bound, and message.
- Invalid input: a value is nonnumeric/nonfinite where presence requires a number,
  a price is nonpositive, volume is negative, OHLC ordering is invalid, or the
  shape/mask/anchor contract is malformed. It fails before target construction.

No Phase 1 path clips a target, anchor, feature, or reconstructed candle. A missing
optional volume is the one permitted source NaN: it must have a matching false
presence bit. A present zero volume has a true bit and numeric zero.

## Float operations and guards

Forward targets are calculated in float64 as differences of logarithms. Inverse
operations are performed in the configured output dtype so float32 behavior is
tested directly. Before each exponential the runtime checks:

1. the recursive log value is finite;
2. it remains within `[-300,300]`; and
3. it is inside the nonzero finite exponential interval of the output dtype.

The third guard is narrower for float32 than the scientific guard. A finite feature
sequence that would overflow float32 or underflow a price to zero therefore returns
`LOG_PRICE_OUTSIDE_DTYPE_RANGE` and emits no candle. `expm1` volume output is
similarly checked for finite nonnegative output. Runtime failures are explicit; no
post-output validator or projection repairs them.

## Locked round-trip tolerances

`numerical_roundtrip_audit` reports per-field maximum absolute, relative, and
log-space error, plus maximum and final recursive close drift. It uses exact volume
presence masks when computing volume error.

| Dtype/field | Absolute | Relative | Log-space | Recursive close relative |
|---|---:|---:|---:|---:|
| float64 OHLCV | `1e-12` | `1e-12` | `1e-12` | `1e-12` |
| float32 OHLC | `1e-6` | `2e-5` | `1e-4` | `1e-4` |
| float32 volume | `1e-3` | `5e-5` | `1e-4` | not used |

The Phase 1 suite exercises lengths 1, 5, 128, 512, and 2,048 in both dtypes.
Synthetic features are deterministic, structurally supported, and require no
provider data.

Against the float64 synthetic reference at 2,048 recursive steps, the float64
round trip is exact for this fixture. The float32 runtime reports maximum absolute
error `0.0014648437518189894`, maximum relative error
`1.2514681899901712e-05`, maximum log-space error
`1.2514760209469955e-05`, maximum relative close drift
`1.244052244697315e-05`, and final relative close drift
`7.81884527896428e-06`; every value is within its locked field/dtype tolerance.

## Shapes and volume modes

Single-sequence tensors use `[T,F]`; batches use `[B,T,F]`. Raw head output is
exactly `[T,5]` or `[B,T,5]`. A single sequence accepts one scalar anchor (or exact
shape `[1]`); a batch requires exact shape `[B]`. The runtime never broadcasts an
anchor, mask, sequence, or batch axis.

- `VOLUME_REQUIRED`: source and output have five columns; every volume is present,
  finite, and nonnegative.
- `VOLUME_OPTIONAL`: source/output have five storage columns plus an exact boolean
  `[T]` or `[B,T]` mask. Missing storage is NaN with a false bit; zero is numeric zero
  with a true bit.
- `PRICE_ONLY`: source/output have four OHLC columns, no volume mask, and no volume
  is represented as data. The unused fifth raw neural channel is explicitly labeled
  `PRICE_ONLY_RAW_VOLUME_CHANNEL_IGNORED`.

Ragged sequence masks are not accepted in Phase 1. Calls are dense and exact; a
later ragged contract would require a separately specified serialization and loss
semantics.

## Canonical serialization

The scoped functions `bridge_canonical_bytes` and `bridge_sha256` serialize only
Bridge configuration, feature tensors, reconstructed sequences, validation results,
typed failures, and numerical audits.

- Container encoding: UTF-8 JSON, sorted keys, compact separators, explicit schema
  and object versions.
- Tensor encoding: C-order IEEE-754 bytes normalized to little-endian, represented
  as lowercase hexadecimal with explicit dtype and shape.
- Negative zero: normalized to positive zero before tensor encoding.
- Timestamps: dates use ISO 8601; datetimes must be timezone-aware and are encoded
  in UTC with `Z`.
- Missing volume: NaN never enters canonical bytes. Missing slots are normalized in
  the tensor bytes and accompanied by an exact `explicit-presence-u8` mask; missing
  and present-zero objects therefore hash differently.
- Hash: lowercase SHA-256 over the exact canonical bytes.

Nonfinite present values, naïve datetimes, implicit missing volume, unsupported
objects, and platform-dependent native tensor dumps are rejected.

## Phase 1 evidence boundary

The deterministic property proof uses 10,000 finite raw-head examples plus 250
batch and 250 optional-volume examples. Every successful output is checked directly
against the financial grammar and independently through the reused Sentinel
validator. The tests also cover explicit structural defects, cap-adjacent values,
NaN/infinities, unsafe exponentials, float32 precision loss, batch/mask/anchor shape
errors, causal future perturbation, serialization replay, and long recursive drift.

These results prove the implemented mathematical/runtime contract only. They do not
show that the future learned head is accurate, that frozen Kronos tokens contain
enough wick/range information, or that Bridge improves any reconstruction or
forecast metric.

## Phase 2 evidence boundary

Phase 2 produced no learned numerical evidence. It stopped
`OPERATIONALLY_BLOCKED` before data or checkpoint access because the locked
512-candle example cannot fit inside the locked validation or reconstruction-test
partitions without crossing a chronological boundary. The Phase 1 caps,
activations, dtype tolerances, recursion, and serialization contract remain
unchanged. No saturation distribution, coverage measurement, learned round trip,
latency, memory, range error, or close-drift result was measured.
