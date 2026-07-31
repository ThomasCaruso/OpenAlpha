# Bridge Mathematical Representation

## Status

This document defines the Phase 1 mathematical contract. It does not implement or
train a learned decoder. Numerical constants are frozen before Bridge-2K evaluation
in `research/bridge-v0/experiment.yaml`.

The contract has two layers:

1. an exact financial representation for supported valid OHLC(V) sequences; and
2. a bounded neural-output map that can emit only supported representation values.

Structural validity comes from the inverse construction, not from a penalty or a
post-output projection.

## Symbols and supported domain

For candle `t`, let `O_t`, `H_t`, `L_t`, `C_t`, and optional `V_t` denote open,
high, low, close, and volume. Let `C_{t-1}` be the causal previous close. The first
output candle receives its anchor from the last observed candle before the output
suffix; later candles use the preceding reconstructed close.

A source candle is supported when:

- all present values are finite float64 values;
- `1e-12 <= O_t,H_t,L_t,C_t <= 1e12`;
- `H_t >= max(O_t,C_t,L_t)`;
- `L_t <= min(O_t,C_t,H_t)`;
- `1e-12 <= C_{t-1} <= 1e12`;
- volume is either absent or `0 <= V_t <= 1e15`;
- `abs(log(O_t/C_{t-1})) <= log(4)`;
- `abs(log(C_t/O_t)) <= log(4)`;
- `log(H_t/max(O_t,C_t)) <= log(4)`;
- `log(min(O_t,C_t)/L_t) <= log(4)`; and
- the output suffix contains at most 64 candles.

Valid candles outside these deliberately broad numerical limits are
`UNSUPPORTED_NUMERICAL_DOMAIN`; they are not clipped into the experiment. This
keeps every emitted finite-float guarantee explicit and testable.

## Forward transform

For supported valid input, define:

```text
gap_t    = log(O_t / C_{t-1})
body_t   = log(C_t / O_t)
upper_t  = log(H_t / max(O_t, C_t))
lower_t  = log(min(O_t, C_t) / L_t)
volume_t = log1p(V_t)                         when volume is present
```

`upper_t`, `lower_t`, and `volume_t` are nonnegative. Use differences of logarithms
rather than direct ratios in the implementation:

```text
gap_t   = log(O_t) - log(C_{t-1})
body_t  = log(C_t) - log(O_t)
upper_t = log(H_t) - max(log(O_t), log(C_t))
lower_t = min(log(O_t), log(C_t)) - log(L_t)
```

After calculation, values within `1e-14` of zero caused only by floating-point
roundoff may be canonicalized to positive zero. A genuinely negative wick value is
a structural input failure, not something the transform repairs.

The representation record is typed and contains:

```text
gap: float64
body: float64
upper: nonnegative float64
lower: nonnegative float64
log1p_volume: optional nonnegative float64
volume_present: bool
```

## Exact inverse transform

For a supported representation and a positive previous close, compute in log space:

```text
log_open  = log(C_{t-1}) + gap_t
log_close = log_open + body_t
log_high  = max(log_open, log_close) + upper_t
log_low   = min(log_open, log_close) - lower_t

O_t = exp(log_open)
C_t = exp(log_close)
H_t = exp(log_high)
L_t = exp(log_low)
V_t = expm1(volume_t)                         when volume is present
```

Every intermediate must be finite and lie inside the frozen representable log-price
guard `[-300, 300]`. The chosen source bounds, parameter caps, and 64-candle suffix
make this guard conservative. A guard failure returns a typed error and emits no
candle.

The next step receives the just-reconstructed `C_t`. It never receives the true
future close or the official decoder's next close.

## Neural-output map

The trainable head emits finite raw values
`a_gap, a_body, a_upper, a_lower, a_volume`. Nonfinite head output is a typed model
failure. Finite raw output is mapped as follows:

```text
gap   = log(4) * tanh(a_gap)
body  = log(4) * tanh(a_body)

softplus_stable(x) = max(x, 0) + log1p(exp(-abs(x)))
bounded_nonnegative(x, cap) = cap * softplus_stable(x) / (cap + softplus_stable(x))

upper  = bounded_nonnegative(a_upper,  log(4))
lower  = bounded_nonnegative(a_lower,  log(4))
volume = bounded_nonnegative(a_volume, log1p(1e15))
```

The stable softplus construction is nonnegative for every finite input. The rational
saturation keeps wick and volume logs below their declared caps without a hidden
projection. Gap and body are bounded symmetrically by `tanh`.

The head may approximate a zero wick or zero volume arbitrarily closely. Exact zero
is supported by the mathematical representation itself and may also occur through
floating-point underflow of the nonnegative map. Structural validity does not depend
on exact zero recovery.

## Proof of structural validity

Assume a supported positive `C_{t-1}`, finite mapped parameters, and successful
finite exponentiation.

1. `exp(x) > 0`, so open, close, high, and low are positive.
2. `upper_t >= 0`, so
   `log_high >= max(log_open, log_close)` and therefore
   `H_t >= max(O_t,C_t)`.
3. `lower_t >= 0`, so
   `log_low <= min(log_open, log_close)` and therefore
   `L_t <= min(O_t,C_t)`.
4. `max(log_open,log_close) >= min(log_open,log_close)` and nonnegative wicks only
   widen the interval, so `H_t >= L_t`.
5. `volume_t >= 0`, so `expm1(volume_t) >= 0` when volume is produced.
6. The same proof applies inductively after assigning reconstructed `C_t` as the
   next anchor.

No validity loss term is needed and no terminal projection is applied to Bridge
output.

## Missing volume and amount

Volume presence is a schema property, not an inferred target.

- When source volume exists, it must be finite and nonnegative. The representation
  contains `log1p_volume`, the learned head produces a bounded nonnegative volume
  parameter, and the inverse emits nonnegative volume.
- When source volume is absent, `volume_present=false`, the volume target and loss
  are masked, and the public safe output omits volume. Official zero placeholders
  required by the pinned tokenizer are retained only in compatibility provenance.
- Amount is not an independent Bridge output. If a consumer requests it and volume
  is present, derive
  `amount = V_t * (O_t + H_t + L_t + C_t) / 4`, label it derived, and validate it as
  finite and nonnegative. Otherwise omit it.

## Causal normalization state

The mathematical representation is expressed in raw financial units. The Bridge
head nevertheless needs the exact causal state that gave meaning to the official
tokens:

- ordered six-feature population mean and standard deviation;
- epsilon `1e-5` and clip bounds `[-5,5]`;
- feature-presence and amount-derivation flags; and
- the last observed close before the supervised or forecast suffix.

For a training example with 448 prefix candles and a 64-candle reconstruction
suffix, mean and standard deviation are calculated from the 448-candle prefix only.
The real suffix is normalized with that frozen prefix state before the official
encoder creates target tokens. At inference, the same state is calculated from
historical data only. No statistic is recomputed from realized future candles.

## Typed batch interfaces for Phase 1

The implementation task will define immutable typed records equivalent to:

```text
FinancialRepresentationBatch
  gap: float64[B,S]
  body: float64[B,S]
  upper: float64[B,S]
  lower: float64[B,S]
  log1p_volume: optional float64[B,S]
  volume_present: bool[B]

ReconstructionAnchorBatch
  previous_close: float64[B]
  maximum_suffix_length: 64

ReconstructedCandleBatch
  open/high/low/close: float64[B,S]
  volume: optional float64[B,S]
  amount: optional derived float64[B,S]
```

Batch dimensions must agree exactly. Ragged suffixes use an explicit boolean mask;
masked positions are never serialized as candles. Broadcasting that could change
the batch or time axis is rejected.

## Phase 1 validator contract

The structural validator runs independently even though Bridge output is valid by
construction. It reports, at minimum:

- nonfinite or nonnumeric values;
- nonpositive OHLC;
- high below open, close, or low;
- low above open, close, or high;
- negative volume or amount;
- missing, duplicate, unordered, or unexpected timestamps;
- shape, mask, column-order, and suffix-length mismatches; and
- provenance or compatibility-manifest mismatches.

Validation never mutates the raw or safe path.

## Round-trip and property requirements

Before any learned decoder work, Phase 1 must demonstrate:

- every finite supported neural-output tensor either produces a structurally valid
  sequence or a typed no-output numerical failure;
- every supported valid source sequence transforms and inverses within
  `rtol=1e-12` and `atol=1e-12` in float64;
- exact zero-wick and zero-volume fixtures remain valid;
- sequence chaining uses reconstructed, not future true, closes;
- changing a later representation value cannot change an earlier reconstruction;
- missing volume remains missing;
- deterministic canonical serialization rejects NaN and infinity and produces
  identical SHA-256 hashes for identical inputs; and
- the independent Sentinel validator reports zero violations for every successful
  reconstruction.

Property-based tests cover random supported sequences, both body directions, zero
and extreme supported wicks, cap-adjacent returns, missing volume, batch masks, and
the 64-step chaining bound. Numerical tests cover anchor bounds, log-price guards,
subnormal-adjacent values, overflow attempts, and malformed tensors.

## Serialization

Canonical artifacts use UTF-8 JSON with sorted keys, compact separators, explicit
schema versions, ISO-8601 timestamps, decimal finite numbers only, and SHA-256 over
the exact bytes. Negative zero is serialized as zero. NaN, infinity, unordered
timestamps, implicit missing values, and platform-dependent tensor dumps are
prohibited.

Raw official output, safe Bridge output, validation, token IDs, compatibility
manifest, normalization state hash, latency, and raw-to-safe differences are
separate fields. Serializing the safe path must never replace the raw path field.
