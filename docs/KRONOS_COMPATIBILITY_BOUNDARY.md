# Kronos Compatibility Boundary

## Scope and evidence

This trace describes only the pinned source and checkpoints used by the preserved
OpenAlpha experiments. It is not a statement about later Kronos revisions.

| Asset | Pinned identity | Verified SHA-256 |
|---|---|---|
| Official source | `shiyu-coder/Kronos@67b630e67f6a18c9e9be918d9b4337c960db1e9a` | source tree is Git-addressed |
| `model/kronos.py` | pinned source file | `638a56e035856c600c9848b368be087cb706a61603a0790124968c95b8c69f3a` |
| `model/module.py` | pinned source file | `a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f` |
| Kronos-mini | `f4e68697d9d5aed55cef5c96aabc3376bcad9f81` | weights `a7d5f37e2e9fbd9891f7d7d4f72574512dd1f704fee14223e0a8cd0fbf54197c` |
| Kronos-mini config | same snapshot | `70daca2cb11e3a979dd6b8ac12ee08e2aace877acf28f5b8dfb4fe5609736201` |
| Kronos-Tokenizer-2k | `26966d0035065a0cae0ebad7af8ece35bc1fb51c` | weights `b97ec46b3b72160509e289183eaf7bdf5f0dac5bb9b49522f6d46638a99a8717` |
| Tokenizer-2k config | same snapshot | `0b30a443affb03e05a876a083857de9164f899feb7b4d261da02c485c9a3e3b6` |
| Kronos-Tokenizer-base | `0e0117387f39004a9016484a186a908917e22426` | weights `59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee` |
| Tokenizer-base config | same snapshot | `2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2` |

The cached source checkout was clean at the pinned revision when this document was
prepared, and both source-file hashes matched the immutable Sentinel v1 trace.

## Encoder and hierarchical identifiers

`KronosTokenizer.encode` is at `model/kronos.py:142-159`. For the audited
checkpoints its input is a float tensor with shape `[batch, sequence, 6]` in this
ordered feature space:

```text
open, high, low, close, volume, amount
```

The encode path is:

```text
[B,T,6]
  -> Linear(6,256)
  -> three causal TransformerBlock instances
  -> Linear(256,20)
  -> L2 normalization and binary spherical quantization
  -> coarse IDs [B,T] + fine IDs [B,T]
```

The config field is `n_enc_layers: 4`, while the implementation constructs
`range(n_enc_layers - 1)`, so the checkpoint contains three encoder blocks.

`BSQuantizer.forward` at `model/module.py:245-254` splits the 20-dimensional sign
vector into the first 10 bits (`s1`, `pre`, or coarse) and final 10 bits (`s2`,
`post`, or fine). `bits_to_indices` uses least-significant-bit-first powers
`2**arange(0, 10)` within each half.

| Property | Tokenizer-2k | Tokenizer-base |
|---|---:|---:|
| Input features | 6 | 6 |
| Hidden dimension | 256 | 256 |
| Coarse bits | 10 | 10 |
| Fine bits | 10 | 10 |
| Coarse vocabulary | 1,024 | 1,024 |
| Fine vocabulary | 1,024 | 1,024 |
| Pair space | 1,048,576 | 1,048,576 |
| Quantized latent width | 20 | 20 |
| BSQ entropy group size | 5 | 4 |

Kronos does not store a conventional learned tokenizer codebook table. The
tokenizer codebook is the implicit Cartesian set of bipolar sign vectors, scaled
by `1/sqrt(20)`. The deterministic compatibility fingerprint for the canonical
specification

```json
{"bit_order":"least_significant_first_within_each_10_bit_token","codebook_dim":20,"normalization":"one_over_sqrt_20","s1_bits":10,"s2_bits":10,"values":[-1,1]}
```

is
`ea1825698624a0835a2238d9bfe8c5ebbd39e074fb4daa9613423f0e4df23af2`.
Bridge manifests call this the implicit codebook fingerprint. The tokenizer config,
weights, and source hashes are still required because the learned encoder,
`post_quant_embed`, and decoder trunk are checkpoint-specific.

## Official continuous decoder

`KronosTokenizer.decode` is at `model/kronos.py:161-177`. With `half=True`, it:

1. converts the two ID tensors back into two 10-bit vectors;
2. concatenates them into `[B,T,20]` bipolar values scaled by `1/sqrt(20)`;
3. applies the learned `post_quant_embed: Linear(20,256)`;
4. applies the same three tokenizer decoder Transformer blocks to the ordered
   sequence; and
5. applies the unrestricted `head: Linear(256,6)`.

`MultiHeadAttentionWithRoPE` in `model/module.py:315-353` sets
`is_causal=True`. Decoder output for candle `t` can therefore depend on all provided
tokens through `t`, not on future tokens. It is not a candle-local lookup. The
decoder receives no calendar timestamp tensor, but it receives sequence order
through causal attention and rotary position state.

Any replacement that decodes each pair in isolation would be incompatible with the
audited architecture. OpenAlpha Bridge must process the full ordered token window
and test future-token perturbation invariance.

## Forecast generation boundary

The pretrained Kronos model consumes the same coarse and fine IDs through learned
forecast-model embeddings. `Kronos.decode_s1` produces coarse logits and a causal
transformer context. `Kronos.decode_s2` conditions fine-token logits on that context
and the selected coarse ID. `auto_regressive_inference` appends one unchanged pair
per future step.

After generation, the official code concatenates historical and generated tokens,
keeps the last `max_context` positions, and calls the tokenizer decoder once on the
full ordered window. For `max_context=512` and a five-step forecast, the final
decoder window contains the last 507 historical pairs and all five generated pairs.

Calendar features (`minute`, `hour`, `weekday`, `day`, and `month`) enter the
forecasting transformer during token generation. They do not enter the tokenizer
continuous decoder.

The official implementation averages decoded samples when `sample_count > 1`.
Bridge auditing cannot infer individual token paths from that average. OpenAlpha
must capture and preserve each generated token sequence and decode paths separately;
the Phase 3 paired experiment uses `sample_count=1` per declared seed.

## Normalization and price restoration

`KronosPredictor.predict` at `model/kronos.py:519-559` prepares the six features in
the order above.

- If volume is absent, volume and amount are filled with zero.
- If volume is present and amount is absent, amount is derived as volume multiplied
  by the row mean of open, high, low, and close.
- A population mean and population standard deviation are calculated for each of
  the six columns over the supplied historical frame.
- Each feature is normalized as `(x - mean) / (std + 1e-5)` and clipped to
  `[-5, 5]` before encoding.
- The tokenizer emits normalized six-feature reconstruction values.
- The predictor restores scale outside the tokenizer as
  `decoded * (std + 1e-5) + mean` using the same historical statistics.

Volume and amount are not decoded through specialized nonnegative heads. They are
the fifth and sixth unrestricted outputs of the same linear head used for OHLC.

OpenAlpha Bridge receives the exact historical normalization bundle and a
feature-presence policy. It does not recompute statistics from forecast values. Its
hard price reconstruction uses the last observed close as the first causal anchor.
When volume was absent at input, Bridge may pass official zero placeholders through
the frozen compatibility path but must omit safe volume and amount, set a warning,
and exclude volume loss. It must not present a synthesized zero as an observed
volume forecast.

When volume is supported, Bridge produces nonnegative volume. If an amount field is
required, OpenAlpha derives it deterministically as volume times the safe candle's
OHLC mean and labels it as derived; it does not learn an independent unrestricted
amount.

## Selected Bridge tensors

The stable public compatibility bundle is:

| Tensor or field | Type and shape | Source | Mutable |
|---|---|---|---|
| `coarse_ids` | `int64[B,T]`, values `0..1023` | official encoder or forecaster | no |
| `fine_ids` | `int64[B,T]`, values `0..1023` | official encoder or forecaster | no |
| `position_ids` | `int64[T]`, ordered `0..T-1` | decoder window | no |
| `normalization_mean` | `float64[B,6]` | historical prefix only | no |
| `normalization_std` | `float64[B,6]` | historical prefix only | no |
| `normalization_epsilon` | scalar `1e-5` | pinned predictor contract | no |
| `clip_bounds` | scalar pair `[-5,5]` | pinned predictor contract | no |
| `previous_close` | positive `float64[B]` | last observation before output suffix | causal state |
| `volume_present` | `bool[B]` | input schema | no |

Inside the frozen compatibility trunk:

| Tensor | Shape | Construction |
|---|---|---|
| `quantized_bits` | `float32[B,T,20]` | official `indices_to_bits(..., half=True)` |
| `decoder_hidden` | `float32[B,T,256]` | frozen `post_quant_embed` plus three frozen causal decoder blocks |
| `scale_features` | `float32[B,T,13]` | causal mean/std ratios and log scales broadcast over time |
| `bridge_input` | `float32[B,T,269]` | `decoder_hidden` concatenated with `scale_features` |
| `bridge_raw` | `float32[B,S,5]` | trainable 269-to-64-to-5 head on the supervised suffix |

`scale_features` contains `log(anchor_close)`; four price-mean values calculated as
`mean_i / anchor_close - 1`; four `log1p(std_i / anchor_close)` values; and
`log1p(mean)` plus `log1p(std)` for volume and amount. The price anchor, means, and
standard deviations must be finite; the anchor must be positive; standard
deviations must be nonnegative; and volume/amount means must be nonnegative.
Violations are typed compatibility failures.

The selected head is `Linear(269,64) -> SiLU -> Linear(64,5)`, with 17,605 trainable
parameters. The five outputs parameterize gap, body, upper wick, lower wick, and
volume. The frozen official six-output head remains available only to produce the
raw comparison forecast; it is not required to generate the Bridge path.

## Smallest stable boundary

For OpenAlpha Bridge, the smallest stable boundary is **after the official ordered
coarse/fine token sequence has been generated and preserved, and before continuous
financial values are accepted downstream**.

At that boundary OpenAlpha can branch without modifying the encoder, forecasting
transformer, token vocabulary, generated IDs, or official decoder:

```text
preserved token IDs
  |-- official full-sequence decoder -> raw forecast -> Sentinel
  `-- frozen tokenizer trunk + OpenAlpha constrained head -> safe forecast -> Sentinel
```

This differs from Sentinel v1's pre-append interception boundary because Bridge does
not change token selection or autoregressive conditioning. It reconstructs the
completed original token sequence.

The decoder can be augmented without modifying the official encoder or pretrained
forecasting transformer. The implicit tokenizer codebook and learned frozen
`post_quant_embed` can be reused directly. The learned forecast-model token
embeddings are not needed and are deliberately excluded to avoid coupling Bridge to
a particular forecasting-backbone size.

## Runtime rejection rules

A Bridge checkpoint is loadable only when its manifest matches all of the following:

- official source repository and revision;
- `model/kronos.py` and `model/module.py` hashes;
- tokenizer repository, revision, config hash, and weights hash;
- implicit codebook fingerprint, bit split, bit order, and vocabulary sizes;
- input dimension, hidden dimension, decoder-block count, and feature order;
- normalization epsilon, clipping, amount derivation, and missing-volume policy;
- supported maximum token window and output suffix;
- Bridge architecture/schema version and checkpoint hash;
- model license and Bridge training-data policy.

Unknown fields, mismatched hashes, out-of-range IDs, nonfinite scale state,
nonpositive anchors, unordered positions, unsupported feature schemas, or suffixes
longer than the manifest limit fail closed. No nearest-version or best-effort loading
is permitted.
