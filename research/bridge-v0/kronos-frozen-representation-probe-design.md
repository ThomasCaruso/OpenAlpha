# Frozen-Representation Probe — Design Note

**Status: superseded by a sealed preregistration. Not implemented, not executed.**

> This note has been formalised as
> [`kronos-frozen-representation-probe-v1.yaml`](kronos-frozen-representation-probe-v1.yaml)
> (SHA-256 `5003f706f9edbbf5473b7b3e224df6dc454a72d51b1733637c5751be9ef7273c`),
> which fixes the partitions, horizon, sampling policy, ridge grid, control
> feature list, thresholds and decision rules. Where the two disagree, the sealed
> YAML governs. This note is retained as the reasoning and the source audit
> behind it.
>
> No code has been written and nothing has executed. Every test session begins
> strictly after the sealed preregistration timestamp
> (`2026-08-04T03:12:08+00:00`); the boundary is derived from that timestamp
> rather than chosen. An earlier revision, digest `a3f7b8fe…3db6a2`, set the
> boundary at 2026-05-13 and justified it by claiming the test data did not yet
> exist. That was false — the document was sealed in August 2026 — and the claim
> has been replaced by the verifiable timestamp relation.

This document describes a study that may be run *later*, and only after the
zero-shot benchmark (`openalpha-kronos-zero-shot-benchmark-v1`) has produced a
result. It has no specification hash, no experiment id, no run-id namespace and
no code. Nothing here authorizes implementation or execution.

> The representation probe tests whether Kronos contains useful internal
> information despite weak direct generation. It is not a continuation of
> structural repair.

---

## 1. Why this is a different hypothesis

The two completed structural-validity diagnostics measured what the model
*emits*. The zero-shot benchmark measures whether what it emits beats a trivial
baseline. Neither examines what the model *encodes*.

A sequence model can generate poorly and still carry useful state. Sampling
noise, tokenizer lossiness and exposure bias all degrade generated paths without
necessarily destroying the information in the hidden layers that produced them.
That is why a direct zero-shot failure is preregistered as still permitting this
probe: `PROCEED_TO_FROZEN_REPRESENTATION_PROBE` is the mapped outcome of rule Z4.

This is emphatically **not** structural repair. It does not decode candles, does
not enforce OHLC inequalities, does not filter or project paths, and does not
touch the generation path at all. It reads hidden states and fits a small linear
model on top of them.

---

## 2. Audit — what is technically accessible today

Audited against the pinned upstream source at revision
`67b630e67f6a18c9e9be918d9b4337c960db1e9a`, vendored at
`vendor/kronos/67b630e6/`, digests
`model/kronos.py` = `0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032`
(as committed) and
`model/module.py` = `a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f`.

**No upstream source was modified in this session, and none needs to be.**

### 2.1 Primary candidate — final transformer hidden state, already public

`Kronos.decode_s1` (`model/kronos.py:278`) returns a **two-tuple**:

```python
def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
    ...
    x = self.norm(x)
    s1_logits = self.head(x)
    return s1_logits, x        # <- x is the hidden state
```

`x` is the post-`norm` output of the full transformer stack, shape
`[batch, seq_len, d_model]`. For Kronos-base, `d_model = 832` and `layers = 12`.
Its own docstring names it "the context representation from the Transformer".

This is the strongest finding of the audit: **hidden-state access requires no
modification, no fork, no monkey-patch and no forward hook.** It is a public
method that the official inference path itself already calls —
`auto_regressive_inference` at `model/kronos.py:436` does
`s1_logits, context = model.decode_s1(...)` and passes `context` straight into
`decode_s2` at line 440. Reading `x` is therefore exercising the same code path
the released model runs, not a private detour.

Practical shape: with a 40-row context, one `decode_s1` call yields 40 hidden
vectors of width 832. The natural per-origin feature is the **last position**
(`x[:, -1, :]`), which is the state from which the model would have predicted the
next candle.

### 2.2 Secondary candidate — logit-space features

`s1_logits` from the same call, shape `[batch, seq_len, 1024]`, and `s2_logits`
from `Kronos.decode_s2(context, s1_ids)` (`model/kronos.py:310`), also
`[batch, seq_len, 1024]`. Both are public returns.

Wide relative to the sample size and highly correlated with the hidden state.
Useful mainly as a distributional summary — entropy, top-k mass, sampled-token
log-probability — rather than as 1024 raw regressors.

### 2.3 Tokenizer-side candidates

- `KronosTokenizer.encode(x, half=False)` (`model/kronos.py:142`) returns
  `z_indices`, the discrete two-stream token ids. Public, cheap, and already used
  by the official path at line 398. Categorical, so it would need embedding or
  one-hot treatment before entering a linear model.
- `KronosTokenizer.forward(x)` (`model/kronos.py:74`) returns
  `(z_pre, z), bsq_loss, quantized, z_indices`. `quantized` is the continuous
  binary-spherical quantized representation — a genuine continuous tokenizer
  feature. Accessible without modification, but `forward` also runs both decoder
  stacks, so it costs more than `encode` for a feature that only needs the
  encoder half.
- The pre-quantization continuous embedding (after `self.embed` and
  `self.encoder`, before `self.quant_embed`) has **no public accessor**. It is
  reachable only by calling the submodules directly, which is read-only attribute
  use rather than a source change, but it is the least clean option here.

### 2.4 Per-layer hidden states — not exposed

`self.transformer` is an `nn.ModuleList` iterated inside `decode_s1`; there is no
`output_hidden_states` flag and no per-layer return. Intermediate layers would
require `register_forward_hook` on the individual layers. A forward hook is
read-only instrumentation and does not modify the checkpoint or the upstream
file, but it is more invasive than the audit needs. **Recommendation: do not use
per-layer states in a first probe.** The final hidden state is public, sufficient
and defensible.

### 2.5 Audit conclusion

| Candidate | Public API | Shape (Kronos-base) | Source change needed | Recommended |
| --- | --- | --- | --- | --- |
| `decode_s1` hidden state `x` | yes | `[B, T, 832]` | none | **primary** |
| `s1_logits` / `s2_logits` | yes | `[B, T, 1024]` | none | summaries only |
| `tokenizer.encode` z_indices | yes | `[B, T, 2]` streams | none | secondary |
| `tokenizer.forward` `quantized` | yes | `[B, T, codebook]` | none | optional |
| pre-quantization embedding | no | `[B, T, d]` | none, but private access | no |
| per-layer hidden states | no | 12 x `[B, T, 832]` | forward hooks | no |

**The probe is technically feasible today with zero upstream modification.**

---

## 3. Proposed design

### 3.1 Extraction

For each asset-origin, run the frozen model under `inference_mode` on the context
window, call `decode_s1`, and take the final-position hidden state. Freeze
everything: no gradient, no optimizer, no parameter update. The parameter digest
must be identical before and after, exactly as in the completed studies.

Features are standardized using **training-partition statistics only**.

### 3.2 Downstream models

Deliberately small and boring, so the result is about the representation rather
than about the head:

- **Linear regression** (ridge) for future return over a fixed horizon.
- **Logistic regression** (L2) for direction of that return.
- Ridge regularization strength selected on the **validation** partition only,
  from a small predeclared grid. Never on test.

### 3.3 Controls — the load-bearing part

Every control uses the **identical downstream model, identical partitions,
identical regularization grid and identical selection procedure**. Only the input
features differ. Without this, a positive result says nothing.

1. Raw OHLCV over the same context window.
2. Simple engineered features: trailing returns at several lags, realized
   volatility, range, volume z-score.
3. A trivial constant/intercept-only control.

A Kronos representation is interesting only if it beats these under the same
harness, on the same partitions, by a preregistered margin.

### 3.4 Partitions

Strictly chronological, with a gap between partitions at least as long as the
prediction horizon so no label straddles a boundary:

- **Train** — earliest block. Fit only.
- **Validation** — middle block. Hyperparameter selection only.
- **Test** — latest block. **Untouched.** Opened exactly once, at the end, and
  only if the preregistered decision rules say so. Any earlier read voids it.

No shuffling. No k-fold across time. No purged-CV variant without preregistering
the purge and embargo explicitly.

### 3.5 Boundaries carried forward

- No fine-tuning. Fine-tuning is considered **only** if this frozen probe
  succeeds, and then only under a further preregistration.
- No structural repair, projection, validity filtering or constrained decoding.
- No trading claim, backtest, transaction-cost model or position sizing.
- Development evidence until the test partition is opened under preregistered
  rules; nothing here pre-authorizes opening it.

---

## 4. What must exist before implementation

1. A completed zero-shot benchmark run with a published terminal artifact.
2. A decision outcome that reaches this probe.
3. Its own preregistration document with its own SHA-256, experiment id, run-id
   namespace, artifact namespace, success and failure schemas, decision rules and
   thresholds — disjoint from `canary_`, `base_` and `zsb_`.
4. Explicit partition dates, horizon, ridge grid and control feature list, fixed
   before any extraction.

Until all four exist, this file is a design note and nothing more.
