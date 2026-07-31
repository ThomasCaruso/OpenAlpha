# Phase 3B experimental decoder implementation

## Source boundary

The pinned official Kronos checkout at revision
`67b630e67f6a18c9e9be918d9b4337c960db1e9a` remained read-only. OpenAlpha did
not alter model weights or silently patch the checkout. The maintained experiment
is implemented in `scripts/kronos_constrained_worker.py` and imports the pinned
model/tokenizer classes and sampling helpers.

The interception point mirrors `model/kronos.py:auto_regressive_inference` after
`decode_s1` has produced coarse logits and context and after `decode_s2` has
produced fine logits, but before the sampled coarse/fine pair is rolled into the
token buffers. The direct official predictor and the experimental raw loop were
required to produce identical path hashes before constrained methods ran.

Conceptually, the maintained difference from the official loop is:

```diff
 raw_pair = sample(coarse_logits, fine_logits)
-append(raw_pair)
+raw_candle = decode(context + raw_pair)
+if method == STEPWISE_PROJECT_REENCODE and invalid(raw_candle):
+    selected_pair = encode(project(raw_candle))
+    require valid(decode(context + selected_pair))
+elif method == VALID_CANDIDATE_RESAMPLING and invalid(raw_candle):
+    selected_pair = probability_weighted_valid_pair_within_budget()
+else:
+    selected_pair = raw_pair
+append(selected_pair)
```

This is a documented experimental loop, not an upstream source modification.

## Method C: stepwise project and re-encode

For an invalid decoded candle, the decoder leaves open and close unchanged, raises
high to the feasible maximum, lowers low to the feasible minimum, causally
re-normalizes the modified candle, and invokes the pinned tokenizer encoder on the
full modified window. It extracts the final hierarchical token pair, appends that
pair to a copy of the pre-intervention buffers, decodes again, and accepts it only
if the actual round trip satisfies the grammar. Otherwise it emits the typed hard
failure `ROUNDTRIP_INVALID`. Raw tokens, raw candle, violations, projected candle,
projected tokens, round-trip candle, changes, and later effects are preserved.

## Method D: bounded valid-candidate resampling

The raw sampled pair remains candidate rank 1. If it is invalid, the decoder forms
a bounded joint pool from coarse log probabilities and fine log probabilities
conditioned on each coarse candidate. It considers at most 16 candidates initially
and 64 after one declared expansion. Each candidate is decoded in the current
token-buffer context, inverse-normalized, and validated. Selection among valid
candidates is weighted by joint model probability using a deterministic auxiliary
fraction derived from origin, seed, and step. The selected valid pair alone is
appended.

The audit records candidate IDs, joint log probabilities, decoded candles,
violations, valid/rejected counts, rank, expansion, validation time, selected pair,
and fallback. The declared fallback is Method C; it was never used in the bounded
run. No unconstrained fallback exists.

## Measured tokenizer limitation

Method C changed the hierarchical token pair in only 2 of 23 projection/re-encode
attempts. Twenty-one paths hard-failed because the encoded projected candle decoded
back to an invalid candle in its sequence context. This supports the source trace's
pre-execution warning that the hierarchical tokenizer is sequence-dependent and
that continuous feasibility is not preserved by encode/decode.

## Audit and claim boundary

Every successful constrained path passed the versioned financial grammar. Raw
paths and failed attempts remain unchanged and independently auditable. This file
documents feasibility behavior only; it does not claim improved forecasting or
generalize beyond the pinned mini checkpoint, 12 origins, daily SPY/QQQ data,
three seeds, context 512, and five-session horizon.
