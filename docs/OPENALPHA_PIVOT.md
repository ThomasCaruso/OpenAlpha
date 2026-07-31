# OpenAlpha for Kronos Pivot

- Decision date: 2026-07-31
- Status: Phase 0 compatibility design; no Bridge training authorized by this commit
- Project identity: OpenAlpha
- Primary integration: OpenAlpha for Kronos
- Preserved research head: `daf347756f468953394fcf5f3eb1969d41f1a406`
- Untouched holdout accessed: no

## Decision

OpenAlpha remains one repository and one project. It is not renamed, split into a
separate decoder repository, or turned into a replacement financial foundation
model.

The product becomes a connected safety and compatibility stack:

1. OpenAlpha Sentinel validates and audits raw financial-model output.
2. OpenAlpha Bridge reconstructs unchanged Kronos hierarchical tokens through a
   constraint-preserving continuous decoder boundary.
3. OpenAlpha Evidence preserves the full investigation, including negative results
   and exact limitations.

The public product statement is:

> OpenAlpha makes financial foundation-model forecasts structurally safe,
> auditable, and deployment-ready. Its first integration adds a
> constraint-preserving compatibility decoder and assurance layer to Kronos without
> retraining the pretrained forecasting backbone.

This is a structural-correctness and safe-use project. It makes no profitability or
universal forecast-improvement claim.

## Preserved research sequence

The following sequence remains the motivation and evidence for this work:

1. OpenAlpha began as an auditable Kronos forecasting study.
2. Sentinel v0 tested whether forecast-time diagnostics could predict later Kronos
   return error.
3. The v0 hypothesis failed honestly. Pooled chronological out-of-fold risk/error
   Spearman was 0.08148; abstention did not reduce accepted error; the zero-return
   baseline beat Kronos at every declared coverage; the untouched holdout was not
   accessed.
4. The study found widespread OHLC structural invalidity in direct official Kronos
   output and verified that OpenAlpha introduced no mapping error.
5. Across the complete v0 development sample, 473 of 936 paths (50.53%), 961 of
   4,680 candles (20.53%), and 57 of 104 canonical forecasts (54.81%) were invalid.
6. Sentinel v1 found that terminal projection guarantees final validity, stepwise
   projection/re-encoding causes excessive hard failures, and validity-only
   resampling degrades high-low range quality.
7. Sentinel v1.1 measured Tokenizer-2k round-trip invalidity of 28.3659%,
   Tokenizer-base round-trip invalidity of 23.1445%, and raw generated-candle
   invalidity of 37.7778% in the fixed audit.
8. Unsupported pairs did not explain the defect; supported pairs were not safer;
   median bounded valid probability mass was approximately 60%.
9. The locked result was `TOKENIZER_CONSTRAINT_DEFECT`, and the inference-only
   constrained-decoding study stopped under its preregistered rules.
10. No untouched holdout result was accessed or used.

No commit or artifact supporting that sequence is rewritten or removed by this
pivot.

## Why the reliability-risk hypothesis stays retired

The v0 signal did not rank error strongly enough to support a reliability product.
Additional feature mining, model refitting, threshold changes, or holdout access
would violate the closure decision and would not address the newly isolated
mechanism. Bridge therefore does not revive Sentinel v0. Sentinel's retained job is
structural assurance and audit.

## Why naive constrained decoding stays stopped

The v1 methods acted during or after autoregressive token selection:

- Terminal projection fixed the returned candle but could not change information
  already represented in the token or undo invalid intermediate conditioning.
- Project/re-encode could send a projected candle through a tokenizer round trip
  that reintroduced invalidity and hard-failed 21 of 36 paths.
- Candidate resampling selected valid decoded candidates but changed the model
  distribution and failed the locked range-error gate.

The v1.1 result explains why these methods were unstable: the official continuous
decoder itself does not preserve the candle constraints. Bridge is therefore not a
new inference-time token filter. It is the separately versioned trained
reconstruction direction that v1.1 explicitly left outside the stopped track.

## Why the tokenizer decoder is the intervention point

Kronos-mini predicts the existing hierarchical coarse and fine identifiers. Those
identifiers are the vocabulary and latent geometry learned by the forecasting
transformer. A completely new tokenizer would produce different identifiers and
would break compatibility unless the forecasting model were substantially
retrained.

OpenAlpha instead freezes and retains:

- the official encoder needed to create training tokens;
- the official 10-bit coarse and 10-bit fine identifier spaces;
- the implicit binary-spherical codebook and frozen tokenizer trunk;
- the pretrained Kronos forecasting transformer and its weights;
- each original generated token sequence;
- the official tokenizer decoder and raw output for comparison.

Only a separately labeled constrained continuous reconstruction head is trainable.
The raw official result remains available and auditable.

## Selected approach

The pinned tokenizer decoder is causal and sequence-level. The selected Bridge-2K
architecture is therefore Candidate C: a sequence-level constrained reconstruction
head over the frozen official tokenizer decoder trunk.

The official token identifiers are converted by the pinned
`indices_to_bits(..., half=True)` path into the same 20-dimensional bipolar latent.
The frozen official `post_quant_embed` projection and causal tokenizer decoder
blocks produce the sequence state. A small OpenAlpha head consumes that state plus
causal normalization features and emits gap return, body return, nonnegative upper
wick, nonnegative lower wick, and optional nonnegative log-volume parameters.

The final OHLC(V) mapping is hard-constrained. There is no post-output repair step.

Candidate A is not selected because rebuilding the complete token-to-sequence trunk
would duplicate working official weights without evidence that it is needed.
Candidate B is not selected as the primary design because a candle-local residual
adapter would make the official sequence dependency implicit and create an unstable
boundary around invalid continuous values. The official decoded output remains a
baseline and audit input, not the sole Bridge representation.

## Bounded feasibility direction

Bridge-2K is the only authorized first target. It uses the pinned
Kronos-Tokenizer-2k and Kronos-mini revisions already audited by OpenAlpha.

The first complete Bridge-2K program is capped at 250,000 retrieved complete
candles, 10 GiB of temporary cache, one accelerator with no more than 24 GiB of
memory, 24 GPU-hours for Phase 2, one fixed small head, and no more than one million
trainable parameters or a 25 MiB Bridge checkpoint. Raw provider data and model
weights remain outside Git.

The reconstruction feasibility gate precedes forecast integration. The fixed
Sentinel development origins may be used only after reconstruction succeeds. The
untouched holdout remains prohibited.

## Continuation boundary

The exact numerical gates are locked in `research/bridge-v0/experiment.yaml`.
Bridge must have zero invalid output, beat terminal projection materially on range
reconstruction, remain competitive with the official decoder on OHLC and
close/return error, replay deterministically, fit the resource caps, and generalize
to declared unseen slices.

If the frozen tokens do not support a competitive safe reconstruction within the
locked architecture, data, and compute budget, Bridge research stops. OpenAlpha
then ships Sentinel validation, immutable audit, deterministic projection,
compatibility profiling, and the complete negative Bridge result. There is no
further conceptual pivot.

## Phase 0 exit

This phase ends after the source trace, mathematical contract, compatibility
boundary, bounded experiment, preregistration, and repository narrative are
committed. It does not implement the representation package, fetch training data,
fit a decoder, run a Bridge benchmark, generate a checkpoint, or access the
untouched holdout.
