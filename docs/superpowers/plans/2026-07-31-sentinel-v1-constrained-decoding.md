# Sentinel v1 Constrained-Decoding Implementation Plan

**Goal:** Determine, on a bounded paired development sample, whether an invalid
Kronos candle can be prevented from contaminating later rollout steps.

**Architecture:** Keep the existing provider, structural validator, projection, and
artifact systems. Add a narrow experimental decoder worker beside the pinned worker,
a small method/metric module in `packages/sentinel`, and one resumable Phase 3B
runner. The cached official source and model weights remain read-only and outside
Git.

## Tasks

1. Add failing grammar and terminal-projection comparison tests.
2. Add failing synthetic token-loop tests for paired raw sampling, stepwise
   projection/re-encoding, bounded candidate selection, explicit fallback, final
   validation, audit records, and deterministic replay.
3. Implement the minimum model-independent method records and metric functions.
4. Implement the experimental worker at the pinned token-append interception point.
5. Prove the tokenizer round trip on one synthetic valid candle before real rollout.
6. Implement the resumable 12-origin runner with private causal data and compact
   committed outputs.
7. Run all four methods, preserve failures, and generate the locked paired metrics.
8. Run the model-size canary only if a mini in-loop method passes every gate.
9. Generate the conclusion and verification receipts; update docs and commit.

Each implementation step follows red-green-refactor. No holdout command or v0 model
fit is part of this plan.
