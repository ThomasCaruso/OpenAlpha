# ADR 0005: Pin Kronos and Evaluate After Its Pretraining Cutoff

- Status: Amended by ADR 0010
- Date: 2026-07-28

## Context

Kronos inference is stochastic, the official branch has changed since checkpoint publication, the released predictor does not set evaluation mode, and the paper reports pretraining through June 2024. Checkpoint metadata does not reconstruct the training dataset.

## Decision

- Pin a reviewed official Kronos commit.
- Pin model/tokenizer Hugging Face revisions and SHA-256 hashes.
- Set model/tokenizer to evaluation mode and record deterministic controls.
- Validate all wrapper assumptions outside the upstream predictor.
- Limit base/small causal context to 512 bars.
- Classify 2024-01-01 through 2024-06-30 as historical replay only, begin candidate sealed evaluation after 2024-06-30, and disclose residual contamination uncertainty.
- Support CPU with explicit progress/resource diagnostics and never substitute a fake forecast.

ADR 0010 fixes `NeoQuasar/Kronos-mini` as the only Sentinel v0 checkpoint because the immediate question is diagnostic feasibility under bounded local inference. Kronos-base is not a v0 flagship or fallback. Results always name the exact checkpoint.

## Consequences

- Runs remain comparable despite moving upstream code.
- Published-paper claims are not conflated with mini-checkpoint demo results.
- Historical evaluation before July 2024 is retained only as replay evidence and is not accepted as clean out-of-pretraining evidence.
- Exact cross-device equality is not promised.

## Sources

- [Kronos repository](https://github.com/shiyu-coder/Kronos)
- [Kronos paper](https://arxiv.org/html/2508.02739v1)
- [Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)

