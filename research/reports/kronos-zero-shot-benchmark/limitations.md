# Limitations and Non-Claims

Expanding §6 and §7 of `report.md`. This document exists so the boundary of the
evidence is recorded at least as carefully as the evidence itself.

---

## 1. Statistical limitations

### 1.1 Low effective sample size — the dominant limitation
Uncertainty rests on approximately 25/4 ≈ 6 independent temporal blocks, not on
100 observations. This is a deliberate consequence of the moving-block design:
with a 40-session context and a 12-session stride, consecutive origins share
information, and pretending otherwise would manufacture precision the data does
not contain.

The practical effect is that only a fairly large, consistent effect could have
satisfied Z1. **A modest real effect would not have been detected by this
benchmark.** That was known and accepted before execution.

The observed result is not close to the boundary — median skill negative, 0 of 4
assets supporting, fraction beating persistence roughly half the threshold — so
low power does not explain this particular outcome. It would matter a great deal
for interpreting a *near-miss*, and there wasn't one.

### 1.2 Block truncation is mildly conservative
Seven blocks of four give 28 positions, truncated to the first 25. Ordinals
appearing early in a drawn block are therefore very slightly over-weighted
relative to a circular scheme. The point estimate is unaffected because it is not
resampled. The interval is marginally wider than a circular-block alternative,
which was rejected because wraparound would join windows two years apart.

### 1.3 Two configurations, no sweep
Temperatures 0.6 and 1.0 only. Top-p fixed at 0.9, top-k at 0, one sample per
call. Adding temperatures after seeing results was prohibited by the
specification and did not happen. A wider configuration space is untested.

### 1.4 One horizon, one context length
40 → 12 only. Chosen before execution to sit materially closer to the published
daily benchmark than the completed 448 → 64 structural diagnostics. Other
context/horizon pairs are untested.

### 1.5 The step-decay decomposition is exploratory
Steps 1–6 beat persistence at 0.537 versus 0.405 for steps 7–12 (Config A). This
was **not preregistered**, is a post-hoc slice of a negative result, and must not
be read as evidence that a shorter horizon would succeed. Testing that requires
its own preregistration.

---

## 2. Data and regime limitations

### 2.1 Sixteen months, one regime
Sessions from 2025-01-02 to 2026-05-12. One macro regime, no crisis period, no
rate-shock, no prolonged bear market. Twenty-five origins inside sixteen months
is thorough coverage of a short window, not coverage of market history.

### 2.2 Four highly correlated instruments
SPY, QQQ, IWM and DIA are all large-cap US equity index ETFs. They are close to a
single market factor, which is precisely why the bootstrap clusters them. Four
assets should not be read as four independent tests. Nothing is established about
other asset classes, single names, international markets, or less liquid
instruments.

### 2.3 Pretraining cutoff is unverifiable
Post-2024 sessions were chosen as a precaution against evaluating on data the
model may have trained on. The Kronos pretraining cutoff is not published and
cannot be confirmed from the released artifacts. This is a best-effort
precaution, documented before execution, not proof of non-contamination.

### 2.4 One provider, one convention
A single data provider, one calendar policy, one adjustment convention. Vendors
disagree on adjusted equity history. Series digests pin exactly what was used but
do not establish vendor-independence.

---

## 3. Protocol and scope limitations

### 3.1 Not the published evaluation protocol
The Kronos paper's harness, preprocessing, universe, horizon definitions and
metric definitions were not replicated. **No disagreement with published results
is established or implied.** Any apparent tension is more likely protocol
difference than model behaviour.

### 3.2 Frozen weights, no adaptation
No fine-tuning, task head, calibration, or prompt engineering beyond the
preregistered configuration. Results say nothing about the model after
adaptation.

### 3.3 Direct generation only
This measured what the model emits. It did not extract or evaluate internal
representations. A model can generate poorly and still encode useful state —
which is exactly why the preregistered Z4 mapping routes to a representation
probe rather than to abandoning the model.

### 3.4 Persistence is a strong daily baseline
Zero-return persistence is hard to beat on daily equity index returns. Two other
context-only baselines also failed to beat it here. Failing to beat persistence
is common and is not, on its own, a remarkable indictment.

### 3.5 Development evidence only
No holdout partition was opened (`held_out_partition_opened = false`). No test
partition exists in this result to be contaminated, and none may be opened on the
strength of it.

### 3.6 Environment dependence
Tesla T4 (capability 7.5), PyTorch `2.13.0+cu126`, CUDA `12.6`, NumPy `2.5.1`.
Sampled paths may differ on other hardware. The analysis layer is deterministic
and hardware-independent.

---

## 4. Verifiability limitation

The terminal artifact lives in a private S3-compatible bucket. Its numeric
contents are mirrored into `results-summary.json` in this repository, generated
directly from the payload rather than transcribed — but an outside reader cannot
independently fetch the artifact and confirm its SHA-256. The specification
digest, the vendored source digests, the model and tokenizer digests, and the
complete analysis code and test suite **are** externally checkable.

---

## 5. Explicit non-claims

None of the following are supported. Each is listed because it is a plausible
misreading.

1. **Not** that Kronos-base is useless at other horizons, assets or frequencies.
2. **Not** that the published Kronos evaluation protocol would reproduce this.
3. **Not** that Kronos-base would fail after task-specific fine-tuning.
4. **Not** that Kronos internal representations carry no usable information.
5. **Not** a trading conclusion. No backtest, transaction costs, slippage,
   position sizing, risk model or execution assumptions exist anywhere in this
   study.
6. **Not** a held-out or out-of-sample scientific result.
7. **Not** a statement that foundation models cannot forecast markets. One model,
   one configuration space, one window.

---

## 6. Authorization state

```text
authorizes_training              = false
authorizes_fine_tuning           = false
authorizes_representation_probe  = false
authorizes_stage_b               = false
authorizes_stage_c               = false
authorizes_test_opening          = false
authorizes_production_inference  = false
authorizes_trading_claims        = false
scientific_result_available      = false

training_performed                     = false
optimizer_constructed                  = false
held_out_partition_opened              = false
market_backtest_performed              = false
forecasts_repaired                     = false
paths_filtered_by_structural_validity  = false
structural_validity_used_in_decision   = false
parameters_unmodified                  = true
trainable_parameter_count              = 0
```

The decision outcome `PROCEED_TO_FROZEN_REPRESENTATION_PROBE` names a possible
next study. It does not authorize one; that requires its own preregistration.

---

## 7. What would strengthen these findings

Recorded for completeness, not as a commitment.

- Many more origins spanning multiple market regimes, including a crisis period
- Genuinely uncorrelated instruments, not four US index ETFs
- A horizon and configuration sweep, preregistered in advance
- Replication of the published evaluation protocol
- Vendor-independent data
- A public evidence store so artifacts are externally verifiable
