# Limitations and Non-Claims

This document expands §5 and §6 of `report.md`. It exists so that the boundary of the evidence is
recorded at least as carefully as the evidence itself.

---

## 1. Statistical and design limitations

### 1.1 Single forecast origin
Both studies evaluated **one** origin. `forecast_origins = 1`. There is no distribution across
origins, no standard error, no confidence interval and no bootstrap. Every headline number is a
point estimate from a single window. A single 64-day window on a single index can be idiosyncratic
in volatility, drift and regime.

This is the single most important limitation. It applies to every quantity in the report, including
the persistence comparison.

### 1.2 Single asset
SPY only. No cross-sectional evidence. Nothing is established about other equity ETFs, individual
equities, other asset classes, or less liquid instruments.

### 1.3 Single horizon configuration
448 context candles to 64 target candles, filling a 512 budget exactly. A 64-step autoregressive
daily rollout is a long horizon, and it is the regime in which sampled paths accumulate the most
drift. The published Kronos daily benchmark uses materially shorter horizons. Our configuration was
chosen to fill the context budget exactly, not to match the paper.

### 1.4 Single sampling configuration for the headline result
Temperature `1.0`, top-p `0.9`, top-k `0`, one sample per call. Lower temperature was not tested and
would be expected to reduce both path dispersion and structural violation rates. The study does not
establish that structural invalidity persists at low temperature.

### 1.5 Rollout population size
64 seeded rollouts per study. Zero were valid in both. This bounds the valid-path rate but does not
prove it is exactly zero — it establishes that it is low enough that 64 diverse samples found none.
The per-rollout invalid candle counts (3 to 21, mean 10.09 for Kronos-base) indicate valid paths
were not near-misses, which makes a much larger sample unlikely to change the conclusion, but that
is an extrapolation, not a measurement.

### 1.6 Size-matched controls are undefined
Because zero rollouts were valid, `k = 0` and the size-matched control block is undefined
(`"no valid rollouts, so there is no size to match"`). Rule R4 could not be evaluated on its merits.
The absence of a valid-only result is a structural feature of the regime, not a null finding about
valid-only ensembling in general.

### 1.7 Persistence is a strong baseline
Zero-return persistence is difficult to beat on daily equity index returns. Failing to beat it is
common for many methods and is not, on its own, a strong indictment. The result is reported because
it was preregistered, not because it is surprising.

---

## 2. Protocol and comparability limitations

### 2.1 Not the published evaluation protocol
We did not replicate the Kronos paper's evaluation harness, preprocessing pipeline, instrument
universe, horizon definitions, normalization scheme or metric definitions. **No disagreement with
published results is established or implied.** Any apparent tension between our numbers and
published numbers is more likely to reflect protocol difference than model behaviour.

### 2.2 Frozen weights only
No fine-tuning, no task adaptation, no head training, no calibration, no prompt or context
engineering beyond the preregistered configuration. Results say nothing about the model's behaviour
after adaptation.

### 2.3 Direct generation only
We measured emitted candle sequences. We did not extract, probe or evaluate internal
representations. Direct generation quality and representation quality are separate hypotheses, and
this work bears only on the former.

### 2.4 Structural validity defined narrowly
"Valid" means the OHLC inequalities hold: `high >= max(open, close)`, `low <= min(open, close)`,
`high >= low`. Other notions of well-formedness — volume/amount consistency, gap plausibility,
realistic intrabar ranges, distributional calibration — were not tested.

### 2.5 Provider dependence
One data provider, one calendar policy (`XNYS`), one adjustment convention. Different vendors
disagree on adjusted equity history. The candle matrix digest pins what we used
(`56c810e5…f88b` for the Kronos-base study) but does not establish vendor-independence.

### 2.6 Two model pairs only
Kronos-mini with Kronos-Tokenizer-2k, and Kronos-base with Kronos-Tokenizer-base. Other released
sizes and other tokenizer pairings were not tested. Notably, the two tested pairs differ in both
model and tokenizer simultaneously, so model size and tokenizer version are confounded in the
mini-versus-base comparison.

### 2.7 Environment dependence
Tesla T4 (capability 7.5), PyTorch `2.13.0+cu126`, CUDA `12.6`, NumPy `2.5.1`. Sampled paths may
differ on other hardware or library versions.

---

## 3. Interpretive limitations

### 3.1 The mini-versus-base comparison is not a clean scaling experiment
Kronos-base showed roughly half the round-trip invalidity of Kronos-mini but *higher* Method B
forecast error. Because model size and tokenizer changed together, and because each study has a
single origin, this pattern cannot be attributed to scale.

### 3.2 The invalidity/error correlation is descriptive only
Spearman `0.0402` (mini) and `0.4309` (base). These are within-study, single-origin, rank
correlations over 64 rollouts. They are recorded for completeness. With zero valid rollouts there is
no valid group to contrast, so the correlation cannot be converted into a selection rule, and it
should not be read as a causal claim in either direction.

### 3.3 Projection is one repair method, not all repair methods
Method C used `TERMINAL_PROJECTION_V0`, a deterministic minimal-adjustment repair. It restored
validity completely with zero unrepairable candles and zero change in the primary metric. A
different repair could in principle change the metric — but it would be changing the forecast, not
merely enforcing structure, and the preregistered finding is specifically that enforcing structure
is metric-neutral here.

### 3.4 Development-stage evidence only
No holdout partition was opened in either study (`held_out_partition_opened = false`). No test
partition exists in these results to be contaminated, and none may be opened on the strength of
them.

---

## 4. Explicit non-claims

The evidence does **not** establish any of the following. Each is listed because it is a plausible
misreading of the results.

1. **Universal Kronos failure.** Two model pairs, one asset, one origin, one horizon, one
   temperature. That is not a general claim about the model family.
2. **Failure across all horizons or assets.** Untested. Shorter horizons and other assets are
   specifically the subject of the successor benchmark.
3. **Failure under the paper's exact evaluation protocol.** We did not run that protocol.
4. **Failure after task-specific fine-tuning.** No adaptation was performed.
5. **Absence of useful internal representations.** Representations were never examined.
6. **A trading conclusion.** No backtest, no transaction costs, no slippage, no position sizing, no
   risk model, no execution assumptions. Nothing here supports any statement about tradability.
7. **A held-out scientific result.** This is development evidence. `scientific_result_available` is
   `false` in both artifacts.

Additionally, these results are **not** production evidence, **not** trading evidence and **not**
holdout evidence, and must never be relabelled as such.

---

## 5. Authorization state

Every authorization field in both terminal artifacts is `false`:

```text
authorizes_training             = false
authorizes_stage_b              = false
authorizes_stage_c              = false
authorizes_test_opening         = false
authorizes_production_inference = false
authorizes_trading_claims       = false
scientific_result_available     = false
```

Supporting integrity fields, Kronos-base study:

```text
training_performed        = false
optimizer_constructed     = false
held_out_partition_opened = false
parameters_unmodified     = true
trainable_parameter_count = 0
```

Nothing in this report authorizes training, Stage B, Stage C, opening a test or held-out partition,
production inference or any trading claim. The successor zero-shot benchmark carries its own
preregistered decision rules and its own authorization boundary; nothing here pre-authorizes its
outcome.

---

## 6. What would be required to strengthen these findings

Recorded for completeness, and explicitly **not** a work plan — the structural direction is closed.

- Many chronological origins, with paired statistics and bootstrap intervals.
- Multiple assets under one identical policy.
- Shorter, paper-style horizons.
- A temperature sweep, including low temperature.
- Replication of the published evaluation protocol.
- Vendor-independent data.

The first three of these are addressed, for the *forecast-skill* question rather than the
structural question, by `openalpha-kronos-zero-shot-benchmark-v1`. None of them reopens the
structural-validity hypothesis.
