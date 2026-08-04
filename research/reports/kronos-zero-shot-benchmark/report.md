# Frozen Kronos-base Zero-Shot Forecasts Do Not Beat Zero-Return Persistence

**Experiment:** `openalpha-kronos-zero-shot-benchmark-v1`
**Run:** `zsb_25e0256eefb2b07a`
**Evidence class:** `development_compatibility_canary`
**Claim boundary:** DEVELOPMENT BENCHMARK — NOT HOLDOUT OR TRADING EVIDENCE
**Status:** complete. One run, one immutable artifact, no retries.

---

## Abstract

Two completed structural-validity diagnostics had shown that Kronos tokenizer
round trips produce materially invalid OHLC candles, and that repairing that
invalidity changes forecast error by exactly zero. Those studies closed the
structural direction but left a fair question open: they tested one asset, one
origin, and a 448 → 64 horizon far longer than the published daily benchmark.
Perhaps the model simply had not been asked properly.

This benchmark asks it properly. Frozen Kronos-base, four liquid US equity index
ETFs, a 40-session context and a 12-session horizon, 25 chronological origins per
asset, two predeclared temperatures, an eight-seed ensemble — 1,600 generations
in total — scored against zero-return persistence and three other baselines that
cannot see the target.

It does not beat persistence. Median relative skill was negative under both
temperatures (−0.042 at T=0.6, −0.084 at T=1.0). Only 31% and 26% of origins beat
persistence against a preregistered 60% requirement. **Zero of four assets**
supported either configuration. Both moving-block confidence intervals excluded
zero on the *unfavorable* side. All four preregistered Z1 conditions failed under
both configurations, so the result is not a marginal miss on a single criterion.

The preregistered outcome is `PROCEED_TO_FROZEN_REPRESENTATION_PROBE` with the
zero-shot generation direction stopped.

---

## 1. What was preregistered, and when

The specification `research/bridge-v0/kronos-zero-shot-benchmark-v1.yaml`
(SHA-256 `6832c0f7befc54cd7ccec382db8cac1e6314f3fb356eb5eed8e9e9eca9a6fd07`) fixed
every asset, date, origin offset, temperature, seed, baseline, metric, threshold
and decision rule **before execution**. The digest is verified inside the
container at run time; a drifted document fails the run closed rather than
producing a result.

The design was corrected twice before it ever ran, both times to *weaken* the
statistical claim it could make. Neither correction touched a threshold, an
origin, an asset or a seed:

| Revision | Resampling unit | Why it was wrong |
| --- | --- | --- |
| initial | 100 independent asset-origin rows | The four ETFs share chronological windows; their errors are cross-sectionally dependent |
| second | 25 independent origin clusters | The origins overlap: a 40-session context on a 12-session stride means adjacent origins share 28 sessions |
| final | moving blocks of 4 consecutive origin clusters | Block length `ceil(40 / 12) = 4` spans every overlapping context relationship |

Each correction widened the interval. The final design's effective sample for
uncertainty is roughly 25/4 ≈ 6 temporal blocks, not 100 rows — the honest cost
of overlapping windows. Both superseded estimators were **removed from the
codebase**, not demoted, so no weaker interval can reach the decision layer.

---

## 2. Design

| | |
| --- | --- |
| Model | `NeoQuasar/Kronos-base` @ `2b554741eca47781b64468546e77fef3e85130e6`, frozen |
| Tokenizer | `NeoQuasar/Kronos-Tokenizer-base` @ `0e0117387f39004a9016484a186a908917e22426` |
| Parameters | 106,268,634 total, **0 trainable** |
| Assets | SPY, QQQ, IWM, DIA (1d, XNYS) |
| Retrieval | `2025-01-02` → `2026-07-01` exclusive; 373 sessions retrieved per asset, first 340 used |
| Window | 40-session context → 12-session horizon (52 of a 512 budget; no truncation) |
| Origins | 25 per asset at index offsets `40, 52, … 328`; target windows exactly disjoint |
| Configurations | A: T=0.6 · B: T=1.0 (both top-p 0.9, top-k 0, one sample per call) |
| Seeds | `20250102 … 20250109`, identical for every asset-origin-configuration |
| Primary candidate | 8-path ensemble mean |
| Secondary candidate | the single deterministic path (ensemble member zero) |
| Primary metric | `close_return_mae` on anchored log returns |
| Scale | 100 asset-origins × 2 configurations × 8 seeds = **1,600 generations** |

Origins are defined as **integer index offsets**, not calendar dates. That is the
only way one policy resolves identically for four symbols whose provider
histories may begin on different sessions, and it cannot be nudged after seeing a
result.

Dates chosen from 2025 onward to sit after the likely pretraining cutoff. That
cutoff is not published and cannot be verified — a precaution, not a proof, and
recorded as such before execution.

---

## 3. Observed evidence

Measurements only. Interpretation is in §4.

### 3.1 Headline

| | Config A (T=0.6) | Config B (T=1.0) |
| --- | --- | --- |
| Median relative skill | **−0.041743** | **−0.083634** |
| Fraction of origins beating persistence | **0.31** | **0.26** |
| Supporting assets (of 4) | **0** | **0** |
| Moving-block 95% interval | **[−0.000645, −0.000223]** | **[−0.001577, −0.000346]** |
| Excludes zero favorably | **false** | **false** |
| Ensemble mean `close_return_mae` | 0.00894610 | 0.00942205 |
| Persistence mean `close_return_mae` | 0.00853200 | (same baseline) |

Both intervals lie entirely **below** zero: persistence beat the model by a margin
the block resampling did not erase.

### 3.2 Distributions over the 100 asset-origins

| Quantity | Config A | Config B |
| --- | --- | --- |
| Ensemble relative skill | mean −0.06283, sd 0.13153, min −0.59302, max +0.20913 | mean −0.11582, sd 0.17673, min −0.79957, max +0.20795 |
| Single-path relative skill | mean −0.17698, median −0.13576 | mean −0.39224, median −0.26792 |
| Paired difference | mean −0.00041410, sd 0.00092742 | mean −0.00089006, sd 0.00130565 |

The 8-path ensemble beat the single sampled path under both temperatures, which
is what the prior study predicted and why the ensemble was preregistered as the
primary candidate.

### 3.3 Secondary metrics (ensemble, pooled)

| Metric | Config A | Config B |
| --- | --- | --- |
| `close_mae` | 9.2692 | 10.1428 |
| `close_return_rmse` | 0.011563 | 0.011994 |
| `directional_accuracy` | 0.52295 | 0.49288 |
| `median_absolute_return_error` | 0.0070054 | 0.0075905 |

### 3.4 Per asset (25 origins each)

| Config | Asset | Median skill | Fraction beating | Supports |
| --- | --- | --- | --- | --- |
| A | SPY | −0.04548 | 0.28 | false |
| A | QQQ | −0.06674 | 0.20 | false |
| A | IWM | −0.01114 | 0.44 | false |
| A | DIA | −0.04472 | 0.32 | false |
| B | SPY | −0.09972 | 0.20 | false |
| B | QQQ | −0.11965 | 0.16 | false |
| B | IWM | −0.05994 | 0.28 | false |
| B | DIA | −0.01492 | 0.40 | false |

No asset supported either configuration. The result is not carried or rescued by
any single instrument.

### 3.5 Baselines (none can access target data)

| Baseline | Mean `close_return_mae` | Mean skill vs persistence |
| --- | --- | --- |
| `zero_return_persistence` (primary) | 0.00853199 | 0.000000 |
| `last_close_level` | 0.00853199 | 0.000000 |
| `context_mean_return` | 0.00858524 | −0.003023 |
| `context_drift` | 0.00859725 | −0.003873 |

`last_close_level` is identical to persistence by construction; the invariant held
exactly. The two drifting baselines also failed to beat persistence, which says
the evaluation window is genuinely hard rather than the model being uniquely bad.

### 3.6 Forecast-step decay (Config A)

| Steps | Mean fraction beating persistence |
| --- | --- |
| 1–6 | 0.537 |
| 7–12 | 0.405 |

Per-step: 0.55, 0.58, 0.47, 0.52, 0.51, 0.59, 0.33, 0.34, 0.40, 0.44, 0.43, 0.49.
**This decomposition was not preregistered** and cannot support a shorter-horizon
claim. It is reported because pooled numbers hide it.

### 3.7 Execution integrity

```
provider requests            4 (exactly one per asset)
sessions retrieved / used    373 / 340 for every asset
first / last session used    2025-01-02 / 2026-05-12 (identical across all four)
cross-asset alignment        25/25 ordinals, target AND context dates aligned,
                             proved from retrieved sessions, not assumed
parameter hash before/after  8a056aaa…08fdb == 8a056aaa…08fdb
clipping                     0 clipped scalars of 24,000 — the clip never engaged
reproducibility              all 200 cells: single-path metrics == ensemble member 0 exactly
runtime                      257.5 s internal on a Tesla T4
peak GPU                     441,826,304 allocated / 463,470,592 reserved
```

### 3.8 Rule evaluations

| Rule | Matched | Finding |
| --- | --- | --- |
| Z5 | false | `BENCHMARK_INCONCLUSIVE` — zero limitations; the run was fully interpretable |
| Z1 | false | `ZERO_SHOT_SKILL_OBSERVED` — 0 of 2 configurations favorable |
| Z2 | false | `ISOLATED_CONFIGURATION_EFFECT` |
| Z3 | false | `ASSET_SPECIFIC_EFFECT` |
| **Z4** | **true** | **`NO_ZERO_SHOT_SKILL`** |

All four Z1 conditions failed under both configurations — median skill, fraction
of origins, asset support, and the interval. Z5 did not fire, so this is a real
negative rather than an uninterpretable run.

---

## 4. Interpretation

**4.1 The model does not forecast these windows better than assuming no change.**
That is the whole finding. It held under both temperatures, on all four assets, at
every level of aggregation, and against a baseline that requires no model at all.

**4.2 Lower temperature was consistently better, but not enough.** T=0.6 beat
T=1.0 on median skill (−0.042 vs −0.084), fraction beating (0.31 vs 0.26), and
every pooled secondary metric. Sampling noise is part of the problem. It is not
the whole problem, because reducing it did not cross zero.

**4.3 Averaging helps for the reason averaging usually helps.** The ensemble beat
the single path by a wide margin under both temperatures. That is variance
reduction, not signal — the ensemble still lost to persistence.

**4.4 The window is hard, not rigged against the model.** Two independent
non-persistence baselines also failed. Zero-return persistence is a strong daily
baseline; failing to beat it is common and is not on its own an indictment.

**4.5 The negative is clean.** Z5 did not fire; no limitation was recorded; the
provider behaved; alignment was proved at all 25 ordinals; parameters were
verifiably unmodified. There is no execution defect to blame.

**4.6 The widened interval was not the binding constraint.** A fair worry about
the moving-block correction was that it would manufacture a null by inflating
error bars. It did not: the point estimates are negative, the medians are
negative, the per-asset support count is zero, and the fraction beating
persistence is roughly half the threshold. Every criterion failed independently
of the interval.

**4.7 Direct generation is not the same hypothesis as representation quality.**
This benchmark measured what the model *emits*. Whether its hidden states encode
usable information is untested here, which is why the preregistered mapping for
Z4 routes to a representation probe rather than to abandoning the model outright.

---

## 5. Strongest supported conclusion

> Under a preregistered 40-session context and 12-session horizon, across four
> liquid US equity index ETFs and 25 chronological development origins each,
> frozen zero-shot Kronos-base forecasts did not beat a zero-return persistence
> baseline on the preregistered primary metric under either tested sampling
> temperature.

---

## 6. Limitations

Expanded in `limitations.md`.

1. **Low effective sample.** Uncertainty rests on ~6 independent temporal blocks.
2. **One horizon, one context length.** 40 → 12 only.
3. **Two temperatures.** No sweep; no top-p or top-k variation.
4. **Development evidence.** No holdout was opened.
5. **Sixteen months, one regime.** 2025-01 to 2026-05, four correlated large-cap
   US index ETFs — effectively close to a single market factor.
6. **Not the paper's protocol.** No claim of contradicting published results.
7. **Frozen weights, no adaptation.** No fine-tuning, calibration, or task head.
8. **One provider, one calendar, one adjustment convention.**
9. **Pretraining cutoff unverifiable.** Post-2024 dates are a precaution only.

---

## 7. What this does NOT establish

- **Not** that Kronos-base is useless at other horizons, assets or frequencies.
- **Not** that the published Kronos evaluation protocol would reproduce this.
- **Not** that Kronos-base would fail after task-specific fine-tuning.
- **Not** that Kronos internal representations carry no usable information.
- **Not** a trading, profitability, or economic conclusion of any kind.
- **Not** a held-out or out-of-sample scientific result.

Every authorization field in the artifact is `false`, including
`authorizes_trading_claims` and `scientific_result_available`.

---

## 8. Structural observations — descriptive only

Recorded because the benchmark saw them, and deliberately excluded from every
decision rule. `decide()` cannot receive a structural quantity: nothing in its
signature can carry one.

| | Config A | Config B |
| --- | --- | --- |
| Mean invalid paths per cell (of 8) | 2.59 | 5.13 |
| Mean per-path invalid candle fraction | 0.0619 | 0.1319 |
| Mean invalidity/error Spearman | 0.0213 | 0.1207 |

Structural invalidity persists at this shorter horizon and rises with
temperature, consistent with the two completed structural studies. **No path was
repaired, filtered, reordered or selected on validity** — `paths_repaired = 0` and
`paths_filtered_by_validity = 0` in every one of the 200 cells.

---

## 9. What follows

The preregistered outcome is `PROCEED_TO_FROZEN_REPRESENTATION_PROBE`, with
`zero_shot_generation_direction = STOP_KRONOS_ZERO_SHOT_DIRECTION`.

Direct generation quality and representation usefulness are different
hypotheses, so a direct failure stops the generation direction without closing
the representation question. A design note exists at
`research/bridge-v0/kronos-frozen-representation-probe-design.md`; it is **not
implemented and not authorized**, and would require its own preregistration.

Nothing in this report authorizes training, fine-tuning, Stage B, Stage C,
opening a holdout, production inference, or any trading claim.

---

## 10. Provenance

```
artifact  openalpha-compatibility/kronos-zero-shot-benchmark/runs/
          zsb_25e0256eefb2b07a/kronos_zero_shot_benchmark_terminal.json
sha256    93688f04ab2d887cacc56fad717d1cd2e018635b70e4a36d4e0cf480b265236a
schema    openalpha.bridge.zero_shot.kronos_zero_shot_benchmark.v1
commit    8a438cc5cf64462fcb04fcad995ef12ea4779246 (source == deployed)
spec      kronos-zero-shot-benchmark-v1.yaml @ 6832c0f7…fd07
completed 2026-08-03T23:51:49Z
```

Complete numeric detail — per-asset, per-origin, per-step, all 25 bootstrap
clusters and both interval blocks — is in `results-summary.json`, generated
directly from the artifact payload rather than transcribed. Reproduction
procedure is in `reproduction.md`.
