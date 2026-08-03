# Structural Validity Is Not Forecast Skill: Frozen-Inference Diagnostics of Kronos-mini and Kronos-base

**Evidence class:** `development_compatibility_canary`
**Status:** closed. Both studies are complete and immutable.
**Claim boundary:** DEVELOPMENT DIAGNOSTIC — NOT HOLDOUT OR TRADING EVIDENCE.

---

## Abstract

We ran two preregistered frozen-inference diagnostics against released Kronos model/tokenizer
pairs — Kronos-mini with Kronos-Tokenizer-2k, and Kronos-base with Kronos-Tokenizer-base — at a
single SPY daily development origin under a 448-candle context and a 64-candle target horizon.
Both diagnostics measured the same four things: whether the tokenizer round trip preserves OHLC
structural validity (Method A), what a frozen official forecast produces and how it compares to a
zero-return persistence baseline (Method B), whether deterministic terminal projection can restore
structural validity and whether doing so improves forecast error (Method C), and whether filtering
a seeded rollout population by structural validity yields a usable ensemble (Method D).

Both models produced materially invalid OHLC candles through the tokenizer round trip alone —
24.6% of reconstructed candles for Kronos-mini and 12.3% for Kronos-base, against a preregistered
1% materiality threshold. Neither model produced a single structurally valid 64-candle rollout in
64 seeded attempts. In both studies, deterministic projection restored complete structural
validity at exactly zero change in the preregistered primary metric. No candidate forecast, in
either study, beat zero-return persistence.

The strongest supported conclusion is stated in §4. In short: structural invalidity was real and
material, and fixing it changed nothing about forecast accuracy. Structural validity and forecast
skill are separate properties, and the former was not the binding constraint.

---

## 1. Preregistration and scope

Both studies were preregistered before execution. Decision rules, thresholds, seeds, metrics,
model pins and source pins were fixed in specification files whose SHA-256 digests are recorded in
`artifact-manifest.json` and verified by the deployment gate at execution time.

| | Kronos-mini study | Kronos-base study |
| --- | --- | --- |
| Experiment ID | `bridge-v0 frozen inference diagnostic` | `openalpha-kronos-base-replication-v1` |
| Operative specification | `phase2-frozen-inference-diagnostic-v4.yaml` | `kronos-base-replication-v1.yaml` |
| Specification SHA-256 | `bd407722adfc3ebf92eb187828d42c2c9cfa57b2fdc0406121f27d39a5c44977` | `47ad7956adce368d5c6c69e54aea939ab6497bb4d411c96f894a4e30b7be58dd` |
| Authoritative branch | `feature/openalpha-kronos-constrained-inference` | `feature/openalpha-kronos-base-diagnostic` |
| Source commit | (historical branch head) | `0e6193baabd747dc0b2610a0ab80c3320e32fdd0` |
| Run ID | `canary_0a92fde788bd685c` | `base_03b08cbc706193d6` |
| Success schema | `openalpha.bridge.diagnostic.*` | `openalpha.bridge.base_study.kronos_base_diagnostic.v1` |
| Model | `NeoQuasar/Kronos-mini` | `NeoQuasar/Kronos-base` @ `2b554741eca47781b64468546e77fef3e85130e6` |
| Tokenizer | `NeoQuasar/Kronos-Tokenizer-2k` | `NeoQuasar/Kronos-Tokenizer-base` @ `0e0117387f39004a9016484a186a908917e22426` |

The two artifacts record `results_are_interchangeable: false`. They are separate studies against
separate released model pairs, with separate specifications, run-ID namespaces and artifact
namespaces. Neither amends the other.

### 1.1 Method definitions

Both studies share the same four-method structure.

- **Method A — tokenizer round trip.** Encode the observed candle window to discrete tokens and
  decode it back, with no forecasting involved. Measure reconstruction error and count OHLC
  inequality violations (`high < open`, `high < close`, `high < low`, `low > open`, `low > close`).
  This isolates the tokenizer from the forecasting model entirely.
- **Method B — official frozen forecast.** One deterministic seeded rollout through the official
  frozen inference path, scored against the target suffix and against a zero-return persistence
  baseline.
- **Method C — deterministic terminal projection.** Apply a deterministic repair
  (`TERMINAL_PROJECTION_V0`) to Method B's path, restoring the OHLC inequalities, and re-score. The
  question is whether structural repair buys forecast accuracy.
- **Method D — valid rollout filtering.** Generate 64 seeded rollouts, partition them by structural
  validity, and evaluate a valid-only ensemble against size-matched random controls.

### 1.2 Shared evaluation setting

Both diagnostics evaluated a single SPY daily development origin with a 448-candle context and a
64-candle target suffix, filling a 512-candle model context budget exactly, with no truncation.
The two studies report an identical zero-return persistence baseline
(`close_return_mae = 0.0033651655739541575`, `close_mae = 3.0446901321411133`, 64 scored steps),
which confirms they scored the same target window on the same data.

For the Kronos-base study, that window is SPY daily, calendar `XNYS`, 512 sessions from
`2015-05-07` through `2017-05-17`, candle data digest
`56c810e5c5d7cfa92062eaced416b6648581ad99b2fdf1563c501cc21ac7f88b`, with the target suffix
beginning `2017-02-15`. Normalization was fitted on the 448-row context only
(state digest `fc8a62300840eccaefd5ca7f26dbbf045546aedc3b433850c27512a0e71d6bc6`).

**This is a single development origin on a single asset.** That is the dominant limitation of both
studies and is treated at length in §5 and in `limitations.md`.

---

## 2. Observed evidence

Everything in this section is a recorded measurement from an immutable terminal artifact. No
interpretation appears here.

### 2.1 Required factual comparison

| Quantity | Kronos-mini | Kronos-base |
| --- | --- | --- |
| Full tokenizer round-trip invalid fraction (Method A) | `0.24609375` | `0.123046875` |
| Target-suffix invalid fraction (Method A) | `0.9375` | `0.609375` |
| Valid rollouts (Method D) | `0 / 64` | `0 / 64` |
| Method B close-return MAE | `0.007306554165097093` | `0.00920398038369002` |
| Persistence close-return MAE | `0.0033651655739541575` | `0.0033651655739541575` |
| Projection primary improvement (Method C) | `0.0` | `0.0` |
| Invalidity/error Spearman (Method D) | `0.04015798783221637` | `0.43092362328095035` |
| Manual ensemble close-return MAE (Method D) | not recorded in this comparison | `0.00487316858023227` |
| Best persistence skill | not recorded in this comparison | `-0.4481214885679963` |

Both round-trip invalid fractions exceed the preregistered materiality threshold of `0.01` — the
mini study by a factor of ~24.6, the base study by a factor of ~12.3.

### 2.2 Additional recorded evidence, Kronos-base study

Method A, tokenizer round trip only (no forecasting):

- 63 of 512 reconstructed candles structurally invalid (`0.123046875`); path invalid.
- Violation counts across the full window: `low_above_open` 38, `high_below_close` 35,
  `high_below_open` 17, `low_above_close` 5.
- Over the 64-row target suffix: 39 of 64 invalid (`0.609375`); violations `low_above_open` 23,
  `high_below_close` 23, `high_below_open` 12, `low_above_close` 5.
- Reconstruction error over the full sequence: internal return MAE
  `0.0028718265429328277`; per-column MAE open `0.7612`, high `0.5738`, low `0.6271`,
  close `0.5422`.

Clipping attribution (Method A) — the standardization clip is **not** a confound:

- Clipped input rows: 1 of 512 (`row_clipped_fraction = 0.001953125`), below the `0.01`
  materiality threshold. Two clipped scalars of 3072, in `volume` and `amount` only, at row 75.
- The 64-row target suffix had **zero** clipped rows and zero clipped scalars.
- Invalid rows with clipped input: `0`. Invalid rows with unclipped input: `63`, out of `511`
  unclipped rows.
- `invalid_fraction_given_clipped_input = 0.0`;
  `invalid_fraction_given_unclipped_input = 0.1232876712328767`.

Method B, official frozen forecast (seed `20150507`, temperature `1.0`, top-p `0.9`, top-k `0`,
one sample per call, `inference_mode`, parameters frozen):

- `close_return_mae 0.00920398038369002`, `close_mae 25.250720262527466`, 64 scored steps.
- Structural validity: 11 of 64 candles invalid (`0.171875`).
- Persistence comparison: `close_return_skill -1.735075045022258`,
  `close_level_skill -7.293362925826031`, directional accuracy `0.4375` (28 of 64).
- Total path sampling log-probability `-354.53529707053775`, measured after temperature scaling and
  after top-k/top-p filtering with renormalization.

Method C, deterministic terminal projection:

- Validity before: 11 of 64 invalid. Validity after: **0 of 64 invalid**, `path_is_invalid false`,
  `unrepairable_candle_count 0`.
- 11 candles adjusted, 15 fields adjusted, maximum absolute adjustment `1.1321563720703125`, total
  absolute adjustment `4.072662353515625`.
- `close_return_mae` before `0.00920398038369002`; after `0.00920398038369002`.
  `primary_error_improvement = 0.0`.

Method D, 64 seeded rollouts (seeds `20150507` … `20150570`):

- `valid_rollout_count 0`, `valid_rollout_fraction 0.0`, below the `0.25` support minimum.
- 64 distinct token paths, zero repeated paths — the rollout population is genuinely diverse.
- Per-rollout invalid candle counts: min 3, max 21, mean 10.09.
- Per-rollout `close_return_mae`: min `0.006944`, max `0.011572`, mean `0.008886`.
- `invalid_group_mean_primary_error 0.008886499571094574`;
  `valid_group_mean_primary_error null` (the valid group is empty).
- Manual seeded ensemble: `close_return_mae 0.00487316858023227`,
  `close_return_skill -0.4481214885679963`, directional accuracy `0.515625` (33 of 64).
- Size-matched controls: undefined, `k = 0`,
  `undefined_reason: "no valid rollouts, so there is no size to match"`.

Frozen-model integrity, Kronos-base study:

- `total_parameter_count 106268634`, `trainable_parameter_count 0`.
- `parameter_sha256` identical before and after inference:
  `8a056aaa47ee31de1b877d3f70ae4bc80d19969215577b90e765ea1541c08fdb`.
- `training_performed false`, `optimizer_constructed false`, `held_out_partition_opened false`.
- Provider requests: `1`. Runtime `181.430315 s` on a Tesla T4; peak GPU allocation
  `470,318,592` bytes, peak reservation `490,733,568` bytes.

### 2.3 Decision-rule evaluations

Kronos-base study, 16 preregistered rules at one forecast origin. Matched findings, in rule order:

| Rule | Finding | Observed |
| --- | --- | --- |
| R1 (primary) | `ROUNDTRIP_MATERIAL_INVALIDITY` | invalid fraction `0.123046875` vs threshold `0.01` |
| R2 | `ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED` | 63 invalid candles |
| R2a | `ROUNDTRIP_INVALIDITY_ON_UNCLIPPED_INPUTS` | 63 invalid rows with unclipped input, of 511 |
| R3 | `NO_VALID_ROLLOUTS_OBSERVED` | 0 valid of 64 |
| R6 | `PROJECTION_RESTORES_VALIDITY_WITHOUT_THRESHOLD_IMPROVEMENT` | error before = after |
| R8 | `NO_SKILL_AGAINST_PERSISTENCE` | best skill `-0.4481214885679963` vs threshold `0.0` |

Rules that did **not** match, and which therefore constrain the interpretation:

- `R0 DIAGNOSTIC_OPERATIONAL_FAILURE` — every method completed. The run is not a crash.
- `R0b REPRODUCIBILITY_FAILURE` — reproducibility check performed and agreed.
- `R2b ROUNDTRIP_INVALIDITY_CONFINED_TO_CLIPPED_INPUTS` — did not match, so the clip does not
  explain the invalidity.
- `R2c MATERIAL_CLIPPING_EXPOSURE_OBSERVED` — clipping exposure `0.001953125` was immaterial.
- `R2d ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING` — did not match; interpretation is not
  confounded by clipping.
- `R9 DIAGNOSTIC_INCONCLUSIVE` — did not match; the required quantities were computable.

Reproducibility, Kronos-base study: `performed true`, `agrees true`, detail `"identical"`. Coarse
tokens, fine tokens, raw decoded suffix, forecast metrics, sampling log-probabilities and validity
all agreed between Method B and the seed-matched Method D rollout zero.

---

## 3. Interpretation

This section is inference, not measurement. It is separated deliberately.

**3.1 The tokenizer, not the forecaster, is the source of structural invalidity.** Method A
involves no forecasting at all — it encodes observed market candles and decodes them straight back.
Both models produced materially invalid candles in that round trip. The discrete tokenization is
lossy in a way that does not respect the OHLC inequalities. Whatever the forecasting stack does
later, it is building on a representation that cannot exactly express the constraint set.

**3.2 Structural invalidity is not caused by the standardization clip.** In the Kronos-base study,
clipping touched one row out of 512, in `volume` and `amount` only, and touched the target suffix
not at all. Every one of the 63 invalid reconstructions came from an unclipped input. The
preregistered confound rules (R2b, R2d) were written to catch exactly this alternative explanation
and neither fired. The invalidity is a property of the tokenizer, not of our preprocessing.

**3.3 Repairing structure does not buy accuracy.** This is the central result. Method C restored
complete structural validity — 11 invalid candles to zero, no unrepairable candles — and the
primary metric did not move at all: `primary_error_improvement = 0.0` in both studies. The
adjustments were small (maximum `1.13`, total `4.07` across 15 fields) and orthogonal to the error
that actually dominates the forecast. A forecast can be perfectly well-formed and still wrong.

**3.4 Validity-based selection has no population to select from.** Method D found zero valid paths
in 64 diverse rollouts in both studies. Not a low fraction — zero. Rejection sampling, valid-only
ensembling and size-matched control comparison are all undefined in this regime, which is why the
size-matched control block reports `undefined_reason: "no valid rollouts, so there is no size to
match"`. Scaling the rollout count would not obviously fix this: the per-rollout invalid candle
count ranged from 3 to 21 with a mean of 10.09, so valid paths are not near-misses.

**3.5 The invalidity/error correlation differs between the two models and is not load-bearing.**
The Spearman correlation between structural invalidity and forecast error was `0.0402` for
Kronos-mini and `0.4309` for Kronos-base. The base value is moderately positive, which is
consistent with worse paths being both more invalid and less accurate — but with zero valid
rollouts there is no valid group to compare against, so this correlation cannot be converted into a
selection rule. It is descriptive.

**3.6 Neither model beat a trivial baseline at this origin.** Every candidate — raw Method B,
projected Method C, and the manual seeded ensemble — was worse than assuming zero return. The
ensemble was the least bad (`close_return_skill -0.448` versus `-1.735` for the single path), and
was the only candidate with directional accuracy above chance (`0.5156`), which is consistent with
averaging removing sampling noise rather than with the model carrying directional signal. At the
tested configuration, the persistence baseline dominates.

**3.7 Why the structural direction is closed.** The hypothesis under test was that structural
invalidity was a binding constraint on usable forecasts, and that repairing it would unlock skill.
Method C falsifies the second half of that directly and at zero cost: validity was fully restored
and the metric was unchanged. Every downstream structural intervention — constrained decoding,
projection variants, rejection sampling, valid-only selection, beam search over valid candles,
structural repair heads, OHLC inequality losses, probability weighting for validity — inherits the
same defect: it optimizes a property that has been measured not to control the primary metric.
Both studies independently recommended `ABANDON_STRUCTURAL_VALIDITY_DIRECTION`.

**3.8 The two studies agree despite differing in magnitude.** Kronos-base is roughly twice as good
as Kronos-mini at structural preservation (12.3% versus 24.6% round-trip invalidity; 60.9% versus
93.8% on the target suffix), and is the larger model — yet its Method B forecast error was
*higher* (`0.00920` versus `0.00731`) and it still produced zero valid rollouts. Improvement in
structural fidelity across the model family did not translate into forecast improvement. That
divergence is itself evidence that the two properties are decoupled.

---

## 4. Strongest supported conclusion

> Across the two tested released Kronos model/tokenizer pairs, OHLC structural invalidity was
> material, but deterministic restoration of complete structural validity produced no improvement
> in the preregistered primary forecast metric at the tested SPY development origin.

That is the whole of what the evidence supports. It is a statement about two model pairs, one
asset, one origin, one horizon configuration and one preregistered metric.

---

## 5. Limitations

Summarized here; expanded in `limitations.md`.

1. **One origin, one asset.** Both studies evaluated a single SPY daily development origin. There
   is no origin-level distribution, no confidence interval across origins, and no cross-asset
   evidence. A single window can be idiosyncratic.
2. **One horizon configuration.** 448 context to 64 target. This is far longer than the horizons
   used in the published Kronos daily benchmark, and long-horizon autoregressive rollout is the
   regime where sampled paths degrade most.
3. **One sampling configuration for the headline path.** Temperature `1.0`, top-p `0.9`, top-k `0`.
   Lower temperature was not tested, and would be expected to reduce both dispersion and
   structural violation rates.
4. **Not the paper's protocol.** We did not replicate the published evaluation harness,
   preprocessing, universe, horizon or metric definitions. Disagreement with published results is
   therefore not established.
5. **Frozen weights only.** No fine-tuning, no task adaptation, no calibration.
6. **Direct generation only.** We measured what the model emits, not what its internal
   representations contain.
7. **Development data only.** No holdout partition was opened, in either study.
8. **Provider-dependent.** One provider, one calendar policy, one price series.
9. **Persistence is a strong daily baseline.** Zero-return persistence is hard to beat on daily
   equity index returns; failing to beat it is common and is not by itself remarkable.
10. **Structural validity was defined by OHLC inequalities only.** Other notions of well-formedness
    were not tested.

---

## 6. What this evidence does NOT establish

Stated explicitly. None of the following are supported by these two studies:

- **Not** universal Kronos failure.
- **Not** failure across all horizons or all assets.
- **Not** failure under the paper's exact evaluation protocol.
- **Not** failure after task-specific fine-tuning.
- **Not** the absence of useful internal representations.
- **Not** a trading conclusion of any kind.
- **Not** a held-out scientific result.

Additionally, and to be unambiguous about the boundary: this evidence is **not** production
evidence, **not** trading evidence and **not** holdout evidence. It is development-stage
compatibility and diagnostic evidence. Every authorization field in both terminal artifacts is
`false` — `authorizes_training`, `authorizes_stage_b`, `authorizes_stage_c`,
`authorizes_test_opening`, `authorizes_production_inference` and `authorizes_trading_claims` — and
`scientific_result_available` is `false`.

---

## 7. What follows

Two hypotheses survive this work, and they are distinct from each other and from the closed
structural question.

**7.1 External validity of zero-shot generation.** The tested configuration (448 → 64, one origin,
one asset, temperature 1.0) is far from the published daily benchmark setting. Whether frozen
Kronos-base beats persistence under shorter, paper-style horizons across multiple chronological
origins and multiple assets is untested and is a fair question. This is the subject of the
successor study, `openalpha-kronos-zero-shot-benchmark-v1`, which is separately preregistered with
a disjoint identity and is **not** an amendment to either completed study.

**7.2 Usefulness of internal representations.** Direct generation quality and representation
quality are different hypotheses. A model can generate poorly and still encode useful state. A
frozen-representation probe would test this and is described as a design note only, in
`research/bridge-v0/kronos-frozen-representation-probe-design.md`. It is not implemented, and it is
explicitly not a continuation of structural repair.

Neither successor is authorized by this report. The zero-shot benchmark's own preregistered
decision rules govern what, if anything, comes after it.

---

## 8. Provenance

Full immutable identities — object keys, artifact digests, run IDs, experiment identities,
specification digests, model and tokenizer pins with config and weight digests, and source commits
— are recorded in `artifact-manifest.json`. Numeric results are duplicated in machine-readable form
in `results-summary.json`. Reproduction procedure is in `reproduction.md`.

Both completed studies are immutable. Their specifications, schemas, run IDs, source pins, model
pins, thresholds, seeds, conclusions and terminal objects must not be amended, retried, reopened,
reinterpreted or optimized. Repository tests pin these values.
