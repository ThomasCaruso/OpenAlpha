# Sentinel v1.1 Financial Token-Manifold Compatibility Design

## Status and claim boundary

This design defines the final bounded constrained-decoding investigation before
OpenAlpha either justifies one larger study or stops decoder research and ships a
validation and assurance library.

The study is **development compatibility research, not holdout evidence**. It may
read no Sentinel holdout origin and may not modify any Sentinel v0 or v1 file,
sealed record, result, or commit. Raw provider responses, reusable market histories,
model weights, and caches remain outside Git.

## Research questions

Primary:

> Does Kronos generate invalid financial candles because the tokenizer itself does
> not preserve OHLC constraints, because the autoregressive model generates
> unsupported hierarchical token combinations, or because too little model
> probability lies inside the valid financial output region?

Secondary:

> Can conditioning Kronos on both hard financial validity and empirical token
> support preserve forecast quality better than validity-only candidate
> resampling?

The implementation must preserve four distinct concepts: mathematical validity,
empirical tokenizer support, conditional model support, and distributional
distortion.

## Considered approaches

### Selected: staged support audit with bounded mass estimation

Audit tokenizer round trips first, then reuse the exact twelve v1 origins for raw
token-support and model-probability analysis. Estimate probability mass over
declared nested candidate grids of 64, 256, and 1,024 joint pairs. Report lower
bounds, upper bounds formed by adding uncovered mass, and an explicit truncation
flag. Implement a support-conditioned decoder only if the audited classification is
`OFF_MANIFOLD_TOKEN_COMBINATIONS`, `LOW_VALID_PROBABILITY_MASS`, or
`CANDIDATE_SEARCH_FAILURE`.

This approach directly tests the competing mechanisms while keeping inference and
candidate decoding bounded.

### Rejected: complete joint-vocabulary enumeration

The hierarchical space contains `vocab_s1 * vocab_s2` pairs, which is 1,048,576 for
the existing 10-bit/10-bit tokenizer. Exact fine-token conditionals for every
coarse token at every step would dominate this small study's runtime and memory.
Exact enumeration is not justified unless the bounded audit proves that uncovered
mass prevents classification; v1.1 does not authorize that expansion.

### Rejected: marginal-token or opaque compatibility score

Marginal coarse/fine occurrence is useful descriptive evidence but cannot establish
that an exact hierarchical pair is empirically supported. Combining validity,
frequency, model probability, and draft distance through a tuned score would make
the result difficult to interpret. The default support method therefore uses one
hard exact-pair floor and original model probabilities only.

## Immutable model and tokenizer identities

The study uses the pinned official source revision
`67b630e67f6a18c9e9be918d9b4337c960db1e9a` and these repository revisions:

| Component | Repository | Revision | Official parameter/context note |
|---|---|---|---|
| Mini model | `NeoQuasar/Kronos-mini` | `f4e68697d9d5aed55cef5c96aabc3376bcad9f81` | 4.1M, context up to 2,048 |
| Tokenizer 2k | `NeoQuasar/Kronos-Tokenizer-2k` | `26966d0035065a0cae0ebad7af8ece35bc1fb51c` | paired with mini |
| Small model | `NeoQuasar/Kronos-small` | `901c26c1332695a2a8f243eb2f37243a37bea320` | 24.7M, context 512 |
| Base model | `NeoQuasar/Kronos-base` | `2b554741eca47781b64468546e77fef3e85130e6` | 102.3M, context 512 |
| Tokenizer base | `NeoQuasar/Kronos-Tokenizer-base` | `0e0117387f39004a9016484a186a908917e22426` | paired with small/base |

File names, file hashes, installed dependencies, cache footprints, Python/PyTorch
versions, OS, and device are recorded after download and before execution. Cache
paths must resolve outside the repository.

## Fixed support corpus

The corpus contains exactly:

`SPY`, `QQQ`, `IWM`, `DIA`, `TLT`, `HYG`, `GLD`, `EFA`, `EEM`, and `XLF`.

The provider remains Yahoo Finance through pinned `yfinance==1.5.2` with the
existing explicit raw-daily request contract. Requests begin 2017-01-01 inclusive
and end 2024-06-29 exclusive. Every retained observation must be on or before
2024-06-28, before the earliest v1 origin. Each instrument must supply the final
1,536 complete expected XNYS sessions. Those sessions are divided into three
chronological, non-overlapping 512-session windows. Missing or duplicate sessions,
invalid OHLCV, post-boundary rows, or fewer than 1,536 complete sessions create an
explicit corpus failure; no replacement instrument is selected.

The corpus therefore contains 30 windows and 15,360 encoded positions per
tokenizer. Raw frames are ephemeral. Git may contain only request metadata, hashes,
quality summaries, token counts/frequencies, reconstruction metrics, and compact
derived audit rows.

## Exact tokenizer preprocessing

Each 512-session window follows the pinned official predictor boundary:

1. order columns as open, high, low, close, volume, amount;
2. derive amount in memory as `volume * mean(open, high, low, close)`, matching the
   pinned predictor;
3. cast to float32;
4. calculate feature-wise mean and population standard deviation over that window;
5. normalize with `(x - mean) / (std + 1e-5)`;
6. clip to the pinned predictor limit;
7. call tokenizer `encode(..., half=True)`;
8. call tokenizer `decode(..., half=True)` on the exact token sequence;
9. inverse-transform with the recorded mean and standard deviation.

The derived amount is model-wrapper state, not provider data, and is not described
as an observed market field. The audit validates reconstructed OHLCV and records
amount only as safe summary metadata.

## Phase 3C-A: tokenizer round-trip audit

For Tokenizer-2k and Tokenizer-base, preserve each window identity, normalized
input hash, token-sequence hash, reconstruction hash, shapes, dtypes, safe numeric
summaries, and structural-validity result. Calculate:

- invalid reconstructed-window and candle fractions;
- violation counts, categories, and normalized severities;
- normalized open, high, low, close, and high-low range reconstruction errors;
- coarse, fine, and exact-pair counts and frequencies;
- unique exact-pair counts;
- results by instrument;
- results by volatility regime.

Violation severity and reconstruction errors use the observed candle close as the
positive row-level denominator. Volatility regime uses causal trailing-20-session
close-return volatility. Low, middle, and high regimes are defined by the pooled
33 1/3% and 66 2/3% quantiles computed once from the fixed observed corpus and
shared across tokenizers.

For a tokenizer with `n` reconstructed candles and `k` invalid candles, calculate a
Wilson 95% interval. A tokenizer has a material round-trip defect when its invalid
candle fraction is at least 1.0% and the interval lower bound is at least 0.5%.
Reconstruction is described as overwhelmingly valid only when the invalid fraction
is at most 0.1% and the interval upper bound is at most 0.25%. Values between these
boundaries remain ambiguous rather than being rounded into a preferred conclusion.

The audit continues through compatibility diagnosis even when a tokenizer defect is
found, but no new decoder is implemented when the final primary classification is
`TOKENIZER_CONSTRAINT_DEFECT`.

## Empirical token support

Support is tokenizer-specific. For a generated token pair `(coarse, fine)`, record:

- whether the coarse token appeared and its count;
- whether the fine token appeared and its count;
- whether the exact pair appeared and its count;
- exact-pair frequency `count / N`;
- Jeffreys-smoothed pair probability
  `(count + 0.5) / (N + 0.5 * vocab_s1 * vocab_s2)`;
- smoothed surprisal as the negative natural logarithm of that probability.

The hard support definition used by the conditional method is exact-pair count at
least two. Marginal support never substitutes for exact-pair support. This floor and
smoothing constant are frozen before any generated-origin audit or outcome access.

## Phase 3C-B: generated token-support audit

Use the exact v1 origins and seeds:

- cutoffs 2024-07-05, 2024-09-13, 2024-11-22, 2025-01-31, 2025-04-11, and
  2025-06-20;
- assets SPY and QQQ;
- seeds 1729, 2027, and 7919;
- context 512 and horizon five.

Rerun raw mini generation only where the required logit/rank evidence was not
preserved. The regenerated raw path and token hashes must match the sealed v1
records. A mismatch is a terminal audit failure, never a replacement of v1.

At each of 180 generated steps record the raw pair, support fields, decoded
validity and severity, unfiltered temperature-one coarse and conditional-fine
log-probabilities/ranks, actual top-p sampling probabilities, asset, origin, seed,
and horizon step. Report valid versus invalid, supported versus unsupported,
frequency/surprisal groups, asset groups, and horizon groups. Later-step drift is
descriptive; no outcome-based threshold is selected.

## Phase 3C-C: bounded probability-mass audit

The original model-support fields use unfiltered temperature-one softmax. The mass
audit and conditional sampler use the actual locked sampling distribution after
temperature 1.0 and top-p 0.9 filtering at both hierarchical stages.

At every step evaluate nested grids:

| Budget | Coarse candidates | Fine candidates per coarse | Maximum pairs |
|---:|---:|---:|---:|
| 64 | 8 | 8 | 64 |
| 256 | 16 | 16 | 256 |
| 1,024 | 32 | 32 | 1,024 |

Candidates are ordered by joint sampling probability. For each budget record:

- considered probability mass;
- valid mass lower bound;
- supported mass lower bound;
- valid-supported mass lower bound;
- uncovered mass `1 - considered_mass`;
- corresponding upper bounds `min(1, lower_bound + uncovered_mass)`;
- raw selected-pair probability;
- renormalization factors;
- constraint tax `-log(valid_mass_lower_bound)`;
- support-constraint tax `-log(valid_supported_mass_lower_bound)`;
- whether the value is exact (always false unless considered mass is numerically
  one within `1e-12`).

Zero lower-bound mass produces positive infinity tax and an explicit
`ZERO_ESTIMATED_MASS` status. JSON artifacts store infinity as `null` plus the
status; no nonfinite JSON number is emitted. The study cannot expand beyond 1,024
pairs. Classification uses the 1,024-pair bounds and reports how much ambiguity the
uncovered tail leaves.

## Phase 3C-D: model-size and tokenizer canary

The canary uses cutoffs 2024-07-05 and 2025-04-11 for both SPY and QQQ, three seeds,
context 512, and horizon five. It compares mini/Tokenizer-2k,
small/Tokenizer-base, and base/Tokenizer-base. It records raw validity, invalid
candles, round-trip results, compatibility support, probability-mass bounds, cache,
runtime, peak process memory, and deterministic replay.

Probability grids stop at 256 pairs in the canary; the full 1,024 grid remains
mini-only. A model is skipped with an immutable operational record when any of
these predeclared limits is reached:

- cache requirement above 4 GiB;
- peak process memory above 8 GiB;
- first five-step path above ten minutes;
- projected remaining model run above four hours.

Small is attempted before base. A base skip does not silently reuse small evidence.

## Root-cause classification

Classification follows this order so overlapping findings cannot be selected
opportunistically:

1. **E - MINI_OR_TOKENIZER_2K_SPECIFIC:** Tokenizer-2k has a material round-trip
   defect or mini has material raw invalidity, while Tokenizer-base is
   overwhelmingly valid and both available larger models have invalid-candle rates
   no more than half mini's and no greater than 5%.
2. **A - TOKENIZER_CONSTRAINT_DEFECT:** at least one deployed tokenizer has a
   material round-trip defect and the evidence does not satisfy E.
3. **C - LOW_VALID_PROBABILITY_MASS:** round trips are not materially defective and
   the median valid-mass upper bound is at most 0.25, or at least 75% of steps have
   valid-mass upper bound below 0.50.
4. **B - OFF_MANIFOLD_TOKEN_COMBINATIONS:** valid-mass scarcity does not satisfy C,
   and either (a) at least 70% of invalid raw steps have exact-pair count below two
   with unsupported-pair invalid rate at least twice the supported-pair rate and at
   least ten observations in each group, or (b) median valid-mass lower bound is at
   least 0.25 while median valid-supported-mass upper bound is at most 0.25.
5. **D - CANDIDATE_SEARCH_FAILURE:** median valid-supported-mass lower bound is at
   least 0.25, median considered mass is at least 0.80, and the prior validity-only
   method still failed its locked range-quality gate.
6. **F - MIXED_OR_UNRESOLVED:** every remaining result, including insufficient
   canary evidence or bounds too wide to distinguish mechanisms.

The primary classification is selected mechanically. Secondary contributing
findings may be reported but cannot replace it.

## Conditional method gate

Implement `SUPPORT_CONDITIONED_SAMPLING_V0` only when the mechanical classification
is B, C, or D. It uses the maximum 1,024-pair grid, rejects invalid candidates and
exact pairs with corpus count below two, renormalizes original top-p joint
probabilities, and samples with a deterministic auxiliary uniform derived from
experiment hash, origin, seed, and step. Only the selected original token pair is
appended. An empty eligible set is a hard failure; there is no projection or
unconstrained fallback.

The optional draft-conditioned method is run only when support-conditioned sampling
has at least two eligible candidates at 80% or more of intervention steps. It
multiplies original eligible probabilities by
`exp(-0.5 * draft_distance_atr_units^2)`, where distance is OHLC root-mean-square
distance from the preserved raw draft candle divided by causal trailing-20-session
ATR. The formula and coefficient are fixed here; no outcome-based weight selection
is permitted.

## Paired method comparison

When the conditional method gate opens, use the same twelve origins, inputs,
timestamps, revisions, seeds, and realized outcomes to compare:

- `RAW_AUTOREGRESSIVE`;
- `TERMINAL_PROJECTION`;
- preserved prior `VALID_CANDIDATE_RESAMPLING`;
- `SUPPORT_CONDITIONED_SAMPLING_V0`;
- optional `DRAFT_CONDITIONED_SUPPORT_SAMPLING`.

Forecast artifacts are sealed before a logically separate resolver reads the
already declared five-session outcomes. Previous v1 outputs are referenced, not
rewritten. All structural, compatibility, path/range/wick/body, horizon, barrier,
return, diversity, probability-distortion, latency, memory, cache, and replay
metrics named in the user specification are reported for every applicable method.

## Locked continuation rule

A method can justify a larger, separately preregistered experiment only when all
conditions pass over 36 paired paths:

1. every returned path is structurally valid;
2. hard failures are at most 1/36;
3. high-low range MAE is no greater than `0.009934683599145366`, exactly 95% of the
   prior validity-only result `0.010457561683310912`;
4. close MAE is no greater than both 110% of raw v1 close MAE and raw v1 close MAE
   plus 0.001;
5. mean pairwise path diversity is at least 50% of raw v1, final-return variance is
   positive, and repeated-path rate is at most 10%;
6. mean path distance from the raw draft is no greater than the prior validity-only
   value `0.00894289586146275`;
7. mean selected-token rank is no greater than the prior value
   `9.318181818181818`;
8. two identical synthetic replays and two identical real replays hash exactly;
9. median latency is at most five times paired raw latency, cache is at most 4 GiB,
   and peak process memory is at most twice raw plus 1 GiB;
10. model weights and tokenizers remain unchanged.

The rule does not require superiority on every forecast metric. Results at every
declared barrier and horizon step are published regardless of direction.

## Final decision

If one conditional method passes all gates, the report recommends one separately
preregistered larger experiment and stops before running it. Otherwise constrained
decoder research stops. Sentinel's supported product becomes:

- a financial forecast structural validator;
- an immutable raw-output audit layer;
- a deterministic terminal-projection gateway;
- a token-manifold compatibility profiler;
- a model-selection safety check.

When A is selected, the report may describe a future constraint-preserving
parameterization based on open/open return, candle body, nonnegative upper/lower
wicks, and nonnegative volume. No tokenizer or model training is authorized in
v1.1.

## Required research outputs

The bounded study creates `docs/SENTINEL_TOKEN_MANIFOLD.md`,
`docs/SENTINEL_CONSTRAINT_COMPATIBILITY.md`, and the versioned
`research/sentinel-v1_1/` tree containing `experiment.yaml`,
`preregistration.md`, `tokenizer_roundtrip/`, `token_support/`,
`probability_mass/`, `model_size_canary/`, `method_comparison/`, and
`report.md`. It updates `docs/MASTER_PLAN.md`, `docs/SENTINEL_DIRECTION.md`,
`docs/SENTINEL_METHODOLOGY.md`, and `docs/STATUS.md` without changing any v0 or
v1 artifact.

Every result directory contains only compact derived records, manifests, and
hashes permitted by project policy. Empty or inapplicable conditional-method
outputs remain explicit gated-status artifacts rather than fabricated results.

## Implementation boundaries

Create only focused v1.1 modules for corpus/token support, round-trip metrics,
probability-mass bounds, classification, and the conditional method when gated.
Reuse the v1 grammar, projection, path metrics, data provider, canonical
serialization, and private model process. Do not extend the v1 runner or edit v1
research artifacts. The v1.1 CLI exposes only fixed operations and has no holdout,
symbol, cutoff, or arbitrary-method override.

Every origin and audit stage is checkpointed with content hashes. Provider/model
failures, hash mismatches, nonfinite outputs, zero estimated mass, unavailable
models, and operational skips are explicit terminal records. No silent provider,
model, tokenizer, candidate-budget, or method fallback exists.

## Testing and verification

Production behavior is developed test-first. Deterministic synthetic fixtures cover:

- fixed corpus/window selection and no post-boundary rows;
- exact pair counts, support floor, smoothing, and surprisal;
- round-trip reconstruction errors and Wilson intervals;
- structural severity and regime grouping;
- probability lower/upper bounds, uncovered mass, zero-mass status, and tax;
- mechanical classification precedence;
- deterministic conditional sampling and hard failure on empty support;
- no-holdout CLI surface;
- forecast-before-outcome sealing;
- manifest hashes and forbidden-artifact policy.

Final verification runs targeted tests, the complete existing suite, Ruff, Pyright,
experiment-hash verification, artifact and ledger verification, policy scans,
unchanged v0/v1 tree checks, and `git diff --check` before the result commit.
