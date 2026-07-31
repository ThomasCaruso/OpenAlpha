# Project Status

Last updated: 2026-07-30

## Product direction

OpenAlpha has made its second and final pivot to **OpenAlpha Sentinel**: a forecast-time reliability, failure-detection, and intervention layer. Kronos is the first forecast provider and case study.

Benchmarking remains necessary for errors, labels, baselines, and evaluation. It is no longer the product.

## Preserved completed work

- `c037bd5efafd728cb4e0df961dd6a93889b0ce56` is preserved.
- `27f0c69d73e2de678300c44a7899de4295a310c5` is preserved.
- `60a0682faf327e2d7ff7cf019a6c41a8e8d6df8c` is preserved.
- `480c0e7e90d2451deaa0463ab253d07826522e8f` is preserved.
- Content-addressed artifacts, path confinement, immutable run journals, verified manifests, and artifact verification remain unchanged.
- Existing tests, Ruff, and Pyright configuration remain unchanged.
- The provider-independent market-data boundary remains.
- Phase 2 uses a pinned yfinance adapter through the same provider-independent boundary; Alpaca remains a later independent verification provider.
- No silent provider fallback exists.

## Sentinel Phase 1

The worktree now defines:

- final direction and value test;
- controlled failure-reason taxonomy;
- exact Sentinel v0 methodology;
- development period 2024-07-01 through 2025-06-30;
- untouched holdout 2025-07-01 through 2026-06-30;
- weekly SPY/QQQ five-session experiment;
- nine-path Kronos ensemble and zero-return baseline;
- twenty-five candidate diagnostics after the Phase 2.5 structural-validity amendment;
- continuous error, worst-development-quartile failure, and baseline-relative labels;
- logistic/ridge risk configuration;
- USE, 50/50 BLEND, and ABSTAIN policy;
- required evaluation outputs and numeric continuation rules;
- minimal internal provider/decision contracts;
- research workspace and experiment YAML;
- archived unimplemented Kronos Reality Check plan and design.

Phase 1 remains locked. Phase 2 added one narrow internal package and completed only the single predeclared SPY origin described below; the SPY/QQQ development sample has not begun.

## Verified provider and model facts

- yfinance 1.5.2 is pinned in `pyproject.toml` and `uv.lock`.
- yfinance is an open-source client using Yahoo Finance as its underlying source; it is not an official Yahoo integration and is intended for research/education.
- Its documented download interface treats start as inclusive and end as exclusive and exposes explicit interval, adjustment, repair, actions, progress, threads, and timeout controls.
- Phase 2 excludes Adj Close and disables automatic/back adjustment and repair.
- Alpaca remains a separately sourced later verification provider; Phase 2 makes no provider-independent claim.
- The official Kronos-mini model card declares Kronos-Tokenizer-2k, time-series forecasting, MIT license, 4,108,192 parameters, and no deployed Hugging Face Inference Provider.
- Current feasibility pins are source `67b630e67f6a18c9e9be918d9b4337c960db1e9a`, model `f4e68697d9d5aed55cef5c96aabc3376bcad9f81`, and tokenizer `26966d0035065a0cae0ebad7af8ece35bc1fb51c`.
- Reported Hub storage is 16,440,776 model bytes plus 15,842,376 tokenizer bytes.

## Resolved Phase 2 operational questions

- Python 3.11.15 and PyTorch 2.13.0+cpu loaded the pinned source/model/tokenizer on this Windows CPU host.
- Every Sentinel call used `sample_count=1` and retained its individual path.
- Resetting Python, NumPy, PyTorch CPU, and accelerator RNGs produced identical A/B hashes for context 512/seed 1729; seed 2027 produced a different hash.
- The nine official calls completed in 163.56 to 680.03 ms of provider-measured inference time per path.
- The exact model/tokenizer filenames and hashes are recorded below and in the audit.
- The Hugging Face cache footprint was 32,283,678 bytes outside Git.
- yfinance returned 526 complete XNYS sessions from 2022-06-01 through 2024-07-05.
- The official Kronos predictor accepts OHLCV from Sentinel but internally derives an `amount` feature. Sentinel did not request, manufacture, persist, or redistribute a provider amount column. This behavior is a disclosed model-wrapper limitation.

## Evidence status

- Protocol v1 is unfrozen and its implementation task is paused.
- One real, historical, development-only SPY forecast origin has completed.
- Model weights and caches exist only under `C:\Users\Tommy\.cache\openalpha-sentinel\phase2`, outside Git and the worktree.
- Yahoo response frames were ephemeral. No raw response, reusable market dataset, CSV export, or yfinance cache is tracked.
- A real Kronos forecast, causal diagnostic vector, separately resolved outcome, append-only ledger, completed journal, and verified manifest now exist for this one origin.
- No Sentinel risk model, action, development sample, holdout result, provider-independent result, or empirical Sentinel efficacy claim exists.

## Phase 2 pre-execution amendment

Before any market-data request, model download, or Kronos inference, Sentinel v0 was amended to use raw OHLCV and raw five-session close-to-close log return. Only the three 512-context paths now form the canonical timestamp-wise averaged close path; 128/256-context paths are stress tests only.

- Previous experiment SHA-256: `c6459409e6b750217b4e81d09d461dc4b8cf303746d3efdcc8de80762e5a763c`.
- Amended experiment SHA-256: `261e4bac51b9b3d6b68b6cf0128d406ca63cbf4bcd9735ccbda1aa04d1fa761a`.
- Exact field changes and rationale: `research/sentinel-v0/amendments/2026-07-30-raw-bars-and-canonical-forecast.md`.
- Dividend omission and possible raw corporate-action discontinuities are known limitations; no corporate-action engine is in scope.
- No Phase 2 action is emitted.

## Verification evidence

Fresh Phase 1 verification on 2026-07-30:

- `uv sync --locked --group dev` — exit 0; resolved 26 packages and checked 25 packages.
- `uv run pytest -q` — exit 0; 220 passed in 2.84 seconds.
- `uv run ruff check .` — exit 0; `All checks passed!`
- `uv run pyright packages/research-core packages/experiment-spec` — exit 0; 0 errors, 0 warnings, 0 informations.
- `git diff --check` — exit 0; Git emitted only working-copy CRLF-to-LF normalization notices for three amended ADR files.
- Sentinel YAML lock assertion — exit 0; `sentinel experiment lock: valid`.
- Active-document scope/encoding assertion — exit 0; 27 files valid.

These gates verify the preserved code and the Phase 1 document/configuration lock. They do not constitute real-data, real-inference, or empirical Sentinel evidence.

Pre-execution amendment verification on 2026-07-30:

- `uv run pytest -q` — exit 0; 220 passed in 3.04 seconds.
- `uv run ruff check .` — exit 0; all checks passed.
- `uv run pyright packages/research-core packages/experiment-spec` — exit 0; 0 errors, 0 warnings, 0 informations.
- `git diff --check` — exit 0.
- Exact amendment contract assertions — exit 0.
- `experiment.sha256` byte check — exit 0; `261e4bac51b9b3d6b68b6cf0128d406ca63cbf4bcd9735ccbda1aa04d1fa761a`.

No external request or inference was made while producing or verifying this amendment.

## Phase 2 yfinance provider amendment

Before any market-data request, model download, or Kronos inference, Sentinel v0 replaced Alpaca as the Phase 2 default with pinned `yfinance==1.5.2` over Yahoo Finance's unofficial public interface. The provider port is preserved, no fallback exists, and Alpaca remains a required later independent verification path.

- Previous experiment SHA-256: `261e4bac51b9b3d6b68b6cf0128d406ca63cbf4bcd9735ccbda1aa04d1fa761a`.
- Amended experiment SHA-256: `587bc36ac1dd941c66d9e94c92d7b26a33faeb8e7e7a38883762f0cb54897950`.
- Exact field changes and rationale: `research/sentinel-v0/amendments/2026-07-30-yfinance-development-provider.md`.
- Start is inclusive, end is exclusive, and 2024-07-05 remains the exact locally enforced XNYS input cutoff.
- Automatic/back adjustment and repair are disabled; actions are audit warnings only; Adj Close is excluded.
- Raw responses, reusable histories, caches, and CSV exports remain outside Git and cannot be redistributed under project policy.
- Cross-provider verification is explicitly deferred and required before serious publication.
- No market-data request, model download, or inference occurred while creating this amendment.

Pre-execution yfinance amendment verification on 2026-07-30:

- `uv lock` — exit 0; resolved 51 packages and locked `yfinance==1.5.2`.
- `uv sync --locked --group dev --group sentinel-phase2` — exit 0; resolved 51 packages, prepared 21, and installed 25.
- `uv run pytest -q` — exit 0; 220 passed in 7.12 seconds.
- `uv run ruff check .` — exit 0; `All checks passed!`
- `uv run pyright packages/research-core packages/experiment-spec` — exit 0; 0 errors, 0 warnings, 0 informations.
- `git diff --check` — exit 0.
- yfinance import/signature check — exit 0; Python 3.13.2, yfinance 1.5.2, and every required download argument is exposed by the installed client.
- Sentinel yfinance YAML/contract/hash assertion — exit 0; `sentinel yfinance experiment lock: valid`.
- `experiment.sha256` byte check — exit 0; `587bc36ac1dd941c66d9e94c92d7b26a33faeb8e7e7a38883762f0cb54897950`.

These checks verify only the pre-execution provider amendment. They are not market-data, inference, or Sentinel-performance evidence.

## Phase 2 one-origin result

**Claim boundary: DEVELOPMENT PROOF — NOT EMPIRICAL EVIDENCE.**

The `create` command fetched only causal history, ran the seed probe and official nine paths, computed diagnostics, and sealed the forecast. An independent reload verified the creation seal before the separately imported `resolve` operation requested any outcome row.

### Causal data

- Provider/client: Yahoo Finance through `yfinance==1.5.2`.
- Explicit request: SPY, `interval=1d`, start 2022-06-01 inclusive, end 2024-07-06 exclusive, `auto_adjust=false`, `back_adjust=false`, `repair=false`, `actions=true`, `progress=false`, `threads=false`, timeout 30.0 seconds.
- Validated rows: 526 complete XNYS sessions.
- First/final session: 2022-06-01 / 2024-07-05.
- Normalized input SHA-256: `f09446b7f7d541907401ca133580eac205c99acb5e27d84bf922943cc4d57bbe`.
- Nine dividends were recorded as warnings. Adj Close was not an input.

### Model environment and files

- Source: `shiyu-coder/Kronos@67b630e67f6a18c9e9be918d9b4337c960db1e9a`.
- Model: `NeoQuasar/Kronos-mini@f4e68697d9d5aed55cef5c96aabc3376bcad9f81`.
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-2k@26966d0035065a0cae0ebad7af8ece35bc1fb51c`.
- Environment: Python 3.11.15, PyTorch 2.13.0+cpu, NumPy 2.2.6, pandas 2.2.2, CPU, Windows `10.0.26200`.
- Model `config.json`: 225 bytes, SHA-256 `70daca2cb11e3a979dd6b8ac12ee08e2aace877acf28f5b8dfb4fe5609736201`.
- Model `model.safetensors`: 16,440,776 bytes, SHA-256 `a7d5f37e2e9fbd9891f7d7d4f72574512dd1f704fee14223e0a8cd0fbf54197c`.
- Tokenizer `config.json`: 301 bytes, SHA-256 `0b30a443affb03e05a876a083857de9164f899feb7b4d261da02c485c9a3e3b6`.
- Tokenizer `model.safetensors`: 15,842,376 bytes, SHA-256 `b97ec46b3b72160509e289183eaf7bdf5f0dac5bb9b49522f6d46638a99a8717`.
- Hugging Face cache path: `C:\Users\Tommy\.cache\openalpha-sentinel\phase2\hf`; 32,283,678 bytes.

### Reproducibility and official paths

- Probe A (512/1729): `fdf503bcb82d38f361c8a1231818e0b1fd1284cd3015344fefad7ba47dea97ed`.
- Probe B (512/1729): `fdf503bcb82d38f361c8a1231818e0b1fd1284cd3015344fefad7ba47dea97ed`.
- Probe C (512/2027): `68d31144b7b53c8f67a2aeb0d54c5a294d5ca9cb894f5a5e6f9ea5172f7f00be`.
- A/B matched; C differed. Exact seeded replay is supported in this recorded environment.

| Context | Seed | Predicted raw log return | Inference ms |
|---:|---:|---:|---:|
| 128 | 1729 | -0.0001255688084 | 166.4748 |
| 128 | 2027 | -0.01352307375 | 163.5625 |
| 128 | 7919 | 0.001245479525 | 192.6445 |
| 256 | 1729 | -0.01867061848 | 309.1686 |
| 256 | 2027 | -0.003944726966 | 339.8556 |
| 256 | 7919 | 0.01707839656 | 305.6819 |
| 512 | 1729 | -0.01764257167 | 472.4725 |
| 512 | 2027 | -0.01301144592 | 512.3851 |
| 512 | 7919 | -0.01833364695 | 680.0327 |

Seven of nine official paths contained at least one OHLC ordering inconsistency. These finite model outputs were preserved without clipping or repair and marked `MODEL_OUTPUT_OHLC_INCONSISTENCY`. The first pre-seal attempt incorrectly rejected three such probe paths; it created no seal and accessed no outcome. That implementation failure and one retry remain disclosed in the forecast and report.

### Canonical forecast and diagnostics

- Canonical averaged 512-context close path: `[544.7853800456, 543.5544026693, 545.7588500977, 546.9165649414, 545.6582438151]`.
- Canonical predicted raw log return: `-0.01632642835274094`.
- Total creation runtime: 12.9268844 seconds.
- Available diagnostics: directional agreement 0.7777778; return dispersion 120.3766301 bp; path dispersion 2.1476472; context direction agreement 1.0; context return spread 122.1420031 bp; baseline disagreement 163.2642835 bp; recent annualized volatility 0.05933293; volatility change 0.55166611; trend strength 0.53844384; gap/outlier score 1.97526236; analogue distance 0.96619730; analogue outcome dispersion 0.02605027; horizon path divergence -0.0002239166.
- `RECENT_MODEL_ERROR`: `not_computable / NO_PRIOR_RESOLVED_FORECASTS`.
- `UNCERTAINTY_MISCALIBRATION`: `not_computable / NO_PRIOR_CALIBRATION_SAMPLE`.
- No USE, BLEND, REPAIR, ABSTAIN, reliability score, failure probability, or failure reason was emitted.

### Separate outcome

- Outcome sessions: 2024-07-08 through 2024-07-12, validated against XNYS.
- Outcome normalized-input SHA-256: `2f21b5e9eeab9a2d02bc4484a27bfdd075d9f2cbb31ee9bd2939cf9499ad9635`.
- Realized raw log return: `0.00959962794302243`.
- Kronos absolute return error: `0.02592605629576337`.
- Flat-baseline absolute return error: `0.00959962794302243`.
- Direction correct: false.
- Closer model at this origin: `BASELINE`.
- Canonical close-path MAE: `12.443314615885424`.

This one negative result does not establish whether Kronos or Sentinel works. It demonstrates only that the real, causal, auditable chain operates and preserves an unfavorable model result.

### Artifact chain

- Forecast ID / creation seal: `973d9d0e30327f88dfe2a0ccd7da2d36fbf23f37407578249a60fc50d77b7021`.
- Forecast artifact: `2d65affb6e16bb152a82ffac43a98deca9b57d13699a8a117fa5558c08d99945`.
- Diagnostics artifact: `1a9a8d3c2074daea16b052e3dd638397447eb3008576b710b8a20da75a385962`.
- Outcome artifact: `50ab72351e9c3928a35574dfbbaa03fd370cccd19e29bb39cd10c6cfd519875e`.
- Methodology audit: `ea8e2c4b5789f969c066adabaf685823528190b2399986a52ec91162f7f1dabf`.
- Completed manifest: `f17c148f18e61078e97fa0507d4d7e8d6eafbf667245ed5b7d10688b549b9286`.
- Independent creation and complete-chain verification: true.
- Private confined store: `C:\Users\Tommy\.cache\openalpha-sentinel\phase2\run-spy-20240705\artifacts`.
- Human-readable audit: `research/sentinel-v0/reports/phase2-spy-2024-07-05.md`.

### Final Phase 2 verification

- `uv sync --locked --group dev --group sentinel-phase2` — exit 0; resolved 56 packages and checked 55.
- `uv run pytest -q` — exit 0; 249 passed in 7.48 seconds.
- `uv run ruff check .` — exit 0; `All checks passed!`
- `uv run pyright packages/research-core packages/experiment-spec packages/sentinel scripts/run_sentinel_v0_origin.py` — exit 0; 0 errors, 0 warnings, 0 informations.
- `git diff --check` — exit 0.
- Forbidden tracked artifact scan — exit 0; no model/checkpoint, CSV, yfinance cache, generated-report, or generated-result file is tracked.
- Generated report ignore check — exit 0; the local generated copy is ignored.
- Offline evidence verification — exit 0; the creation seal and resolved chain verify, the resolved descriptor embeds the same creation evidence, and the tracked audit matches the artifact-generated report after normalizing platform line endings.

## Phase 2.5 structural-validity and integration audit

**Claim boundary: DEVELOPMENT INTEGRATION AUDIT - NOT EMPIRICAL EVIDENCE.**

Conclusion category: **B. OFFICIAL RAW-PATH STRUCTURAL INVALIDITY CONFIRMED.**

The pinned official predictor was executed directly outside OpenAlpha's artifact layer and compared with the typed provider response and immutable Phase 2 path. All three canonical path hashes were identical: 2155d9da024b0d0b878a029cca4a180520f59b67acab879eea98e6499cc7fe2c. The input hash also matched the sealed record: f09446b7f7d541907401ca133580eac205c99acb5e27d84bf922943cc4d57bbe. The direct official output was already invalid; OpenAlpha introduced no column, timestamp, inverse-transform, canonicalization, or storage discrepancy.

The audited source was model/kronos.py at pinned source revision 67b630e67f6a18c9e9be918d9b4337c960db1e9a. The trace covered KronosPredictor.predict, KronosPredictor.generate, auto_regressive_inference, KronosTokenizer.encode/decode, and calc_time_stamps. Model/tokenizer pairing, named OHLCV order, internal amount derivation, normalization, inverse transformation, and XNYS timestamp alignment were verified. Deliberate column permutations and shifted, duplicate, missing, or extra timestamps are rejected by regression tests.

### Structural measurements

Sealed Phase 2 nine-path ensemble:

- invalid paths: 7/9 (0.7777777778)
- invalid candles: 24/45 (0.5333333333)
- total violations: 52
- maximum / mean normalized severity: 0.0140865932 / 0.0034681328
- high-low / high-below-body / low-above-body counts: 11 / 36 / 5
- nonfinite / nonpositive counts: 0 / 0

Averaging did not remove the issue. The offline three-path 512 canonical average was invalid in 5/5 candles; the diagnostic all-nine average was invalid in 4/5. Official sample_count=3 and sample_count=5 calls were each invalid in 5/5 candles, with 12 and 11 violations respectively.

Identical seeded reruns reproduced exact output hashes, violation locations, codes, gaps, and severities for all three selected 512-context paths. A first repeatability receipt incorrectly included path labels in equality and was preserved as superseded; the corrected comparison ignores identity labels only and retains exact numerical and violation comparisons.

CONSTRAINT_PROJECTION_V0 restored OHLC ordering by changing high and low only. It preserved every open, close, and implied return exactly. It remains a separate experimental artifact and is not an accuracy claim.

The six-origin canary used three 512-context seeds at three additional cutoffs for each of SPY and QQQ: 18 paths total.

- invalid paths: 12/18 (0.6666666667)
- invalid candles: 31/90 (0.3444444444)
- total violations: 53
- maximum / mean normalized severity: 0.0137362017 / 0.0037325163
- high-low / high-below-body / low-above-body counts: 8 / 38 / 7
- nonfinite / nonpositive counts: 0 / 0

Both assets reproduced the phenomenon, while one QQQ origin had no invalid path. This tiny canary cannot estimate population frequency or establish a relationship with forecast error.

### Preservation and governance

- Sealed Phase 2 creation descriptor SHA-256: 72016ee3e8074da2e54c6cbabf73e932b8aaa8090a62d61b853659360d77eaeb (unchanged).
- Sealed Phase 2 resolved descriptor SHA-256: 6920056d303a9e3f15821aa538ccaf4c77c4421b04a258da7784d8f57ef0b561 (unchanged).
- Sealed Phase 2 artifact inventory SHA-256: 0a40d0fab1b57e2bfcd43b0b2d8203e23c59cf29ad0e2ccff781cf0e78e0781b (unchanged).
- Original creation and resolved chains reverified successfully.
- Previous experiment SHA-256: 587bc36ac1dd941c66d9e94c92d7b26a33faeb8e7e7a38883762f0cb54897950.
- Structural-validity-amended SHA-256: fd50c208f465ce99750b11d4de5d5bc8c391bd73cb9ff450037c596c3e37d8bc.
- Eleven structural diagnostics and their expected error relationships are locked before the development sample and holdout.
- No full development sample or Sentinel risk-model fitting occurred.

Private immutable receipts include final trace c9a5fc1994b496921144713daf68fd0fbf3c8a91630a4879c51a11b99599bbd7, repeatability 7b9754e4519f8cf5286a80857ef09fe8472a642573000606291e5dbdf1606ff2, averaging/projection 7bac665b0ea5e01b79b36c74615b515c01f334338062b1a2d4e4d40127f0b767, canary forecast 435940afb200f0ccdf68bd73b80cb0a8fa1afa5a6e0018d6870e80e2d8c8b6ef, canary outcome 4f240f4b3fe44f11ca7358c0384b59be90eaec10d1ca502e542116f37aee5284, and compact report 4e73825b225ebe100b349eea68bf982354b2de8d5dfe34adde48bc598815772b.

### Final Phase 2.5 verification

- uv sync --locked --group dev --group sentinel-phase2 - exit 0; resolved 56 packages and checked 55.
- Targeted Phase 2.5 pytest command - exit 0; 35 passed in 1.83 seconds.
- uv run pytest -q - exit 0; 284 passed in 6.76 seconds.
- uv run ruff check . - exit 0; all checks passed.
- uv run pyright packages/research-core packages/experiment-spec packages/sentinel scripts/run_sentinel_v0_origin.py scripts/kronos_inference_worker.py scripts/kronos_phase2_5_worker.py scripts/run_sentinel_phase2_5.py - exit 0; 0 errors, 0 warnings, 0 informations.
- Offline report operation - exit 0; receipt 4e73825b225ebe100b349eea68bf982354b2de8d5dfe34adde48bc598815772b reproduced, Phase 2 fingerprint unchanged, source inventory unchanged, and creation/resolved chains verified.
- Sentinel v0.4 lock assertion - exit 0; fd50c208f465ce99750b11d4de5d5bc8c391bd73cb9ff450037c596c3e37d8bc.
- Public JSON parse and policy scan - exit 0; six JSON files valid; no raw observation arrays, patch markers, model weights, reusable CSVs, or cache artifacts tracked.
- git diff --check - exit 0.

## Exact next task

Phase 2.5 is complete. The next justified task is the chronological Sentinel development sample using experiment hash fd50c208f465ce99750b11d4de5d5bc8c391bd73cb9ff450037c596c3e37d8bc. It should test, rather than assume, whether the locked structural diagnostics predict later Kronos error. The untouched holdout and risk-model fitting remain out of scope until the development workflow is complete and frozen.

## Superseded Phase 2 pause

Phase 2 is complete and stops at this single origin. Do not begin the approximately 208-origin development/holdout experiment. The next possible implementation task is to review the Phase 2 evidence—especially widespread model-output OHLC inconsistencies and the official predictor's internal amount derivation—then explicitly decide whether Phase 3 should proceed unchanged or requires a pre-development experiment amendment.
