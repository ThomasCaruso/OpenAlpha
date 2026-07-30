# Project Status

Last updated: 2026-07-30

## Product direction

OpenAlpha has made its second and final pivot to **OpenAlpha Sentinel**: a forecast-time reliability, failure-detection, and intervention layer. Kronos is the first forecast provider and case study.

Benchmarking remains necessary for errors, labels, baselines, and evaluation. It is no longer the product.

## Preserved completed work

- `c037bd5efafd728cb4e0df961dd6a93889b0ce56` is preserved.
- `27f0c69d73e2de678300c44a7899de4295a310c5` is preserved.
- Content-addressed artifacts, path confinement, immutable run journals, verified manifests, and artifact verification remain unchanged.
- Existing tests, Ruff, and Pyright configuration remain unchanged.
- The provider-independent market-data boundary remains.
- Alpaca remains the first documented provider; no Yahoo adapter or silent fallback exists.

## Sentinel Phase 1

The worktree now defines:

- final direction and value test;
- controlled failure-reason taxonomy;
- exact Sentinel v0 methodology;
- development period 2024-07-01 through 2025-06-30;
- untouched holdout 2025-07-01 through 2026-06-30;
- weekly SPY/QQQ five-session experiment;
- nine-path Kronos ensemble and zero-return baseline;
- fourteen diagnostics;
- continuous error, worst-development-quartile failure, and baseline-relative labels;
- logistic/ridge risk configuration;
- USE, 50/50 BLEND, and ABSTAIN policy;
- required evaluation outputs and numeric continuation rules;
- minimal internal provider/decision contracts;
- research workspace and experiment YAML;
- archived unimplemented Kronos Reality Check plan and design.

No Sentinel package or experiment implementation has begun.

## Verified provider and model facts

- Alpaca documents the historical endpoint, inclusive dates, daily timeframe, adjustment values, as-of mapping, SIP/IEX feed choices, pagination, authentication errors, and rate limits.
- Alpaca states API market data cannot be redistributed.
- The official Kronos-mini model card declares Kronos-Tokenizer-2k, time-series forecasting, MIT license, 4,108,192 parameters, and no deployed Hugging Face Inference Provider.
- Current feasibility pins are source `67b630e67f6a18c9e9be918d9b4337c960db1e9a`, model `f4e68697d9d5aed55cef5c96aabc3376bcad9f81`, and tokenizer `26966d0035065a0cae0ebad7af8ece35bc1fb51c`.
- Reported Hub storage is 16,440,776 model bytes plus 15,842,376 tokenizer bytes.

## Unresolved real-inference questions

- Whether the pinned official source and PyTorch dependencies run correctly on this Python 3.13 Windows CPU host.
- Whether seed control produces reproducible independent paths through the official predictor.
- Whether `sample_count=1` exposes the raw path needed for disagreement diagnostics without hidden averaging.
- Actual median and tail latency for nine requests at 128/256/512 contexts.
- Exact downloaded file hashes and total transient cache footprint.
- Whether the user's Alpaca account has historical SIP entitlement.
- Whether a documented ephemeral hosted alternative is cheaper than local execution if the host is too slow.

These are Phase 2 feasibility questions, not permission to substitute fake output.

## Evidence status

- Protocol v1 is unfrozen and its implementation task is paused.
- No Alpaca request has run in this repository.
- No model weights have been downloaded into Git or the worktree.
- No real Kronos forecast, Sentinel diagnostic, outcome, risk model, holdout result, or empirical Sentinel claim exists.

## Phase 2 pre-execution amendment

Before any Alpaca request, model download, or Kronos inference, Sentinel v0 was amended to use raw SIP OHLCV and raw five-session close-to-close log return. Only the three 512-context paths now form the canonical timestamp-wise averaged close path; 128/256-context paths are stress tests only.

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

## Exact next task

Implement the minimal `ForecastProvider` and causal raw-market-context contracts test-first, validate a synthetic five-session inference shape, then run the 512/1729 replay probe and one real SPY 2024-07-05 nine-path origin. Seal the three-path 512-context canonical mean and diagnostics before separately resolving 2024-07-08 through 2024-07-12. Do not emit an action or begin the development sample; stop on any declared data, model, sealing, or infrastructure blocker.
