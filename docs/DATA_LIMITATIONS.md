# Sentinel v0 Data and Inference Limitations

## Provider history

Phase 2 retrieves Yahoo Finance data through the unofficial open-source yfinance client. Yahoo does not sponsor or guarantee this integration, and availability, schemas, corrections, and response behavior may change. The source is not an institutional point-in-time archive. Sentinel v0 therefore tests a fixed retrieved representation, not a perfect reconstruction of what a market-data user saw at the historical cutoff.

No cross-provider verification is performed for the single Phase 2 origin. Strong claims require a representative rerun through an independent source such as Alpaca and comparison of bars, returns, forecasts, diagnostics, and conclusions.

## Raw representation

`auto_adjust=False`, `back_adjust=False`, and `repair=False` avoid using adjusted prices or yfinance repair as model inputs. `Adj Close` is excluded. Raw close-to-close return omits distributions, can contain corporate-action discontinuities, and is not total shareholder return. Dividend and split fields are warnings only; Sentinel v0 performs no corporate-action reconstruction or execution simulation.

## Universe and sample

SPY and QQQ are current, liquid ETFs with related US equity exposure. Two assets and roughly 104 weekly cutoffs per asset are enough for signal feasibility, not broad generalization. Weekly rows remain serially dependent and the two assets are correlated.

## Kronos contamination and model scope

Evaluation starts after the reported June 2024 pretraining boundary, reducing direct temporal overlap but not proving absence of related patterns or data. Kronos-mini may behave differently from larger Kronos checkpoints and other forecasting models.

## Stochastic inference

Recorded seeds and settings improve auditability but may not guarantee bit-identical paths across PyTorch versions, devices, or kernels. The v0 question concerns diagnostic predictiveness under the recorded execution environment.

## Analogue diagnostics

Historical analogue distance depends on a short, ETF-specific eligible history and one fixed normalization/distance rule. Nearest windows are descriptive support, not proof that regimes are causally equivalent.

## Recent-error availability

`RECENT_MODEL_ERROR` is unavailable for the first eight scheduled resolved forecasts per asset. Development-fitted missingness handling preserves those rows, but early-period behavior may differ from mature live operation.

## Failure labels

The worst-development-quartile label is relative to this dataset. It is not a universal definition of a bad forecast. The baseline-relative label depends on a deliberately simple zero-return baseline.

## Events and omitted information

News, earnings, filings, macro releases, and unscheduled events are excluded. Sentinel v0 cannot conclude that market/model-output diagnostics capture exogenous-event risk.

## Holdout and multiplicity

The later year is untouched for configuration, but many diagnostics and outputs remain a multiple-comparison risk. Continuation uses the locked aggregate criteria rather than selecting a favorable chart or asset.

## Storage and reproducibility

Raw data and model caches are not committed. Reproduction requires network availability, the pinned yfinance/code/model identities, and sufficient local resources. Hashing proves which artifacts were used, not that their sources were correct.
