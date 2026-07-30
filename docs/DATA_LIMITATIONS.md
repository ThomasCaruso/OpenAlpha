# Sentinel v0 Data and Inference Limitations

## Provider history

Alpaca historical bars retrieved now may contain corrections, adjustments, and symbol mappings unavailable in identical form at a historical cutoff. `asof` addresses symbol identity, not complete point-in-time data vintage. Sentinel v0 therefore tests its method on a fixed retrieved historical representation, not a perfect reconstruction of the provider's past state.

## Raw representation

`adjustment=raw` avoids presenting a modern adjusted series as exact point-in-time history. Raw close-to-close return omits distributions, can contain corporate-action discontinuities, and is not a total shareholder return. Sentinel v0 performs no corporate-action reconstruction or execution simulation, so its results cannot support dividend, fill-price, or trading-profit claims.

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

Raw data and model caches are not committed. Reproduction requires provider credentials, network availability, exact pinned code/model identities, and sufficient local resources. Hashing proves which artifacts were used, not that their sources were correct.
