"""The Kronos-base frozen-representation probe.

Asks whether the frozen Kronos-base hidden state carries information about
future returns that raw OHLCV and simple engineered features do not, when all
are fed to an identical downstream model.

Governed entirely by the sealed preregistration
``research/bridge-v0/kronos-frozen-representation-probe-v1.yaml``
(SHA-256 ``3c3656f0...b95ece``), whose digest is verified inside the execution
container before any work begins.

Three properties are structural rather than promised:

**Fitting and testing cannot be confused.** They are separate execution paths
writing separate immutable artifacts. The fit artifact publishes the selected
hyperparameters and preprocessing state before the test partition is opened, so
"we did not tune on test" is a matter of record rather than of trust.

**The test partition cannot be opened early.** Every gate in :mod:`test` must
pass -- sealed specification digest, exact fit-artifact digest, the 2026-08-05
boundary, at least 232 aligned post-boundary sessions -- before a single test
observation is retrieved.

**Generation assumptions do not leak in.** This package decodes no candles,
enforces no OHLC inequality, repairs and filters nothing, and never samples. It
reads one hidden state per origin and fits a linear model on it.
"""

from __future__ import annotations
