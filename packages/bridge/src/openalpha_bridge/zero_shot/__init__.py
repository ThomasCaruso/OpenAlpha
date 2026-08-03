"""The Kronos-base zero-shot forecasting benchmark.

A new study, not an amendment. It asks whether frozen Kronos-base zero-shot
forecasts beat simple persistence under shorter, paper-style horizons across
several chronological development origins and a small liquid ETF panel.

Identity is separate; measurement is shared. The experiment id, specification,
run-id namespace, artifact namespace, schemas, worker and terminal object are
all its own and are disjoint from both completed structural-validity studies.
The primitives it measures with -- normalization, anchored return metrics,
persistence comparison, structural validity -- come from
``openalpha_bridge.diagnostic`` so two studies cannot drift apart on arithmetic.

What this package must never do, by construction and by test: repair a forecast,
filter or select paths by structural validity, let a structural quantity enter a
decision rule, substitute the model or tokenizer, download a weight locally,
open a held-out partition, construct an optimizer, or reuse either completed
study's namespace.
"""

from __future__ import annotations
