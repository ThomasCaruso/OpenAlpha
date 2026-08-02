"""The Kronos-base replication study.

A separate study, not an amendment to the completed Kronos-mini diagnostic. It
has its own preregistration document and hash, its own model and tokenizer
pins, its own run-identifier pattern, its own artifact namespace, and its own
success and failure schemas. Nothing here may read, write, rename or invalidate
anything belonging to the mini study.

What it deliberately shares with mini is the measurement code: normalization,
structural validity, methods A to D, the metrics and the decision rules all
come from ``openalpha_bridge.diagnostic``. Sharing the measurement is the point
-- if the arithmetic differed, a difference in result could not be attributed
to the model family.
"""

from __future__ import annotations
