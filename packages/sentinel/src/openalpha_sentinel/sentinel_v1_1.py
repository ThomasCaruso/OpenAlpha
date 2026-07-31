from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

from .phase3b import Phase3BOrigin, locked_phase3b_origins

SUPPORT_SYMBOLS = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "HYG",
    "GLD",
    "EFA",
    "EEM",
    "XLF",
)
SUPPORT_START = date(2017, 1, 1)
SUPPORT_END_EXCLUSIVE = date(2024, 6, 29)
SUPPORT_LAST_SESSION = date(2024, 6, 28)
SUPPORT_WINDOW_LENGTH = 512
SUPPORT_WINDOWS_PER_SYMBOL = 3
SEEDS = (1729, 2027, 7919)
MASS_BUDGETS = ((8, 8), (16, 16), (32, 32))
CANARY_MASS_BUDGETS = ((8, 8), (16, 16))
CANARY_CUTOFFS = (date(2024, 7, 5), date(2025, 4, 11))
HOLDOUT_START = date(2025, 7, 1)

SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
MODEL_REVISIONS = {
    "NeoQuasar/Kronos-mini": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
    "NeoQuasar/Kronos-small": "901c26c1332695a2a8f243eb2f37243a37bea320",
    "NeoQuasar/Kronos-base": "2b554741eca47781b64468546e77fef3e85130e6",
}
TOKENIZER_REVISIONS = {
    "NeoQuasar/Kronos-Tokenizer-2k": (
        "26966d0035065a0cae0ebad7af8ece35bc1fb51c"
    ),
    "NeoQuasar/Kronos-Tokenizer-base": (
        "0e0117387f39004a9016484a186a908917e22426"
    ),
}
MODEL_TOKENIZER_PAIRS = {
    "NeoQuasar/Kronos-mini": "NeoQuasar/Kronos-Tokenizer-2k",
    "NeoQuasar/Kronos-small": "NeoQuasar/Kronos-Tokenizer-base",
    "NeoQuasar/Kronos-base": "NeoQuasar/Kronos-Tokenizer-base",
}


def locked_v1_1_origins() -> tuple[Phase3BOrigin, ...]:
    origins = locked_phase3b_origins()
    if any(
        session >= HOLDOUT_START
        for origin in origins
        for session in origin.forecast_sessions
    ):
        raise ValueError("Sentinel v1.1 origin entered the untouched holdout")
    return origins


def verify_experiment_hash(experiment_path: Path, digest_path: Path) -> bool:
    expected = digest_path.read_text(encoding="utf-8").strip().lower()
    if len(expected) != 64:
        return False
    actual = hashlib.sha256(experiment_path.read_bytes()).hexdigest()
    return actual == expected
