"""Amendment 3 conformance: the implemented formulas must equal the amended ones.

Reads the amendment YAML as text and checks the code against it, so a drift in
either direction fails rather than silently diverging.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from openalpha_bridge.phase2.features import compute_scale_features
from openalpha_bridge.phase2.identity import AMENDMENT_3_SHA256, verify_locked_hashes
from openalpha_bridge.phase2.kronos import SCALE_FEATURE_ORDER
from openalpha_bridge.windowing import CONTEXT_PREFIX_LENGTH

ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "research" / "bridge-v0"
AMENDMENT = RESEARCH / "phase2-amendment-3-scale-features.yaml"

CANARY = {
    "sequence_id": "ecd7fd5797a9147a",
    "prefix_start": "2015-05-07",
    "prefix_end": "2017-02-14",
    "target_start": "2017-02-15",
    "target_end": "2017-05-17",
}


def test_amendment_3_hash_is_locked() -> None:
    observed = verify_locked_hashes(RESEARCH)
    assert observed["phase2-amendment-3-scale-features.yaml"] == AMENDMENT_3_SHA256


def test_prior_locks_remain_byte_identical() -> None:
    observed = verify_locked_hashes(RESEARCH)
    assert observed["experiment.yaml"] == (
        "d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2"
    )
    assert observed["phase2-preregistration-amendment.yaml"] == (
        "4c60846e8cb791d922c29a5e704fe3a473740171dd2bbb7b3390796ceeeef3c8"
    )
    assert observed["phase2-amendment-2-context-prefix.yaml"] == (
        "66f3c8171c2805bccf4b924125bd206edad27c957dfff40a0abf3864fbab10c1"
    )


def test_amendment_declares_every_named_feature() -> None:
    text = AMENDMENT.read_text(encoding="utf-8")
    for name in SCALE_FEATURE_ORDER:
        assert f"{name}:" in text, f"{name} has no amended formula"


def test_amendment_pins_ddof_zero_and_the_prefix_anchor() -> None:
    text = AMENDMENT.read_text(encoding="utf-8")
    assert "population_standard_deviation_ddof: 0" in text
    assert "anchor_close: final_close_of_the_448_candle_prefix" in text


def test_amendment_does_not_overclaim_prefix_causality() -> None:
    """The amendment must state the narrow claim, not the broad one."""
    text = AMENDMENT.read_text(encoding="utf-8")
    assert "not_claimed:" in text
    assert "later prefix candle" in text


def test_amendment_pins_the_exact_canary_sequence() -> None:
    text = AMENDMENT.read_text(encoding="utf-8")
    for value in CANARY.values():
        assert value in text
    assert "STAGE_A_OFFICIAL_CANARY_PASSED" in text
    assert "authorizes_stage_b: false" in text


def _matrix(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    level = 200.0
    for _ in range(CONTEXT_PREFIX_LENGTH + 64):
        level = max(10.0, level * float(1.0 + rng.normal(0.0, 0.005)))
        close = max(10.0, level * float(1.0 + rng.normal(0.0, 0.005)))
        high = max(level, close) * 1.003
        low = min(level, close) / 1.003
        volume = 2.0e6 * float(1.0 + rng.random())
        rows.append((level, high, low, close, volume, volume * close))
        level = close
    return np.array(rows, dtype=np.float64)


def test_implementation_matches_every_amended_formula() -> None:
    matrix = _matrix()
    values = compute_scale_features(matrix)
    prefix = matrix[:CONTEXT_PREFIX_LENGTH]
    anchor = float(prefix[-1, 3])
    mean = prefix.mean(axis=0)
    std = prefix.std(axis=0, ddof=0)  # amended ddof

    expected = {
        "log_anchor_close": math.log(anchor),
        "open_mean_relative_to_anchor": mean[0] / anchor,
        "high_mean_relative_to_anchor": mean[1] / anchor,
        "low_mean_relative_to_anchor": mean[2] / anchor,
        "close_mean_relative_to_anchor": mean[3] / anchor,
        "log1p_open_std_over_anchor": math.log1p(std[0] / anchor),
        "log1p_high_std_over_anchor": math.log1p(std[1] / anchor),
        "log1p_low_std_over_anchor": math.log1p(std[2] / anchor),
        "log1p_close_std_over_anchor": math.log1p(std[3] / anchor),
        "log1p_volume_mean": math.log1p(mean[4]),
        "log1p_volume_std": math.log1p(std[4]),
        "log1p_amount_mean": math.log1p(mean[5]),
        "log1p_amount_std": math.log1p(std[5]),
    }
    for name, want in expected.items():
        got = float(values[SCALE_FEATURE_ORDER.index(name)])
        assert got == pytest.approx(want, rel=1e-6), name


def test_ddof_one_would_not_satisfy_the_amendment() -> None:
    """Guards the specific ddof choice rather than assuming it."""
    matrix = _matrix()
    prefix = matrix[:CONTEXT_PREFIX_LENGTH]
    anchor = float(prefix[-1, 3])
    values = compute_scale_features(matrix)
    sample = math.log1p(float(prefix.std(axis=0, ddof=1)[3]) / anchor)
    population = math.log1p(float(prefix.std(axis=0, ddof=0)[3]) / anchor)
    got = float(values[SCALE_FEATURE_ORDER.index("log1p_close_std_over_anchor")])
    # The stored feature is float32, so compare at float32 precision.
    assert got == pytest.approx(population, rel=1e-6)
    # ddof=1 differs by sqrt(n/(n-1)) ~ 0.1% here, far above float32 noise, so
    # the assertion above genuinely discriminates between the two choices.
    assert abs(sample - population) > 1e-5
    assert got != pytest.approx(sample, rel=1e-6)


def test_scored_suffix_cannot_influence_the_features() -> None:
    matrix = _matrix()
    baseline = compute_scale_features(matrix)
    perturbed = matrix.copy()
    perturbed[CONTEXT_PREFIX_LENGTH:, :] *= 7.0
    assert np.array_equal(baseline, compute_scale_features(perturbed))


def test_a_prefix_change_does_move_earlier_rows_as_documented() -> None:
    """The amendment explicitly declines to claim prefix-row independence."""
    matrix = _matrix()
    baseline = compute_scale_features(matrix)
    perturbed = matrix.copy()
    perturbed[400, :] *= 1.5  # a late prefix candle
    assert not np.array_equal(baseline, compute_scale_features(perturbed))
