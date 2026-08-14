"""The published research record must stay byte-identical and self-reproducing.

`results-summary.json` is the file a reader consults for the numbers quoted in the
README and in each report. Unlike the terminal artifacts it is regenerated from an
artifact payload rather than fetched, so a well-meaning regeneration could silently
restate a completed study. These guards pin the exact bytes, the exact published
numbers, and -- for the zero-shot benchmark -- the property that the record still
recomputes its own decision interval from the clusters it carries.

Nothing here executes a study, opens a partition, or touches a provider.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from openalpha_kronos.studies.zero_shot.aggregation import (
    OriginCluster,
    paired_origin_moving_block_bootstrap,
)
from openalpha_kronos.studies.zero_shot.spec import (
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
)

ROOT = Path(__file__).resolve().parents[2]

STRUCTURAL_SUMMARY = "research/reports/kronos-structural-validity/results-summary.json"
ZERO_SHOT_SUMMARY = "research/reports/kronos-zero-shot-benchmark/results-summary.json"

SUMMARY_DIGESTS = {
    STRUCTURAL_SUMMARY: "fabd86207ca37dd7d6ec1e04e6cca916813c5e324d9fa18f03d368b821d1f25a",
    ZERO_SHOT_SUMMARY: "1d5eef1b293addb55fbc10dc2e16eddda4c60c68b3be80ebd2041f2369217c20",
}


def _summary(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


@pytest.mark.parametrize("relative, expected", SUMMARY_DIGESTS.items())
def test_published_numeric_summary_is_byte_identical(relative: str, expected: str) -> None:
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_the_published_structural_numbers_are_unchanged() -> None:
    """The two structural studies reported these numbers; they are terminal."""
    studies = _summary(STRUCTURAL_SUMMARY)["studies"]

    assert studies["kronos_mini"]["roundtrip_full_invalid_fraction"] == 0.24609375
    assert studies["kronos_mini"]["method_b_close_return_mae"] == 0.007306554165097093
    assert studies["kronos_mini"]["invalidity_error_spearman"] == 0.04015798783221637
    assert studies["kronos_base"]["roundtrip_full_invalid_fraction"] == 0.123046875
    assert studies["kronos_base"]["method_b_close_return_mae"] == 0.00920398038369002
    assert studies["kronos_base"]["invalidity_error_spearman"] == 0.43092362328095035
    assert studies["kronos_base"]["best_persistence_skill"] == -0.4481214885679963

    # The finding that closed the direction: full structural repair moved nothing.
    for study in studies.values():
        assert study["projection_primary_improvement"] == 0.0
        assert study["valid_rollout_count"] == 0
        assert study["persistence_close_return_mae"] == 0.0033651655739541575


def test_the_published_zero_shot_numbers_are_unchanged() -> None:
    """Every quoted zero-shot headline number, including both intervals."""
    summary = _summary(ZERO_SHOT_SUMMARY)
    configurations = summary["configurations"]

    assert configurations["A"]["temperature"] == 0.6
    assert configurations["A"]["relative_skill"]["median"] == -0.04174286800947691
    assert configurations["A"]["fraction_beating_persistence"] == 0.31
    assert configurations["A"]["moving_block_bootstrap"]["lower"] == -0.000645048876271538
    assert configurations["A"]["moving_block_bootstrap"]["upper"] == -0.000223311215065983

    assert configurations["B"]["temperature"] == 1.0
    assert configurations["B"]["relative_skill"]["median"] == -0.0836337619605827
    assert configurations["B"]["fraction_beating_persistence"] == 0.26
    assert configurations["B"]["moving_block_bootstrap"]["lower"] == -0.0015770473336209681
    assert configurations["B"]["moving_block_bootstrap"]["upper"] == -0.00034561031764769075

    for configuration in configurations.values():
        assert configuration["origins_scored"] == 100
        # No asset supported either configuration, so the interval excludes zero
        # on the unfavorable side under both temperatures.
        assert [asset["supports_configuration"] for asset in configuration["per_asset"]] == [
            False,
            False,
            False,
            False,
        ]
        assert configuration["moving_block_bootstrap"]["excludes_zero"] is True
        assert configuration["moving_block_bootstrap"]["excludes_zero_favorably"] is False


def test_the_published_decision_is_the_preregistered_one() -> None:
    """The recorded outcome must stay the one the sealed rules produced."""
    decision = _summary(ZERO_SHOT_SUMMARY)["decision"]

    assert decision["matched_findings"] == ["NO_ZERO_SHOT_SKILL"]
    assert decision["zero_shot_generation_direction"] == "STOP_KRONOS_ZERO_SHOT_DIRECTION"
    assert decision["structural_validity_used_in_any_rule"] is False
    assert decision["limitations"] == []
    assert decision["thresholds"] == {
        "bootstrap_confidence_level": 0.95,
        "minimum_fraction_of_origins_beating_persistence": 0.6,
        "minimum_median_relative_skill": 0.0,
        "minimum_supporting_assets": 3,
        "total_assets": 4,
    }


def test_the_published_record_authorizes_nothing() -> None:
    summary = _summary(ZERO_SHOT_SUMMARY)

    for field, value in summary["authorizations"].items():
        assert value is False, field
    for field, value in summary["integrity"].items():
        assert value is False, field


@pytest.mark.parametrize("label", ["A", "B"])
def test_the_published_clusters_reproduce_the_published_interval(label: str) -> None:
    """The record carries enough to recompute the interval its decision read."""
    configuration = _summary(ZERO_SHOT_SUMMARY)["configurations"][label]
    published = configuration["moving_block_bootstrap"]

    clusters = tuple(
        OriginCluster(
            ordinal=entry["ordinal"],
            assets=tuple(entry["assets"]),
            paired_differences=tuple(entry["paired_differences"]),
        )
        for entry in configuration["origin_clusters"]
    )
    assert len(clusters) == 25

    recomputed = paired_origin_moving_block_bootstrap(
        clusters,
        seed=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES,
        confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
    )
    assert recomputed.lower == pytest.approx(published["lower"])
    assert recomputed.upper == pytest.approx(published["upper"])
    assert recomputed.point_estimate == pytest.approx(published["point_estimate"])
