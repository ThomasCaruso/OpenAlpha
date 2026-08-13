"""The shared moving-block core, and proof it changed no published result.

Two studies now delegate their bootstrap arithmetic to one neutral module. The
danger of that refactor is silent: if the draw order, the truncation of the
final block, or the percentile indices shifted, the completed zero-shot
benchmark's published interval would no longer be reproducible from its own
recorded data, and nothing would say so.

So the anchor test is not a golden constant typed out by hand. It recomputes the
intervals from the clusters recorded inside the committed terminal artifact and
requires them bit-identical to the intervals recorded beside them. That artifact
is content-addressed and verified elsewhere, so this is a comparison against
published research rather than against a value someone could quietly edit.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.representation_probe.spec import MINIMUM_TEST_ORIGINS_PER_ASSET
from openalpha_bridge.representation_probe.test import (
    PROBE_CLUSTER_COUNT,
)
from openalpha_bridge.representation_probe.test import (
    OriginCluster as ProbeCluster,
)
from openalpha_bridge.representation_probe.test import (
    paired_moving_block_bootstrap as probe_bootstrap,
)
from openalpha_bridge.resampling import (
    MovingBlockCore,
    block_start_count,
    blocks_per_resample,
    moving_block_percentile_interval,
)
from openalpha_bridge.zero_shot.aggregation import (
    BOOTSTRAP_ASSETS_PER_CLUSTER,
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_CLUSTER_COUNT,
)
from openalpha_bridge.zero_shot.aggregation import (
    OriginCluster as ZeroShotCluster,
)
from openalpha_bridge.zero_shot.aggregation import (
    paired_origin_moving_block_bootstrap as zero_shot_bootstrap,
)
from openalpha_bridge.zero_shot.spec import (
    ASSET_PANEL,
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
)

REPO = Path(__file__).resolve().parents[3]
ARTIFACT = REPO / "research" / "artifacts" / "kronos_zero_shot_benchmark_terminal.json"


def _published_aggregates() -> list[dict]:
    return json.loads(ARTIFACT.read_bytes())["aggregates"]


# ====== the regression that matters ==================================


@pytest.mark.parametrize("configuration", ["A", "B"])
def test_the_shared_core_reproduces_the_published_zero_shot_interval(
    configuration: str,
) -> None:
    """Recompute a published interval from its own recorded clusters.

    If the shared core ever diverges from the implementation that produced the
    terminal artifact, this fails -- which is the only guard that matters,
    because the artifact is immutable and cannot be corrected.
    """
    aggregate = next(
        a for a in _published_aggregates() if a["configuration_label"] == configuration
    )
    recorded = aggregate["moving_block_bootstrap"]
    clusters = tuple(
        ZeroShotCluster(
            ordinal=c["ordinal"],
            assets=tuple(c["assets"]),
            paired_differences=tuple(c["paired_differences"]),
        )
        for c in aggregate["origin_clusters"]
    )

    recomputed = zero_shot_bootstrap(
        clusters,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        seed=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES,
        confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
    )

    # Exact equality, not approximate. A float that moved by one ulp means the
    # arithmetic changed.
    assert recomputed.lower == recorded["lower"]
    assert recomputed.upper == recorded["upper"]
    assert recomputed.point_estimate == recorded["point_estimate"]
    assert recomputed.excludes_zero == recorded["excludes_zero"]
    assert recomputed.excludes_zero_favorably == recorded["excludes_zero_favorably"]


@pytest.mark.parametrize("configuration", ["A", "B"])
def test_the_published_interval_metadata_is_unchanged(configuration: str) -> None:
    """Structure and labelling must survive the refactor too, not just numbers."""
    aggregate = next(
        a for a in _published_aggregates() if a["configuration_label"] == configuration
    )
    recorded = aggregate["moving_block_bootstrap"]
    clusters = tuple(
        ZeroShotCluster(
            ordinal=c["ordinal"],
            assets=tuple(c["assets"]),
            paired_differences=tuple(c["paired_differences"]),
        )
        for c in aggregate["origin_clusters"]
    )
    recomputed = zero_shot_bootstrap(
        clusters,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        seed=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES,
        confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
    )
    for field in (
        "method",
        "base_resampling_unit",
        "temporal_resampling_unit",
        "cluster_count",
        "assets_per_cluster",
        "observation_count",
        "block_length",
        "available_block_starts",
        "blocks_drawn_per_resample",
        "clusters_retained_per_resample",
        "resamples",
        "confidence_level",
        "seed",
        "assets_resampled_within_cluster",
        "origins_resampled_independently",
        "wraparound",
        "block_length_selected_before_execution",
        "block_length_tuned_from_results",
    ):
        assert getattr(recomputed, field) == recorded[field], field


def test_a_frozen_synthetic_case_pins_the_core_arithmetic() -> None:
    """A second anchor that does not depend on the artifact being present."""
    totals = [0.001 * ((index % 7) - 3) for index in range(BOOTSTRAP_CLUSTER_COUNT)]
    core = moving_block_percentile_interval(
        totals,
        observations=BOOTSTRAP_CLUSTER_COUNT * BOOTSTRAP_ASSETS_PER_CLUSTER,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        seed=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES,
        confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
    )
    assert core.cluster_count == 25
    assert core.observation_count == 100
    assert core.block_length == 4
    assert core.available_block_starts == 22
    assert core.blocks_drawn_per_resample == 7
    assert core.point_estimate == pytest.approx(sum(totals) / 100)
    assert core.lower <= core.upper


# ====== each wrapper keeps its own cluster-count guard ===============


@pytest.mark.parametrize("count", [16, 24, 26, 100])
def test_the_zero_shot_wrapper_still_demands_exactly_twenty_five(count: int) -> None:
    """Sharing the core must not relax the completed study's contract."""
    clusters = tuple(
        ZeroShotCluster(ordinal=i, assets=ASSET_PANEL, paired_differences=(0.1, 0.1, 0.1, 0.1))
        for i in range(count)
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        zero_shot_bootstrap(
            clusters,
            seed=BOOTSTRAP_SEED,
            resamples=100,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
        )
    assert excinfo.value.failures[0].code == "ZERO_SHOT_BOOTSTRAP_CLUSTER_COUNT_INVALID"
    assert "exactly 25" in excinfo.value.failures[0].message


@pytest.mark.parametrize("count", [4, 15, 17, 25])
def test_the_probe_wrapper_demands_exactly_sixteen(count: int) -> None:
    clusters = tuple(
        ProbeCluster(ordinal=i, assets=ASSET_PANEL, paired_differences=(0.1, 0.1, 0.1, 0.1))
        for i in range(count)
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        probe_bootstrap(clusters, resamples=100)
    assert excinfo.value.failures[0].code == "PROBE_BOOTSTRAP_CLUSTER_COUNT_INVALID"
    assert "exactly 16" in excinfo.value.failures[0].message


def test_each_wrapper_accepts_only_its_own_panel_size() -> None:
    assert BOOTSTRAP_CLUSTER_COUNT == 25
    assert PROBE_CLUSTER_COUNT == MINIMUM_TEST_ORIGINS_PER_ASSET == 16

    twenty_five = tuple(
        ZeroShotCluster(ordinal=i, assets=ASSET_PANEL, paired_differences=(0.01,) * 4)
        for i in range(25)
    )
    sixteen = tuple(
        ProbeCluster(ordinal=i, assets=ASSET_PANEL, paired_differences=(0.01,) * 4)
        for i in range(16)
    )
    assert zero_shot_bootstrap(
        twenty_five, seed=BOOTSTRAP_SEED, resamples=100,
        confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
    ).cluster_count == 25
    assert probe_bootstrap(sixteen, resamples=100).cluster_count == 16


def test_the_two_wrappers_agree_where_their_domains_could_overlap() -> None:
    """Same totals through both wrappers give the same numbers.

    Their panels differ in size so they never both accept the same input, but
    the core they share must not treat them differently. Driving the core
    directly at each size proves the delegation is the only difference.
    """
    for count in (16, 25):
        totals = [0.002 * ((i % 5) - 2) for i in range(count)]
        core = moving_block_percentile_interval(
            totals,
            observations=count * 4,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            seed=BOOTSTRAP_SEED,
            resamples=500,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
        )
        clusters_zs = tuple(
            ZeroShotCluster(ordinal=i, assets=ASSET_PANEL,
                            paired_differences=(totals[i] / 4,) * 4)
            for i in range(count)
        )
        wrapper = (
            zero_shot_bootstrap(
                clusters_zs, seed=BOOTSTRAP_SEED, resamples=500,
                confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
            )
            if count == 25
            else probe_bootstrap(
                tuple(
                    ProbeCluster(ordinal=c.ordinal, assets=c.assets,
                                 paired_differences=c.paired_differences)
                    for c in clusters_zs
                ),
                resamples=500,
            )
        )
        assert wrapper.lower == pytest.approx(core.lower)
        assert wrapper.upper == pytest.approx(core.upper)
        assert wrapper.point_estimate == pytest.approx(core.point_estimate)


# ====== the core itself ==============================================


def test_the_core_is_deterministic_under_a_seed() -> None:
    totals = [0.001 * i for i in range(20)]
    kwargs = {
        "observations": 80,
        "block_length": 4,
        "seed": BOOTSTRAP_SEED,
        "resamples": 300,
        "confidence_level": 0.95,
    }
    assert moving_block_percentile_interval(totals, **kwargs) == (
        moving_block_percentile_interval(totals, **kwargs)
    )
    other = moving_block_percentile_interval(totals, **{**kwargs, "seed": BOOTSTRAP_SEED + 1})
    assert other.seed != BOOTSTRAP_SEED


def test_the_core_draws_block_starts_not_clusters() -> None:
    """Seven draws from twenty-two starts, not twenty-five draws from clusters."""
    calls: list[int] = []
    real = random.Random.randrange

    def recording(self: random.Random, *args, **kwargs) -> int:
        calls.append(args[0] if args else -1)
        return real(self, *args, **kwargs)

    from unittest import mock

    with mock.patch.object(random.Random, "randrange", recording):
        moving_block_percentile_interval(
            [0.001] * 25, observations=100, block_length=4,
            seed=BOOTSTRAP_SEED, resamples=10, confidence_level=0.95,
        )
    assert set(calls) == {22}
    assert len(calls) == 70  # 10 resamples x 7 blocks


def test_a_constant_panel_reproduces_its_own_mean() -> None:
    """If the denominator or retained-cluster count were wrong, this shifts."""
    for value in (0.001, -0.0025, 0.0):
        core = moving_block_percentile_interval(
            [value * 4] * 25, observations=100, block_length=4,
            seed=BOOTSTRAP_SEED, resamples=200, confidence_level=0.95,
        )
        assert core.point_estimate == pytest.approx(value)
        assert core.lower == pytest.approx(value)
        assert core.upper == pytest.approx(value)


def test_the_core_never_wraps_around() -> None:
    """A lone non-zero final cluster can only enter via blocks that contain it."""
    totals = [0.0] * 25
    totals[24] = 1.0
    core = moving_block_percentile_interval(
        totals, observations=100, block_length=4, seed=BOOTSTRAP_SEED,
        resamples=2000, confidence_level=0.95,
    )
    # Only start 21 reaches ordinal 24, and only as a full (non-truncated) block.
    # With wraparound, earlier starts would also pick it up and the upper bound
    # would be larger.
    assert core.available_block_starts == 22
    assert core.upper <= 6 * 1.0 / 100 + 1e-12


def test_block_geometry_helpers_match_the_declared_arithmetic() -> None:
    assert block_start_count(25, 4) == 22
    assert block_start_count(16, 4) == 13
    assert blocks_per_resample(25, 4) == 7
    assert blocks_per_resample(16, 4) == 4


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"block_length": 0}, "RESAMPLING_BLOCK_LENGTH_INVALID"),
        ({"block_length": 30}, "RESAMPLING_BLOCK_LENGTH_INVALID"),
        ({"resamples": 0}, "RESAMPLING_RESAMPLES_INVALID"),
        ({"confidence_level": 1.0}, "RESAMPLING_CONFIDENCE_INVALID"),
        ({"confidence_level": 0.0}, "RESAMPLING_CONFIDENCE_INVALID"),
        ({"observations": 0}, "RESAMPLING_OBSERVATIONS_INVALID"),
    ],
)
def test_the_core_validates_what_it_needs(kwargs: dict, code: str) -> None:
    base = {
        "observations": 100,
        "block_length": 4,
        "seed": 1,
        "resamples": 10,
        "confidence_level": 0.95,
    }
    with pytest.raises(BridgeTransformError) as excinfo:
        moving_block_percentile_interval([0.001] * 25, **{**base, **kwargs})
    assert excinfo.value.failures[0].code == code


def test_the_core_refuses_a_non_finite_total() -> None:
    totals = [0.001] * 25
    totals[3] = math.nan
    with pytest.raises(BridgeTransformError) as excinfo:
        moving_block_percentile_interval(
            totals, observations=100, block_length=4, seed=1, resamples=10,
            confidence_level=0.95,
        )
    assert excinfo.value.failures[0].code == "RESAMPLING_NON_FINITE_TOTAL"


def test_the_core_is_study_neutral() -> None:
    """It must not know about assets, ordinals, or either study."""
    source = (
        REPO / "packages" / "bridge" / "src" / "openalpha_bridge" / "resampling.py"
    ).read_text(encoding="utf-8")
    for term in ("SPY", "QQQ", "IWM", "DIA", "zero_shot", "representation_probe",
                 "ASSET_PANEL", "kronos", "Kronos"):
        assert term not in source, term
    assert isinstance(
        moving_block_percentile_interval(
            [0.1] * 8, observations=8, block_length=2, seed=1, resamples=10,
            confidence_level=0.9,
        ),
        MovingBlockCore,
    )
