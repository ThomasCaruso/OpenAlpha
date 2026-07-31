from datetime import date

import pytest
from openalpha_sentinel.development_manifest import (
    EXPERIMENT_SHA256,
    build_development_manifest,
    require_development_cutoff,
    require_outcome_sessions,
)


def test_manifest_has_exact_locked_population() -> None:
    manifest = build_development_manifest()

    assert EXPERIMENT_SHA256 == (
        'fd50c208f465ce99750b11d4de5d5bc8c391bd73cb9ff450037c596c3e37d8bc'
    )
    assert len(manifest.cutoffs) == 52
    assert len(manifest.origins) == 104
    assert manifest.cutoffs[0] == date(2024, 7, 5)
    assert manifest.cutoffs[-1] == date(2025, 6, 27)
    assert [(item.cutoff, item.asset) for item in manifest.origins[:2]] == [
        (date(2024, 7, 5), 'SPY'),
        (date(2024, 7, 5), 'QQQ'),
    ]
    assert len(manifest.canonical_sha256) == 64
    assert manifest.stale_target_override is True


def test_holdout_cutoff_is_rejected_but_final_outcome_is_allowed() -> None:
    with pytest.raises(ValueError, match='holdout'):
        require_development_cutoff(date(2025, 7, 1))

    assert require_outcome_sessions(date(2025, 6, 27))[-1] == date(2025, 7, 7)


def test_outcome_sessions_require_a_locked_development_origin() -> None:
    with pytest.raises(ValueError, match='locked development cutoff'):
        require_outcome_sessions(date(2024, 7, 3))
