"""Unit tests for the canonical range_projection vocabulary and haircut."""

from __future__ import annotations

import pytest

from minimappr.core.range_projection import (
    LEGACY_BOUNDED_GRID_BOUNDARY,
    LEGACY_PRIOR_PROJECTED,
    RANGE_ASYMPTOTIC,
    RANGE_BOUNDARY,
    RANGE_REFINED,
    UNOBSERVABLE_CONFIDENCE_CAP,
    UNOBSERVABLE_RANGE_OBSERVABILITY_CAP,
    apply_unobservable_range_haircut,
    normalize_range_mode,
    range_mode_is_unobservable,
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (RANGE_REFINED, False),
        (RANGE_ASYMPTOTIC, True),
        (RANGE_BOUNDARY, True),
        (LEGACY_PRIOR_PROJECTED, True),
        (LEGACY_BOUNDED_GRID_BOUNDARY, True),
        (None, False),
        ("something_unknown", False),
    ],
)
def test_range_mode_is_unobservable(mode: str | None, expected: bool) -> None:
    assert range_mode_is_unobservable(mode) is expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (LEGACY_PRIOR_PROJECTED, RANGE_ASYMPTOTIC),
        (LEGACY_BOUNDED_GRID_BOUNDARY, RANGE_BOUNDARY),
        (RANGE_REFINED, RANGE_REFINED),
        (RANGE_ASYMPTOTIC, RANGE_ASYMPTOTIC),
        (RANGE_BOUNDARY, RANGE_BOUNDARY),
        (None, None),
        ("unknown", "unknown"),
    ],
)
def test_normalize_range_mode(mode: str | None, expected: str | None) -> None:
    assert normalize_range_mode(mode) == expected


def test_haircut_caps_unobservable_modes() -> None:
    for mode in (RANGE_ASYMPTOTIC, RANGE_BOUNDARY, LEGACY_PRIOR_PROJECTED):
        confidence, observability = apply_unobservable_range_haircut(
            mode=mode, confidence=0.87, range_observability=0.42
        )
        assert confidence == UNOBSERVABLE_CONFIDENCE_CAP
        assert observability == UNOBSERVABLE_RANGE_OBSERVABILITY_CAP


def test_haircut_leaves_observable_mode_untouched() -> None:
    confidence, observability = apply_unobservable_range_haircut(
        mode=RANGE_REFINED, confidence=0.87, range_observability=0.42
    )
    assert confidence == pytest.approx(0.87)
    assert observability == pytest.approx(0.42)


def test_haircut_does_not_raise_confidence() -> None:
    # Already below the cap: haircut must not increase it.
    confidence, observability = apply_unobservable_range_haircut(
        mode=RANGE_ASYMPTOTIC, confidence=0.05, range_observability=None
    )
    assert confidence == pytest.approx(0.05)
    assert observability is None
