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


def test_rust_python_cap_constants_match() -> None:
    """Phase 6 parity: the Rust range_projection.rs caps must equal the Python ones.

    Reads the Rust source directly (no cargo needed) so a drift in either language's
    cap constant fails fast, keeping the RANGE_PROJECTION_CONTRACT single-sourced.
    """
    import re
    from pathlib import Path

    from minimappr.core.range_projection import (
        BEARING_PROJECTED_CONFIDENCE_CAP,
        BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP,
        UNOBSERVABLE_CONFIDENCE_CAP as PY_UNOBS_CONF,
        UNOBSERVABLE_RANGE_OBSERVABILITY_CAP as PY_UNOBS_OBS,
    )

    rust_src = (
        Path(__file__).resolve().parents[1]
        / "minimappr-ingest-sidecar"
        / "src"
        / "range_projection.rs"
    ).read_text()

    def _const(name: str) -> float:
        match = re.search(rf"pub const {name}: f32 = ([0-9.]+);", rust_src)
        assert match is not None, f"Rust constant {name} not found"
        return float(match.group(1))

    assert _const("UNOBSERVABLE_CONFIDENCE_CAP") == pytest.approx(PY_UNOBS_CONF)
    assert _const("UNOBSERVABLE_RANGE_OBSERVABILITY_CAP") == pytest.approx(PY_UNOBS_OBS)
    assert _const("BEARING_PROJECTED_CONFIDENCE_CAP") == pytest.approx(BEARING_PROJECTED_CONFIDENCE_CAP)
    assert _const("BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP") == pytest.approx(
        BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP
    )


def test_rust_python_amplitude_prior_defaults_match() -> None:
    """Phase 6 parity: the amplitude-prior CLI defaults in the Rust sidecar match the
    Python config defaults (reference level, min/max range, std factor)."""
    import re
    from pathlib import Path

    from minimappr.config import Settings

    settings = Settings()
    main_src = (
        Path(__file__).resolve().parents[1]
        / "minimappr-ingest-sidecar"
        / "src"
        / "main.rs"
    ).read_text()

    def _default(env_name: str) -> float:
        # Find the clap arg block for the env var and read its default_value_t.
        block = re.search(
            rf'env = "{env_name}",\s*default_value_t = ([0-9.]+)', main_src
        )
        assert block is not None, f"Rust default for {env_name} not found"
        return float(block.group(1))

    assert _default("MINIMAPPR_LOCALIZATION_AMPLITUDE_REFERENCE_LEVEL_DB") == pytest.approx(
        settings.localization_amplitude_reference_level_db
    )
    assert _default("MINIMAPPR_LOCALIZATION_AMPLITUDE_PRIOR_MIN_RANGE_M") == pytest.approx(
        settings.localization_amplitude_prior_min_range_m
    )
    assert _default("MINIMAPPR_LOCALIZATION_AMPLITUDE_PRIOR_MAX_RANGE_M") == pytest.approx(
        settings.localization_amplitude_prior_max_range_m
    )
    assert _default("MINIMAPPR_LOCALIZATION_AMPLITUDE_PRIOR_STD_FACTOR") == pytest.approx(
        settings.localization_amplitude_prior_std_factor
    )
    # Envelope default parity: far-field max range (1 km).
    assert _default("MINIMAPPR_LOCALIZATION_FAR_FIELD_MAX_RANGE_M") == pytest.approx(
        settings.localization_far_field_max_range_m
    )
