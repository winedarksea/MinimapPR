from __future__ import annotations

import math

import pytest

from minimappr.core.amplitude_range import (
    amplitude_range_prior_m,
    received_level_db_from_rms,
)


@pytest.mark.parametrize(
    ("received_db", "expected_m", "clamped"),
    [
        (80.0, 10.0, False),  # 20 dB down → 10×
        (60.0, 100.0, False),  # 40 dB down → 100×
        (100.0, 5.0, True),  # at reference → 1 m, clamped up to floor
        (40.0, 1000.0, False),  # 60 dB down → 1000 m at ceiling boundary
        (0.0, 1000.0, True),  # far below → clamped to ceiling
    ],
)
def test_amplitude_prior_inverse_square_table(received_db, expected_m, clamped) -> None:
    r, was_clamped = amplitude_range_prior_m(
        received_db,
        reference_source_level_db=100.0,
        min_range_m=5.0,
        max_range_m=1000.0,
    )
    assert r == pytest.approx(expected_m, rel=1e-4, abs=1e-2)
    assert was_clamped is clamped


def test_amplitude_prior_non_finite_falls_back_to_floor() -> None:
    r, clamped = amplitude_range_prior_m(
        float("nan"),
        reference_source_level_db=100.0,
        min_range_m=5.0,
        max_range_m=1000.0,
    )
    assert r == pytest.approx(5.0)
    assert clamped is True


def test_received_level_db_from_rms_matches_spl_formula() -> None:
    # Shared with assembly.py's SPL proxy: 20·log10(rms) + gain_offset.
    assert received_level_db_from_rms(0.1, 3.0) == pytest.approx(20.0 * math.log10(0.1) + 3.0)
    # Floors at 1e-9 to avoid log10(0).
    assert math.isfinite(received_level_db_from_rms(0.0, 0.0))


def test_amplitude_prior_matches_rust_helper_constants() -> None:
    # Cross-language parity: identical formula/clamps to
    # minimappr-ingest-sidecar/src/range_projection.rs::amplitude_range_prior_m.
    for received_db in (95.0, 80.0, 55.0, 30.0):
        r, _ = amplitude_range_prior_m(
            received_db,
            reference_source_level_db=100.0,
            min_range_m=5.0,
            max_range_m=1000.0,
        )
        raw = 10.0 ** ((100.0 - received_db) / 20.0)
        assert r == pytest.approx(min(max(raw, 5.0), 1000.0), rel=1e-5)
