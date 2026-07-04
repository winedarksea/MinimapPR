from __future__ import annotations

import numpy as np

from minimappr.core.localization_uncertainty import (
    apply_frequency_covariance_scaling,
    clamp_covariance_eigenvalues_range_proportional,
)


def test_identity_at_or_above_alias_cutoff() -> None:
    """Item B: at or above the alias cutoff angular resolution is not degraded,
    so the covariance is returned unchanged."""
    cov = np.diag([3.0, 1.0, 1.5])
    bearing = np.array([1.0, 0.0, 0.0])
    out = apply_frequency_covariance_scaling(
        covariance_m2=cov,
        bearing_unit_vec=bearing,
        dominant_frequency_hz=4_000.0,
        alias_cutoff_hz=3_400.0,
    )
    np.testing.assert_allclose(out, cov)


def test_lateral_inflation_below_cutoff_preserves_radial_axis() -> None:
    """Item B: below the cutoff, variance perpendicular to the bearing inflates by
    (alias_cutoff / dominant)^2 while the radial (along-bearing) axis is unchanged."""
    # Radial axis = x; lateral plane = y, z. Distinct eigenvalues avoid degeneracy.
    cov = np.diag([3.0, 1.0, 1.5])
    bearing = np.array([1.0, 0.0, 0.0])
    # dominant/alias = 0.5 → inflation = (1/0.5)^2 = 4×
    out = apply_frequency_covariance_scaling(
        covariance_m2=cov,
        bearing_unit_vec=bearing,
        dominant_frequency_hz=1_000.0,
        alias_cutoff_hz=2_000.0,
    )
    # Along-bearing (radial) variance is preserved.
    assert abs(out[0, 0] - 3.0) < 1e-6
    # Lateral variances inflate ~4×.
    assert abs(out[1, 1] - 4.0) < 1e-6
    assert abs(out[2, 2] - 6.0) < 1e-6
    # Output stays symmetric.
    np.testing.assert_allclose(out, out.T, atol=1e-9)


def test_inflation_capped_at_frequency_floor() -> None:
    """Item B: a vanishing dominant frequency clamps to the floor (0.25), capping
    lateral inflation at 1/floor^2 = 16× rather than blowing up."""
    cov = np.diag([5.0, 1.0, 2.0])
    bearing = np.array([1.0, 0.0, 0.0])
    out = apply_frequency_covariance_scaling(
        covariance_m2=cov,
        bearing_unit_vec=bearing,
        dominant_frequency_hz=1.0,
        alias_cutoff_hz=1.0e6,
    )
    assert abs(out[1, 1] - 16.0) < 1e-4
    assert abs(out[2, 2] - 32.0) < 1e-4
    assert abs(out[0, 0] - 5.0) < 1e-6


def test_degenerate_inputs_returned_unchanged() -> None:
    """Item B: non-positive frequencies or a zero bearing vector are no-ops."""
    cov = np.diag([1.0, 2.0, 3.0])
    bearing = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        apply_frequency_covariance_scaling(cov, bearing, 0.0, 3_400.0), cov
    )
    np.testing.assert_allclose(
        apply_frequency_covariance_scaling(cov, bearing, 1_000.0, 0.0), cov
    )
    np.testing.assert_allclose(
        apply_frequency_covariance_scaling(cov, np.zeros(3), 1_000.0, 3_400.0), cov
    )


def test_range_proportional_clamp_scales_ceiling_with_range() -> None:
    """Phase 1b: the effective std ceiling grows with range, so an honest large
    covariance at 800 m survives that would be clipped by a fixed 250 m ceiling."""
    # Isotropic std = 400 m (variance 160000). At range 800 m with factor 1.0 the
    # effective ceiling is min(max(1.0*800, floor), ceiling) = min(800, 1000) = 800,
    # so a 400 m std passes through untouched.
    cov = np.diag([400.0**2, 400.0**2, 400.0**2])
    out, capped = clamp_covariance_eigenvalues_range_proportional(
        cov,
        range_m=800.0,
        std_factor=1.0,
        floor_std_m=30.0,
        ceiling_std_m=1000.0,
    )
    assert capped is False
    np.testing.assert_allclose(np.asarray(out), cov)


def test_range_proportional_clamp_bites_close_in() -> None:
    """Near-field: a bogus 400 m std at 10 m range is clamped to the floor (30 m)."""
    cov = np.diag([400.0**2, 1.0, 1.0])
    out, capped = clamp_covariance_eigenvalues_range_proportional(
        cov,
        range_m=10.0,
        std_factor=1.0,
        floor_std_m=30.0,
        ceiling_std_m=1000.0,
    )
    assert capped is True
    eigs = np.linalg.eigvalsh(np.asarray(out))
    assert float(np.max(eigs)) <= 30.0**2 + 1e-6


def test_range_proportional_clamp_absolute_ceiling_still_binds() -> None:
    """The absolute ceiling caps the range term: at 5000 m the ceiling stays 1000 m."""
    cov = np.diag([3000.0**2, 1.0, 1.0])
    out, capped = clamp_covariance_eigenvalues_range_proportional(
        cov,
        range_m=5000.0,
        std_factor=1.0,
        floor_std_m=30.0,
        ceiling_std_m=1000.0,
    )
    # Range term (5000) exceeds the absolute ceiling (1000), so the ceiling — not the
    # range term — is what bit; range_capped is False (absolute ceiling, not range).
    assert capped is False
    eigs = np.linalg.eigvalsh(np.asarray(out))
    assert float(np.max(eigs)) <= 1000.0**2 + 1e-3


def test_range_proportional_clamp_factor_zero_is_legacy_fixed() -> None:
    """std_factor <= 0 reproduces the legacy fixed clamp at the absolute ceiling."""
    cov = np.diag([400.0**2, 1.0, 1.0])
    out, capped = clamp_covariance_eigenvalues_range_proportional(
        cov,
        range_m=1000.0,
        std_factor=0.0,
        floor_std_m=30.0,
        ceiling_std_m=250.0,
    )
    assert capped is False
    eigs = np.linalg.eigvalsh(np.asarray(out))
    assert float(np.max(eigs)) <= 250.0**2 + 1e-3


def test_range_proportional_clamp_none_and_degenerate() -> None:
    out, capped = clamp_covariance_eigenvalues_range_proportional(
        None, range_m=100.0, std_factor=1.0, floor_std_m=30.0, ceiling_std_m=1000.0
    )
    assert out is None and capped is False
    # ceiling_std_m <= 0 passes list through unchanged.
    cov_list = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    out, capped = clamp_covariance_eigenvalues_range_proportional(
        cov_list, range_m=100.0, std_factor=1.0, floor_std_m=30.0, ceiling_std_m=0.0
    )
    assert out == cov_list and capped is False
