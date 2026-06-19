from __future__ import annotations

import numpy as np

from minimappr.core.localization_uncertainty import apply_frequency_covariance_scaling


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
