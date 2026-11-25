"""The radial range search stops at the far-field envelope (TODO "Re-solve radial divergence").

A plane-wave (infinitely far) source makes the radial cost fall forever: the
unbounded search used to run the radius to float saturation (10^8-10^9 m),
which every sanity gate then dropped. Bounded, it is ``range_asymptotic`` at
the envelope with a radial std of at least that radius: bearing kept, range
honestly unknown.
"""

from __future__ import annotations

import numpy as np

from minimappr.core.radial_range_refinement import adaptive_radial_refinement
from minimappr.core.range_projection import RANGE_ASYMPTOTIC
from minimappr.spatial_audio.geometry import SIRITH_MIC_POSITIONS_M

SOUND_SPEED_MPS = 343.0
ENVELOPE_M = 5_000.0
TIME_STD_S = 1.0 / (48_000 * 4)


def _plane_wave_residual_function():
    sensors = np.asarray(SIRITH_MIC_POSITIONS_M, dtype=np.float64)
    direction = np.array([0.8, 0.5, 0.2]) / np.linalg.norm([0.8, 0.5, 0.2])
    # Arrival at sensor i minus sensor 0 for a plane wave from `direction`.
    measured = -(sensors[1:] - sensors[0]) @ direction / SOUND_SPEED_MPS

    def residual(position_m: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(sensors - position_m, axis=1)
        return (distances[1:] - distances[0]) / SOUND_SPEED_MPS - measured

    return residual, direction, sensors.mean(axis=0)


def _refine(maximum_radius_m):
    residual, direction, centroid = _plane_wave_residual_function()
    return adaptive_radial_refinement(
        centroid_m=centroid,
        direction=direction,
        initial_radius_m=50.0,
        minimum_radius_m=0.05,
        residual_function=residual,
        effective_time_std_s=TIME_STD_S,
        maximum_radius_m=maximum_radius_m,
    )


def test_unbounded_search_diverges_on_a_plane_wave() -> None:
    unbounded = _refine(None)
    assert np.linalg.norm(unbounded.position_m) > 1.0e6


def test_bounded_search_is_asymptotic_at_the_envelope_with_honest_radial_std() -> None:
    _residual, direction, centroid = _plane_wave_residual_function()
    bounded = _refine(ENVELOPE_M)
    radius_m = float(np.linalg.norm(bounded.position_m - centroid))
    assert bounded.projection_mode == RANGE_ASYMPTOTIC
    assert radius_m <= ENVELOPE_M * (1.0 + 1e-9)
    assert radius_m >= 50.0
    assert bounded.radial_std_m >= radius_m
    # The bearing survives the bound.
    assert float((bounded.position_m - centroid) @ direction) / radius_m > 0.999


def test_a_near_source_inside_the_envelope_is_unaffected() -> None:
    sensors = np.asarray(SIRITH_MIC_POSITIONS_M, dtype=np.float64)
    source = np.array([0.6, 0.3, 0.1])
    arrival = np.linalg.norm(sensors - source, axis=1) / SOUND_SPEED_MPS
    measured = arrival[1:] - arrival[0]

    def residual(position_m: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(sensors - position_m, axis=1)
        return (distances[1:] - distances[0]) / SOUND_SPEED_MPS - measured

    centroid = sensors.mean(axis=0)
    kwargs = dict(
        centroid_m=centroid,
        direction=source - centroid,
        initial_radius_m=0.2,
        minimum_radius_m=0.05,
        residual_function=residual,
        effective_time_std_s=TIME_STD_S,
    )
    np.testing.assert_allclose(
        adaptive_radial_refinement(**kwargs, maximum_radius_m=ENVELOPE_M).position_m,
        adaptive_radial_refinement(**kwargs).position_m,
    )
