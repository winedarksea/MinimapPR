"""Planar 5-mic array geometry helpers: min-pair aliasing cutoff, coplanarity,
FOA suitability, and NodeSpec.half_space defaulting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from minimappr.models import GeoPoint, NodeSpec, NodeType
from minimappr.spatial_audio.geometry import (
    SIRITH_MIC_POSITIONS_M,
    alias_cutoff_from_positions,
    array_out_of_plane_extent_m,
    foa_geometry_suitable,
    is_coplanar,
    min_pair_spacing_m,
    reflect_covariance_into_half_space,
    reflect_position_into_half_space,
)

C = 343.2
_R = 0.025 * math.sqrt(0.5)
_PLANAR = np.array(
    [
        [_R, _R, 0.0],
        [-_R, _R, 0.0],
        [-_R, -_R, 0.0],
        [_R, -_R, 0.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def test_min_pair_spacing_is_corner_to_center_25mm() -> None:
    assert min_pair_spacing_m(_PLANAR) == pytest.approx(0.025, abs=1e-4)


def test_min_pair_mode_roughly_doubles_cutoff_vs_max_baseline() -> None:
    max_baseline_cutoff = alias_cutoff_from_positions(_PLANAR, C)  # default mode
    min_pair_cutoff = alias_cutoff_from_positions(_PLANAR, C, mode="min_pair")
    assert max_baseline_cutoff == pytest.approx(C / (2 * 0.05), rel=1e-3)
    assert min_pair_cutoff == pytest.approx(C / (2 * 0.025), rel=1e-3)
    assert min_pair_cutoff > 1.9 * max_baseline_cutoff


def test_tetra_cutoff_unchanged_by_new_mode_param() -> None:
    # Default (unspecified mode) must remain the max-baseline cutoff for tetra.
    default_cutoff = alias_cutoff_from_positions(SIRITH_MIC_POSITIONS_M, C)
    explicit_max = alias_cutoff_from_positions(
        SIRITH_MIC_POSITIONS_M, C, mode="max_baseline"
    )
    assert default_cutoff == pytest.approx(explicit_max)
    # Tetra's widest pair is ~52.5 mm -> ~3266 Hz.
    assert default_cutoff == pytest.approx(3266.0, abs=20.0)


def test_coplanarity_distinguishes_planar_from_tetra() -> None:
    assert is_coplanar(_PLANAR)
    assert array_out_of_plane_extent_m(_PLANAR) < 1e-4
    assert not is_coplanar(SIRITH_MIC_POSITIONS_M)
    assert array_out_of_plane_extent_m(SIRITH_MIC_POSITIONS_M) > 1e-2


def test_foa_rejects_coplanar_array() -> None:
    ok, reason = foa_geometry_suitable(_PLANAR, max_baseline_m=0.1)
    assert not ok
    assert "coplanar" in reason
    ok_tetra, _ = foa_geometry_suitable(SIRITH_MIC_POSITIONS_M, max_baseline_m=0.1)
    assert ok_tetra


def test_nodespec_half_space_defaults_by_node_type() -> None:
    geo = GeoPoint(lat=44.987, lon=-93.258, alt_m=281.5)
    planar = NodeSpec(
        id="planar-1",
        node_type=NodeType.SIRITH_PLANAR,
        position_geo=geo,
        sensor_offsets_m=[tuple(row) for row in _PLANAR.tolist()],
    )
    assert planar.half_space == "upper"

    tetra = NodeSpec(
        id="tetra-1",
        node_type=NodeType.SIRITH_TETRA,
        position_geo=geo,
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
    )
    assert tetra.half_space == "none"

    explicit = NodeSpec(
        id="planar-2",
        node_type=NodeType.SIRITH_PLANAR,
        half_space="lower",
        position_geo=geo,
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
    )
    assert explicit.half_space == "lower"


def test_reflect_position_mirrors_across_array_plane_when_on_wrong_side() -> None:
    plane_z = 2.0
    below_plane = np.array([1.0, 0.5, 1.0])  # z=1.0 < plane_z, wrong side for "upper"
    reflected = reflect_position_into_half_space(below_plane, plane_z, "upper")
    assert reflected[2] == pytest.approx(3.0)  # mirrored: 2*2.0 - 1.0
    assert reflected[0] == pytest.approx(1.0)
    assert reflected[1] == pytest.approx(0.5)

    above_plane = np.array([1.0, 0.5, 3.0])  # already on the correct side
    unchanged = reflect_position_into_half_space(above_plane, plane_z, "upper")
    assert unchanged[2] == pytest.approx(3.0)


def test_reflect_position_none_half_space_is_noop() -> None:
    position = np.array([1.0, 2.0, -5.0])
    assert np.allclose(reflect_position_into_half_space(position, 0.0, None), position)
    assert np.allclose(reflect_position_into_half_space(position, 0.0, "none"), position)


def test_reflect_covariance_flips_only_cross_z_terms_when_reflected() -> None:
    cov = np.array(
        [
            [1.0, 0.2, 0.3],
            [0.2, 1.5, 0.4],
            [0.3, 0.4, 2.0],
        ]
    )
    reflected = reflect_covariance_into_half_space(cov, was_reflected=True)
    assert reflected[0, 0] == pytest.approx(1.0)
    assert reflected[1, 1] == pytest.approx(1.5)
    assert reflected[2, 2] == pytest.approx(2.0)
    assert reflected[0, 1] == pytest.approx(0.2)
    assert reflected[0, 2] == pytest.approx(-0.3)
    assert reflected[2, 0] == pytest.approx(-0.3)
    assert reflected[1, 2] == pytest.approx(-0.4)

    unchanged = reflect_covariance_into_half_space(cov, was_reflected=False)
    assert np.allclose(unchanged, cov)
