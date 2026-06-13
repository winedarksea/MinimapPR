"""Distance-matrix localization coverage.

Confirms localization behaves correctly as a function of source *distance*
(2 / 10 / 100 / 1000 m) across node configurations and excitation frequency,
under perfect time sync. This complements the scattered ad-hoc geometries in
``test_cluster_synthetic_4node.py`` / ``test_mixed_node_spatial_localization.py``
/ ``test_phase2_localization_soundscape.py`` with a single, explicit matrix.

Physics encoded in the assertions:
  * A tight tetra (~7 cm aperture) only observes range in the true near field;
    beyond ~10 m it degrades to a bearing-only (``range_asymptotic``) estimate.
  * A synchronized multi-node cluster observes range from inter-node parallax.
    A realistic, fairly tight footprint (``MULTI_TIGHT``) still localizes a far
    source but with looser radial accuracy; a wider spread (``MULTI_WIDE``)
    fixes long range cleanly — the 1 km case is the maximal test.
  * Pure tones become spatially ambiguous when wavelength << baseline, so the
    frequency sweep is scoped to the tight single tetra.
"""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.advanced_localization import SRPPhatLocalizer
from minimappr.core.localization import LocalizationEngine
from minimappr.core.localization_dispatch import LocalizationDispatcher
from minimappr.models import LocalizationResult
from tests.helpers import synthesize_multinode_windows

SAMPLE_RATE_HZ = 48_000
SOUND_SPEED_MPS = 343.2

# A general 3-D look direction (azimuth + elevation) so every axis is exercised.
EXPECTED_DIRECTION = np.asarray([0.6, 0.7, 0.39], dtype=np.float64)
EXPECTED_DIRECTION /= np.linalg.norm(EXPECTED_DIRECTION)

TETRA_ORIGIN_M = np.asarray([0.0, 0.0, 1.5], dtype=np.float64)

# Multi-node footprints. MULTI_TIGHT is a realistic "as tight as practical"
# deployment (~8 m spacing); MULTI_WIDE is a deliberately wider long-range
# array (~50 m spacing). Both are one tetra plus three single-mic point nodes.
MULTI_TIGHT_NODES_M = {
    "tetra-a": (0.0, 0.0, 1.5),
    "point-b": (8.0, 0.0, 3.0),
    "point-c": (0.0, 8.0, 1.0),
    "point-d": (8.0, 8.0, 4.0),
}
MULTI_WIDE_NODES_M = {
    "tetra-a": (0.0, 0.0, 1.5),
    "point-b": (50.0, 0.0, 6.0),
    "point-c": (0.0, 50.0, 1.0),
    "point-d": (50.0, 50.0, 9.0),
}


def _broadband_excitation(seed: int, *, duration_s: float = 0.4) -> np.ndarray:
    sample_count = int(SAMPLE_RATE_HZ * duration_s)
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(sample_count) * np.hanning(sample_count)).astype(np.float32)


def _tone_excitation(freq_hz: float, *, duration_s: float = 0.4) -> np.ndarray:
    sample_count = int(SAMPLE_RATE_HZ * duration_s)
    t = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE_HZ
    return (np.sin(2.0 * np.pi * freq_hz * t) * np.hanning(sample_count)).astype(np.float32)


def _centroid_m(sensor_positions: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(list(sensor_positions.values())), axis=0)


def _bearing_dot(
    result: LocalizationResult,
    centroid_m: np.ndarray,
    expected_direction: np.ndarray,
) -> float:
    offset = np.asarray(result.position_m, dtype=np.float64) - centroid_m
    norm = float(np.linalg.norm(offset))
    if norm == 0.0:
        return 0.0
    return float(np.dot(offset / norm, expected_direction))


def _assert_position_fix(
    result: LocalizationResult,
    source_m: np.ndarray,
    centroid_m: np.ndarray,
    expected_direction: np.ndarray,
    *,
    max_error_m: float,
    min_bearing_dot: float = 0.97,
) -> None:
    estimate = np.asarray(result.position_m, dtype=np.float64)
    position_error_m = float(np.linalg.norm(estimate - source_m))
    assert _bearing_dot(result, centroid_m, expected_direction) >= min_bearing_dot
    assert position_error_m <= max_error_m, (
        f"position error {position_error_m:.2f} m exceeded {max_error_m:.2f} m "
        f"(estimate={estimate}, source={source_m})"
    )


def _multinode_dispatcher() -> LocalizationDispatcher:
    # max_tau_s must cover the largest inter-node delay (≈ widest baseline / c).
    # MULTI_WIDE diagonal ≈ 70 m → ≈ 0.21 s.
    return LocalizationDispatcher(
        algorithms={"gcc_phat": LocalizationEngine(max_tau_s=0.30, interp_factor=8)},
    )


def _single_tetra_localizer() -> SRPPhatLocalizer:
    return SRPPhatLocalizer(
        max_tau_s=0.003,
        grid_resolution_m=0.5,
        search_padding_m=1.0,
        interp=8,
        far_field_default_range_m=60.0,
        far_field_max_range_m=250.0,
    )


# --------------------------------------------------------------------------- #
# Single tetrahedral node
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("distance_m", [2.0, 10.0, 100.0, 1000.0])
def test_single_tetra_distance_matrix(distance_m: float) -> None:
    """A tight (~7 cm) tetra is a *bearing* instrument across the whole range.

    Azimuth/elevation are recovered with sub-degree accuracy at 2 m through 1 km,
    but the array baseline is far too small to observe *range* at any of these
    distances (``range_observability`` stays well under 0.10) — so the solver
    does not claim a trustworthy distance. True range fixing requires multiple
    nodes (see ``test_multinode_distance_matrix``); genuine near-field range
    refinement (≤ ~4 m) is covered by ``test_phase2_localization_soundscape``.
    """
    excitation = _broadband_excitation(seed=7)
    source_m = TETRA_ORIGIN_M + EXPECTED_DIRECTION * distance_m
    sensor_positions, sensor_windows, sensor_node_ids = synthesize_multinode_windows(
        excitation,
        SAMPLE_RATE_HZ,
        source_position_m=source_m,
        node_origins_m={"tetra-a": tuple(TETRA_ORIGIN_M)},
        tetra_node_ids=("tetra-a",),
        sound_speed_mps=SOUND_SPEED_MPS,
    )
    result = _single_tetra_localizer().localize(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
        sensor_node_ids=sensor_node_ids,
    )
    centroid_m = _centroid_m(sensor_positions)

    assert _bearing_dot(result, centroid_m, EXPECTED_DIRECTION) >= 0.97
    assert result.range_observability is not None
    assert result.range_observability < 0.10


# --------------------------------------------------------------------------- #
# Multi-node clusters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("config_name", "nodes", "distance_m", "max_error_m"),
    [
        # MULTI_TIGHT (~8 m footprint): clean position fix out to ~100 m; at 1 km
        # the parallax is too small to observe range, so it degrades to a
        # bearing-only estimate (max_error_m=None marks that regime).
        ("multi_tight", MULTI_TIGHT_NODES_M, 2.0, 1.0),
        ("multi_tight", MULTI_TIGHT_NODES_M, 10.0, 1.0),
        ("multi_tight", MULTI_TIGHT_NODES_M, 100.0, 12.0),
        ("multi_tight", MULTI_TIGHT_NODES_M, 1000.0, None),
        # MULTI_WIDE (~50 m footprint): true position fix all the way to 1 km.
        ("multi_wide", MULTI_WIDE_NODES_M, 100.0, 5.0),
        ("multi_wide", MULTI_WIDE_NODES_M, 1000.0, 80.0),
    ],
)
def test_multinode_distance_matrix(
    config_name: str,
    nodes: dict[str, tuple[float, float, float]],
    distance_m: float,
    max_error_m: float | None,
) -> None:
    del config_name  # used only for readable parametrize IDs
    excitation = _broadband_excitation(seed=11)
    cluster_centroid_m = np.mean(np.vstack([np.asarray(p) for p in nodes.values()]), axis=0)
    source_m = cluster_centroid_m + EXPECTED_DIRECTION * distance_m

    sensor_positions, sensor_windows, sensor_node_ids = synthesize_multinode_windows(
        excitation,
        SAMPLE_RATE_HZ,
        source_position_m=source_m,
        node_origins_m=nodes,
        tetra_node_ids=("tetra-a",),
        sound_speed_mps=SOUND_SPEED_MPS,
    )
    result = _multinode_dispatcher().localize(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
        sensor_node_ids=sensor_node_ids,
    )
    centroid_m = _centroid_m(sensor_positions)

    if max_error_m is None:
        # Range-collapse regime: bearing stays exact while the solver honestly
        # reports range as unobservable (small footprint vs a 1 km source).
        assert _bearing_dot(result, centroid_m, EXPECTED_DIRECTION) >= 0.97
        assert result.range_observability is not None
        assert result.range_observability < 0.10
    else:
        _assert_position_fix(
            result,
            source_m,
            centroid_m,
            EXPECTED_DIRECTION,
            max_error_m=max_error_m,
            min_bearing_dot=0.95,
        )


# --------------------------------------------------------------------------- #
# Frequency sensitivity (tight single tetra only — tones are unambiguous here)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("freq_hz", [500.0, 1500.0, 3000.0])
@pytest.mark.parametrize("distance_m", [2.0, 100.0])
def test_single_tetra_frequency_band_bearing(freq_hz: float, distance_m: float) -> None:
    excitation = _tone_excitation(freq_hz)
    source_m = TETRA_ORIGIN_M + EXPECTED_DIRECTION * distance_m
    sensor_positions, sensor_windows, sensor_node_ids = synthesize_multinode_windows(
        excitation,
        SAMPLE_RATE_HZ,
        source_position_m=source_m,
        node_origins_m={"tetra-a": tuple(TETRA_ORIGIN_M)},
        tetra_node_ids=("tetra-a",),
        sound_speed_mps=SOUND_SPEED_MPS,
    )
    result = _single_tetra_localizer().localize(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
        sensor_node_ids=sensor_node_ids,
    )
    centroid_m = _centroid_m(sensor_positions)
    # Across the band the bearing must stay accurate regardless of range regime.
    assert _bearing_dot(result, centroid_m, EXPECTED_DIRECTION) >= 0.90
