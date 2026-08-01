"""Tests for the two-node localization health fixes.

Covers four discrete fixes:
  1. Stationary-node GPS Kalman convergence + outlier jump gate (ingest).
  2. local_to_geo clamps out-of-range positions instead of raising (geo).
  3. Position sanity gate drops unphysical localizations (fusion_node).
  4. Covariance eigenvalue ceiling keeps track uncertainty physical (uncertainty).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core.fusion_node import FusionMetrics, FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.ingest import IngestProcessor
from minimappr.core.localization_uncertainty import clamp_covariance_eigenvalues
from minimappr.models import GeoPoint, NodeSpec, NodeType


# --------------------------------------------------------------------------- #
# Fix 1: stationary-node GPS Kalman hardening
# --------------------------------------------------------------------------- #


def _make_processor(coordinate_frame: LocalCoordinateFrame, **kwargs) -> IngestProcessor:
    settings = Settings()
    return IngestProcessor(
        localization_config=settings.localization_config(),
        fusion_config=settings.fusion_config(),
        registry=SimpleNamespace(),
        buffer=SimpleNamespace(),
        storage=SimpleNamespace(),
        coordinate_frame=coordinate_frame,
        preprocessor_factory=SimpleNamespace(),
        **kwargs,
    )


def _stationary_node(coordinate_frame: LocalCoordinateFrame, local_xyz) -> NodeSpec:
    return NodeSpec(
        id="stationary-1",
        node_type=NodeType.SIRITH_TETRA,
        position_geo=coordinate_frame.local_to_geo(local_xyz),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        mobility="stationary",
        metadata={"gps": {"signal": "fix_2d", "position_source": "gps_nmea_uart"}},
    )


def test_stationary_kde_averages_out_gps_noise() -> None:
    """A stationary node fed noisy fixes around a true point converges near it and
    does not chase the noise the way the mobile Q would."""
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=44.98, lon=-93.26, alt_m=250.0), mode="flat")
    processor = _make_processor(
        frame,
        node_position_kde_warmup_fixes=30,
    )
    rng = np.random.default_rng(7)
    true_xyz = (2.0, -1.0, 0.5)
    last_local = None
    for _ in range(800):
        noisy = (
            true_xyz[0] + rng.normal(0.0, 4.0),
            true_xyz[1] + rng.normal(0.0, 4.0),
            true_xyz[2] + rng.normal(0.0, 4.0),
        )
        spec = _stationary_node(frame, noisy)
        normalized, _ = processor._normalize_node_spec(spec)
        last_local = normalized.position_m
    # After heavy averaging the estimate sits within ~1 m of truth on each axis.
    assert abs(last_local[0] - true_xyz[0]) < 1.0
    assert abs(last_local[1] - true_xyz[1]) < 1.0
    assert abs(last_local[2] - true_xyz[2]) < 1.0


def test_explicit_stationary_kalman_rejects_single_large_jump() -> None:
    """Once initialized, a fix that jumps more than the gate is dropped, holding the
    prior estimate."""
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=44.98, lon=-93.26, alt_m=250.0), mode="flat")
    processor = _make_processor(
        frame,
        node_position_kalman_q_stationary=0.001,
        node_position_kalman_r=25.0,
        node_position_gps_gate_m=5.0,
    )
    # Initialize at origin.
    init = _stationary_node(frame, (0.0, 0.0, 0.0))
    normalized_init, _ = processor._normalize_node_spec(init, position_filter="kalman")
    assert normalized_init.position_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)

    # A 100 m jump (>> 5 m gate) must be rejected.
    jump = _stationary_node(frame, (100.0, 0.0, 0.0))
    normalized_jump, _ = processor._normalize_node_spec(jump, position_filter="kalman")
    assert normalized_jump.position_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


def test_mobile_node_still_uses_reactive_q() -> None:
    """A mobile node keeps the larger process noise and tracks a real move."""
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=44.98, lon=-93.26, alt_m=250.0), mode="flat")
    processor = _make_processor(
        frame,
        node_position_kalman_q=0.5,
        node_position_kalman_q_stationary=0.001,
        node_position_kalman_r=25.0,
        node_position_gps_gate_m=50.0,
    )
    spec0 = NodeSpec(
        id="mobile-1",
        node_type=NodeType.SIRITH_TETRA,
        position_geo=frame.local_to_geo((0.0, 0.0, 0.0)),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        mobility="mobile",
        metadata={"gps": {"signal": "fix_3d", "position_source": "gps_nmea_uart"}},
    )
    processor._normalize_node_spec(spec0)
    spec1 = spec0.model_copy(update={"position_geo": frame.local_to_geo((20.0, 0.0, 0.0))})
    normalized, _ = processor._normalize_node_spec(spec1)
    # Reactive Q moves a meaningful fraction toward the new fix in one step.
    assert 0.0 < normalized.position_m[0] < 20.0


# --------------------------------------------------------------------------- #
# Fix 2: local_to_geo real-world bounds guard
# --------------------------------------------------------------------------- #


def test_local_to_geo_clamps_out_of_range_position() -> None:
    """An absurd local position must produce an Earth-bounded GeoPoint, not raise."""
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=44.98, lon=-93.26, alt_m=250.0), mode="flat")
    # ~1e7 m east/north drives lat/lon far outside valid bounds.
    geo = frame.local_to_geo((50_000_000.0, 50_000_000.0, 500_000.0))
    assert -90.0 <= geo.lat <= 90.0
    assert -180.0 <= geo.lon <= 180.0
    assert -12_000.0 <= geo.alt_m <= 100_000.0


def test_local_to_geo_normal_position_unchanged() -> None:
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=44.98, lon=-93.26, alt_m=250.0), mode="flat")
    geo = frame.local_to_geo((10.0, -5.0, 2.0))
    assert geo.lat == pytest.approx(44.98, abs=1e-2)
    assert geo.lon == pytest.approx(-93.26, abs=1e-2)


# --------------------------------------------------------------------------- #
# Fix 3: position sanity gate in _build_localization_branch
# --------------------------------------------------------------------------- #


def _gate_stub(max_range_m: float = 500.0) -> SimpleNamespace:
    """Minimal stub exposing only what the sanity gate touches."""
    dropped: list[tuple[str, str]] = []
    return SimpleNamespace(
        localization_config=SimpleNamespace(
            localization_max_range_m=max_range_m,
            localization_max_position_std_m=250.0,
        ),
        _metrics=FusionMetrics(),
        _record_silent_drop=lambda *, stage, reason: dropped.append((stage, reason)),
        _dropped=dropped,
        _current_localizer_name=lambda: "gcc_phat",
        _record_range_projection_metrics=lambda mode: None,
    )


def test_sanity_gate_drops_out_of_range_localization() -> None:
    stub = _gate_stub(max_range_m=500.0)
    loc = SimpleNamespace(position_m=(600.0, 300.0, 50.0), reference_sensor="s0")
    result = FusionNode._build_localization_branch(
        stub,
        localization=loc,
        selected_windows={"s0": np.zeros(16, dtype=np.float32)},
        classification_windows={},
        capability_tier="full_3d",
    )
    assert result is None
    assert stub._metrics.localization_rejected_out_of_range_count == 1
    assert ("localization", "position_out_of_range") in stub._dropped


def test_sanity_gate_drops_non_finite_localization() -> None:
    stub = _gate_stub(max_range_m=500.0)
    loc = SimpleNamespace(position_m=(float("nan"), 0.0, 0.0), reference_sensor="s0")
    result = FusionNode._build_localization_branch(
        stub,
        localization=loc,
        selected_windows={"s0": np.zeros(16, dtype=np.float32)},
        classification_windows={},
        capability_tier="full_3d",
    )
    assert result is None
    assert stub._metrics.localization_rejected_out_of_range_count == 1


# --------------------------------------------------------------------------- #
# Fix 4: covariance eigenvalue ceiling
# --------------------------------------------------------------------------- #


def test_clamp_covariance_caps_large_eigenvalues() -> None:
    # 1e6 m² eigenvalue (σ = 1 km) must be capped to (250 m)² = 62_500 m².
    cov = np.diag([1_000_000.0, 4.0, 9.0])
    out = clamp_covariance_eigenvalues(cov, maximum_std_m=250.0)
    out = np.asarray(out)
    eigvals = np.linalg.eigvalsh(out)
    assert eigvals.max() <= 250.0**2 + 1e-6
    # Small axes are preserved.
    assert sorted(round(v, 3) for v in eigvals)[:2] == [4.0, 9.0]


def test_clamp_covariance_passthrough_when_within_ceiling() -> None:
    cov = [[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 1.0]]
    out = clamp_covariance_eigenvalues(cov, maximum_std_m=250.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(cov))


def test_clamp_covariance_handles_missing_and_disabled() -> None:
    assert clamp_covariance_eigenvalues(None, maximum_std_m=250.0) is None
    # maximum_std_m <= 0 disables the cap (returns input untouched).
    cov_list = [[1e9, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert clamp_covariance_eigenvalues(cov_list, maximum_std_m=0.0) == cov_list


# --------------------------------------------------------------------------- #
# Fix 5: a 2D GPS fix does not carry a usable altitude
# --------------------------------------------------------------------------- #


def _node_with_fix(coordinate_frame: LocalCoordinateFrame, local_xyz, signal: str) -> NodeSpec:
    return NodeSpec(
        id="gps-fix-node",
        node_type=NodeType.SIRITH_TETRA,
        position_geo=coordinate_frame.local_to_geo(local_xyz),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        mobility="stationary",
        metadata={"gps": {"signal": signal, "position_source": "gps_nmea_uart"}},
    )


def test_two_d_fix_holds_altitude_but_still_tracks_horizontally() -> None:
    """A fix_2d reading must not move the node vertically.

    Two co-sited production nodes reported altitudes 25 m apart because a 2D fix's
    altitude — which the receiver never solved — was smoothed like a real
    measurement and became the vertical baseline for every track.
    """
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=44.98, lon=-93.26, alt_m=250.0), mode="flat")
    processor = _make_processor(frame)

    # A real 3D fix establishes the trusted altitude.
    solved = _node_with_fix(frame, (0.0, 0.0, 25.0), "fix_3d")
    normalized, _ = processor._normalize_node_spec(solved, position_filter="raw")
    assert normalized.position_m[2] == pytest.approx(25.0, abs=1e-6)

    # The receiver degrades to a 2D fix reporting a wildly different altitude.
    degraded = _node_with_fix(frame, (0.0, 0.0, 0.7), "fix_2d")
    held, geo = processor._normalize_node_spec(degraded, position_filter="raw")
    assert held.position_m[2] == pytest.approx(25.0, abs=1e-6)
    assert geo.alt_m == pytest.approx(275.0, abs=1e-3)

    # Latitude/longitude ARE solved by a 2D fix and must still track.
    moved = _node_with_fix(frame, (40.0, -15.0, 0.7), "fix_2d")
    tracked, _ = processor._normalize_node_spec(moved, position_filter="raw")
    assert tracked.position_m[0] == pytest.approx(40.0, abs=1e-3)
    assert tracked.position_m[1] == pytest.approx(-15.0, abs=1e-3)
    assert tracked.position_m[2] == pytest.approx(25.0, abs=1e-6)


def test_two_d_fix_without_prior_three_d_fix_pins_to_site_datum() -> None:
    """A node that has NEVER had a 3D fix has no measured altitude at all.

    Receivers emit a GGA altitude even under a GSA 2D fix; passing it through
    fabricated a 5.7 m vertical spread between two co-sited nodes on the
    2026-08-01 live box. Default mode pins ENU z to the site datum instead.
    """
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=44.98, lon=-93.26, alt_m=250.0), mode="flat")
    processor = _make_processor(frame)

    only_2d = _node_with_fix(frame, (0.0, 0.0, 12.0), "fix_2d")
    normalized, geo = processor._normalize_node_spec(only_2d, position_filter="raw")
    assert normalized.position_m[2] == pytest.approx(0.0, abs=1e-6)
    assert geo.alt_m == pytest.approx(250.0, abs=1e-3)


def test_two_d_fix_without_prior_three_d_fix_raw_mode_passthrough() -> None:
    """gps_2d_altitude_mode="raw" restores the legacy passthrough-to-filters."""
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=44.98, lon=-93.26, alt_m=250.0), mode="flat")
    processor = _make_processor(frame, gps_2d_altitude_mode="raw")

    only_2d = _node_with_fix(frame, (0.0, 0.0, 12.0), "fix_2d")
    normalized, _ = processor._normalize_node_spec(only_2d, position_filter="raw")
    assert normalized.position_m[2] == pytest.approx(12.0, abs=1e-6)


def test_datum_hold_releases_once_three_d_fix_arrives() -> None:
    """The datum pin is only a stand-in: a later real 3D fix replaces it, and a
    subsequent 2D degradation holds the measured altitude, not the datum."""
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=44.98, lon=-93.26, alt_m=250.0), mode="flat")
    processor = _make_processor(frame)

    pinned, _ = processor._normalize_node_spec(
        _node_with_fix(frame, (0.0, 0.0, 12.0), "fix_2d"), position_filter="raw"
    )
    assert pinned.position_m[2] == pytest.approx(0.0, abs=1e-6)
    solved, _ = processor._normalize_node_spec(
        _node_with_fix(frame, (0.0, 0.0, 25.0), "fix_3d"), position_filter="raw"
    )
    assert solved.position_m[2] == pytest.approx(25.0, abs=1e-6)
    held, _ = processor._normalize_node_spec(
        _node_with_fix(frame, (0.0, 0.0, 0.7), "fix_2d"), position_filter="raw"
    )
    assert held.position_m[2] == pytest.approx(25.0, abs=1e-6)


def test_datum_pin_keeps_kde_z_at_datum_on_two_d_only_input() -> None:
    """The KDE filter must never advance z off the datum while only 2D fixes
    arrive — the substituted value reaches the estimator, not the raw GGA z."""
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=44.98, lon=-93.26, alt_m=250.0), mode="flat")
    processor = _make_processor(frame, node_position_kde_warmup_fixes=5)

    normalized = None
    for i in range(30):
        node = _node_with_fix(frame, (0.0, 0.0, 10.0 + (i % 3)), "fix_2d")
        normalized, _ = processor._normalize_node_spec(node, position_filter="kde")
    assert normalized is not None
    assert normalized.position_m[2] == pytest.approx(0.0, abs=1e-6)


def test_three_d_fix_refreshes_the_trusted_altitude() -> None:
    """A recovered 3D fix replaces the held altitude rather than staying pinned."""
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=44.98, lon=-93.26, alt_m=250.0), mode="flat")
    processor = _make_processor(frame)

    processor._normalize_node_spec(
        _node_with_fix(frame, (0.0, 0.0, 25.0), "fix_3d"), position_filter="raw"
    )
    processor._normalize_node_spec(
        _node_with_fix(frame, (0.0, 0.0, 0.7), "fix_2d"), position_filter="raw"
    )
    recovered, _ = processor._normalize_node_spec(
        _node_with_fix(frame, (0.0, 0.0, 31.0), "fix_3d"), position_filter="raw"
    )
    assert recovered.position_m[2] == pytest.approx(31.0, abs=1e-6)

    # And the newly solved altitude becomes the one held on the next degradation.
    degraded_again, _ = processor._normalize_node_spec(
        _node_with_fix(frame, (0.0, 0.0, 0.7), "fix_2d"), position_filter="raw"
    )
    assert degraded_again.position_m[2] == pytest.approx(31.0, abs=1e-6)
