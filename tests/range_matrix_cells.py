"""Cell definitions for the realistic range matrix (plan item E5).

A cell is one source range x node layout x clock class x position source.
This module turns a cell into (a) the true scene the synthesis renders, (b)
what the nodes *report* (packet clock offsets, hand-measured positions with a
stated accuracy) and (c) the observability oracle's inputs. It never runs the
pipeline, so the pass-class table can be derived from it alone.

Amplitude: the source level is set per cell so the nearest sensor hears it at
``NEAREST_SENSOR_IN_BAND_SNR_DB``; every other sensor follows spherical
spreading plus broadband air absorption from there. E5 isolates geometry and
timing; SNR-limited admission is measured by the E7 funnel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import speed_of_sound_mps
from minimappr.core.localization_overhaul_config import default_localization_clock_class_uncertainty_seconds
from minimappr.models import GeoPoint, NodeType, TimeQuality
from minimappr.sim.replay_gate_config import LABEL_MATCH_MODE_EXACT, TierScoringConfig
from minimappr.spatial_audio.geometry import SIRITH_MIC_POSITIONS_M
from tests.acoustic_scene_synthesis import AcousticScene, SceneNode, scene_node_near_enu
from tests.realistic_capture_synthesis import spreading_gains_relative_to_nearest
from tests.range_observability_oracle import (
    ObservabilityOracleConfig,
    ObservabilityVerdict,
    OracleNode,
    assess_observability,
)

# --- Site and signal ---
SITE_ORIGIN = GeoPoint(lat=44.98698840878797, lon=-93.2579197515542, alt_m=285.75)
COORDINATE_MODE = "flat"
SAMPLE_RATE_HZ = 48_000
SCENE_TEMPERATURE_C = 20.0
SCENE_HUMIDITY_FRACTION = 0.5
SOUND_SPEED_MPS = speed_of_sound_mps(SCENE_TEMPERATURE_C, SCENE_HUMIDITY_FRACTION)
GCC_PHAT_INTERPOLATION_FACTOR = 4
LOCALIZATION_WINDOW_SECONDS = 0.08
SOURCE_BAND_HZ = (400.0, 3200.0)
NEAREST_SENSOR_IN_BAND_SNR_DB = 30.0
# Broadband ISO 9613-1 absorption near 1-2 kHz at 20 C / 50 % RH.
AIR_ABSORPTION_DB_PER_M = 0.005
# Same SNR bar the noise-floor-relative trigger uses by default.
HEARD_MIN_IN_BAND_SNR_DB = 8.0

# --- Geometry ---
PLANAR_CORNER_RADIUS_M = 0.025
PLANAR_SENSOR_OFFSETS_BODY_M = tuple(
    (PLANAR_CORNER_RADIUS_M * x_sign * np.sqrt(0.5), PLANAR_CORNER_RADIUS_M * y_sign * np.sqrt(0.5), 0.0)
    for x_sign, y_sign in ((1, 1), (-1, 1), (-1, -1), (1, -1))
) + ((0.0, 0.0, 0.0),)
TETRA_SENSOR_OFFSETS_BODY_M = tuple(tuple(float(axis) for axis in row) for row in SIRITH_MIC_POSITIONS_M)
POINT_SENSOR_OFFSETS_BODY_M = ((0.0, 0.0, 0.0),)
LAYOUT_ANCHOR_ENU_M = (20.0, -10.0, 1.5)
# Source direction from the observer; 10 degrees up keeps a ground-mounted
# planar node's half-space prior on the true side (the mirror is the oracle's job).
SOURCE_AZIMUTH_DEG = 35.0
SOURCE_ELEVATION_DEG = 10.0

LAYOUT_SINGLE_TETRA = "single_tetra"
LAYOUT_SINGLE_PLANAR = "single_planar"
# The point sits in the planar array's own plane: all six sensors coplanar,
# so the source and its mirror across that plane give identical TDOAs.
LAYOUT_PLANAR_POINT_COPLANAR = "planar_point_coplanar"
# Four tetras across ~10 m with 3.5 m of height relief; the longest baseline
# keeps cross-node lags inside half the 80 ms localization window.
LAYOUT_FOUR_NODE_SPREAD = "four_node_spread"

_LAYOUT_NODES: dict[str, tuple[tuple[str, NodeType, tuple, tuple[float, float, float]], ...]] = {
    LAYOUT_SINGLE_TETRA: (("tetra-a", NodeType.SIRITH_TETRA, TETRA_SENSOR_OFFSETS_BODY_M, (0.0, 0.0, 0.0)),),
    LAYOUT_SINGLE_PLANAR: (("planar-a", NodeType.SIRITH_PLANAR, PLANAR_SENSOR_OFFSETS_BODY_M, (0.0, 0.0, 0.0)),),
    LAYOUT_PLANAR_POINT_COPLANAR: (
        ("planar-a", NodeType.SIRITH_PLANAR, PLANAR_SENSOR_OFFSETS_BODY_M, (0.0, 0.0, 0.0)),
        ("point-b", NodeType.POINT, POINT_SENSOR_OFFSETS_BODY_M, (6.0, 3.0, 0.0)),
    ),
    LAYOUT_FOUR_NODE_SPREAD: (
        ("tetra-a", NodeType.SIRITH_TETRA, TETRA_SENSOR_OFFSETS_BODY_M, (0.0, 0.0, 0.0)),
        ("tetra-b", NodeType.SIRITH_TETRA, TETRA_SENSOR_OFFSETS_BODY_M, (7.0, 1.0, 2.0)),
        ("tetra-c", NodeType.SIRITH_TETRA, TETRA_SENSOR_OFFSETS_BODY_M, (1.0, 6.0, 1.0)),
        ("tetra-d", NodeType.SIRITH_TETRA, TETRA_SENSOR_OFFSETS_BODY_M, (7.0, 7.0, 3.5)),
    ),
}
LAYOUTS = tuple(_LAYOUT_NODES)

# Off-matrix layouts for the kilometre-range tests (tests/test_localization_km_range_e2e.py);
# the matrix and its oracle table stay as they are.
# Five GPS point nodes across ~1.5 km (corner diagonal) with height relief, so
# cross-node lags reach ~4.4 s: past the old fixed 1 s coverage cap.
LAYOUT_POINT_NETWORK_KM = "point_network_km"
POINT_NETWORK_KM_SIDE_M = 1_060.0
_OFF_MATRIX_LAYOUT_NODES: dict[str, tuple[tuple[str, NodeType, tuple, tuple[float, float, float]], ...]] = {
    LAYOUT_POINT_NETWORK_KM: (
        ("point-a", NodeType.POINT, POINT_SENSOR_OFFSETS_BODY_M, (0.0, 0.0, 0.0)),
        ("point-b", NodeType.POINT, POINT_SENSOR_OFFSETS_BODY_M, (POINT_NETWORK_KM_SIDE_M, 0.0, 3.0)),
        ("point-c", NodeType.POINT, POINT_SENSOR_OFFSETS_BODY_M, (0.0, POINT_NETWORK_KM_SIDE_M, 8.0)),
        ("point-d", NodeType.POINT, POINT_SENSOR_OFFSETS_BODY_M, (POINT_NETWORK_KM_SIDE_M, POINT_NETWORK_KM_SIDE_M, 14.0)),
        ("point-e", NodeType.POINT, POINT_SENSOR_OFFSETS_BODY_M, (0.5 * POINT_NETWORK_KM_SIDE_M, 0.5 * POINT_NETWORK_KM_SIDE_M, 25.0)),
    ),
}

# --- Clock classes (packet timestamps) ---
CLOCK_GPS_LOCKED = "gps_locked"
CLOCK_NTP_TIMED = "ntp_timed"
CLOCK_TIME_QUALITY = {CLOCK_GPS_LOCKED: TimeQuality.GPS_LOCKED, CLOCK_NTP_TIMED: TimeQuality.NTP_DISCIPLINED}
CLOCK_CLASSES = tuple(CLOCK_TIME_QUALITY)
# Each node's actual packet-clock error in units of its class sigma: within
# about 1 sigma, of mixed sign, so cross-node errors neither cancel nor vanish.
# (The fifth entry serves only the off-matrix five-node layout.)
NODE_CLOCK_ERROR_IN_SIGMA = (0.8, -0.6, 1.1, -0.9, 0.5)

# --- Position sources ---
POSITION_GPS = "gps"
POSITION_HAND_MEASURED = "hand_measured"
POSITION_SOURCES = (POSITION_GPS, POSITION_HAND_MEASURED)
# Tape measure from a surveyed corner; the operator enters this accuracy.
HAND_MEASURED_POSITION_STD_M = 0.3
# Packet GPS is placed on the float32 grid, so the GPS fix is the truth; this
# models a converged stationary-node estimate, not raw single-fix jitter.
GPS_POSITION_STD_M = 0.0
# Actual entry error per node in units of the stated sigma (about 1 sigma).
NODE_HAND_MEASURED_ERROR_IN_SIGMA = (
    (0.7, -0.5, 0.3),
    (-0.6, 0.6, -0.2),
    (0.5, 0.8, 0.2),
    (-0.8, -0.4, -0.3),
    (0.4, -0.7, 0.1),
)

RANGES_M = (1.0, 10.0, 100.0, 1000.0)

TIER_SCORING_CONFIG = TierScoringConfig(
    bearing_correct_max_error_deg=10.0,
    range_resolved_max_absolute_error_m=0.5,
    range_resolved_max_relative_error=0.25,
    label_match_mode=LABEL_MATCH_MODE_EXACT,
)
ORACLE_CONFIG = ObservabilityOracleConfig(
    sound_speed_mps=SOUND_SPEED_MPS,
    lag_quantum_seconds=1.0 / (SAMPLE_RATE_HZ * GCC_PHAT_INTERPOLATION_FACTOR),
    correlation_window_seconds=LOCALIZATION_WINDOW_SECONDS,
    band_hz=SOURCE_BAND_HZ,
    heard_min_in_band_snr_db=HEARD_MIN_IN_BAND_SNR_DB,
    tiers=TIER_SCORING_CONFIG,
)


@dataclass(frozen=True)
class RangeMatrixCell:
    range_m: float
    layout: str
    clock_class: str
    position_source: str

    @property
    def cell_id(self) -> str:
        return f"{int(self.range_m)}m-{self.layout}-{self.clock_class}-{self.position_source}"


def all_cells() -> list[RangeMatrixCell]:
    return [
        RangeMatrixCell(range_m, layout, clock_class, position_source)
        for range_m in RANGES_M
        for layout in LAYOUTS
        for clock_class in CLOCK_CLASSES
        for position_source in POSITION_SOURCES
    ]


@dataclass(frozen=True)
class ReportedNode:
    """What one node tells the server, next to the truth it was built from."""

    scene_node: SceneNode
    time_quality: TimeQuality
    clock_std_seconds: float
    packet_clock_error_ns: int
    # Hand-measured nodes carry no packet GPS; the operator enters this.
    entered_position_enu_m: tuple[float, float, float] | None
    stated_position_std_m: float


@dataclass(frozen=True)
class RealizedCell:
    cell: RangeMatrixCell
    scene: AcousticScene
    reported_nodes: tuple[ReportedNode, ...]
    sensor_gain_relative_to_nearest: dict[str, float]
    observer_position_m: np.ndarray

    def oracle_nodes(self) -> list[OracleNode]:
        positions = self.scene.sensor_positions_enu_m()
        nodes = []
        for reported in self.reported_nodes:
            sensor_ids = reported.scene_node.sensor_ids
            nodes.append(
                OracleNode(
                    node_id=reported.scene_node.node_id,
                    sensor_positions_m=tuple(tuple(float(v) for v in positions[sensor_id]) for sensor_id in sensor_ids),
                    sensor_in_band_snr_db=tuple(
                        NEAREST_SENSOR_IN_BAND_SNR_DB + 20.0 * np.log10(self.sensor_gain_relative_to_nearest[sensor_id])
                        for sensor_id in sensor_ids
                    ),
                    clock_offset_std_seconds=reported.clock_std_seconds,
                    position_std_m=reported.stated_position_std_m,
                )
            )
        return nodes

    def oracle_verdict(self) -> ObservabilityVerdict:
        return assess_observability(
            self.oracle_nodes(),
            [self.scene.node_position_enu_m(reported.scene_node.node_id) for reported in self.reported_nodes],
            self.scene.source_position_enu_m,
            ORACLE_CONFIG,
        )


def _source_direction_unit() -> np.ndarray:
    azimuth, elevation = np.radians(SOURCE_AZIMUTH_DEG), np.radians(SOURCE_ELEVATION_DEG)
    return np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])


def realize_cell(cell: RangeMatrixCell) -> RealizedCell:
    frame = LocalCoordinateFrame(origin=SITE_ORIGIN, mode=COORDINATE_MODE)
    clock_std_seconds = default_localization_clock_class_uncertainty_seconds()[CLOCK_TIME_QUALITY[cell.clock_class].value]
    scene_nodes: list[SceneNode] = []
    reported: list[ReportedNode] = []
    layout_nodes = _LAYOUT_NODES.get(cell.layout) or _OFF_MATRIX_LAYOUT_NODES[cell.layout]
    for node_index, (node_id, node_type, offsets, anchor_delta) in enumerate(layout_nodes):
        scene_node = scene_node_near_enu(
            frame,
            node_id=node_id,
            node_type=node_type,
            approximate_position_enu_m=tuple(a + d for a, d in zip(LAYOUT_ANCHOR_ENU_M, anchor_delta)),
            sensor_offsets_body_m=offsets,
        )
        scene_nodes.append(scene_node)
        true_position_m = np.asarray(frame.geo_to_local(scene_node.position_geo), dtype=np.float64)
        hand_measured = cell.position_source == POSITION_HAND_MEASURED
        entered = (
            tuple(
                float(v)
                for v in true_position_m
                + HAND_MEASURED_POSITION_STD_M * np.asarray(NODE_HAND_MEASURED_ERROR_IN_SIGMA[node_index])
            )
            if hand_measured
            else None
        )
        reported.append(
            ReportedNode(
                scene_node=scene_node,
                time_quality=CLOCK_TIME_QUALITY[cell.clock_class],
                clock_std_seconds=clock_std_seconds,
                packet_clock_error_ns=int(round(NODE_CLOCK_ERROR_IN_SIGMA[node_index] * clock_std_seconds * 1e9)),
                entered_position_enu_m=entered,
                stated_position_std_m=HAND_MEASURED_POSITION_STD_M if hand_measured else GPS_POSITION_STD_M,
            )
        )
    provisional = AcousticScene(
        site_origin=SITE_ORIGIN,
        coordinate_mode=COORDINATE_MODE,
        nodes=tuple(scene_nodes),
        source_position_enu_m=(0.0, 0.0, 0.0),
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
    )
    node_positions = [provisional.node_position_enu_m(node.node_id) for node in scene_nodes]
    if len(scene_nodes) == 1:
        observer_m = node_positions[0]
    else:
        observer_m = np.mean(np.stack(list(provisional.sensor_positions_enu_m().values())), axis=0)
    source_m = observer_m + cell.range_m * _source_direction_unit()
    scene = AcousticScene(
        site_origin=SITE_ORIGIN,
        coordinate_mode=COORDINATE_MODE,
        nodes=tuple(scene_nodes),
        source_position_enu_m=tuple(float(v) for v in source_m),
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
    )
    return RealizedCell(
        cell=cell,
        scene=scene,
        reported_nodes=tuple(reported),
        sensor_gain_relative_to_nearest=spreading_gains_relative_to_nearest(scene, AIR_ABSORPTION_DB_PER_M),
        observer_position_m=np.asarray(observer_m, dtype=np.float64),
    )
