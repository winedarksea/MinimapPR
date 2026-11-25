"""Realistic range matrix, 1 m to 1 km, through normal ingest (plan item E5).

Matrix: source range x node layout (single tetra, single planar, planar +
point in one plane, four-node spread) x clock class (GPS-locked, NTP-timed)
x position source (packet GPS, hand-measured with a stated accuracy). Each
cell streams firmware-shaped uploads with packet-time stamps (each node's
own clock error applied) through the real sidecar and the full app, and is
judged on the stored detections (tests/range_matrix_scoring.py).

The pass class of every cell is fixed in ``EXPECTED_PASS_CLASSES`` below,
derived from the physical observability oracle
(``tests/range_observability_oracle.py``) before any cell was run;
``test_pass_class_table_is_the_oracle_verdict`` keeps the two in lockstep,
so a table edit is a deliberate oracle or scene change, never a tweak to
match the pipeline. Each entry is (guaranteed, max honest): the pipeline must
reach the first and must not claim more than the second.

Default suite: the six-cell diagonal (every range, layout, clock class and
position source appears). Full matrix: set ``MINIMAPPR_ENABLE_RANGE_MATRIX_FULL=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.range_matrix_cells import (
    CLOCK_CLASSES,
    LAYOUTS,
    ORACLE_CONFIG,
    POSITION_SOURCES,
    RANGES_M,
    RangeMatrixCell,
    all_cells,
    realize_cell,
)
from tests.range_observability_oracle import OracleNode, PassClass, assess_observability, position_covariance_m2

FULL_MATRIX_ENV = "MINIMAPPR_ENABLE_RANGE_MATRIX_FULL"

NOT_HEARD = PassClass.NOT_HEARD
HEARD = PassClass.HEARD
BEARING = PassClass.BEARING_CORRECT
RANGE = PassClass.RANGE_RESOLVED
FIX_3D = PassClass.FIX_3D

# (guaranteed, max honest) per cell, from the oracle. Do not edit to match a run.
EXPECTED_PASS_CLASSES: dict[str, tuple[PassClass, PassClass]] = {
    "1m-single_tetra-gps_locked-gps": (BEARING, BEARING),
    "1m-single_tetra-gps_locked-hand_measured": (HEARD, HEARD),
    "1m-single_tetra-ntp_timed-gps": (BEARING, BEARING),
    "1m-single_tetra-ntp_timed-hand_measured": (HEARD, HEARD),
    "1m-single_planar-gps_locked-gps": (BEARING, BEARING),
    "1m-single_planar-gps_locked-hand_measured": (HEARD, HEARD),
    "1m-single_planar-ntp_timed-gps": (BEARING, BEARING),
    "1m-single_planar-ntp_timed-hand_measured": (HEARD, HEARD),
    "1m-planar_point_coplanar-gps_locked-gps": (RANGE, FIX_3D),
    "1m-planar_point_coplanar-gps_locked-hand_measured": (HEARD, HEARD),
    "1m-planar_point_coplanar-ntp_timed-gps": (BEARING, FIX_3D),
    "1m-planar_point_coplanar-ntp_timed-hand_measured": (HEARD, HEARD),
    "1m-four_node_spread-gps_locked-gps": (FIX_3D, FIX_3D),
    "1m-four_node_spread-gps_locked-hand_measured": (HEARD, FIX_3D),
    "1m-four_node_spread-ntp_timed-gps": (FIX_3D, FIX_3D),
    "1m-four_node_spread-ntp_timed-hand_measured": (HEARD, HEARD),
    "10m-single_tetra-gps_locked-gps": (BEARING, BEARING),
    "10m-single_tetra-gps_locked-hand_measured": (BEARING, BEARING),
    "10m-single_tetra-ntp_timed-gps": (BEARING, BEARING),
    "10m-single_tetra-ntp_timed-hand_measured": (BEARING, BEARING),
    "10m-single_planar-gps_locked-gps": (BEARING, BEARING),
    "10m-single_planar-gps_locked-hand_measured": (BEARING, BEARING),
    "10m-single_planar-ntp_timed-gps": (BEARING, BEARING),
    "10m-single_planar-ntp_timed-hand_measured": (BEARING, BEARING),
    "10m-planar_point_coplanar-gps_locked-gps": (BEARING, BEARING),
    "10m-planar_point_coplanar-gps_locked-hand_measured": (BEARING, BEARING),
    "10m-planar_point_coplanar-ntp_timed-gps": (BEARING, BEARING),
    "10m-planar_point_coplanar-ntp_timed-hand_measured": (BEARING, BEARING),
    "10m-four_node_spread-gps_locked-gps": (FIX_3D, FIX_3D),
    "10m-four_node_spread-gps_locked-hand_measured": (FIX_3D, FIX_3D),
    "10m-four_node_spread-ntp_timed-gps": (FIX_3D, FIX_3D),
    "10m-four_node_spread-ntp_timed-hand_measured": (FIX_3D, FIX_3D),
    "100m-single_tetra-gps_locked-gps": (BEARING, BEARING),
    "100m-single_tetra-gps_locked-hand_measured": (BEARING, BEARING),
    "100m-single_tetra-ntp_timed-gps": (BEARING, BEARING),
    "100m-single_tetra-ntp_timed-hand_measured": (BEARING, BEARING),
    "100m-single_planar-gps_locked-gps": (BEARING, BEARING),
    "100m-single_planar-gps_locked-hand_measured": (BEARING, BEARING),
    "100m-single_planar-ntp_timed-gps": (BEARING, BEARING),
    "100m-single_planar-ntp_timed-hand_measured": (BEARING, BEARING),
    "100m-planar_point_coplanar-gps_locked-gps": (BEARING, BEARING),
    "100m-planar_point_coplanar-gps_locked-hand_measured": (BEARING, BEARING),
    "100m-planar_point_coplanar-ntp_timed-gps": (HEARD, BEARING),
    "100m-planar_point_coplanar-ntp_timed-hand_measured": (HEARD, BEARING),
    "100m-four_node_spread-gps_locked-gps": (FIX_3D, FIX_3D),
    "100m-four_node_spread-gps_locked-hand_measured": (BEARING, FIX_3D),
    "100m-four_node_spread-ntp_timed-gps": (BEARING, FIX_3D),
    "100m-four_node_spread-ntp_timed-hand_measured": (BEARING, FIX_3D),
    "1000m-single_tetra-gps_locked-gps": (BEARING, BEARING),
    "1000m-single_tetra-gps_locked-hand_measured": (BEARING, BEARING),
    "1000m-single_tetra-ntp_timed-gps": (BEARING, BEARING),
    "1000m-single_tetra-ntp_timed-hand_measured": (BEARING, BEARING),
    "1000m-single_planar-gps_locked-gps": (BEARING, BEARING),
    "1000m-single_planar-gps_locked-hand_measured": (BEARING, BEARING),
    "1000m-single_planar-ntp_timed-gps": (BEARING, BEARING),
    "1000m-single_planar-ntp_timed-hand_measured": (BEARING, BEARING),
    "1000m-planar_point_coplanar-gps_locked-gps": (BEARING, BEARING),
    "1000m-planar_point_coplanar-gps_locked-hand_measured": (BEARING, BEARING),
    "1000m-planar_point_coplanar-ntp_timed-gps": (BEARING, BEARING),
    "1000m-planar_point_coplanar-ntp_timed-hand_measured": (BEARING, BEARING),
    "1000m-four_node_spread-gps_locked-gps": (FIX_3D, FIX_3D),
    "1000m-four_node_spread-gps_locked-hand_measured": (BEARING, BEARING),
    "1000m-four_node_spread-ntp_timed-gps": (BEARING, BEARING),
    "1000m-four_node_spread-ntp_timed-hand_measured": (BEARING, BEARING),
}

DEFAULT_DIAGONAL_CELL_IDS = (
    "1m-single_tetra-gps_locked-gps",
    "10m-single_planar-ntp_timed-hand_measured",
    "10m-planar_point_coplanar-gps_locked-gps",
    "10m-four_node_spread-ntp_timed-gps",
    "100m-single_tetra-gps_locked-hand_measured",
    # The one passing four-node cross-node cell (4 of 4 runs on the merged
    # overhaul, either reporter); its gps sibling stays a known gap.
    "1000m-four_node_spread-gps_locked-hand_measured",
)

# Measured with --runxfail on the milestone-2 integration head (R4, R5, R7, P9,
# E4 merged), re-measured after the relative-trigger onset window and the
# compact-array guard fix (which cleared the 100 m lone-tetra regression, the
# 1 km planar + point GPS bearing, and the 100 m hand-measured planar cells).
# Every reason below is an open defect, not pending milestone-2 work: remove an
# entry when the fix lands (strict: an unexpected pass fails). The GCC-PHAT
# band split (2026-11-25: wide, tapered PHAT-beta cross-node pairs) cleared
# the 100 m NTP hand-measured planar + point collapse (every run since) and
# made the 100 m NTP GPS one flaky.
_AFTER_M2 = "open after milestone 2 (R4/R5/R7/P9 on base)"
# M4 made position_std_m widen the published covariance (both solvers); the
# hand-measured cells below still fail, but no longer for an overconfident
# covariance.
_AFTER_M4 = "open after M4 (position_std_m widens the covariance)"
_HAND_MEASURED_RANGE_CLAIM = (
    f"{_AFTER_M4}: the stated covariance now contains the truth, but the solve still reports range_refined "
    "with range observability over the weak threshold, a range claim the oracle allows only as heard "
    "(0.3 m placement errors at 1 m); the range-claim rule ignores the stated position accuracy"
)
_HAND_MEASURED_SPREAD_RANGE = (
    f"{_AFTER_M4}: the four-node solve on 0.3 m hand-measured positions at 10 m keeps the bearing but reports "
    "weak range observability (a bearing-only claim) where the oracle guarantees a 3-D fix"
)
_NTP_CROSS_NODE_UNUSABLE = f"{_AFTER_M2}: the NTP-timed (2 ms class) cross-node solve is still unusable or overconfident"
_PLANAR_FAR_FIELD = (
    f"{_AFTER_M2}: a planar array at 100 m+ (alone or with its coplanar point node) is stored omni "
    "(low_localization_confidence) or points the wrong way, where a tetra at the same range localizes"
)
_SPREAD_FAR_FIELD = (
    f"{_AFTER_M2}: the four-node cross-node solve at 100 m+ is unusable (stored omni, no detection covering "
    "the event, or the right bearing at about half the range with a covariance that excludes the truth)"
)
_COMPACT_PLANAR_POINT_WEAK_CLOCK = (
    "solver gap reported by P9: compact planar + one point node with a biased NTP clock collapses near the "
    "array with a too-small covariance"
)


def _gap(base: str, detail: str) -> str:
    return f"{base}; measured: {detail}"


KNOWN_PIPELINE_GAPS: dict[str, str] = {
    # -- diagonal --
    "10m-four_node_spread-ntp_timed-gps": _gap(
        _NTP_CROSS_NODE_UNUSABLE, "stored omni (render fallback); earlier 6.8 m off at 6.2 deg; oracle guarantees a 3-D fix"
    ),
    "1000m-four_node_spread-gps_locked-gps": _gap(
        _SPREAD_FAR_FIELD,
        "bearing right (0.03 deg) but 458 m short; stated angular std 0.01 deg vs 0.77 deg elevation error "
        "(Mahalanobis^2 3551 > 13.8); earlier runs 460-710 m short or no covering detection",
    ),
    # -- near range, cross-node, weak clock / hand-measured --
    "1m-planar_point_coplanar-gps_locked-hand_measured": _gap(
        _HAND_MEASURED_RANGE_CLAIM,
        "claims RANGE_RESOLVED (0.45 m off, contained) with the cross-node PHAT-beta front end; before M4 it "
        "claimed a 3-D fix on a 0.05 m covariance",
    ),
    "1m-planar_point_coplanar-ntp_timed-gps": _gap(
        _COMPACT_PLANAR_POINT_WEAK_CLOCK, "1.74 m off vs stated std [0.20, 0.14, 0.25] m (Mahalanobis^2 47 > 16.3)"
    ),
    "1m-four_node_spread-ntp_timed-gps": _gap(
        _NTP_CROSS_NODE_UNUSABLE, "no stored detection covers the event (twice); earlier 1.5 m off vs stated 0.05 m"
    ),
    "10m-planar_point_coplanar-ntp_timed-gps": _gap(
        _COMPACT_PLANAR_POINT_WEAK_CLOCK, "9.4 m off vs stated std [0.18, 0.13, 0.24] m, claims a 3-D fix"
    ),
    "10m-four_node_spread-gps_locked-hand_measured": _gap(
        _HAND_MEASURED_SPREAD_RANGE, "2.1 m off, bearing 0.3 deg, bearing-only claim (3 of 3 runs on the M4 head)"
    ),
    "10m-four_node_spread-ntp_timed-hand_measured": _gap(
        _NTP_CROSS_NODE_UNUSABLE, "stored omni (render fallback); oracle guarantees a 3-D fix"
    ),
    # -- planar far field --
    # (The four 1000m-single_planar cells pass since loc/km-range: bearing right
    # and inside the stated uncertainty, after the far-field envelope, bounded
    # radial search and unobservable-range covariance changes.)
    "100m-planar_point_coplanar-gps_locked-gps": _gap(_PLANAR_FAR_FIELD, "stored omni (low_localization_confidence)"),
    "100m-planar_point_coplanar-gps_locked-hand_measured": _gap(
        _PLANAR_FAR_FIELD, "bearing 51 deg wrong vs stated angular std 8 deg (Mahalanobis^2 46 > 13.8)"
    ),
    "1000m-planar_point_coplanar-ntp_timed-gps": _gap(
        _COMPACT_PLANAR_POINT_WEAK_CLOCK, "1001 m off (at the array) vs stated std [1.46, 1.11, 2.66] m, claims a 3-D fix"
    ),
    "1000m-planar_point_coplanar-ntp_timed-hand_measured": _gap(
        _COMPACT_PLANAR_POINT_WEAK_CLOCK, "1001 m off (at the array) vs stated std [2.63, 3.01, 3.14] m, claims a 3-D fix"
    ),
    # -- spread far field --
    "100m-four_node_spread-gps_locked-gps": _gap(_SPREAD_FAR_FIELD, "no stored detection covers the event"),
    "100m-four_node_spread-ntp_timed-gps": _gap(_SPREAD_FAR_FIELD, "stored omni"),
    "100m-four_node_spread-ntp_timed-hand_measured": _gap(_SPREAD_FAR_FIELD, "stored omni"),
    "1000m-four_node_spread-ntp_timed-gps": _gap(_SPREAD_FAR_FIELD, "stored omni"),
    "1000m-four_node_spread-ntp_timed-hand_measured": _gap(_SPREAD_FAR_FIELD, "stored omni"),
}
# Cells that both passed and failed across full-matrix runs (opt-in only):
# non-strict, so they neither hide nor break the run.
KNOWN_FLAKY_CELLS: dict[str, str] = {
    "100m-four_node_spread-gps_locked-hand_measured": (
        f"flaky after M4 ({_SPREAD_FAR_FIELD}): passes when tetra-b reports (65 m off, bearing right, inside the "
        "now-widened covariance), stored omni (render fallback) when tetra-d reports; 1 of 3 runs on the M4 head "
        "passed, so it left the default diagonal (100m-single_tetra-gps_locked-hand_measured took its place)"
    ),
    "1m-planar_point_coplanar-gps_locked-gps": "flaky: the 6-sensor solve's range claim comes and goes run to run",
    "1m-four_node_spread-ntp_timed-hand_measured": (
        "flaky: sometimes stored omni (passes), sometimes a 0.4-1.2 m-off 3-D fix with stated std 0.05-0.08 m"
    ),
    "10m-four_node_spread-gps_locked-gps": (
        "flaky on the milestone-2 head: usually a 0.15 m 3-D fix, once no stored detection covered the event"
    ),
    "100m-planar_point_coplanar-ntp_timed-gps": (
        f"flaky after the GCC-PHAT band split ({_COMPACT_PLANAR_POINT_WEAK_CLOCK}): passes as stored omni when "
        "point-b reports (render fallback, honest HEARD), otherwise the 96 m collapse at the array; 3 of 5 runs passed"
    ),
    "10m-planar_point_coplanar-ntp_timed-hand_measured": (
        f"flaky after the onset-window fix ({_COMPACT_PLANAR_POINT_WEAK_CLOCK}): passes when planar-a reports "
        "(3.9 m off, bearing right), fails when point-b reports (9.9 m off, bearing 68 deg wrong); 1 of 3 runs passed"
    ),
}

CELLS_BY_ID = {cell.cell_id: cell for cell in all_cells()}


class RangeMatrixCellViolation(AssertionError):
    """Raised only by the cell verdict, so a strict xfail never absorbs a harness
    failure (upload rejected, app start-up, sidecar timeout)."""


def _cell_param(cell_id: str):
    marks = []
    if cell_id in KNOWN_PIPELINE_GAPS:
        marks.append(
            pytest.mark.xfail(strict=True, raises=RangeMatrixCellViolation, reason=KNOWN_PIPELINE_GAPS[cell_id])
        )
    if cell_id in KNOWN_FLAKY_CELLS:
        marks.append(
            pytest.mark.xfail(strict=False, raises=RangeMatrixCellViolation, reason=KNOWN_FLAKY_CELLS[cell_id])
        )
    return pytest.param(cell_id, id=cell_id, marks=marks)


# ---- oracle and table ---------------------------------------------------------------------


def test_matrix_covers_every_combination_and_the_diagonal_spans_every_axis():
    assert set(EXPECTED_PASS_CLASSES) == set(CELLS_BY_ID)
    assert len(CELLS_BY_ID) == len(RANGES_M) * len(LAYOUTS) * len(CLOCK_CLASSES) * len(POSITION_SOURCES)
    diagonal = [CELLS_BY_ID[cell_id] for cell_id in DEFAULT_DIAGONAL_CELL_IDS]
    assert {cell.range_m for cell in diagonal} == set(RANGES_M)
    assert {cell.layout for cell in diagonal} == set(LAYOUTS)
    assert {cell.clock_class for cell in diagonal} == set(CLOCK_CLASSES)
    assert {cell.position_source for cell in diagonal} == set(POSITION_SOURCES)
    assert set(KNOWN_PIPELINE_GAPS) <= set(CELLS_BY_ID)
    assert set(KNOWN_FLAKY_CELLS) <= set(CELLS_BY_ID)
    assert not set(KNOWN_FLAKY_CELLS) & set(KNOWN_PIPELINE_GAPS)
    assert not set(KNOWN_FLAKY_CELLS) & set(DEFAULT_DIAGONAL_CELL_IDS)


def test_pass_class_table_is_the_oracle_verdict():
    mismatches = {}
    for cell_id, cell in CELLS_BY_ID.items():
        verdict = realize_cell(cell).oracle_verdict()
        computed = (verdict.guaranteed_class, verdict.max_honest_class)
        if computed != EXPECTED_PASS_CLASSES[cell_id]:
            mismatches[cell_id] = (EXPECTED_PASS_CLASSES[cell_id], computed)
    assert not mismatches, mismatches


def _oracle_node(node_id, positions, clock_std_s, *, snr_db=30.0, position_std_m=0.0) -> OracleNode:
    return OracleNode(
        node_id=node_id,
        sensor_positions_m=tuple(tuple(map(float, p)) for p in positions),
        sensor_in_band_snr_db=(snr_db,) * len(positions),
        clock_offset_std_seconds=clock_std_s,
        position_std_m=position_std_m,
    )


def _ring(center, radius_m, count, z_offsets=None):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    z_offsets = z_offsets if z_offsets is not None else [0.0] * count
    return [np.asarray(center) + [radius_m * np.cos(a), radius_m * np.sin(a), z] for a, z in zip(angles, z_offsets)]


def test_node_clock_error_is_shared_by_the_nodes_pairs():
    """One node: its clock error is common to every channel and cancels. Two
    nodes: adding channels to each node cannot average the cross-node clock
    error away, because every cross-node pair carries the same two offsets."""
    tetra = _ring((0.0, 0.0, 1.5), 0.03, 4, z_offsets=[0.0, 0.03, 0.0, 0.03])
    source = np.array([3.0, 2.0, 2.0])
    lone_gps = position_covariance_m2([_oracle_node("a", tetra, 1e-6)], source, ORACLE_CONFIG)
    lone_ntp = position_covariance_m2([_oracle_node("a", tetra, 2e-3)], source, ORACLE_CONFIG)
    np.testing.assert_allclose(lone_ntp, lone_gps, rtol=1e-6)

    def range_std(sensors_per_node: int) -> float:
        nodes = [
            _oracle_node("a", _ring((0.0, 0.0, 1.5), 0.0001, sensors_per_node), 2e-3),
            _oracle_node("b", _ring((40.0, 0.0, 1.5), 0.0001, sensors_per_node), 2e-3),
        ]
        far_source = np.array([20.0, 30.0, 1.5])
        covariance = position_covariance_m2(nodes, far_source, ORACLE_CONFIG)
        direction = np.array([1.0, 0.0, 0.0])
        return float(np.sqrt(direction @ covariance @ direction))

    clock_floor_m = ORACLE_CONFIG.sound_speed_mps * 2e-3 * np.sqrt(2.0) / 2.0
    assert range_std(16) > 0.9 * range_std(1)
    assert range_std(16) > 0.9 * clock_floor_m


def test_coplanar_mirror_blocks_a_3d_fix_that_height_relief_restores():
    """Four GPS-locked tetra-sized nodes all in one horizontal plane: the
    local information is sharp but the mirror across that plane fits equally
    well, so no 3-D fix is honest. Lifting nodes out of the plane restores it."""
    source = np.array([12.0, 9.0, 6.0])
    corners = [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0), (20.0, 20.0)]

    def verdict_for(heights):
        nodes, positions = [], []
        for index, ((x, y), z) in enumerate(zip(corners, heights)):
            sensors = _ring((x, y, z), 0.03, 4)
            nodes.append(_oracle_node(f"n{index}", sensors, 1e-6))
            positions.append(np.array([x, y, z]))
        return assess_observability(nodes, positions, source, ORACLE_CONFIG)

    flat = verdict_for([1.5, 1.5, 1.5, 1.5])
    assert flat.mirror_separation_sigma is not None and flat.mirror_separation_sigma < 1e-6
    # Locally every axis is resolved well inside tolerance; only the mirror blocks it.
    assert ORACLE_CONFIG.guaranteed_sigma_multiplier * flat.max_axis_std_m < flat.range_tolerance_m
    assert flat.max_honest_class == RANGE
    relief = verdict_for([1.5, 4.0, 2.5, 6.0])
    assert relief.guaranteed_class == FIX_3D


def test_aperture_limits_range_with_distance():
    """A 5 cm array never resolves range past arm's length; a 10 m spread does to 1 km."""
    for range_m in RANGES_M:
        assert EXPECTED_PASS_CLASSES[f"{int(range_m)}m-single_tetra-gps_locked-gps"][1] <= BEARING
        assert EXPECTED_PASS_CLASSES[f"{int(range_m)}m-four_node_spread-gps_locked-gps"][0] == FIX_3D


# ---- live cells ---------------------------------------------------------------------------


def _run_and_assert_cell(monkeypatch, tmp_path: Path, cell: RangeMatrixCell) -> None:
    from tests.range_matrix_live_run import run_range_matrix_cell

    guaranteed, max_honest = EXPECTED_PASS_CLASSES[cell.cell_id]
    result = run_range_matrix_cell(monkeypatch, tmp_path, cell)
    outcome = result.outcome
    print(
        f"[range-matrix] {cell.cell_id}: expected {guaranteed.name}..{max_honest.name}, "
        f"achieved {outcome.achieved_class.name}, violations {len(outcome.violations)}"
    )
    for line in outcome.detection_lines:
        print(f"[range-matrix]   {line}")
    if outcome.violations:
        raise RangeMatrixCellViolation(
            f"{cell.cell_id} ({result.detection_count} detections, {result.manifest_count} manifests):\n"
            + "\n".join(outcome.violations)
            + "\nevent detections:\n"
            + "\n".join(outcome.detection_lines)
        )


@pytest.mark.parametrize("cell_id", [_cell_param(cell_id) for cell_id in DEFAULT_DIAGONAL_CELL_IDS])
def test_range_matrix_diagonal_cell(monkeypatch, tmp_path: Path, cell_id: str) -> None:
    _run_and_assert_cell(monkeypatch, tmp_path, CELLS_BY_ID[cell_id])


@pytest.mark.skipif(os.environ.get(FULL_MATRIX_ENV) != "1", reason=f"full range matrix is opt-in ({FULL_MATRIX_ENV}=1)")
@pytest.mark.parametrize(
    "cell_id", [_cell_param(cell_id) for cell_id in CELLS_BY_ID if cell_id not in DEFAULT_DIAGONAL_CELL_IDS]
)
def test_range_matrix_full_cell(monkeypatch, tmp_path: Path, cell_id: str) -> None:
    _run_and_assert_cell(monkeypatch, tmp_path, CELLS_BY_ID[cell_id])
