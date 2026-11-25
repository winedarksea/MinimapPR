"""Kilometre-range localization through normal ingest (real sidecar + full app).

Two owner requirements ("localize from 1 m out to a kilometre or more"; "as
long range as possible for an array, even if the uncertainty is fairly wide"):

(a) A point network spread over ~1.5 km solves an event with every node and
    every cross-node pair, at their true multi-second lags, with no setting
    changed: the event span, coverage wait and same-event window come from
    the node geometry (core/localization_event_timing.py). The old fixed 1 s
    cap held the span to ~340 m of baseline.
(b) A small site (one tetra; a ~10 m four-tetra cluster) hearing an impulsive
    source 2 km away stores a bearing (or fix) whose stated uncertainty
    contains the truth, never a silent drop.

Cells reuse the E5 harness (tests/range_matrix_live_run.py) and the E5 scorer
against the physical observability oracle, computed for the off-matrix scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.range_matrix_cells import (
    CLOCK_GPS_LOCKED,
    LAYOUT_FOUR_NODE_SPREAD,
    LAYOUT_POINT_NETWORK_KM,
    LAYOUT_SINGLE_TETRA,
    POSITION_GPS,
    SOURCE_BAND_HZ,
    RangeMatrixCell,
    realize_cell,
)
from tests.range_observability_oracle import PassClass
from tests.realistic_capture_synthesis import SourceBurstSpec

# Inside the network, so every node hears it and range is resolvable.
POINT_NETWORK_SOURCE_RANGE_M = 300.0
POINT_NETWORK_NODE_COUNT = 5
FAR_SOURCE_RANGE_M = 2_000.0
# The buffers must already hold a km network's ~4.4 s cross-node lag of
# history before the trigger (the span reaches that far back).
POINT_NETWORK_PRE_ROLL_SECONDS = 6.0
# Past the noise-floor-relative trigger's warm-up, so a faint impulse can be
# admitted relative to the floor rather than only by the absolute gate.
FAR_IMPULSE_PRE_ROLL_SECONDS = 3.0
# A km network's solve reads ~9 s of 48 kHz audio per channel: seconds of
# 4x-interpolated GCC-PHAT on the debug-build sidecar.
POINT_NETWORK_DETECTION_WAIT_SECONDS = 150.0
# A muzzle blast's broadband crack: tens of milliseconds.
IMPULSIVE_BURST = SourceBurstSpec(duration_seconds=0.03, band_hz=SOURCE_BAND_HZ, edge_ramp_seconds=0.002, seed=20261125)
# A gunshot 2 km out: ~150 dB SPL at 1 m, less 66 dB spreading and ~10 dB air
# absorption, peaks ~30 dB over suburban ambient in band. (At 20 dB a 30 ms
# impulse is diluted below the 8 dB relative trigger by the 100 ms upload's
# RMS, an admission limiter recorded in TODO.md.)
FAR_IMPULSE_NEAREST_SENSOR_SNR_DB = 30.0
# Every cross-node pair at its geometric lag (GCC-PHAT, 4x interpolation at 48 kHz).
CROSS_NODE_LAG_TOLERANCE_SECONDS = 2.0e-4
POINT_NETWORK_HORIZONTAL_TOLERANCE_M = 5.0


def _run_cell(monkeypatch, tmp_path: Path, cell: RangeMatrixCell, **run_options):
    from tests.range_matrix_live_run import run_range_matrix_cell

    verdict = realize_cell(cell).oracle_verdict()
    result = run_range_matrix_cell(monkeypatch, tmp_path, cell, **run_options)
    print(
        f"[km-range] {cell.cell_id}: oracle {verdict.guaranteed_class.name}..{verdict.max_honest_class.name}, "
        f"achieved {result.outcome.achieved_class.name}, violations {len(result.outcome.violations)}"
    )
    for line in result.outcome.detection_lines + result.outcome.violations:
        print(f"[km-range]   {line}")
    return result, verdict


def _event_solves(manifests: dict[str, dict]) -> list[dict]:
    return [
        manifest
        for manifest in manifests.values()
        if manifest.get("manifest_type") == "localization_result"
        and ((manifest.get("localization") or {}).get("solve_provenance") or {}).get("attempt_kind", "").startswith("event_trigger")
    ]


@pytest.fixture(scope="module")
def point_network_run(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    try:
        cell = RangeMatrixCell(POINT_NETWORK_SOURCE_RANGE_M, LAYOUT_POINT_NETWORK_KM, CLOCK_GPS_LOCKED, POSITION_GPS)
        result, verdict = _run_cell(
            monkeypatch,
            tmp_path_factory.mktemp("km_point_network"),
            cell,
            pre_roll_seconds=POINT_NETWORK_PRE_ROLL_SECONDS,
            detection_wait_seconds=POINT_NETWORK_DETECTION_WAIT_SECONDS,
            replay_manifests_over_sse=True,
        )
        yield cell, result, verdict
    finally:
        monkeypatch.undo()


def test_kilometre_point_network_solves_with_every_cross_node_pair(point_network_run) -> None:
    cell, result, _verdict = point_network_run
    realized = realize_cell(cell)
    solves = [
        solve
        for solve in _event_solves(result.manifests_by_id)
        if len(solve["localization"]["solve_provenance"]["active_sensor_indices"]) == POINT_NETWORK_NODE_COUNT
    ]
    assert solves, "no event solve used every node"
    solve = solves[0]
    localization = solve["localization"]
    provenance = localization["solve_provenance"]
    positions = {entry["sensor_id"]: np.asarray(entry["position_m"], dtype=np.float64) for entry in provenance["sensors"]}
    sound_speed_mps = float(localization["sound_speed_mps"])
    max_cross_node_lag_seconds = max(
        float(np.linalg.norm(positions[a] - positions[b])) for a in positions for b in positions
    ) / sound_speed_mps
    assert max_cross_node_lag_seconds > 4.0
    # The span reaches the whole lag either side of the trigger.
    assert provenance["window_seconds"] == pytest.approx(2.0 * max_cross_node_lag_seconds, rel=1e-6)
    pairs = localization["pair_tdoas"]
    expected_pair_count = POINT_NETWORK_NODE_COUNT * (POINT_NETWORK_NODE_COUNT - 1) // 2
    assert len(pairs) == expected_pair_count and all(pair["is_cross_node"] for pair in pairs)
    truth_m = np.asarray(realized.scene.source_position_enu_m, dtype=np.float64)
    sensor_ids = [entry["sensor_id"] for entry in provenance["sensors"]]
    for pair in pairs:
        position_a, position_b = positions[sensor_ids[pair["ch_a"]]], positions[sensor_ids[pair["ch_b"]]]
        expected_lag_seconds = (np.linalg.norm(truth_m - position_a) - np.linalg.norm(truth_m - position_b)) / sound_speed_mps
        assert pair["lag_seconds"] == pytest.approx(expected_lag_seconds, abs=CROSS_NODE_LAG_TOLERANCE_SECONDS)
    # The solve's fix is on the source horizontally (the stored detection is
    # judged by the oracle test below).
    horizontal_error_m = float(np.linalg.norm(np.asarray(localization["position_m"][:2]) - truth_m[:2]))
    print(f"[km-range] event solve fix {localization['position_m']} vs truth {truth_m.tolist()}, mode {localization.get('range_projection_mode')}")
    assert horizontal_error_m < POINT_NETWORK_HORIZONTAL_TOLERANCE_M


@pytest.mark.xfail(
    strict=True,
    reason=(
        "open after loc/km-range: the 1.5 km point network's cross-node pairs are exact and the fix is right "
        "horizontally (<1 m), but the near-coplanar layout (25 m of relief over 1.5 km) solves 74 m low in z "
        "with a stated z std of 0.4 m (Mahalanobis^2 ~3e4). Its pair lags are 0.25 % off against the sidecar's "
        "fixed 343.2 m/s (the scene's 20 C air is 344.04 m/s; TODO 'Far-source limiters'), which the near-coplanar "
        "geometry absorbs in z; "
        "in 1 of 2 runs on the debug-build sidecar no stored render covered the event at all (the ~9 s solves' "
        "4x-interpolated GCC-PHAT saturate the compute pool)"
    ),
)
def test_kilometre_point_network_meets_the_oracle(point_network_run) -> None:
    _cell, result, verdict = point_network_run
    assert not result.outcome.violations, "\n".join(result.outcome.violations)
    assert result.outcome.achieved_class >= verdict.guaranteed_class


_FOUR_NODE_FAR_ELEVATION_OVERCONFIDENT = (
    "open after loc/km-range (same gap as E5 1000m-four_node_spread-gps_locked-gps): the ~10 m cluster stores "
    "a range_refined fix 1.3 km short with weak range observability (a bearing claim) whose elevation is 0.77 deg "
    "off against a stated angular std of 0.01 deg; the 2 km impulse is admitted, solved and stored, not dropped"
)


@pytest.mark.parametrize(
    "layout",
    [
        LAYOUT_SINGLE_TETRA,
        pytest.param(
            LAYOUT_FOUR_NODE_SPREAD,
            marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=_FOUR_NODE_FAR_ELEVATION_OVERCONFIDENT),
        ),
    ],
)
def test_small_site_hears_a_far_impulse_as_an_honest_bearing(monkeypatch, tmp_path: Path, layout: str) -> None:
    cell = RangeMatrixCell(FAR_SOURCE_RANGE_M, layout, CLOCK_GPS_LOCKED, POSITION_GPS)
    result, verdict = _run_cell(
        monkeypatch,
        tmp_path,
        cell,
        source_burst=IMPULSIVE_BURST,
        nearest_sensor_in_band_snr_db=FAR_IMPULSE_NEAREST_SENSOR_SNR_DB,
        pre_roll_seconds=FAR_IMPULSE_PRE_ROLL_SECONDS,
    )
    outcome = result.outcome
    assert not outcome.violations, "\n".join(outcome.violations + outcome.detection_lines)
    assert outcome.achieved_class >= min(verdict.guaranteed_class, PassClass.BEARING_CORRECT)
