"""Server-track S gate (Field Mode W4): all server-track flags on together, on sim scenes.

Fixed-seed Rust-simulator scenes replayed through a full FusionNode with every
server-track stream enabled (``minimappr/sim/server_track_gate.py``; full
matrix and verdicts: ``scripts/server_track_gate.py``,
``docs/field_mode_progress/server_track_gate.md``). Slow (~1 min per replay on
an idle M2); the scenes are the ones where ALL-ON passes the plan's gate today,
so any regression in a stream's interaction with the others shows up here.
"""

from __future__ import annotations

import pytest

from minimappr.sim import server_track_gate as gate
from tests.sim_scenario_binary import ensure_sim_binary

# scene -> (min individual recall, max median horizontal error m) measured ALL-ON.
ALL_ON_SCENE_EXPECTATIONS = {
    "onset_same_band_pair": (1.0, 1.0),
    "onset_same_band_two_species": (1.0, 1.0),
    "gate_countersinging_pair": (1.0, 1.0),
    "gate_collinear_trail": (1.0, 1.0),
    # Field Mode S-fixes tracking: the flock no longer collapses into one entity
    # and the far bird's 391 m fix no longer starts a duplicate.
    "gate_flock_echo": (1.0, 1.0),
    "onset_faint_far_bird": (1.0, 1.0),
    # Localization core overhaul: 0.44 m measured (0.33 m before milestone 3; see
    # docs/tracking.md "Hold rule for perched tracks"). Locked in with headroom.
    "onset_overlap_two_species": (1.0, 0.6),
}
BASELINE_COMPARISON_SCENE = "onset_same_band_two_species"


@pytest.fixture(scope="module", autouse=True)
def sim_binary() -> None:
    ensure_sim_binary("server-track S gate")


@pytest.mark.asyncio
@pytest.mark.parametrize("scene", sorted(ALL_ON_SCENE_EXPECTATIONS))
async def test_all_on_scene_gate(scene, tmp_path) -> None:
    row = await gate.replay_scene_configuration(scene, gate.ALL_ON, tmp_path)
    min_recall, max_median_error_m = ALL_ON_SCENE_EXPECTATIONS[scene]
    assert row["localized_and_classified_individual_recall"] >= min_recall, row["detections_brief"]
    assert row["label_swap_count"] == 0, row["detections_brief"]
    assert row["duplicate_entities"] == 0, row["detections_brief"]
    assert row["echo_promotions"] == 0, row["detections_brief"]
    assert row["entities_spanning_multiple_individuals"] == 0, row["detections_brief"]
    assert row["median_position_error_m"] <= max_median_error_m, row["detections_brief"]


@pytest.mark.asyncio
async def test_all_on_beats_baseline(tmp_path) -> None:
    baseline = await gate.replay_scene_configuration(BASELINE_COMPARISON_SCENE, gate.BASELINE, tmp_path)
    all_on = await gate.replay_scene_configuration(BASELINE_COMPARISON_SCENE, gate.ALL_ON, tmp_path)
    assert gate.all_on_gate_failures([all_on], [baseline]) == []
