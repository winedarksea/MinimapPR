"""W2-onset-srp gate on Rust-simulator scenes, scored with the entity metrics.

Each `minimappr-sim/scenarios/onset_*.toml` scene is rendered, replayed through
a full FusionNode with `localization_onset_clustering_enabled=true` and scored.
Reporting window 1 s isolates per-call localization + classification from the
30 s ReportingFusionPolicy bucketing (W3-entity's scope). Flag-off numbers for
the same scenes are in docs/field_mode_progress/onset_srp.md
(scripts/measure_onset_srp_scenarios.py).
"""

from __future__ import annotations

import pytest

from minimappr.sim import sim_scenario_scoring as scoring
from tests.sim_scenario_binary import ensure_sim_binary

FLAG_ON = {"localization_onset_clustering_enabled": True, "reporting_window_seconds": 1.0}
FLAG_OFF = {"localization_onset_clustering_enabled": False, "reporting_window_seconds": 1.0}
MIN_RECALL = 0.75


@pytest.fixture(scope="module", autouse=True)
def sim_binary() -> None:
    ensure_sim_binary("W2-onset-srp")


async def _score(name: str, tmp_path, overrides):
    report, _ = await scoring.replay_and_score_scenario(
        scoring.SCENARIO_DIRECTORY / f"{name}.toml", tmp_path / name, settings_overrides=overrides
    )
    return report.entity_metrics


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    ["onset_overlap_two_species", "onset_trunk_echo", "onset_faint_far_bird", "onset_same_band_pair"],
)
async def test_onset_clustering_scene_gate(scenario, tmp_path) -> None:
    entity = await _score(scenario, tmp_path, FLAG_ON)
    assert entity.localized_and_classified_recall >= MIN_RECALL
    assert entity.label_swap_rate in (None, 0.0)
    assert entity.duplicate_entities == 0
    assert entity.echo_promotions == 0


@pytest.mark.asyncio
async def test_flag_on_beats_flag_off_on_overlap(tmp_path) -> None:
    off = await _score("onset_overlap_two_species", tmp_path / "off", FLAG_OFF)
    on = await _score("onset_overlap_two_species", tmp_path / "on", FLAG_ON)
    assert on.localized_and_classified_recall > off.localized_and_classified_recall
