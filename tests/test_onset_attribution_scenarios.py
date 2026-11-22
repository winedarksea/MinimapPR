"""W3-attribution gate on Rust-simulator scenes (onset clustering on, 1 s reporting window).

* single tetra, two species overlapping (the W2 open issue): attribution must
  surface both birds with the right labels (W2 alone: recall 0.5, the robin
  held tentative by persistence);
* multi-node scenes: attribution must not cost recall or introduce swaps,
  duplicates or echo promotions (it exercises both the SNR-gradient and LOCO
  tiers there).
Numbers and flag-off comparisons: docs/field_mode_progress/attribution.md
(scripts/measure_attribution_scenarios.py).
"""

from __future__ import annotations

import pytest

from minimappr.sim import sim_scenario_scoring as scoring
from tests.sim_scenario_binary import ensure_sim_binary

ATTRIBUTION_ON = {
    "localization_onset_clustering_enabled": True,
    "onset_attribution_enabled": True,
    "reporting_window_seconds": 1.0,
}
MIN_RECALL = 0.75


@pytest.fixture(scope="module", autouse=True)
def sim_binary() -> None:
    ensure_sim_binary("W3-attribution")


async def _score(name: str, tmp_path, overrides):
    report, detections = await scoring.replay_and_score_scenario(
        scoring.SCENARIO_DIRECTORY / f"{name}.toml", tmp_path / name, settings_overrides=overrides
    )
    return report, detections


@pytest.mark.asyncio
async def test_single_tetra_overlap_labels_both_birds(tmp_path) -> None:
    report, detections = await _score("onset_single_tetra_overlap", tmp_path, ATTRIBUTION_ON)
    entity = report.entity_metrics
    assert entity.localized_and_classified_recall >= MIN_RECALL
    assert report.label_accuracy == 1.0
    assert entity.label_swap_rate in (None, 0.0)
    assert {"American Robin", "Scarlet Tanager"} <= {d.get("label") for d in detections}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    ["onset_overlap_two_species", "onset_same_band_pair", "onset_same_band_two_species", "onset_trunk_echo"],
)
async def test_attribution_keeps_multi_node_scene_gates(scenario, tmp_path) -> None:
    report, _ = await _score(scenario, tmp_path, ATTRIBUTION_ON)
    entity = report.entity_metrics
    assert entity.localized_and_classified_recall >= MIN_RECALL
    assert entity.label_swap_rate in (None, 0.0)
    assert entity.echo_promotions == 0
