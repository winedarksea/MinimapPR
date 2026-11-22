"""Relative-trigger gate on the faint-far-bird simulator scene (Field Mode W3-attribution).

With an absolute `trigger_rms` of 0.001 (the env default, ~20x the scene's floor)
the far pewee never opens a window; the noise-floor-relative trigger lets it in
without the hand-tuned 3x-floor threshold the W2 scene needed.
"""

from __future__ import annotations

import pytest

from minimappr.sim import sim_scenario_scoring as scoring
from tests.sim_scenario_binary import ensure_sim_binary

UNTUNED_TRIGGER_RMS = 0.001
BASE = {
    "localization_onset_clustering_enabled": True,
    "reporting_window_seconds": 1.0,
    "trigger_rms": UNTUNED_TRIGGER_RMS,
}
FAR_BIRD = "Eastern Wood-Pewee"


@pytest.fixture(scope="module", autouse=True)
def sim_binary() -> None:
    ensure_sim_binary("W3-attribution relative trigger")


async def _labels_and_recall(tmp_path, relative: bool):
    report, detections = await scoring.replay_and_score_scenario(
        scoring.SCENARIO_DIRECTORY / "onset_faint_far_bird.toml",
        tmp_path / f"relative-{relative}",
        settings_overrides={**BASE, "trigger_noise_floor_relative_enabled": relative},
    )
    return {d.get("label") for d in detections}, report.entity_metrics.localized_and_classified_recall


@pytest.mark.asyncio
async def test_relative_trigger_opens_windows_for_the_faint_far_bird(tmp_path) -> None:
    labels_off, recall_off = await _labels_and_recall(tmp_path, relative=False)
    labels_on, recall_on = await _labels_and_recall(tmp_path, relative=True)
    assert FAR_BIRD not in labels_off
    assert FAR_BIRD in labels_on
    assert recall_on >= 0.75 > recall_off
