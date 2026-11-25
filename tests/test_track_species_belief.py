"""Per-track species log-odds (Field Mode W2-tracker, S3)."""

from __future__ import annotations

import pytest

from minimappr.config import Settings
from minimappr.core.track_species_belief import (
    SpeciesBeliefConfig,
    SpeciesLogOddsBelief,
    species_evidence_from_detection,
)
from minimappr.core.tracking import TrackManager
from minimappr.sim.tracker_scenario_catalog import WILDLIFE_CONTINUOUS_MOTION

T0 = 1_700_000_000_000_000_000
SECOND = 1_000_000_000


def test_single_detection_belief_equals_its_score():
    belief = SpeciesLogOddsBelief(SpeciesBeliefConfig())
    belief.update({"robin": 0.3})
    label, probability = belief.top_label()
    assert label == "robin"
    assert probability == pytest.approx(0.3, abs=1e-6)


def test_three_weak_bouts_confirm_species():
    """Plan example: 0.30, 0.45, 0.40 from one tree confirm the species."""
    belief = SpeciesLogOddsBelief(SpeciesBeliefConfig(display_min_belief=0.5))
    belief.update({"robin": 0.30})
    assert not belief.passes_display_threshold()
    belief.update({"robin": 0.45})
    belief.update({"robin": 0.40})
    label, probability = belief.top_label()
    assert label == "robin"
    assert probability > 0.9
    assert belief.passes_display_threshold()


def test_per_update_cap_limits_one_bout():
    config = SpeciesBeliefConfig(per_update_log_odds_cap=1.0)
    belief = SpeciesLogOddsBelief(config)
    belief.update({"robin": 0.999})
    # logit(0.1) + 1.0 -> about 0.23, not 0.999.
    assert belief.probability_for("robin") < 0.25


def test_total_cap_bounds_belief_so_it_can_recover():
    config = SpeciesBeliefConfig(total_log_odds_cap=3.0)
    belief = SpeciesLogOddsBelief(config)
    for _ in range(50):
        belief.update({"robin": 0.95})
    saturated = belief.probability_for("robin")
    assert saturated < 0.75  # logit(0.1)+3 = 0.69
    for _ in range(3):
        belief.update({"robin": 0.01, "tanager": 0.9})
    assert belief.top_label()[0] == "tanager"


def test_non_evidence_labels_ignored():
    belief = SpeciesLogOddsBelief(SpeciesBeliefConfig())
    belief.update({"unknown": 0.9, "background": 0.9})
    assert belief.top_label() == (None, 0.0)


def test_nan_score_skipped_without_poisoning_label():
    belief = SpeciesLogOddsBelief(SpeciesBeliefConfig())
    belief.update({"robin": 0.3})
    belief.update({"robin": float("nan"), "tanager": float("nan")})
    assert belief.top_labels(5) == ["robin"]
    assert belief.probability_for("robin") == pytest.approx(0.3, abs=1e-6)
    belief.update({"robin": 0.45})
    assert belief.probability_for("robin") > 0.3


def test_evidence_uses_winner_member_scores_only():
    scores = {"Bird": 0.9, "Animal": 0.8, "birdnet:robin": 0.4, "birdnet:wren": 0.2}
    evidence = species_evidence_from_detection(
        label="robin", confidence=0.4, classifier_scores=scores, winner_member="birdnet", top_k=3
    )
    assert evidence == {"robin": 0.4, "wren": 0.2}
    primary = species_evidence_from_detection(
        label="Bird", confidence=0.9, classifier_scores=scores, winner_member="yamnet", top_k=1
    )
    assert primary == {"Bird": 0.9}


def _manager(**overrides) -> TrackManager:
    return TrackManager(Settings(**overrides))


async def test_flag_off_keeps_single_label_overwrite():
    manager = _manager()
    await manager.update(timestamp_ns=T0, position_m=(5, 5, 1), label="robin",
                         label_category="wildlife", confidence=0.8)
    track = await manager.update(timestamp_ns=T0 + SECOND, position_m=(5, 5, 1), label="wren",
                                 label_category="wildlife", confidence=0.3)
    assert track.label == "wren"
    assert track.species_belief is None


async def test_flag_on_belief_resists_one_wrong_bout_and_is_exposed():
    manager = _manager(tracking_species_log_odds_enabled=True)
    for index, (label, confidence) in enumerate(
        [("robin", 0.7), ("robin", 0.6), ("wren", 0.3), ("robin", 0.5)]
    ):
        track = await manager.update(timestamp_ns=T0 + index * SECOND, position_m=(5, 5, 1),
                                     label=label, label_category="wildlife", confidence=confidence)
    assert track.label == "robin"
    assert track.species_top_label == "robin"
    assert track.species_belief is not None and "wren" in track.species_belief
    assert track.confidence == pytest.approx(track.species_top_probability)


async def test_belief_only_for_configured_categories():
    manager = _manager(tracking_species_log_odds_enabled=True)
    await manager.update(timestamp_ns=T0, position_m=(5, 5, 1), label="drone",
                         label_category="vehicle", confidence=0.2)
    snapshot = await manager.snapshot()
    assert len(snapshot) == 1 and snapshot[0].species_belief is None


async def test_hidden_tentative_track_never_in_snapshot_until_belief_passes():
    manager = _manager(tracking_species_log_odds_enabled=True, tracking_species_display_min_belief=0.5)
    track = await manager.update(timestamp_ns=T0, position_m=(5, 5, 1), label="robin",
                                 label_category="wildlife", confidence=0.3, below_display_threshold=True)
    assert manager.is_track_hidden(track.id)
    assert await manager.snapshot() == []
    assert len(await manager.snapshot(include_hidden=True)) == 1
    await manager.update(timestamp_ns=T0 + 4 * SECOND, position_m=(5.5, 5, 1), label="robin",
                         label_category="wildlife", confidence=0.45, below_display_threshold=True)
    visible = await manager.snapshot()
    assert [t.id for t in visible] == [track.id]


async def test_display_level_detection_reveals_track_even_with_weak_belief():
    """Nothing that surfaces today is hidden: a display-level hit reveals the track."""
    manager = _manager(tracking_species_log_odds_enabled=True, tracking_species_display_min_belief=0.9)
    track = await manager.update(timestamp_ns=T0, position_m=(5, 5, 1), label="robin",
                                 label_category="wildlife", confidence=0.6)
    assert not manager.is_track_hidden(track.id)
    assert len(await manager.snapshot()) == 1



def test_non_evidence_labels_ignored_case_insensitively():
    """YAMNet reports "Silence"; it must never become a track's species."""
    belief = SpeciesLogOddsBelief(SpeciesBeliefConfig())
    belief.update({"Silence": 0.9, "Unknown": 0.9, "robin": 0.3})
    assert belief.top_labels(5) == ["robin"]


async def test_surfaced_track_stays_visible_after_contradicting_weak_bout():
    """Once a hidden track surfaces it was persisted and shown; it must not flicker off."""
    manager = _manager(tracking_species_log_odds_enabled=True, tracking_species_display_min_belief=0.5)
    track = None
    for index, confidence in enumerate((0.3, 0.45, 0.4)):
        track = await manager.update(timestamp_ns=T0 + index * 4 * SECOND, position_m=(5, 5, 1),
                                     label="robin", label_category="wildlife",
                                     confidence=confidence, below_display_threshold=True)
    assert not manager.is_track_hidden(track.id)
    for offset_s in (16, 20):
        await manager.update(timestamp_ns=T0 + offset_s * SECOND, position_m=(5, 5, 1), label="robin",
                             label_category="wildlife", confidence=0.001, below_display_threshold=True,
                             species_evidence={"robin": 0.001})
    assert manager.field_mode._beliefs[track.id].probability_for("robin") < 0.5
    assert not manager.is_track_hidden(track.id)
    assert [t.id for t in await manager.snapshot()] == [track.id]


async def test_hidden_track_not_counted_in_multi_node_metric():
    manager = _manager(tracking_species_log_odds_enabled=True)
    for index, node_id in enumerate(("node-a", "node-b")):
        await manager.update(timestamp_ns=T0 + index * SECOND, position_m=(5, 5, 1), label="robin",
                             label_category="wildlife", confidence=0.2, source_node_id=node_id,
                             below_display_threshold=True)
    assert len(await manager.snapshot(include_hidden=True)) == 1
    assert manager.multi_node_active_count() == 0


def test_flags_off_skips_evidence_work():
    # Stationary_hold (wildlife's default) consumes evidence too, so it is off here.
    manager = _manager(**WILDLIFE_CONTINUOUS_MOTION)
    evidence = manager.field_mode.detection_evidence(
        label="robin", confidence=0.8, classifier_scores={"robin": 0.8}, species_evidence=None
    )
    assert evidence == {}
