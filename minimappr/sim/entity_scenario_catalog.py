"""Field Mode W3-entity scenarios (driven by ``tracker_detection_scenarios``).

- ``omni_to_localized_dedup``: test-strategy row "Omni-to-localized dedup" — a
  localized robin sings 5 times and 2 of the 5 bouts fail to localize (omni,
  from different nodes). Expect one entity, zero duplicates.
- ``heard_nearby_then_localized``: a tanager is first heard only (omni), then
  localized on the next two bouts. Tracker-only shows a "heard nearby" row and
  a track (a duplicate); the entity model promotes the same entity.
- ``two_species_one_omni_each``: robin and tanager both mapped, each with an
  omni bout from the node nearest to it; absorption must pick the right bird.
"""

from __future__ import annotations

from minimappr.sim.tracker_detection_scenarios import SyntheticBout, TrackerScenario
from minimappr.sim.tracker_scenario_catalog import ROBIN, ROBIN_PERCH_M, TANAGER, TANAGER_PERCH_M


def omni_to_localized_dedup() -> TrackerScenario:
    bouts = [
        SyntheticBout(0.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.6}),
        SyntheticBout(8.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.62}, mode="omni"),
        SyntheticBout(16.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.58}),
        SyntheticBout(24.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.6}, mode="omni", reporting_node_id="node-c"),
        SyntheticBout(32.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.64}),
    ]
    return TrackerScenario("omni_to_localized_dedup", {"robin-1": ROBIN}, bouts)


def heard_nearby_then_localized() -> TrackerScenario:
    bouts = [
        SyntheticBout(0.0, "tanager-1", TANAGER_PERCH_M, {TANAGER: 0.6}, mode="omni", reporting_node_id="node-b"),
        SyntheticBout(3.0, "tanager-1", TANAGER_PERCH_M, {TANAGER: 0.65}, reporting_node_id="node-b"),
        SyntheticBout(11.0, "tanager-1", TANAGER_PERCH_M, {TANAGER: 0.6}, reporting_node_id="node-b"),
    ]
    return TrackerScenario("heard_nearby_then_localized", {"tanager-1": TANAGER}, bouts)


def two_species_one_omni_each() -> TrackerScenario:
    bouts = [
        SyntheticBout(0.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.6}),
        SyntheticBout(2.0, "tanager-1", TANAGER_PERCH_M, {TANAGER: 0.6}, reporting_node_id="node-b"),
        SyntheticBout(9.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.6}, mode="omni"),
        SyntheticBout(11.0, "tanager-1", TANAGER_PERCH_M, {TANAGER: 0.6}, mode="omni", reporting_node_id="node-b"),
        SyntheticBout(18.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.6}),
        SyntheticBout(20.0, "tanager-1", TANAGER_PERCH_M, {TANAGER: 0.6}, reporting_node_id="node-b"),
    ]
    return TrackerScenario(
        "two_species_one_omni_each", {"robin-1": ROBIN, "tanager-1": TANAGER}, bouts
    )


ENTITY_SCENARIOS = (omni_to_localized_dedup, heard_nearby_then_localized, two_species_one_omni_each)
