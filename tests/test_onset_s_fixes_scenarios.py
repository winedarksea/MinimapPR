"""Field Mode S-fixes onset: localization on rendered simulator scenes (no fusion replay).

Each `minimappr-sim/scenarios/sfix_*.toml` scene is rendered and every scripted
call's analysis span goes straight through `localize_onset_clusters`, so the
three S-gate localization defects are pinned without classifier/tracker noise:

* planar canopy bird: never reported below ground (no ground-image flip), and the
  reported height uncertainty covers the truth;
* far bird seen by a small layout: never a confident far range (bearing kept);
* loud bird 2 m from one node: localized next to that node, no far ghost.
"""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core.localization import speed_of_sound_mps
from minimappr.core.onset_srp_clustering import localize_onset_clusters
from minimappr.core.onset_srp_types import CLUSTER_STATUS_ACCEPTED
from minimappr.core.range_projection import RANGE_BEARING_PROJECTED
from minimappr.sim import sim_scenario_scoring as scoring
from tests.sim_scenario_binary import ensure_sim_binary

SPAN_BEFORE_CALL_S = 0.3
SPAN_AFTER_CALL_S = 1.3
SCENE_TEMPERATURE_C = 18.0
SCENE_HUMIDITY = 0.6
GROUND_Z_M = 0.0


@pytest.fixture(scope="module", autouse=True)
def sim_binary() -> None:
    ensure_sim_binary("S-fixes onset")


def _accepted_by_call(name: str, tmp_path) -> list[tuple[np.ndarray, list]]:
    bundle_path = scoring.render_scenario(scoring.SCENARIO_DIRECTORY / f"{name}.toml", tmp_path / f"{name}.zip", seed=0)
    bundle = scoring.load_bundle_with_echo_truth(bundle_path)
    signals, positions, node_ids = {}, {}, {}
    sample_rate_hz, start_ns = 0, 0
    for node in bundle.manifest["nodes"]:
        channels, sample_rate_hz = bundle.node_audio(node["node_id"])
        start_ns = node["audio_start_time_ns"]
        for index, sensor_id in enumerate(node["channel_sensor_ids"]):
            signals[sensor_id] = channels[index].astype(np.float64)
            positions[sensor_id] = np.asarray(node["position_m"]) + np.asarray(node["sensor_offsets_m"][index])
            node_ids[sensor_id] = node["node_id"]
    settings = Settings(localization_onset_clustering_enabled=True)
    results = []
    for event in bundle.events:
        if event.get("is_echo_source"):
            continue
        call_s = (event["start_ns"] - start_ns) / 1e9
        first = int((call_s - SPAN_BEFORE_CALL_S) * sample_rate_hz)
        last = int((call_s + SPAN_AFTER_CALL_S) * sample_rate_hz)
        result = localize_onset_clusters(
            sensor_signals={sid: samples[first:last] for sid, samples in signals.items()},
            sensor_positions=positions,
            sensor_node_ids=node_ids,
            sample_rate_hz=sample_rate_hz,
            sound_speed_mps=speed_of_sound_mps(SCENE_TEMPERATURE_C, SCENE_HUMIDITY),
            config=settings.onset_srp_clustering_config(),
            interpolation_factor=settings.gcc_phat_interp_factor,
            far_field_default_range_m=settings.localization_far_field_default_range_m,
        )
        accepted = [c for c in result.clusters if c.status == CLUSTER_STATUS_ACCEPTED]
        results.append((np.asarray(event["position_m"], dtype=np.float64), accepted))
    return results


def test_planar_canopy_bird_is_never_flipped_below_ground(tmp_path) -> None:
    for truth, accepted in _accepted_by_call("sfix_planar_canopy_bird", tmp_path):
        assert accepted
        for cluster in accepted:
            position = np.asarray(cluster.position_m)
            assert np.linalg.norm(position[:2] - truth[:2]) < 1.5
            assert position[2] >= GROUND_Z_M
            vertical_std_m = float(np.sqrt(np.asarray(cluster.covariance_m2)[2][2]))
            assert abs(position[2] - truth[2]) <= 2.5 * vertical_std_m


def test_far_bird_range_is_not_overclaimed(tmp_path) -> None:
    centroid = np.array([5.0, 3.0, 1.5])
    for truth, accepted in _accepted_by_call("sfix_far_bird_small_baseline", tmp_path):
        assert accepted
        truth_range_m = float(np.linalg.norm(truth - centroid))
        for cluster in accepted:
            offset = np.asarray(cluster.position_m) - centroid
            bearing_error = np.degrees(np.arccos(np.clip(offset @ (truth - centroid) / (np.linalg.norm(offset) * truth_range_m), -1, 1)))
            assert bearing_error < 10.0
            if cluster.range_projection_mode != RANGE_BEARING_PROJECTED:
                # A refined range must carry radial uncertainty that reaches the truth.
                radial = offset / np.linalg.norm(offset)
                radial_std_m = float(np.sqrt(radial @ np.asarray(cluster.covariance_m2) @ radial))
                assert abs(np.linalg.norm(offset) - truth_range_m) <= 2.5 * radial_std_m
            else:
                low_m, high_m = cluster.range_interval_m
                assert low_m <= truth_range_m <= high_m


def test_loud_bird_two_metres_from_a_node_localizes_near(tmp_path) -> None:
    for truth, accepted in _accepted_by_call("sfix_loud_bird_near_node", tmp_path):
        assert accepted
        errors = [float(np.linalg.norm(np.asarray(c.position_m)[:2] - truth[:2])) for c in accepted]
        # Was ~5 m plus a 125-1457 m ghost; a ~2 m radial bias along the node's ray remains.
        assert min(errors) < 3.0
        assert max(errors) < 5.0  # no ghost along the node's bearing ray
