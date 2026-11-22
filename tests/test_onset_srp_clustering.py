"""Onset-frame SRP clustering end to end on synthetic node audio, plus grid and
echo-check units (Field Mode W2-onset-srp)."""

from __future__ import annotations

import numpy as np

from minimappr.config import Settings
from minimappr.core.onset_cluster_echo_checks import (
    apply_onset_echo_and_persistence_checks,
)
from minimappr.core.onset_srp_clustering import localize_onset_clusters
from minimappr.core.onset_srp_grid import build_direction_grid, build_spatial_grid
from minimappr.core.onset_srp_types import (
    CLUSTER_STATUS_ACCEPTED,
    CLUSTER_STATUS_ECHO,
    CLUSTER_STATUS_TENTATIVE,
    FramePeak,
    OnsetCluster,
    OnsetFrame,
)
from minimappr.sim.synthetic_entity_scenarios import (
    SYNTHETIC_ENTITY_SCENARIOS,
    _sensor_positions,
    render_node_audio,
)

SAMPLE_RATE_HZ = 16_000
SOUND_SPEED_MPS = 343.2
SPAN_S = (0.7, 2.2)


def _scenario(name):
    return next(s for s in SYNTHETIC_ENTITY_SCENARIOS if s.name == name)


def _run(name: str, nodes: set[str] | None = None):
    scenario = _scenario(name)
    audio = render_node_audio(scenario, SAMPLE_RATE_HZ, 0)
    positions = _sensor_positions(scenario)
    signals, sensor_positions, node_ids = {}, {}, {}
    start, stop = (int(t * SAMPLE_RATE_HZ) for t in SPAN_S)
    for node, channels in audio.items():
        if nodes and node not in nodes:
            continue
        for index in range(channels.shape[0]):
            sensor = f"{node}:ch{index}"
            signals[sensor] = channels[index, start:stop]
            sensor_positions[sensor] = positions[node][index]
            node_ids[sensor] = node
    result = localize_onset_clusters(
        sensor_signals=signals,
        sensor_positions=sensor_positions,
        sensor_node_ids=node_ids,
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
        config=Settings().onset_srp_clustering_config(),
        interpolation_factor=4,
        far_field_default_range_m=50.0,
    )
    return scenario, result


def _accepted(result):
    return [c for c in result.clusters if c.status == CLUSTER_STATUS_ACCEPTED]


def test_multi_node_fix_lands_on_the_bird_not_the_node() -> None:
    scenario, result = _run("one_bird_two_bouts")
    accepted = _accepted(result)
    assert len(accepted) == 1
    truth = np.asarray(scenario.calls[0].position_m)
    assert np.linalg.norm(np.asarray(accepted[0].position_m) - truth) < 1.5


def test_single_tetra_gives_a_bearing_with_unobservable_range() -> None:
    scenario, result = _run("one_bird_two_bouts", nodes={"node-a"})
    primary = _accepted(result)[0]
    direction = np.asarray(primary.position_m) / np.linalg.norm(primary.position_m)
    truth = np.asarray(scenario.calls[0].position_m) / np.linalg.norm(scenario.calls[0].position_m)
    assert np.degrees(np.arccos(np.clip(direction @ truth, -1, 1))) < 10.0
    assert primary.range_projection_mode == "range_bearing_projected"


def test_overlapping_species_give_two_clusters_at_their_own_positions() -> None:
    scenario, result = _run("two_species_overlap")
    accepted = _accepted(result)
    truths = {call.species_label: np.asarray(call.position_m) for call in scenario.calls[:2]}
    for truth in truths.values():
        assert min(np.linalg.norm(np.asarray(c.position_m) - truth) for c in accepted) < 2.0
    assert len(accepted) == 2


def test_trunk_echo_is_not_accepted_as_a_bird() -> None:
    scenario, result = _run("one_bird_trunk_echo")
    echo_position = np.asarray(next(c.position_m for c in scenario.calls if c.is_echo_source))
    for cluster in _accepted(result):
        assert np.linalg.norm(np.asarray(cluster.position_m) - echo_position) > 5.0


def test_consumed_onsets_are_not_relocalized() -> None:
    _, first = _run("one_bird_two_bouts")
    scenario = _scenario("one_bird_two_bouts")
    audio = render_node_audio(scenario, SAMPLE_RATE_HZ, 0)
    positions = _sensor_positions(scenario)
    start, stop = (int(t * SAMPLE_RATE_HZ) for t in SPAN_S)
    signals = {f"{n}:ch{i}": a[i, start:stop] for n, a in audio.items() for i in range(4)}
    second = localize_onset_clusters(
        sensor_signals=signals,
        sensor_positions={f"{n}:ch{i}": positions[n][i] for n in audio for i in range(4)},
        sensor_node_ids={sid: sid.split(":")[0] for sid in signals},
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
        config=Settings().onset_srp_clustering_config(),
        interpolation_factor=4,
        far_field_default_range_m=50.0,
        consumed_onset_samples_by_node=first.consumed_onset_samples_by_node,
    )
    assert second.clusters == ()


def test_grids_cover_directions_and_rings() -> None:
    direction_grid = build_direction_grid(np.zeros(3), azimuth_step_deg=10.0, elevation_step_deg=30.0, far_range_m=50.0)
    assert np.allclose(np.linalg.norm(direction_grid.directions, axis=1), 1.0)
    spatial = build_spatial_grid(
        np.zeros(3), azimuth_step_deg=10.0, elevation_step_deg=30.0, min_range_m=1.0, max_range_m=100.0, ring_count=5
    )
    assert spatial.ranges_m.max() > 100.0 and spatial.ranges_m.min() == 0.0


def _frame(index, onset, node="n1", band=(2000.0, 3000.0)):
    return OnsetFrame(index, node, f"{node}:ch0", onset, 1.0, band[0], band[1], curves=[])


def _peak(index, primary, weight=1.0):
    return FramePeak(index, np.zeros(3), np.array([1.0, 0, 0]), 10.0, 1.0, weight, primary)


def test_echo_and_persistence_checks() -> None:
    frames = {0: _frame(0, 1000), 1: _frame(1, 1000 + 160), 2: _frame(2, 9000, band=(5000.0, 6000.0))}
    bird = OnsetCluster(peaks=[_peak(0, True, 3.0)])
    echo = OnsetCluster(peaks=[_peak(0, False), _peak(1, True)])  # led frame trails the bird by 10 ms
    lone_note = OnsetCluster(peaks=[_peak(2, True, 0.5)])
    clusters = [bird, echo, lone_note]
    kwargs = dict(
        sample_rate_hz=SAMPLE_RATE_HZ,
        echo_min_lag_seconds=0.003,
        echo_max_lag_seconds=0.03,
        echo_min_independent_fraction=0.5,
        persistence_min_independent_onsets=2,
    )
    apply_onset_echo_and_persistence_checks(clusters, frames, **kwargs)
    assert bird.status == CLUSTER_STATUS_ACCEPTED
    assert echo.status == CLUSTER_STATUS_ECHO
    assert lone_note.status == CLUSTER_STATUS_TENTATIVE
    lone_note.status, lone_note.status_reasons = CLUSTER_STATUS_ACCEPTED, []
    apply_onset_echo_and_persistence_checks([bird, lone_note], frames, apply_persistence=False, **kwargs)
    assert lone_note.status == CLUSTER_STATUS_ACCEPTED


def test_level_inconsistency_never_suppresses_the_strongest_cluster(monkeypatch) -> None:
    # Tentative clusters produce no detection downstream, so a level failure on
    # the window's strongest cluster would silently drop a real (faint) bird.
    import minimappr.core.onset_srp_clustering as clustering

    monkeypatch.setattr(clustering, "level_consistency_holds", lambda **_: False)
    _, result = _run("one_bird_two_bouts")
    strongest = result.clusters[0]
    assert strongest.status == CLUSTER_STATUS_ACCEPTED
    assert "level_inconsistent" in strongest.status_reasons
    for secondary in result.clusters[1:]:
        assert secondary.status != CLUSTER_STATUS_ACCEPTED


def test_stage_consumed_onset_memory_expires_and_drops_idle_nodes() -> None:
    from minimappr.core.onset_srp_localization_stage import OnsetSrpLocalizationStage
    from minimappr.core.onset_srp_types import OnsetClusteringResult

    config = Settings().onset_srp_clustering_config()
    stage = OnsetSrpLocalizationStage(config)
    ttl_ns = int(config.dedupe_ttl_seconds * 1e9)
    stage._remember(OnsetClusteringResult((), 1, 1, {"node-a": (100,)}), 0, SAMPLE_RATE_HZ, now_ns=0)
    assert "node-a" in stage._consumed_onset_ns_by_node
    stage._remember(OnsetClusteringResult((), 0, 0, {}), 0, SAMPLE_RATE_HZ, now_ns=ttl_ns * 2)
    assert stage._consumed_onset_ns_by_node == {}


class _SlowBuffer:
    """Buffer whose end time advances `step_ns` per poll for `advancing` sensors."""

    def __init__(self, advancing: set[str], stalled: set[str], step_ns: int) -> None:
        self.end_ns = {sensor: 0 for sensor in advancing | stalled}
        self.advancing = advancing
        self.step_ns = step_ns

    async def snapshot_state(self, sensor_ids):
        for sensor in self.advancing:
            self.end_ns[sensor] += self.step_ns
        return [
            {"sensor_id": s, "present": True, "start_time_ns": 0, "end_time_ns": self.end_ns[s]} for s in sensor_ids
        ]


def _fusion_with_buffer(buffer):
    from types import SimpleNamespace

    return SimpleNamespace(buffer=buffer, classifier_config=SimpleNamespace(stage_timeout_seconds=30.0))


async def test_span_wait_follows_slow_ingest_instead_of_a_wall_clock_deadline() -> None:
    # Ingest slower than real time (loaded replay) must not drop the window:
    # total wait far exceeds the stall budget while the buffer keeps advancing.
    import dataclasses

    from minimappr.core.onset_srp_localization_stage import OnsetSrpLocalizationStage

    config = dataclasses.replace(Settings().onset_srp_clustering_config(), buffer_wait_max_seconds=0.05)
    stage = OnsetSrpLocalizationStage(config)
    buffer = _SlowBuffer(advancing={"a", "b"}, stalled=set(), step_ns=1_000_000)
    covered = await stage._await_span(_fusion_with_buffer(buffer), ["a", "b"], 0, 30_000_000)
    assert covered == ["a", "b"]


async def test_span_wait_releases_a_stalled_sensor_after_the_budget() -> None:
    import dataclasses

    from minimappr.core.onset_srp_localization_stage import OnsetSrpLocalizationStage

    config = dataclasses.replace(Settings().onset_srp_clustering_config(), buffer_wait_max_seconds=0.05)
    stage = OnsetSrpLocalizationStage(config)
    buffer = _SlowBuffer(advancing=set(), stalled={"a"}, step_ns=0)
    buffer.end_ns["b"] = 10**12
    covered = await stage._await_span(_fusion_with_buffer(buffer), ["a", "b"], 0, 1_000_000)
    assert covered == ["b"]
