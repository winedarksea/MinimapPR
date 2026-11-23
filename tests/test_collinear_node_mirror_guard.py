"""Collinear-node mirror guard (closeout B3): the geometry test, the rigid-array
exemption, and the single-mic variant of the S gate's collinear trail replayed
through a full FusionNode."""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.collinear_node_mirror_guard import fix_is_collinear_mirror_ambiguous, node_set_is_collinear
from minimappr.sim import server_track_gate as gate
from tests.sim_scenario_binary import ensure_sim_binary

MAX_ASPECT_RATIO = 0.2
TRAIL_NODES_M = {"phone-a": (0.0, 0.0, 1.5), "phone-b": (20.0, 0.3, 1.5), "phone-c": (40.0, -0.2, 1.5)}
TRIANGLE_NODES_M = {"phone-a": (0.0, 0.0, 1.5), "phone-b": (20.0, 0.0, 1.5), "phone-c": (10.0, 17.0, 1.5)}
TETRA_OFFSETS_M = ((0.05, 0.0, 0.0), (-0.025, 0.043, 0.0), (-0.025, -0.043, 0.0), (0.0, 0.0, 0.06))
SINGLE_MIC_TRAIL_SCENE = "gate_collinear_trail_single_mic"
TETRA_TRAIL_SCENE = "gate_collinear_trail"
# Truth side of the trail (y = 0) for each bird in both trail scenes.
TRUTH_NORTH_SIGN_BY_LABEL = {"Hermit Thrush": 1.0, "Red-eyed Vireo": -1.0}
# Single-mic sensors cannot feed the beamformed classifier at its default (2), so
# the replay lowers it to see the localized rows the guard acts on.
SINGLE_MIC_CLASSIFICATION_OVERRIDE = {"beamformed_classification_min_sensor_count": 1}


def _point_nodes(nodes_m: dict[str, tuple]) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    return {node: node for node in nodes_m}, {node: np.asarray(p) for node, p in nodes_m.items()}


def _with_rigid_array(nodes_m: dict[str, tuple], rigid_node: str) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    node_ids, positions = _point_nodes({n: p for n, p in nodes_m.items() if n != rigid_node})
    for index, offset in enumerate(TETRA_OFFSETS_M):
        sensor = f"{rigid_node}-m{index}"
        node_ids[sensor] = rigid_node
        positions[sensor] = np.asarray(nodes_m[rigid_node]) + np.asarray(offset)
    return node_ids, positions


def test_single_file_nodes_are_collinear_and_a_ring_is_not():
    # Same layouts as the Rust unit test (host_entity_tracking.rs).
    assert node_set_is_collinear([(0, 0, 1.5), (8, 0.3, 1.5), (16, -0.2, 1.5), (24, 0.1, 1.5)], MAX_ASPECT_RATIO)
    ring = [(15 * np.cos(a), 15 * np.sin(a), 1.4) for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    assert not node_set_is_collinear(ring, MAX_ASPECT_RATIO)
    assert node_set_is_collinear([(3.0, 4.0, 1.0)], MAX_ASPECT_RATIO)


def test_point_nodes_on_a_trail_are_mirror_ambiguous_but_a_triangle_is_not():
    assert fix_is_collinear_mirror_ambiguous(TRAIL_NODES_M, *_point_nodes(TRAIL_NODES_M), MAX_ASPECT_RATIO)
    assert not fix_is_collinear_mirror_ambiguous(TRIANGLE_NODES_M, *_point_nodes(TRIANGLE_NODES_M), MAX_ASPECT_RATIO)


def test_any_rigid_array_among_the_contributors_disables_the_guard():
    node_ids, positions = _with_rigid_array(TRAIL_NODES_M, "phone-b")
    assert not fix_is_collinear_mirror_ambiguous(TRAIL_NODES_M, node_ids, positions, MAX_ASPECT_RATIO)
    # A rigid node that did not contribute does not exempt the point nodes that did.
    assert fix_is_collinear_mirror_ambiguous(["phone-a", "phone-c"], node_ids, positions, MAX_ASPECT_RATIO)


def test_zero_aspect_ratio_disables_the_guard():
    assert not fix_is_collinear_mirror_ambiguous(TRAIL_NODES_M, *_point_nodes(TRAIL_NODES_M), 0.0)


@pytest.fixture(scope="module")
def sim_binary() -> None:
    ensure_sim_binary("collinear mirror guard")


async def _replay_rows(scene: str, tmp_path, aspect_ratio: float) -> list[dict]:
    overrides = {**SINGLE_MIC_CLASSIFICATION_OVERRIDE, "localization_collinear_mirror_max_aspect_ratio": aspect_ratio}
    row = await gate.replay_scene_configuration(scene, gate.ALL_ON, tmp_path / f"{scene}-{aspect_ratio}", extra_overrides=overrides)
    return row["raw"]["rows"]


def _placed_labelled(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["spatial_display_mode"] == "localized" and r["label"] in TRUTH_NORTH_SIGN_BY_LABEL]


@pytest.mark.asyncio
async def test_single_mic_trail_fixes_are_not_placed(sim_binary, tmp_path):
    unguarded = await _replay_rows(SINGLE_MIC_TRAIL_SCENE, tmp_path, 0.0)
    mirrored = [r for r in _placed_labelled(unguarded) if np.sign(r["position_m"][1]) != TRUTH_NORTH_SIGN_BY_LABEL[r["label"]]]
    assert mirrored, "the unguarded single-mic trail should place at least one bird at its mirror image"
    guarded = await _replay_rows(SINGLE_MIC_TRAIL_SCENE, tmp_path, MAX_ASPECT_RATIO)
    assert guarded, "the guard reports bearing rays, it does not drop detections"
    assert not [r for r in guarded if r["spatial_display_mode"] == "localized"]


@pytest.mark.asyncio
async def test_rigid_array_trail_is_unaffected_by_the_guard(sim_binary, tmp_path):
    # Judged on the guard's own per-product tag within one replay: comparing two
    # replays' positions is load-sensitive (window timing), the tag is not.
    guarded = await _replay_rows(TETRA_TRAIL_SCENE, tmp_path, MAX_ASPECT_RATIO)
    onset_rows = [r for r in guarded if "onset_cluster_collinear_mirror_ambiguous" in (r["feature_summary"] or {})]
    assert onset_rows
    assert not [r for r in onset_rows if r["feature_summary"]["onset_cluster_collinear_mirror_ambiguous"]]
    assert all(r["feature_summary"].get("localization_range_projection_mode") != "range_asymptotic" for r in onset_rows)
    placed = _placed_labelled(guarded)
    assert placed
    assert all(np.sign(r["position_m"][1]) == TRUTH_NORTH_SIGN_BY_LABEL[r["label"]] for r in placed)
