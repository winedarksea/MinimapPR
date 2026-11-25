"""P7: a point network classifies the clearest node's aligned clip, end to end.

Rust path: point-only cluster renders through ``FusionNode.ingest_localized_render``
to the stored detection, with the per-sensor audio already mirrored into the
fusion buffer (as ``api/stream_consumer.py`` does from ``raw_audio_frame``).
The clip classification runs on a classification worker, so detections are
read after the stage queues drain.
Python-ingest path: ``_localize_candidate`` (with a truth localizer, since the
solve itself is not under test) then ``_classify_and_assemble``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from minimappr.core.fusion_node import FusionNode
from minimappr.core.point_network_call_classification import (
    POINT_NETWORK_CLIP_CLASSIFICATION_PATH,
    POINT_NETWORK_CLIP_FEATURE_KEY,
    SELECTION_REASON_HIGHEST_IN_BAND_SNR,
    SPECIES_UNCERTAIN_LABEL,
    UNAVAILABLE_REASON_NO_BUFFERED_CLIP,
    UNCERTAINTY_REASON_OVERLAPPING_CALLER,
)
from minimappr.core.point_network_call_requests import register_python_ingest_call
from minimappr.storage.db import Storage
from tests.point_network_fusion_harness import (
    RENDER_HALF_SPAN_SECONDS,
    RecordingClassifier,
    TruthLocalizer,
    point_network_render_request,
    python_ingest_candidate,
    register_point_nodes,
    started_point_network_fusion,
)
from tests.point_network_fusion_harness import stored_detections_by_solve as _stored_detections_by_solve
from tests.point_network_scene import (
    NANOSECONDS_PER_SECOND,
    POINT_NODE_POSITIONS_M,
    SAMPLE_RATE_HZ,
    SOUND_SPEED_MPS,
    Caller,
    append_captures,
    clip_onset_sample,
    point_sensor_id,
    render_point_node_captures,
)

HEURISTIC_BIRD_LABEL = "bird_like"
# Caller A sits by pt-a; pt-d (farthest) reports, so pt-a's clip is moved earlier.
CALLER_NEAR_A = Caller(position_m=(8.0, 6.0, 3.0), emission_offset_seconds=0.5, seed=11)
# Caller B sits by pt-c, 34 m from A, loud enough to be within a few dB of A at pt-a,
# and calls 50 ms after A reaches pt-a, so the two overlap in pt-a's clip.
CALLER_B_POSITION_M = (5.0, 40.0, 3.0)
CALLER_B_AMPLITUDE_SCALE = 3.0
ALIGNMENT_TOLERANCE_SAMPLES = 1


def _caller_b_overlapping_a_at_pt_a(caller_a: Caller = CALLER_NEAR_A) -> Caller:
    sound_speed_mps = caller_a.sound_speed_mps
    caller_b = Caller(position_m=CALLER_B_POSITION_M, emission_offset_seconds=0.0)
    arrival_a_at_pt_a_seconds = caller_a.emission_offset_seconds + caller_a.distance_m("pt-a") / sound_speed_mps
    return Caller(
        position_m=CALLER_B_POSITION_M,
        emission_offset_seconds=arrival_a_at_pt_a_seconds + 0.05 - caller_b.distance_m("pt-a") / sound_speed_mps,
        amplitude_scale=CALLER_B_AMPLITUDE_SCALE,
        seed=29,
        sound_speed_mps=sound_speed_mps,
    )


_render_request = point_network_render_request


async def test_rust_point_network_render_classifies_the_clearest_nodes_aligned_clip(tmp_path: Path) -> None:
    fusion, storage, classifier = await started_point_network_fusion(tmp_path)
    try:
        await append_captures(fusion.buffer, render_point_node_captures([CALLER_NEAR_A]))
        await fusion.ingest_localized_render(_render_request(fusion, CALLER_NEAR_A, reporting_node_id="pt-d", solve_id="solve-a"))

        detection = (await _stored_detections_by_solve(storage, fusion))["solve-a"]
        clip = detection["feature_summary"][POINT_NETWORK_CLIP_FEATURE_KEY]
        assert detection["label"] == HEURISTIC_BIRD_LABEL
        assert detection["feature_summary"]["classification_path"] == POINT_NETWORK_CLIP_CLASSIFICATION_PATH
        # Clearest node, why, and the runner-up.
        assert clip["node_id"] == "pt-a" and clip["sensor_id"] == point_sensor_id("pt-a")
        assert clip["selection_reason"] == SELECTION_REASON_HIGHEST_IN_BAND_SNR
        assert clip["runner_up_node_id"] == "pt-b"
        assert clip["in_band_snr_db"] > clip["runner_up_in_band_snr_db"]
        assert set(clip["in_band_snr_db_by_node"]) == set(POINT_NODE_POSITIONS_M)
        # Aligned on the solve's arrival time at pt-a, relative to the reporter.
        expected_offset_seconds = (CALLER_NEAR_A.distance_m("pt-a") - CALLER_NEAR_A.distance_m("pt-d")) / SOUND_SPEED_MPS
        assert clip["alignment_reference_sensor_id"] == point_sensor_id("pt-d")
        assert clip["alignment_offset_seconds"] == pytest.approx(expected_offset_seconds, abs=1.0 / SAMPLE_RATE_HZ)
        # The classifier heard pt-a's call exactly where the reporter's render held it.
        classified_clip = classifier.inputs[-1]
        assert classified_clip.size == int(round(2 * RENDER_HALF_SPAN_SECONDS * SAMPLE_RATE_HZ))
        expected_onset_sample = int(round(RENDER_HALF_SPAN_SECONDS * SAMPLE_RATE_HZ))
        assert abs(clip_onset_sample(classified_clip, CALLER_NEAR_A) - expected_onset_sample) <= ALIGNMENT_TOLERANCE_SAMPLES
        # A single caller keeps its species.
        assert clip["species_uncertain"] is False
        assert clip["overlapping_callers"] == []
        assert fusion._metrics.point_network_clip_classified_count == 1
    finally:
        await fusion.stop()
        await storage.close()


async def test_overlapping_callers_leave_the_masked_callers_species_uncertain(tmp_path: Path) -> None:
    caller_b = _caller_b_overlapping_a_at_pt_a()
    fusion, storage, _ = await started_point_network_fusion(tmp_path)
    try:
        await append_captures(fusion.buffer, render_point_node_captures([CALLER_NEAR_A, caller_b]))
        request_a = _render_request(fusion, CALLER_NEAR_A, reporting_node_id="pt-d", solve_id="solve-a")
        request_b = _render_request(fusion, caller_b, reporting_node_id="pt-c", solve_id="solve-b")
        # The sidecar's event-only solves usually precede the renders.
        await fusion.observe_event_localization(request_a)
        await fusion.observe_event_localization(request_b)
        await fusion.ingest_localized_render(request_a)
        await fusion.ingest_localized_render(request_b)

        detections = await _stored_detections_by_solve(storage, fusion)
        clip_a = detections["solve-a"]["feature_summary"][POINT_NETWORK_CLIP_FEATURE_KEY]
        clip_b = detections["solve-b"]["feature_summary"][POINT_NETWORK_CLIP_FEATURE_KEY]
        # A's clearest node (pt-a) also hears B at a comparable level: no species is asserted.
        assert clip_a["node_id"] == "pt-a"
        assert detections["solve-a"]["label"] == SPECIES_UNCERTAIN_LABEL
        assert detections["solve-a"]["label_confidence"] == 0.0
        assert clip_a["species_uncertain"] is True
        assert clip_a["uncertainty_reason"] == UNCERTAINTY_REASON_OVERLAPPING_CALLER
        assert clip_a["unattributed_label"] == HEURISTIC_BIRD_LABEL
        assert [caller["call_id"] for caller in clip_a["overlapping_callers"]] == ["solve-b"]
        # B dominates its own clearest node (pt-c), where A is ~30 dB down: B keeps its species.
        assert clip_b["node_id"] == "pt-c"
        assert clip_b["species_uncertain"] is False
        assert detections["solve-b"]["label"] == HEURISTIC_BIRD_LABEL
        assert fusion._metrics.point_network_clip_species_uncertain_count == 1
    finally:
        await fusion.stop()
        await storage.close()


async def test_a_later_render_alone_still_sees_an_earlier_overlapping_caller(tmp_path: Path) -> None:
    caller_b = _caller_b_overlapping_a_at_pt_a()
    fusion, storage, _ = await started_point_network_fusion(tmp_path)
    try:
        await append_captures(fusion.buffer, render_point_node_captures([CALLER_NEAR_A, caller_b]))
        await fusion.ingest_localized_render(_render_request(fusion, caller_b, reporting_node_id="pt-c", solve_id="solve-b"))
        await fusion.ingest_localized_render(_render_request(fusion, CALLER_NEAR_A, reporting_node_id="pt-d", solve_id="solve-a"))

        detections = await _stored_detections_by_solve(storage, fusion)
        assert detections["solve-a"]["label"] == SPECIES_UNCERTAIN_LABEL
        assert detections["solve-b"]["label"] == HEURISTIC_BIRD_LABEL
    finally:
        await fusion.stop()
        await storage.close()


async def test_point_network_clip_off_keeps_the_render_classification(tmp_path: Path) -> None:
    fusion, storage, _ = await started_point_network_fusion(tmp_path, point_network_clip_classification_enabled=False)
    try:
        await append_captures(fusion.buffer, render_point_node_captures([CALLER_NEAR_A]))
        await fusion.ingest_localized_render(_render_request(fusion, CALLER_NEAR_A, reporting_node_id="pt-d", solve_id="solve-a"))

        detection = (await _stored_detections_by_solve(storage, fusion))["solve-a"]
        assert POINT_NETWORK_CLIP_FEATURE_KEY not in detection["feature_summary"]
        assert detection["feature_summary"]["classification_path"] == "omni"
    finally:
        await fusion.stop()
        await storage.close()


async def test_unbuffered_point_network_keeps_the_render_and_says_why(tmp_path: Path) -> None:
    fusion, storage, _ = await started_point_network_fusion(tmp_path)
    try:
        await fusion.ingest_localized_render(_render_request(fusion, CALLER_NEAR_A, reporting_node_id="pt-d", solve_id="solve-a"))

        detection = (await _stored_detections_by_solve(storage, fusion))["solve-a"]
        assert detection["feature_summary"]["classification_path"] == "omni"
        assert detection["feature_summary"][POINT_NETWORK_CLIP_FEATURE_KEY] == {
            "unavailable_reason": UNAVAILABLE_REASON_NO_BUFFERED_CLIP,
            "candidate_node_ids": sorted(POINT_NODE_POSITIONS_M),
        }
        assert fusion._metrics.point_network_clip_unavailable_count == 1
    finally:
        await fusion.stop()
        await storage.close()


# -- Python ingest -------------------------------------------------------------------

PYTHON_LOCALIZATION_WINDOW_SECONDS = 0.4
# The trigger fires a little into the call at the triggering node.
PYTHON_TRIGGER_DELAY_SECONDS = 0.05


def _python_candidate(caller: Caller, *, candidate_id: str, triggering_node_id: str):
    return python_ingest_candidate(
        candidate_id=candidate_id,
        triggering_node_id=triggering_node_id,
        event_time_ns=caller.arrival_time_ns(triggering_node_id) + int(PYTHON_TRIGGER_DELAY_SECONDS * NANOSECONDS_PER_SECOND),
        sample_rate_hz=SAMPLE_RATE_HZ,
    )


async def _python_fusion(tmp_path: Path) -> tuple[FusionNode, Storage, RecordingClassifier, Caller]:
    fusion, storage, classifier = await started_point_network_fusion(
        tmp_path,
        localizer=TruthLocalizer(CALLER_NEAR_A.position_m),
        localization_window_seconds=PYTHON_LOCALIZATION_WINDOW_SECONDS,
        classification_window_seconds=PYTHON_LOCALIZATION_WINDOW_SECONDS,
        min_sensors_for_3d=4,
        min_sensors_for_2d=3,
        # Every point node joins the solve (full 3-D tier), as in a live point network.
        trigger_rms=1e-6,
    )
    await register_point_nodes(fusion, storage)
    caller_a = replace(CALLER_NEAR_A, sound_speed_mps=fusion.environment_provider.get_speed_of_sound())
    return fusion, storage, classifier, caller_a


async def test_python_ingest_point_network_classifies_the_clearest_nodes_aligned_clip(tmp_path: Path) -> None:
    fusion, storage, classifier, caller_a = await _python_fusion(tmp_path)
    try:
        await append_captures(fusion.buffer, render_point_node_captures([caller_a]))
        product = await fusion._localize_candidate(_python_candidate(caller_a, candidate_id="evt-a", triggering_node_id="pt-d"))
        assert product is not None and product.localization_branch is not None
        [detection_product] = await fusion._classify_and_assemble(product)

        features = detection_product.detection.feature_summary
        clip = features[POINT_NETWORK_CLIP_FEATURE_KEY]
        assert detection_product.detection.label == HEURISTIC_BIRD_LABEL
        assert features["classification_path"] == POINT_NETWORK_CLIP_CLASSIFICATION_PATH
        assert clip["node_id"] == "pt-a" and clip["runner_up_node_id"] == "pt-b"
        assert clip["alignment_reference_sensor_id"] == point_sensor_id("pt-d")
        expected_offset_seconds = (caller_a.distance_m("pt-a") - caller_a.distance_m("pt-d")) / caller_a.sound_speed_mps
        assert clip["alignment_offset_seconds"] == pytest.approx(expected_offset_seconds, abs=1.0 / SAMPLE_RATE_HZ)
        # Centred window: the call starts half a window minus the trigger delay into the clip.
        expected_onset_sample = int(round((PYTHON_LOCALIZATION_WINDOW_SECONDS / 2 - PYTHON_TRIGGER_DELAY_SECONDS) * SAMPLE_RATE_HZ))
        assert abs(clip_onset_sample(classifier.inputs[-1], caller_a) - expected_onset_sample) <= ALIGNMENT_TOLERANCE_SAMPLES
        assert clip["species_uncertain"] is False
    finally:
        await fusion.stop()
        await storage.close()


async def test_python_ingest_overlapping_caller_leaves_the_species_uncertain(tmp_path: Path) -> None:
    fusion, storage, _, caller_a = await _python_fusion(tmp_path)
    caller_b = _caller_b_overlapping_a_at_pt_a(caller_a)
    try:
        await append_captures(fusion.buffer, render_point_node_captures([caller_a, caller_b]))
        product_a = await fusion._localize_candidate(_python_candidate(caller_a, candidate_id="evt-a", triggering_node_id="pt-d"))
        fusion.localizer = TruthLocalizer(caller_b.position_m)
        product_b = await fusion._localize_candidate(_python_candidate(caller_b, candidate_id="evt-b", triggering_node_id="pt-c"))
        # As the localization worker does, before either reaches classification.
        await register_python_ingest_call(fusion, product_a)
        await register_python_ingest_call(fusion, product_b)
        [detection_a] = await fusion._classify_and_assemble(product_a)
        [detection_b] = await fusion._classify_and_assemble(product_b)

        clip_a = detection_a.detection.feature_summary[POINT_NETWORK_CLIP_FEATURE_KEY]
        assert detection_a.detection.label == SPECIES_UNCERTAIN_LABEL
        assert clip_a["species_uncertain"] is True
        assert [caller["call_id"] for caller in clip_a["overlapping_callers"]] == ["evt-b"]
        assert detection_b.detection.label == HEURISTIC_BIRD_LABEL
    finally:
        await fusion.stop()
        await storage.close()
