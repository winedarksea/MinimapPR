"""P7 milestone-4 review fixes: point-network clip classification edge cases.

- a node that lost packets (zero-filled gaps) is measured over the frames it
  received, so it cannot win "clearest node" on a -200 dB noise floor;
- two reporters' solves of one long-range call (tens of metres apart, within
  their covariances) are one caller, not overlapping callers; two real callers
  still leave the species uncertain;
- a failed clip classifier keeps the sidecar's label;
- the clip classification runs on a classification worker, not inline on the
  SSE consumer, and that worker releases the held fix;
- the call registry survives a far-future span;
- the Python-ingest alignment reference is the triggering node even when the
  solve did not select it;
- a held fix is released when registering the call fails.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import minimappr.core.fusion_node as fusion_node_module
from minimappr.classifiers.base import ClassificationResult
from minimappr.core.point_network_call_classification import (
    POINT_NETWORK_CLIP_CLASSIFICATION_PATH,
    POINT_NETWORK_CLIP_FEATURE_KEY,
    SPECIES_UNCERTAIN_LABEL,
    UNAVAILABLE_REASON_CLIP_CLASSIFIER_FAILED,
    PointNetworkCallRequest,
)
from minimappr.core.point_network_call_requests import localized_call_for_python_ingest_product
from minimappr.core.point_network_localized_calls import LocalizedCall, LocalizedCallRegistry, seconds_to_nanoseconds
from tests.point_network_fusion_harness import (
    RENDER_HALF_SPAN_SECONDS,
    SOLVE_WINDOW_SECONDS,
    TruthLocalizer,
    point_network_render_request,
    python_ingest_candidate,
    register_point_nodes,
    started_point_network_fusion,
    stored_detections_by_solve,
)
from tests.point_network_scene import (
    CAPTURE_SECONDS,
    CAPTURE_START_NS,
    NANOSECONDS_PER_SECOND,
    POINT_NETWORK_CLUSTER_ID,
    POINT_NODE_POSITIONS_M,
    SAMPLE_RATE_HZ,
    SOUND_SPEED_MPS,
    Caller,
    append_captures,
    point_sensor_id,
    render_point_node_captures,
)

HEURISTIC_BIRD_LABEL = "bird_like"
CALLER_NEAR_A = Caller(position_m=(8.0, 6.0, 3.0), emission_offset_seconds=0.5, seed=11)
RENDER_SOLVE_ID = "solve-a"
RENDER_CANDIDATE_ID = f"rust-render-{RENDER_SOLVE_ID}"


def _render_near_a(fusion):
    return point_network_render_request(fusion, CALLER_NEAR_A, reporting_node_id="pt-d", solve_id=RENDER_SOLVE_ID)


def _record_held_fix_releases(fusion) -> list[str]:
    released_candidate_ids: list[str] = []
    original_release = fusion._release_held_position_without_classification

    async def recording_release(product) -> None:
        released_candidate_ids.append(product.candidate.id)
        await original_release(product)

    fusion._release_held_position_without_classification = recording_release
    return released_candidate_ids


# -- Finding 1: a lossy node cannot win on a zero-filled noise floor ----------------

# 50 ms packets, 3 in every 10 lost: Pico nodes show 30-45 % loss.
PACKET_SECONDS = 0.05
LOST_PACKET_INDICES_MOD_10 = {1, 4, 7}
LOST_PACKET_FRACTION = len(LOST_PACKET_INDICES_MOD_10) / 10
# A real in-band SNR for this scene is tens of dB; a -200 dB floor gives ~170.
PLAUSIBLE_MAX_SNR_DB = 60.0


async def _append_with_lost_packets(buffer, sensor_id: str, samples: np.ndarray) -> None:
    packet_samples = int(round(PACKET_SECONDS * SAMPLE_RATE_HZ))
    for packet_index, start in enumerate(range(0, samples.size, packet_samples)):
        if packet_index % 10 in LOST_PACKET_INDICES_MOD_10:
            continue
        start_time_ns = CAPTURE_START_NS + seconds_to_nanoseconds(start / SAMPLE_RATE_HZ)
        await buffer.append(
            sensor_id=sensor_id,
            sample_rate_hz=SAMPLE_RATE_HZ,
            start_time_ns=start_time_ns,
            samples=samples[start : start + packet_samples],
        )


async def test_a_lossy_quieter_node_does_not_win_the_clearest_node(tmp_path: Path) -> None:
    fusion, storage, _ = await started_point_network_fusion(tmp_path)
    try:
        captures = render_point_node_captures([CALLER_NEAR_A])
        # pt-b is ~14 dB quieter than pt-a for this caller (the clean runner-up).
        lossy_sensor_id = point_sensor_id("pt-b")
        lossy_samples = captures.pop(lossy_sensor_id)
        await append_captures(fusion.buffer, captures)
        await _append_with_lost_packets(fusion.buffer, lossy_sensor_id, lossy_samples)
        await fusion.ingest_localized_render(_render_near_a(fusion))

        clip = (await stored_detections_by_solve(storage, fusion))[RENDER_SOLVE_ID]["feature_summary"][
            POINT_NETWORK_CLIP_FEATURE_KEY
        ]
        assert clip["node_id"] == "pt-a"
        assert clip["covered_frame_ratio_by_node"]["pt-a"] == 1.0
        assert 0.5 <= clip["covered_frame_ratio_by_node"]["pt-b"] <= 1.0 - LOST_PACKET_FRACTION + 0.05
        lossy_snr_db = clip["in_band_snr_db_by_node"]["pt-b"]
        assert lossy_snr_db < clip["in_band_snr_db_by_node"]["pt-a"]
        assert lossy_snr_db < PLAUSIBLE_MAX_SNR_DB
    finally:
        await fusion.stop()
        await storage.close()


# -- Finding 2: one long-range call solved by two reporters is one caller -----------

# Loud enough at ~400 m for the heuristic classifier's clip RMS gate (it labels quieter clips ambient).
LONG_RANGE_CALLER = Caller(position_m=(330.0, 250.0, 3.0), emission_offset_seconds=0.1, amplitude_scale=60.0, seed=11)
# ~580 m from the first caller, at a similar range from every node, calling 30 ms later.
SECOND_LONG_RANGE_CALLER = Caller(position_m=(-250.0, 330.0, 3.0), emission_offset_seconds=0.13, amplitude_scale=60.0, seed=29)
# The two reporters' solves of one call land 40 m apart; each states sigma = 30 m per axis.
REPORTER_SOLVE_OFFSET_M = np.array([20.0, 0.0, 0.0])
LONG_RANGE_POSITION_COVARIANCE_M2 = np.diag([30.0**2, 30.0**2, 30.0**2])
CLIP_SECONDS = 2 * RENDER_HALF_SPAN_SECONDS


def _reporter_solve(caller: Caller, *, call_id: str, reporting_node_id: str, position_offset_m, covariance_m2) -> LocalizedCall:
    window_center_ns = caller.arrival_time_ns(reporting_node_id)
    half_window_ns = seconds_to_nanoseconds(SOLVE_WINDOW_SECONDS / 2)
    return LocalizedCall(
        call_id=call_id,
        source_position_m=np.asarray(caller.position_m, dtype=np.float64) + position_offset_m,
        alignment_reference_sensor_id=point_sensor_id(reporting_node_id),
        alignment_reference_position_m=np.asarray(POINT_NODE_POSITIONS_M[reporting_node_id], dtype=np.float64),
        event_span_start_ns=window_center_ns - half_window_ns,
        event_span_end_ns=window_center_ns + half_window_ns,
        sound_speed_mps=SOUND_SPEED_MPS,
        position_covariance_m2=covariance_m2,
        cluster_id=POINT_NETWORK_CLUSTER_ID,
    )


def _clip_request(call: LocalizedCall) -> PointNetworkCallRequest:
    window_center_ns = (call.event_span_start_ns + call.event_span_end_ns) // 2
    return PointNetworkCallRequest(
        call=call,
        sensor_node_ids={point_sensor_id(node_id): node_id for node_id in POINT_NODE_POSITIONS_M},
        sensor_positions_m={
            point_sensor_id(node_id): np.asarray(position_m, dtype=np.float64) for node_id, position_m in POINT_NODE_POSITIONS_M.items()
        },
        base_clip_start_ns=window_center_ns - seconds_to_nanoseconds(RENDER_HALF_SPAN_SECONDS),
        clip_seconds=CLIP_SECONDS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        event_time_ns=window_center_ns,
    )


async def _clip_must_be_buffered():
    raise AssertionError("the clip was buffered: the default classification must not run")


async def _classify_long_range(tmp_path: Path, callers: list[Caller], solves: list[LocalizedCall]):
    fusion, storage, _ = await started_point_network_fusion(tmp_path)
    try:
        assert CAPTURE_SECONDS >= 2.0
        await append_captures(fusion.buffer, render_point_node_captures(callers))
        classifier = fusion._point_network_call_classifier
        for other_solve in solves[1:]:
            classifier.register_localized_call(other_solve)
        classified = await classifier.classify(_clip_request(solves[0]), default_classification=_clip_must_be_buffered)
        return classified.classification
    finally:
        await fusion.stop()
        await storage.close()


def _two_reporter_solves(covariance_m2) -> list[LocalizedCall]:
    return [
        _reporter_solve(LONG_RANGE_CALLER, call_id="solve-by-pt-d", reporting_node_id="pt-d",
                        position_offset_m=REPORTER_SOLVE_OFFSET_M, covariance_m2=covariance_m2),
        _reporter_solve(LONG_RANGE_CALLER, call_id="solve-by-pt-b", reporting_node_id="pt-b",
                        position_offset_m=-REPORTER_SOLVE_OFFSET_M, covariance_m2=covariance_m2),
    ]


async def test_two_reporters_solves_of_one_long_range_call_are_one_caller(tmp_path: Path) -> None:
    classification = await _classify_long_range(
        tmp_path, [LONG_RANGE_CALLER], _two_reporter_solves(LONG_RANGE_POSITION_COVARIANCE_M2)
    )
    clip = classification.features[POINT_NETWORK_CLIP_FEATURE_KEY]
    assert clip["species_uncertain"] is False and clip["overlapping_callers"] == []
    assert classification.label == HEURISTIC_BIRD_LABEL


async def test_without_stated_covariances_the_40_m_apart_solves_would_mask_each_other(tmp_path: Path) -> None:
    # The scene really exercises the covariance gate: separation alone (> 15 m) calls them two callers.
    classification = await _classify_long_range(tmp_path, [LONG_RANGE_CALLER], _two_reporter_solves(None))
    assert classification.label == SPECIES_UNCERTAIN_LABEL


async def test_two_real_long_range_callers_still_leave_the_species_uncertain(tmp_path: Path) -> None:
    solves = [
        _reporter_solve(LONG_RANGE_CALLER, call_id="first-caller", reporting_node_id="pt-d",
                        position_offset_m=REPORTER_SOLVE_OFFSET_M, covariance_m2=LONG_RANGE_POSITION_COVARIANCE_M2),
        _reporter_solve(SECOND_LONG_RANGE_CALLER, call_id="second-caller", reporting_node_id="pt-c",
                        position_offset_m=np.zeros(3), covariance_m2=LONG_RANGE_POSITION_COVARIANCE_M2),
    ]
    classification = await _classify_long_range(tmp_path, [LONG_RANGE_CALLER, SECOND_LONG_RANGE_CALLER], solves)
    clip = classification.features[POINT_NETWORK_CLIP_FEATURE_KEY]
    assert classification.label == SPECIES_UNCERTAIN_LABEL and classification.confidence == 0.0
    assert [caller["call_id"] for caller in clip["overlapping_callers"]] == ["second-caller"]


# -- Finding 3: a failed clip classifier keeps the sidecar's label ------------------

SIDECAR_LABEL = "American Robin"
SIDECAR_CONFIDENCE = 0.8


async def test_a_clip_classifier_failure_keeps_the_sidecar_label(tmp_path: Path) -> None:
    fusion, storage, classifier = await started_point_network_fusion(tmp_path)

    def failing_backend(samples, sample_rate_hz):
        raise RuntimeError("classifier backend unavailable")

    classifier.classify = failing_backend
    try:
        await append_captures(fusion.buffer, render_point_node_captures([CALLER_NEAR_A]))
        request = _render_near_a(fusion)
        request.authoritative_classification = ClassificationResult(
            label=SIDECAR_LABEL, confidence=SIDECAR_CONFIDENCE, scores={SIDECAR_LABEL: SIDECAR_CONFIDENCE}, features={}
        )
        await fusion.ingest_localized_render(request)

        detection = (await stored_detections_by_solve(storage, fusion))[RENDER_SOLVE_ID]
        features = detection["feature_summary"]
        assert detection["label"] == SIDECAR_LABEL
        assert detection["label_confidence"] == pytest.approx(SIDECAR_CONFIDENCE)
        assert features["rust_classification_authoritative"] is True
        clip = features[POINT_NETWORK_CLIP_FEATURE_KEY]
        assert clip["unavailable_reason"] == UNAVAILABLE_REASON_CLIP_CLASSIFIER_FAILED
        assert clip["clip_node_id"] == "pt-a" and clip["failure_reason"] == "classification_error"
        assert fusion._metrics.point_network_clip_classified_count == 0
    finally:
        await fusion.stop()
        await storage.close()


# -- Finding 4: the clip classification runs off the SSE consumer --------------------

BLOCKED_CLASSIFIER_MAX_WAIT_SECONDS = 10.0
INGEST_DEADLINE_SECONDS = 5.0


async def test_ingest_returns_before_the_clip_classification_and_the_worker_releases_the_held_fix(tmp_path: Path) -> None:
    fusion, storage, classifier = await started_point_network_fusion(tmp_path)
    release_classifier = threading.Event()
    unblocked_classify = classifier.classify

    def blocked_until_released(samples, sample_rate_hz):
        release_classifier.wait(timeout=BLOCKED_CLASSIFIER_MAX_WAIT_SECONDS)
        return unblocked_classify(samples, sample_rate_hz)

    classifier.classify = blocked_until_released
    released_candidate_ids = _record_held_fix_releases(fusion)
    try:
        await append_captures(fusion.buffer, render_point_node_captures([CALLER_NEAR_A]))
        await asyncio.wait_for(fusion.ingest_localized_render(_render_near_a(fusion)), timeout=INGEST_DEADLINE_SECONDS)

        # Ingest is back while the clip is still being classified: nothing reported yet,
        # and the classification worker (not ingest) owns the held-fix release.
        assert RENDER_SOLVE_ID not in await stored_detections_by_solve(storage)
        assert released_candidate_ids == []
        release_classifier.set()

        detection = (await stored_detections_by_solve(storage, fusion))[RENDER_SOLVE_ID]
        assert detection["feature_summary"]["classification_path"] == POINT_NETWORK_CLIP_CLASSIFICATION_PATH
        assert detection["label"] == HEURISTIC_BIRD_LABEL
        assert released_candidate_ids == [RENDER_CANDIDATE_ID]
    finally:
        release_classifier.set()
        await fusion.stop()
        await storage.close()


# -- Finding 5: one far-future span cannot disable the call registry ----------------

REGISTRY_RETENTION_SECONDS = 120.0


def _registry_call(call_id: str, span_end_ns: int) -> LocalizedCall:
    origin = np.zeros(3)
    return LocalizedCall(
        call_id=call_id,
        source_position_m=origin,
        alignment_reference_sensor_id="s",
        alignment_reference_position_m=origin,
        event_span_start_ns=span_end_ns - NANOSECONDS_PER_SECOND // 10,
        event_span_end_ns=span_end_ns,
        sound_speed_mps=SOUND_SPEED_MPS,
    )


def test_a_far_future_span_does_not_disable_the_call_registry() -> None:
    registry = LocalizedCallRegistry(retention_seconds=REGISTRY_RETENTION_SECONDS, max_records=512)
    now_ns = CAPTURE_START_NS
    registry.register(_registry_call("bad-clock", now_ns + 3_600 * NANOSECONDS_PER_SECOND))
    registry.register(_registry_call("call-1", now_ns))
    registry.register(_registry_call("call-2", now_ns + NANOSECONDS_PER_SECOND))
    remembered = {call.call_id for call in registry.calls_other_than("none")}
    assert {"call-1", "call-2"} <= remembered

    # Retention still applies, measured back from the call being registered.
    registry.register(_registry_call("call-3", now_ns + seconds_to_nanoseconds(REGISTRY_RETENTION_SECONDS + 10.0)))
    remembered = {call.call_id for call in registry.calls_other_than("none")}
    assert "call-1" not in remembered and "call-3" in remembered


# -- Findings 6 and 10 (Python ingest) ------------------------------------------------

PYTHON_LOCALIZATION_WINDOW_SECONDS = 0.4
PYTHON_TRIGGER_DELAY_SECONDS = 0.05


async def _python_point_network_fusion(tmp_path: Path):
    fusion, storage, _ = await started_point_network_fusion(
        tmp_path,
        localizer=TruthLocalizer(CALLER_NEAR_A.position_m),
        localization_window_seconds=PYTHON_LOCALIZATION_WINDOW_SECONDS,
        classification_window_seconds=PYTHON_LOCALIZATION_WINDOW_SECONDS,
        min_sensors_for_3d=4,
        min_sensors_for_2d=3,
        trigger_rms=1e-6,
    )
    await register_point_nodes(fusion, storage)
    caller = replace(CALLER_NEAR_A, sound_speed_mps=fusion.environment_provider.get_speed_of_sound())
    await append_captures(fusion.buffer, render_point_node_captures([caller]))
    candidate = python_ingest_candidate(
        candidate_id="evt-a",
        triggering_node_id="pt-d",
        event_time_ns=caller.arrival_time_ns("pt-d") + int(PYTHON_TRIGGER_DELAY_SECONDS * NANOSECONDS_PER_SECOND),
        sample_rate_hz=SAMPLE_RATE_HZ,
    )
    return fusion, storage, candidate


async def test_python_alignment_reference_is_the_triggering_node_even_when_not_selected(tmp_path: Path) -> None:
    fusion, storage, candidate = await _python_point_network_fusion(tmp_path)
    try:
        product = await fusion._localize_candidate(candidate)
        triggering_sensor_id = point_sensor_id("pt-d")
        assert triggering_sensor_id in product.selected_sensor_ids
        # e.g. excluded for its rate or coverage: the clip timeline is still pt-d's.
        without_trigger = replace(
            product,
            selected_sensor_ids=[sensor_id for sensor_id in product.selected_sensor_ids if sensor_id != triggering_sensor_id],
            selected_positions={
                sensor_id: position for sensor_id, position in product.selected_positions.items() if sensor_id != triggering_sensor_id
            },
        )
        call = await localized_call_for_python_ingest_product(fusion, without_trigger)
        assert call.alignment_reference_sensor_id == triggering_sensor_id
        np.testing.assert_allclose(call.alignment_reference_position_m, POINT_NODE_POSITIONS_M["pt-d"])
    finally:
        await fusion.stop()
        await storage.close()


async def test_python_ingest_releases_the_held_fix_when_registering_the_call_fails(tmp_path: Path, monkeypatch) -> None:
    fusion, storage, candidate = await _python_point_network_fusion(tmp_path)

    async def failing_registration(host, product) -> None:
        raise RuntimeError("call registry unavailable")

    monkeypatch.setattr(fusion_node_module, "register_python_ingest_call", failing_registration)
    released_candidate_ids = _record_held_fix_releases(fusion)
    try:
        assert await fusion._enqueue_stage(fusion._localization_queue, candidate)
        await asyncio.wait_for(fusion._localization_queue.join(), timeout=INGEST_DEADLINE_SECONDS)
        assert released_candidate_ids == ["evt-a"]
        assert fusion._classification_queue.qsize() == 0
    finally:
        await fusion.stop()
        await storage.close()


async def test_rust_render_releases_the_held_fix_when_registering_the_call_fails(tmp_path: Path) -> None:
    fusion, storage, _ = await started_point_network_fusion(tmp_path)

    def failing_registration(call) -> None:
        raise RuntimeError("call registry unavailable")

    fusion._point_network_call_classifier.register_localized_call = failing_registration
    released_candidate_ids = _record_held_fix_releases(fusion)
    try:
        with pytest.raises(RuntimeError, match="call registry unavailable"):
            await fusion.ingest_localized_render(_render_near_a(fusion))
        assert released_candidate_ids == [RENDER_CANDIDATE_ID]
    finally:
        await fusion.stop()
        await storage.close()
