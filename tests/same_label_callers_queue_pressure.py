"""Two same-label callers under classification-queue pressure (E6).

The classification queue lives on the Python-ingest path, so this case drives
``FusionNode``'s own worker loops: each scene solve is an ``EventCandidate``
put on the localization queue, the localization worker hands its product to
the position-observation seam and then to the bounded, drop-oldest
classification queue, and a slow classifier keeps that queue overflowing.

Two stand-ins, both at the ends of the stage under test:
- ``_localize_candidate`` returns the scene's solve (the Python solver is
  covered elsewhere; here only what happens after a solve matters);
- classification inference is a fixed-label result after a delay longer than
  the callers' solve spacing, adopted through the real orchestrator so
  taxonomy, reporting and assembly run unchanged.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from minimappr.core.birdnet_label_categories import BIRDNET_V24_MODEL_NAME
from minimappr.core.fusion_node import FusionNode
from minimappr.core.fusion_stage_carriers import EventCandidate, LocalizationBranch, LocalizedCandidate
from minimappr.models import ClassificationResult, NodeSpec, TimeQuality
from minimappr.sim.continuous_source_scenes import ContinuousSourceScene, SceneSolve
from tests.continuous_source_manifests import SCENE_START_NS, window_centre_ns
from tests.continuous_source_tracking_harness import (
    ContinuousSourceHarnessError,
    TrackSnapshotEntry,
    started_scene_fusion,
)
from tests.localization_solve_manifest_fixtures import MIXED_SAMPLE_RATE_HZ, planar_node_payload

CLASSIFICATION_QUEUE_SIZE = 1
# Each classification takes longer than the gap between the callers' solves
# as they are fed, so the one-slot queue must shed products.
SLOW_CLASSIFICATION_SECONDS = 0.15
CANDIDATE_FEED_SPACING_SECONDS = 0.02
CLASSIFICATION_SIGNAL_SAMPLES = 1600
SLOW_CLASSIFIER_CONFIDENCE = 0.9
QUEUE_DRAIN_TIMEOUT_SECONDS = 10.0
SOLVE_CONFIDENCE = 0.9
SOLVE_GDOP = 1.0
SOLVE_METHOD = "srp_phat"
SOLVE_RANGE_PROJECTION_MODE = "range_refined"
CANDIDATE_SOURCE_TYPE = "raw_sensor"


@dataclass(slots=True)
class QueuePressureRunRecord:
    final_tracks: dict[str, TrackSnapshotEntry] = field(default_factory=dict)
    classification_queue_drops: int = 0
    classifications_completed: int = 0


def _node_spec() -> NodeSpec:
    return NodeSpec.model_validate(planar_node_payload())


def _localized_product(candidate: EventCandidate, solve: SceneSolve, sensor_positions: dict) -> LocalizedCandidate:
    reference_sensor = next(iter(sensor_positions))
    silent_signal = np.zeros(CLASSIFICATION_SIGNAL_SAMPLES, dtype=np.float32)
    variance_m2 = solve.position_std_m**2
    branch = LocalizationBranch(
        localization_position_m=solve.measured_position_m,
        localization_confidence=SOLVE_CONFIDENCE,
        localization_gdop=SOLVE_GDOP,
        localization_position_covariance_m2=[
            [variance_m2 if row == column else 0.0 for column in range(3)] for row in range(3)
        ],
        localization_range_observability=None,
        localization_residual_rms_seconds=None,
        localization_range_projection_mode=SOLVE_RANGE_PROJECTION_MODE,
        reference_sensor=reference_sensor,
        reference_signal=silent_signal,
        classification_reference_signal=silent_signal,
        tdoa_s={},
        localization_method=SOLVE_METHOD,
        capability_tier="full_3d",
    )
    return LocalizedCandidate(
        candidate=candidate,
        localization_branch=branch,
        selected_sensor_ids=list(sensor_positions),
        selected_windows={},
        selected_positions=sensor_positions,
        classification_selected_windows={},
        omni_reference_sensor=reference_sensor,
        omni_reference_signal=silent_signal,
        omni_position_m=solve.measured_position_m,
        omni_classification_reference_signal=silent_signal,
        localization_audio_quality={},
        classification_audio_quality={},
        classification_audio_quality_source="localization_centered",
        environment={},
    )


def _install_scene_stand_ins(
    fusion: FusionNode, scene: ContinuousSourceScene, sensor_positions: dict, record: QueuePressureRunRecord
) -> None:
    solves_by_candidate_id = {f"candidate-{solve.solve_id}": solve for solve in scene.solves}

    async def scene_solve_for(candidate: EventCandidate) -> LocalizedCandidate:
        return _localized_product(candidate, solves_by_candidate_id[candidate.id], sensor_positions)

    async def slow_fixed_label_classification(product: LocalizedCandidate):
        await asyncio.sleep(SLOW_CLASSIFICATION_SECONDS)
        label = scene.sources[solves_by_candidate_id[product.candidate.id].source_id].label
        classified = await fusion._classification_orchestrator.adopt_authoritative_classification(
            classification=ClassificationResult(
                label=label,
                confidence=SLOW_CLASSIFIER_CONFIDENCE,
                scores={label: SLOW_CLASSIFIER_CONFIDENCE},
                features={"model": BIRDNET_V24_MODEL_NAME},
            ),
            event_time_ns=product.candidate.event_time_ns,
            classification_signal=product.omni_classification_reference_signal,
            sample_rate_hz=product.candidate.sample_rate_hz,
        )
        record.classifications_completed += 1
        return classified

    fusion._localize_candidate = scene_solve_for
    fusion._classify_and_attribute_localized = slow_fixed_label_classification


async def run_callers_under_classification_queue_pressure(
    tmp_path: Path, scene: ContinuousSourceScene, *, scene_start_ns: int = SCENE_START_NS
) -> QueuePressureRunRecord:
    fusion, storage = await started_scene_fusion(
        tmp_path, fusion_classification_queue_size=CLASSIFICATION_QUEUE_SIZE, fusion_worker_count=1
    )
    record = QueuePressureRunRecord()
    try:
        node = _node_spec()
        await fusion.registry.upsert(node, scene_start_ns)
        # Detection rows reference the node row.
        await storage.upsert_node(
            spec=node, last_seen_ns=scene_start_ns, position_geo=fusion.coordinate_frame.local_to_geo(node.position_m)
        )
        sensor_positions = {
            descriptor.sensor_id: descriptor.position_m for descriptor in await fusion.registry.sensors_for_node(node.id)
        }
        _install_scene_stand_ins(fusion, scene, sensor_positions, record)
        for solve in scene.solves:
            candidate = EventCandidate(
                id=f"candidate-{solve.solve_id}",
                source_node_id=node.id,
                event_time_ns=window_centre_ns(solve, scene_start_ns),
                sample_rate_hz=MIXED_SAMPLE_RATE_HZ,
                source_type=CANDIDATE_SOURCE_TYPE,
                time_quality=TimeQuality.GPS_LOCKED,
                source_observation_ids=[],
                enqueued_ns=time.time_ns(),
            )
            if not await fusion._enqueue_stage(fusion._localization_queue, candidate):
                raise ContinuousSourceHarnessError(f"localization queue refused {candidate.id}")
            await asyncio.sleep(CANDIDATE_FEED_SPACING_SECONDS)
        await asyncio.wait_for(fusion._localization_queue.join(), timeout=QUEUE_DRAIN_TIMEOUT_SECONDS)
        await asyncio.wait_for(fusion._classification_queue.join(), timeout=QUEUE_DRAIN_TIMEOUT_SECONDS)
        await asyncio.wait_for(fusion._rules_queue.join(), timeout=QUEUE_DRAIN_TIMEOUT_SECONDS)
        # Products the localization stage queued that classification never took: shed by the full queue.
        record.classification_queue_drops = (
            fusion._metrics.localization_stage_out - fusion._metrics.classification_stage_in
        )
        tracks = await fusion.tracker.snapshot(include_hidden=True)
        record.final_tracks = {track.id: TrackSnapshotEntry.from_track(track) for track in tracks}
        if fusion._metrics.localization_failures or fusion._metrics.classification_failures:
            raise ContinuousSourceHarnessError(f"pipeline worker failures: {fusion._last_error}")
    finally:
        await fusion.stop()
        await storage.close()
    return record
