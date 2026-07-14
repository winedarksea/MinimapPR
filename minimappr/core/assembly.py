"""Detection assembly — builds DetectionEvent records, writes audio snippets,
and persists detections, pings, and track updates to storage.

Extracted from FusionNode to isolate the detection-building logic as an
independently testable component.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from minimappr.core.amplitude_range import received_level_db_from_rms
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.node_registry import NodeRegistry
from minimappr.audio_processing.chain import NodePreprocessorFactory
from minimappr.core.retention import RetentionPolicy
from minimappr.core.zones import ZoneMatcher
from minimappr.interfaces import StorageBackend
from minimappr.models import (
    DetectionEvent,
    LabelId,
    RetentionTier,
    TrackState,
    spatial_display_mode_for_detection,
)
from minimappr.utils.audio import rms, write_wav_mono


@dataclass(slots=True)
class AssemblyResult:
    """Result of assembling and persisting a detection."""

    detection: DetectionEvent
    track: TrackState | None
    suppressed_by_zone: bool
    suppression_reasons: list[str]


def _collapse_long_exact_zero_runs(
    signal,
    *,
    sample_rate_hz: int,
    min_gap_seconds: float = 0.03,
    crossfade_seconds: float = 0.005,
):
    """Remove long exact-zero spans that represent uncovered buffer gaps.

    Coverage holes are inserted as literal zeros by the audio buffer. Compacting
    long zero runs for snippet rendering avoids playback dropouts. Boundaries
    are briefly faded around removed holes so snippets do not splice unrelated
    waveform phases with a hard step.
    """
    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    if arr.size == 0 or sample_rate_hz <= 0:
        return arr

    zero_mask = arr == 0.0
    if not bool(np.any(zero_mask)):
        return arr

    min_gap_samples = max(1, int(round(min_gap_seconds * float(sample_rate_hz))))
    if min_gap_samples <= 1:
        return arr

    keep_mask = np.ones(arr.size, dtype=np.bool_)
    removed_run_starts: list[int] = []
    in_zero_run = False
    run_start = 0

    for index, is_zero in enumerate(zero_mask):
        if is_zero and not in_zero_run:
            in_zero_run = True
            run_start = index
            continue
        if not is_zero and in_zero_run:
            run_length = index - run_start
            if run_length >= min_gap_samples:
                removed_run_starts.append(run_start)
                keep_mask[run_start:index] = False
            in_zero_run = False

    if in_zero_run:
        run_length = arr.size - run_start
        if run_length >= min_gap_samples:
            removed_run_starts.append(run_start)
            keep_mask[run_start:] = False

    compacted = arr[keep_mask]
    if compacted.size == 0:
        return arr
    fade_samples = int(round(crossfade_seconds * float(sample_rate_hz)))
    if fade_samples < 2 or not removed_run_starts:
        return compacted

    kept_prefix_counts = np.cumsum(keep_mask, dtype=np.int64)
    for run_start in removed_run_starts:
        boundary = int(kept_prefix_counts[run_start - 1]) if run_start > 0 else 0
        fade_len = min(fade_samples, boundary, compacted.size - boundary)
        if fade_len < 2:
            continue
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        compacted[boundary - fade_len:boundary] *= fade_out
        compacted[boundary:boundary + fade_len] *= fade_in
    return compacted


# Drone-head labels that represent a non-detection: audio is never materialized
# for these. Positive classes (drone, coyote, ...) keep their snippet.
_DRONE_HEAD_NEGATIVE_LABELS = {"unknown", "ambient", "no_drone"}


def _drone_head_retains_audio(classifier_source: str, label: str) -> bool:
    """Whether a drone-head result should keep its audio snippet.

    Only negative labels (unknown/ambient/no_drone) are discarded; every positive
    class is retained. Non-drone-head sources are unaffected (always True here).
    """
    if classifier_source != "drone_head":
        return True
    return label.strip().lower() not in _DRONE_HEAD_NEGATIVE_LABELS


class DetectionAssembler:
    """Assembles DetectionEvent records from classification output.

    Responsibilities:
      - Zone matching and suppression evaluation
      - Track association (delegates to TrackManager)
      - SPL computation with per-node gain offset
      - Retention tier determination (delegates to RetentionPolicy)
      - Feature summary composition
      - Audio snippet writing
      - Storage persistence (detection, ping, track, track_update)
    """

    def __init__(
        self,
        *,
        storage: StorageBackend,
        coordinate_frame: LocalCoordinateFrame,
        zone_matcher: ZoneMatcher,
        registry: NodeRegistry,
        preprocessor_factory: NodePreprocessorFactory,
        retention_policy: RetentionPolicy,
        snippet_dir: Path | str,
        snippet_retention_seconds: float,
        classifier_audio_retention_seconds: dict[str, float] | None = None,
        event_stale_seconds: float,
    ) -> None:
        self._storage = storage
        self._coordinate_frame = coordinate_frame
        self._zone_matcher = zone_matcher
        self._registry = registry
        self._preprocessor_factory = preprocessor_factory
        self._retention_policy = retention_policy
        self._snippet_dir = Path(snippet_dir)
        self._snippet_retention_seconds = snippet_retention_seconds
        self._classifier_audio_retention_seconds = classifier_audio_retention_seconds or {}
        self._event_stale_seconds = event_stale_seconds

    def replace_coordinate_frame(self, coordinate_frame: LocalCoordinateFrame) -> None:
        self._coordinate_frame = coordinate_frame

    async def assemble(
        self,
        *,
        # Localization data
        localization_position_m: tuple[float, float, float],
        localization_confidence: float,
        localization_gdop: float,
        localization_position_covariance_m2: list[list[float]] | None,
        localization_range_observability: float | None,
        localization_residual_rms_seconds: float | None,
        localization_range_projection_mode: str | None,
        reference_sensor: str,
        tdoa_s: dict[str, float],
        selected_sensor_ids: list[str],
        reference_signal,
        capability_tier: str,
        localization_method: str,
        environment: dict[str, Any],
        # Classification data
        classification_label: str,
        classification_confidence: float,
        classification_scores: dict[str, float],
        classification_features: dict[str, Any],
        classification_path: str,
        omni_confidence: float,
        beamformed_classification_confidence: float | None,
        beamformed_classification_label: str | None,
        beamforming_error: str | None,
        classification_signal,
        label_category: str,
        iff_category: str,
        label_id: LabelId | None,
        report_window_start_ns: int | None,
        report_window_end_ns: int | None,
        reporting_modality: str,
        branch_evidence: dict[str, Any] | None,
        # Event metadata
        event_time_ns: int,
        source_type: str,
        time_quality,
        source_observation_ids: list[str],
        sample_rate_hz: int,
        audio_quality: dict[str, Any] | None = None,
        existing_detection: dict[str, Any] | None = None,
        persist_mode: str = "insert",
        # Tracker
        tracker,
        # Storage batch context
        storage_batch_ctx,
    ) -> AssemblyResult:
        """Build a DetectionEvent, persist it, and return the result."""
        detection_geo = self._coordinate_frame.local_to_geo(localization_position_m)

        # -- zone matching -----------------------------------------------------
        zone_ids = await self._zone_matcher.match_geo_point(
            lat=detection_geo.lat,
            lon=detection_geo.lon,
            now_ns=event_time_ns,
        )
        suppressed_by_zone, suppression_reasons = await self._zone_matcher.evaluate_detection_policies(
            zone_ids=zone_ids,
            label=classification_label,
            label_category=label_category,
            now_ns=event_time_ns,
        )

        # -- feature summary ---------------------------------------------------
        feature_summary = {}
        if existing_detection is not None and isinstance(existing_detection.get("feature_summary"), dict):
            feature_summary.update(existing_detection["feature_summary"])
        feature_summary.update(classification_features)
        feature_summary["capability_tier"] = capability_tier
        feature_summary["localization_method"] = localization_method
        if localization_range_observability is not None:
            feature_summary["localization_range_observability"] = localization_range_observability
        if localization_residual_rms_seconds is not None:
            feature_summary["localization_residual_rms_seconds"] = localization_residual_rms_seconds
        if localization_range_projection_mode is not None:
            feature_summary["localization_range_projection_mode"] = localization_range_projection_mode
        geo_uncertainty = self._coordinate_frame.local_covariance_to_geo_uncertainty(
            localization_position_covariance_m2
        )
        if geo_uncertainty is not None:
            feature_summary["position_geo_uncertainty"] = geo_uncertainty
        feature_summary["classification_path"] = classification_path
        feature_summary["omni_confidence"] = omni_confidence
        feature_summary["environment"] = environment
        feature_summary["branch_evidence"] = branch_evidence or {}
        if audio_quality is not None:
            feature_summary["audio_quality"] = audio_quality
        if beamformed_classification_confidence is not None:
            feature_summary["beamformed_confidence"] = beamformed_classification_confidence
            feature_summary["beamformed_label"] = beamformed_classification_label
        if beamforming_error:
            feature_summary["beamforming_error"] = beamforming_error
        if suppression_reasons:
            feature_summary["zone_suppression"] = suppression_reasons
        spatial_display_mode = spatial_display_mode_for_detection(
            reporting_modality=reporting_modality,
            feature_summary=feature_summary,
        )

        # -- track update ------------------------------------------------------
        source_node_id = await self._registry.node_id_for_sensor(reference_sensor)
        track: TrackState | None = None
        localizable_capability_tiers = {"2d", "full_3d"}
        if (
            spatial_display_mode == "localized"
            and not suppressed_by_zone
            and capability_tier in localizable_capability_tiers
        ):
            track = await tracker.update(
                timestamp_ns=event_time_ns,
                position_m=localization_position_m,
                measurement_covariance_m2=localization_position_covariance_m2,
                label=classification_label,
                label_category=label_category,
                iff_category=iff_category,
                label_id=label_id,
                confidence=classification_confidence,
                sensor_count=len(selected_sensor_ids),
                capability_tier=capability_tier,
                source_node_id=source_node_id,
            )
            track.position_geo = self._coordinate_frame.local_to_geo(track.position_m)

        # -- observation ID resolution ----------------------------------------
        latest_observation_ids = await self._registry.latest_observation_ids(selected_sensor_ids)
        all_observation_ids = list(
            dict.fromkeys([*source_observation_ids, *latest_observation_ids])
        )
        if existing_detection is not None:
            all_observation_ids = list(
                dict.fromkeys([*existing_detection.get("source_observation_ids", []), *all_observation_ids])
            )
        # -- SPL ---------------------------------------------------------------
        gain_offset_db = await self._registry.gain_offset_db_for_sensor(reference_sensor)
        digital_trim_db = self._preprocessor_factory.fixed_gain_db_for_sensor(reference_sensor)
        spl_db = received_level_db_from_rms(
            rms(reference_signal),
            gain_offset_db - digital_trim_db,
        )

        # -- retention ---------------------------------------------------------
        retention_tier = self._retention_policy.tier_for_detection(
            label=classification_label,
            label_category=label_category,
            confidence=classification_confidence,
            suppressed_by_zone=suppressed_by_zone,
        )

        # -- build DetectionEvent ----------------------------------------------
        tor_ns = time.time_ns()
        stale_ns = event_time_ns + int(self._event_stale_seconds * 1_000_000_000)

        detection = DetectionEvent(
            id=(
                existing_detection["id"]
                if existing_detection is not None and isinstance(existing_detection.get("id"), str)
                else f"det-{uuid.uuid4().hex[:12]}"
            ),
            source_type=source_type,
            source_node_id=source_node_id,
            timestamp_ns=event_time_ns,
            toa_ns=event_time_ns,
            tor_ns=tor_ns,
            time_quality=time_quality,
            stale_ns=stale_ns,
            report_window_start_ns=report_window_start_ns,
            report_window_end_ns=report_window_end_ns,
            reporting_modality=reporting_modality,
            spatial_display_mode=spatial_display_mode,
            position_m=localization_position_m,
            position_geo=detection_geo,
            position_covariance_m2=(
                localization_position_covariance_m2
                if localization_position_covariance_m2 is not None
                else (track.position_covariance_m2 if track is not None else None)
            ),
            confidence=localization_confidence,
            gdop=localization_gdop,
            label_id=label_id,
            label=classification_label,
            label_category=label_category,
            iff_category=iff_category,
            label_confidence=classification_confidence,
            spl_db=spl_db,
            track_id=track.id if track is not None and not suppressed_by_zone else None,
            source_sensors=sorted(selected_sensor_ids),
            source_observation_ids=all_observation_ids,
            zone_ids=zone_ids,
            reference_sensor=reference_sensor,
            tdoa_s=tdoa_s,
            classifier_scores=classification_scores,
            feature_summary=feature_summary,
            retention_tier=RetentionTier(retention_tier),
            snippet_path=None,
        )

        # -- snippet writing ---------------------------------------------------
        snippet_path: str | None = None
        snippet_expires_ns: int | None = None
        classifier_source = str(
            classification_features.get("winner_member") or "yamnet"
        ).strip().lower()
        # An explicit zero global retention is a privacy/storage opt-out for all
        # classifier outputs; per-classifier defaults must not override it.
        classifier_audio_retention_seconds = (
            0
            if self._snippet_retention_seconds <= 0
            else self._classifier_audio_retention_seconds.get(
                classifier_source, self._snippet_retention_seconds
            )
        )
        # The drone head reports a negative label ("unknown"/"ambient"/"no_drone")
        # for non-detections. Never materialize an audio file for those; positive
        # classes (drone, coyote, ...) keep their audio. Metadata remains available.
        if not _drone_head_retains_audio(classifier_source, classification_label):
            classifier_audio_retention_seconds = 0
        if classifier_audio_retention_seconds > 0 and retention_tier not in {"ephemeral", "experiment"}:
            snippet_signal = classification_signal
            if isinstance(audio_quality, dict):
                missing_ratio = float(audio_quality.get("missing_ratio") or 0.0)
                if missing_ratio > 0.05:
                    snippet_signal = _collapse_long_exact_zero_runs(
                        snippet_signal,
                        sample_rate_hz=sample_rate_hz,
                    )
            snippet_file = self._snippet_dir / f"{detection.id}.wav"
            await asyncio.to_thread(
                write_wav_mono, snippet_file, snippet_signal, sample_rate_hz
            )
            snippet_path = str(snippet_file)
            snippet_expires_ns = event_time_ns + int(
                classifier_audio_retention_seconds * 1_000_000_000
            )
            detection.snippet_path = snippet_path

        # -- storage persistence -----------------------------------------------
        async with storage_batch_ctx():
            if track is not None and not suppressed_by_zone:
                await self._storage.upsert_track(track)

            if persist_mode == "update":
                await self._storage.update_detection(
                    detection=detection,
                    snippet_path=snippet_path,
                    snippet_expires_ns=snippet_expires_ns,
                    retention_tier=retention_tier,
                )
            else:
                await self._storage.insert_detection(
                    detection=detection,
                    snippet_path=snippet_path,
                    snippet_expires_ns=snippet_expires_ns,
                    retention_tier=retention_tier,
                )
                await self._storage.insert_ping(
                    timestamp_ns=detection.timestamp_ns,
                    ping_type="acoustic",
                    label=detection.label,
                    label_id=detection.label_id,
                    spl_db=detection.spl_db,
                    position_m=detection.position_m,
                    position_geo=detection.position_geo,
                    source_detection_id=detection.id,
                    source_observation_id=(
                        detection.source_observation_ids[0]
                        if detection.source_observation_ids
                        else None
                    ),
                    source_track_id=detection.track_id,
                    retention_tier=retention_tier,
                    metadata={
                        "label_category": detection.label_category,
                        "confidence": detection.label_confidence,
                        "zone_ids": detection.zone_ids,
                        "capability_tier": capability_tier,
                    },
                )
            if track is not None and not suppressed_by_zone:
                await self._storage.insert_track_update(
                    track=track,
                    timestamp_ns=event_time_ns,
                    event_id=detection.event_id,
                    update_type="detection",
                    detection_id=detection.id,
                    observation_ids=all_observation_ids,
                    metadata={
                        "gdop": detection.gdop,
                        "reference_sensor": detection.reference_sensor,
                        "capability_tier": capability_tier,
                    },
                )

        return AssemblyResult(
            detection=detection,
            track=track,
            suppressed_by_zone=suppressed_by_zone,
            suppression_reasons=suppression_reasons,
        )
