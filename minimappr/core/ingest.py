"""Ingest processing — PCM decode, preprocessing, buffer insertion, observation
persistence, environment extraction, and trigger evaluation.

Extracted from FusionNode to isolate the ingest-to-trigger path as an
independently testable component.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)

# Log a summary after this many accepted (non-duplicate) frames.
_INGEST_LOG_INTERVAL_FRAMES = 5000
_AUDIO_SUMMARY_PUBLISH_INTERVAL_NS = 500_000_000
# Receipt time is metadata for freshness. GPS/NTP packet timestamps remain the
# audio processing timeline even when transport latency is high.
_MAX_TRUSTED_NODE_CLOCK_SKEW_NS = 300_000_000_000

import numpy as np

from minimappr.config import FusionConfig, LocalizationConfig
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.ingest_health import IngestHealthClassifier, runner_stats_from_timing_diagnostics
from minimappr.core.live_ingest_state import FrameIdentity, LiveIngestState
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.node_position_estimator import StationaryKdeState, compute_stationary_kde
from minimappr.core.preprocessing import NodePreprocessorFactory
from minimappr.interfaces import AudioPreprocessor, StorageBackend
from minimappr.models import (
    EnvironmentSampleIn,
    GeoPoint,
    IngestFrameRequest,
    IngestFrameResponse,
    NodeSpec,
    TimeQuality,
    sync_grade_from_time_quality,
)
from minimappr.utils.audio import decode_pcm16le_b64, trigger_rms


# Firmware NodeRunner telemetry counters carried in frame timing_diagnostics.
# Surfaced verbatim on the API so publish-queue/Wi-Fi health can be monitored
# (queue overflows drive server-side frame sequence gaps; the publish-failure
# breakdown distinguishes a saturated link from a slow consumer).
# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EnvironmentCounts:
    ingested: int = 0
    persisted: int = 0


@dataclass(slots=True)
class _NodePositionKalman:
    """Per-node 1-D Kalman state for each ENU axis (x=east, y=north, z=up)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    px: float = 100.0
    py: float = 100.0
    pz: float = 100.0
    initialized: bool = False


@dataclass(slots=True)
class IngestResult:
    """Result of processing a single ingest frame."""

    response: IngestFrameResponse
    triggered: bool
    event_time_ns: int | None = None
    sample_rate_hz: int | None = None
    source_type: str | None = None
    time_quality: TimeQuality | None = None
    observation_ids: list[str] | None = None
    environment_counts: EnvironmentCounts | None = None
    sequence_gap_count: int = 0


class EnvironmentUpdater:
    """Thin adapter that forwards environment samples to an EnvironmentProvider."""

    def __init__(self, environment_provider) -> None:
        self._provider = environment_provider

    def update(self, sample: dict[str, Any]) -> None:
        ingest_fn = getattr(self._provider, "ingest_sample", None)
        if not callable(ingest_fn):
            return
        ingest_fn(
            node_id=sample["node_id"],
            timestamp_ns=sample["timestamp_ns"],
            temperature_c=sample["temperature_c"],
            humidity_fraction=sample["humidity_fraction"],
            pressure_pa=sample["pressure_pa"],
            wind_speed_mps=sample["wind_speed_mps"],
            wind_dir_deg=sample["wind_dir_deg"],
            solar_lux=sample["solar_lux"],
            location_m=sample.get("location_m"),
            metadata=sample["metadata"],
        )


class IngestProcessor:
    """Handles the ingest path: decode → preprocess → buffer → observe → trigger.

    Responsibilities:
      - PCM base64 decoding and channel validation
      - Per-node audio preprocessing (highpass/lowpass)
      - Multi-sensor buffer insertion
            - Optional observation persistence plus environment sample persistence
      - Frame energy trigger evaluation and cooldown gating
    """

    def __init__(
        self,
        *,
        localization_config: LocalizationConfig,
        fusion_config: FusionConfig,
        registry: NodeRegistry,
        buffer: MultiSensorBuffer,
        storage: StorageBackend,
        coordinate_frame: LocalCoordinateFrame,
        preprocessor_factory: NodePreprocessorFactory,
        live_ingest_state: LiveIngestState | None = None,
        environment_updater: EnvironmentUpdater | None = None,
        persist_observations_on_ingest: bool = True,
        node_position_kalman_q: float = 0.5,
        node_position_kalman_q_stationary: float = 0.001,
        node_position_kalman_r: float = 25.0,
        node_position_kalman_init_p: float = 100.0,
        node_position_gps_gate_m: float = 5.0,
        node_position_kde_bandwidth_m: float = 2.5,
        node_position_kde_reservoir_capacity: int = 2048,
        node_position_kde_warmup_fixes: int = 30,
        node_position_kde_recompute_seconds: float = 30.0,
        node_position_kde_checkpoint_seconds: float = 60.0,
        node_position_kde_acceptance_radius_m: float = 100.0,
    ) -> None:
        self._localization_config = localization_config
        self._fusion_config = fusion_config
        self._registry = registry
        self._buffer = buffer
        self._storage = storage
        self._coordinate_frame = coordinate_frame
        self._preprocessor_factory = preprocessor_factory
        self._live_ingest_state = live_ingest_state or LiveIngestState()
        self._environment_updater = environment_updater
        self._persist_observations_on_ingest = persist_observations_on_ingest
        self._kalman_q = node_position_kalman_q
        self._kalman_q_stationary = node_position_kalman_q_stationary
        self._kalman_r = node_position_kalman_r
        self._kalman_init_p = node_position_kalman_init_p
        self._gps_gate_m = node_position_gps_gate_m
        self._kde_bandwidth_m = node_position_kde_bandwidth_m
        self._kde_reservoir_capacity = max(1, node_position_kde_reservoir_capacity)
        self._kde_warmup_fixes = max(1, node_position_kde_warmup_fixes)
        self._kde_recompute_seconds = max(0.1, node_position_kde_recompute_seconds)
        self._kde_checkpoint_seconds = max(1.0, node_position_kde_checkpoint_seconds)
        self._kde_acceptance_radius_m = max(0.0, node_position_kde_acceptance_radius_m)

        self._last_trigger_ns = 0
        self._accepted_frame_count = 0
        self._last_audio_summary_publish_ns_by_node: dict[str, int] = {}
        self._preprocess_locks_by_node: dict[str, asyncio.Lock] = {}
        self._position_kalman: dict[str, _NodePositionKalman] = {}
        self._position_kde: dict[str, StationaryKdeState] = {}
        self._position_filter_by_node: dict[str, str] = {}
        # Last altitude that came from an actual 3D fix, held so a later 2D fix
        # (which does not solve altitude at all) cannot move the node vertically.
        self._last_trusted_altitude_m: dict[str, float] = {}
        self._untrusted_altitude_warned_node_ids: set[str] = set()
        self._kde_evaluation_tasks: dict[str, asyncio.Task] = {}
        # Rate-limit the per-frame sequence-gap warning per node; on a lossy
        # link this fires on most frames and saturates the log ring buffer.
        self._seq_gap_warning_last_logged_s: dict[str, float] = {}
        self._seq_gap_warning_interval_seconds = 10.0
        self._unknown_capability_warning_node_ids: set[str] = set()
        self._ingest_health = IngestHealthClassifier()

    @property
    def last_trigger_ns(self) -> int:
        return self._last_trigger_ns

    @property
    def accepted_frame_count(self) -> int:
        return self._accepted_frame_count

    def replace_coordinate_frame(self, coordinate_frame: LocalCoordinateFrame) -> None:
        self._coordinate_frame = coordinate_frame

    async def hydrate_position_estimator_states(self) -> None:
        loader = getattr(self._storage, "list_node_position_estimator_states", None)
        if not callable(loader):
            return
        for node_id, row in (await loader()).items():
            state = row.get("state") if isinstance(row, dict) else None
            if isinstance(state, dict):
                self._position_kde[node_id] = StationaryKdeState.from_snapshot(state)
                self._position_filter_by_node[node_id] = "kde"

    async def reset_position_estimator(self, node_id: str) -> None:
        self._position_kde.pop(node_id, None)
        self._position_kalman.pop(node_id, None)
        self._position_filter_by_node.pop(node_id, None)
        task = self._kde_evaluation_tasks.pop(node_id, None)
        if task is not None:
            task.cancel()
        deleter = getattr(self._storage, "delete_node_position_estimator_state", None)
        if callable(deleter):
            await deleter(node_id)

    def position_estimator_diagnostics(self, node_id: str) -> dict[str, Any]:
        state = self._position_kde.get(node_id)
        effective = self._position_filter_by_node.get(node_id)
        if state is None:
            return {"effective_filter": effective, "status": "inactive", "sample_count": 0}
        return {
            "effective_filter": effective or "kde",
            "status": "ready" if state.estimate is not None else "warming",
            "sample_count": state.seen_count,
            "horizontal_std_m": state.horizontal_std_m,
        }

    def confirm_trigger(self, event_time_ns: int) -> None:
        """Commit the cooldown timestamp after a trigger was successfully enqueued."""
        self._last_trigger_ns = event_time_ns

    async def process_frame(
        self,
        request: IngestFrameRequest,
        *,
        storage_batch_ctx,
        decoded_audio: np.ndarray | None = None,
    ) -> IngestResult:
        """Decode, preprocess, persist observations, and evaluate trigger.

        Returns an ``IngestResult`` containing the response to send back to the
        node, along with trigger metadata if the frame qualified for downstream
        processing.
        """
        frame = request.frame
        raw_node = request.node
        mobility, position_filter = self._registry.position_policy_for(raw_node.id, raw_node.mobility)
        raw_node = raw_node.model_copy(update={"mobility": mobility})
        normalized_node, geo_position = self._normalize_node_spec(raw_node, position_filter=position_filter)
        server_received_ns = time.time_ns()
        runtime = await self._registry.upsert(normalized_node, server_received_ns)
        frame_sync_grade = sync_grade_from_time_quality(frame.time_quality)
        await self._registry.update_node_sensor_sync_grade(
            runtime.spec.id,
            frame_sync_grade,
        )
        boot_session = _node_boot_session(normalized_node)
        toa_ns = frame.toa_ns or frame.start_time_ns
        tor_ns = frame.tor_ns if frame.tor_ns is not None else server_received_ns

        # -- decode PCM --------------------------------------------------------
        audio = await asyncio.to_thread(
            _decode_audio_frame,
            samples_b64=frame.samples_b64,
            channels=frame.channels,
            decoded_audio=decoded_audio,
        )
        if audio.ndim != 2:
            raise ValueError("Decoded audio must be channels-first")
        if audio.shape[0] != frame.channels:
            raise ValueError("Decoded channel count does not match frame.channels")
        if frame.samples_per_channel is not None and audio.shape[1] != frame.samples_per_channel:
            raise ValueError("Decoded sample count does not match frame.samples_per_channel")

        # -- preprocess --------------------------------------------------------
        preprocessor: AudioPreprocessor = self._preprocessor_factory.for_node(normalized_node)
        preprocess_lock = self._preprocess_locks_by_node.setdefault(normalized_node.id, asyncio.Lock())
        async with preprocess_lock:
            processed, processing_metrics = await asyncio.to_thread(
                _preprocess_audio_frame,
                audio,
                frame.sample_rate_hz,
                normalized_node.id,
                preprocessor,
            )
        self._preprocessor_factory.record_frame_metrics(normalized_node.id, processing_metrics)

        # -- environment extraction -------------------------------------------
        environment_sample = _extract_environment_sample(
            request=request,
            node=normalized_node,
            toa_ns=toa_ns,
            tor_ns=tor_ns,
        )

        buffer_start_time_ns, buffer_end_time_ns, buffer_uses_receipt_time = _buffer_timestamps_for_frame(
            frame_start_time_ns=frame.start_time_ns,
            frame_end_time_ns=frame.utc_end_ns,
            sample_count=processed.shape[1],
            sample_rate_hz=frame.sample_rate_hz,
            time_quality=frame.time_quality,
            server_received_ns=server_received_ns,
            # Firmware nodes advertise GPS optional; only they need receipt-time
            # debug audio while waiting for a real NTP/GPS discipline source.
            allow_receipt_time_fallback="gps_optional" in normalized_node.capabilities,
        )

        # Reserve before writing frame-derived metadata. A replay must not leave
        # duplicate observation provenance even though it is excluded from audio.
        frame_identity = FrameIdentity.from_frame(
            node_id=normalized_node.id,
            boot_session=boot_session,
            source_type=frame.source_type,
            start_sample_index=frame.start_sample_index,
            end_sample_index=frame.end_sample_index,
            start_time_ns=frame.start_time_ns,
            frame_sequence=frame.sequence,
        )
        if not await self._live_ingest_state.reserve_frame(frame_identity):
            return IngestResult(
                response=IngestFrameResponse(
                    accepted=True,
                    duplicate=True,
                    triggered=False,
                    frame_energy=0.0,
                    detection_id=None,
                    queued_event_id=None,
                    queue_depth=0,
                ),
                triggered=False,
            )

        # -- persist bounded metadata after reserving buffer insertion ---------
        observation_ids: list[str] = []
        persist_node_registration = await self._live_ingest_state.should_persist_node_row(
            normalized_node,
            now_ns=server_received_ns,
        )
        persist_environment_sample = environment_sample is not None and await self._live_ingest_state.should_persist_environment_sample(
            node_id=normalized_node.id,
            timestamp_ns=int(environment_sample["timestamp_ns"]),
        )
        sequence_gap_count = 0

        try:
            async with storage_batch_ctx():
                if persist_node_registration:
                    await self._storage.upsert_node(
                        spec=normalized_node,
                        last_seen_ns=server_received_ns,
                        position_geo=geo_position,
                    )
                sequence_gap_count = await self._registry.record_frame_sequence(
                    node_id=normalized_node.id,
                    boot_session=boot_session,
                    frame_sequence=frame.sequence,
                )
                if sequence_gap_count > 0 and frame.sequence is not None:
                    # The cumulative gap count is aggregated downstream from the
                    # IngestResult and is unaffected by this log throttle.
                    now_s = time.monotonic()
                    last_logged_s = self._seq_gap_warning_last_logged_s.get(
                        normalized_node.id, 0.0
                    )
                    if now_s - last_logged_s >= self._seq_gap_warning_interval_seconds:
                        self._seq_gap_warning_last_logged_s[normalized_node.id] = now_s
                        expected_sequence = frame.sequence - sequence_gap_count
                        _logger.warning(
                            "Detected ingest frame sequence gap",
                            extra={
                                "node_id": normalized_node.id,
                                "boot_session": boot_session or None,
                                "expected_sequence": expected_sequence,
                                "received_sequence": frame.sequence,
                                "gap_size": sequence_gap_count,
                            },
                        )
                if environment_sample is not None:
                    if persist_environment_sample:
                        await self._storage.insert_environment(
                            node_id=normalized_node.id,
                            timestamp_ns=environment_sample["timestamp_ns"],
                            temperature_c=environment_sample["temperature_c"],
                            pressure_pa=environment_sample["pressure_pa"],
                            humidity_fraction=environment_sample["humidity_fraction"],
                            wind_speed_mps=environment_sample["wind_speed_mps"],
                            wind_dir_deg=environment_sample["wind_dir_deg"],
                            solar_lux=environment_sample["solar_lux"],
                            metadata=environment_sample["metadata"],
                        )
                    if self._environment_updater is not None:
                        self._environment_updater.update(environment_sample)
                if self._persist_observations_on_ingest:
                    for channel_index, sensor_id in enumerate(runtime.sensor_ids):
                        observation_id = await self._storage.insert_observation(
                            node_id=normalized_node.id,
                            sensor_id=sensor_id,
                            sensor_type="audio",
                            source_type=frame.source_type,
                            toa_ns=toa_ns,
                            tor_ns=tor_ns,
                            time_quality=frame.time_quality.value,
                            sample_rate_hz=frame.sample_rate_hz,
                            channel_index=channel_index,
                            frame_sequence=frame.sequence,
                            metadata={
                                "frame_start_ns": frame.start_time_ns,
                                "frame_end_ns": frame.utc_end_ns,
                                "start_sample_index": frame.start_sample_index,
                                "end_sample_index": frame.end_sample_index,
                                "frame_channels": frame.channels,
                                "samples_per_channel": audio.shape[1],
                                "encoding": frame.encoding,
                                "server_received_ns": server_received_ns,
                                "buffer_uses_receipt_time": buffer_uses_receipt_time,
                                "timing_diagnostics": frame.timing_diagnostics,
                                "preprocess": normalized_node.properties.get("preprocess", {}),
                            },
                        )
                        observation_ids.append(observation_id)
        except BaseException:
            await self._live_ingest_state.release_reserved_frame(frame_identity)
            raise

        # -- insert, then commit live audio -----------------------------------
        # A retransmit that races this request is held in-memory. Crucially, a
        # failed insert releases its reservation so it cannot turn a transient
        # error into a permanent audio gap.
        try:
            for channel_index, sensor_id in enumerate(runtime.sensor_ids):
                await self._buffer.append(
                    sensor_id=sensor_id,
                    sample_rate_hz=frame.sample_rate_hz,
                    start_time_ns=buffer_start_time_ns,
                    samples=processed[channel_index],
                    start_sample_index=frame.start_sample_index,
                    end_sample_index=frame.end_sample_index,
                    end_time_ns=buffer_end_time_ns,
                )
                if self._persist_observations_on_ingest:
                    await self._registry.record_observation(
                        sensor_id=sensor_id,
                        observation_id=observation_ids[channel_index],
                    )
        except BaseException:
            await self._live_ingest_state.release_reserved_frame(frame_identity)
            raise
        await self._live_ingest_state.commit_reserved_frame(frame_identity)

        # -- trigger evaluation ------------------------------------------------
        # `trigger_rms` is an O(N) numpy reduction over the processed frame.
        # Concurrent-ingest audit measured this at ~1 ms on a 32-channel frame —
        # not catastrophic, but worth keeping off the event loop alongside the
        # other to_thread'd hot-path numpy work.
        frame_energy = await asyncio.to_thread(trigger_rms, processed)
        await self._publish_audio_summary_if_due(
            node_id=normalized_node.id,
            sensor_ids=list(runtime.sensor_ids),
            now_ns=time.time_ns(),
            frame_energy=frame_energy,
            frame_sequence=frame.sequence,
            source_type=frame.source_type,
            time_quality=frame.time_quality.value,
            buffer_uses_receipt_time=buffer_uses_receipt_time,
            timing_diagnostics=frame.timing_diagnostics,
            sequence_gap_count=sequence_gap_count,
        )
        half_window_ns = int(self._localization_config.localization_window_seconds * 0.5 * 1_000_000_000)
        # Anchor event_time_ns to the buffer's own sample-count timeline rather
        # than the firmware GPS clock.  The GPS clock drifts from the ADC crystal
        # oscillator (~40 ppm); after ~4 hours of uptime frame.start_time_ns
        # advances ~1.2 s ahead of buffer.end_time_ns(), causing every trigger
        # to time out with buffer_lag_timeout.  Using the buffer's sample-count
        # end time keeps event_time_ns consistent with what is actually stored
        # while still using the GPS-originated timeline anchor — this is NOT
        # server receipt time and does not break TDOA alignment.
        first_sensor_id = runtime.sensor_ids[0] if runtime.sensor_ids else None
        buf_end_ns: int | None = (
            await self._buffer.sensor_end_time_ns(first_sensor_id)
            if first_sensor_id is not None
            else None
        )
        if buf_end_ns is not None:
            center_time_ns = buf_end_ns - half_window_ns
        else:
            frame_duration_ns = int((processed.shape[1] / frame.sample_rate_hz) * 1_000_000_000)
            center_time_ns = frame.start_time_ns + max(0, frame_duration_ns - half_window_ns)
        cooldown_ns = int(self._localization_config.trigger_cooldown_seconds * 1_000_000_000)

        triggered = False
        if (
            not buffer_uses_receipt_time
            and frame_energy >= self._localization_config.trigger_rms
            and center_time_ns - self._last_trigger_ns >= cooldown_ns
        ):
            # Don't update _last_trigger_ns here — the caller decides whether
            # the trigger was actually accepted (enqueue succeeded). Call
            # ``confirm_trigger()`` to commit the cooldown timestamp.
            triggered = True

        env_counts = EnvironmentCounts(
            ingested=1 if environment_sample is not None else 0,
            persisted=1 if persist_environment_sample else 0,
        )

        self._accepted_frame_count += 1
        if self._accepted_frame_count % _INGEST_LOG_INTERVAL_FRAMES == 0:
            _logger.info(
                "Ingest summary: %d frames accepted; last node=%s energy=%.4f triggered=%s",
                self._accepted_frame_count,
                normalized_node.id,
                frame_energy,
                triggered,
            )

        return IngestResult(
            response=IngestFrameResponse(
                accepted=True,
                duplicate=False,
                triggered=triggered,
                frame_energy=frame_energy,
                detection_id=None,
                queued_event_id=None,
                queue_depth=0,
            ),
            triggered=triggered,
            event_time_ns=center_time_ns if triggered else None,
            sample_rate_hz=frame.sample_rate_hz if triggered else None,
            source_type=frame.source_type if triggered else None,
            time_quality=frame.time_quality if triggered else None,
            observation_ids=observation_ids,
            environment_counts=env_counts,
            sequence_gap_count=sequence_gap_count,
        )

    async def _publish_audio_summary_if_due(
        self,
        *,
        node_id: str,
        sensor_ids: list[str],
        now_ns: int,
        frame_energy: float,
        frame_sequence: int | None,
        source_type: str,
        time_quality: str,
        buffer_uses_receipt_time: bool,
        timing_diagnostics: dict | None = None,
        sequence_gap_count: int = 0,
    ) -> None:
        last_publish_ns = self._last_audio_summary_publish_ns_by_node.get(node_id, 0)
        if now_ns - last_publish_ns < _AUDIO_SUMMARY_PUBLISH_INTERVAL_NS:
            return
        self._last_audio_summary_publish_ns_by_node[node_id] = now_ns

        audio_summary = await self._buffer.summarize_sensors(sensor_ids=sensor_ids, now_ns=now_ns)
        summary = {
            "sensor_count": len(sensor_ids),
            "active_sensor_count": int(audio_summary["active_sensor_count"] or 0),
            "sample_rate_hz": audio_summary["sample_rate_hz"],
            "last_sample_time_ns": audio_summary["last_sample_time_ns"],
            "age_seconds": audio_summary["age_seconds"],
            "rms": audio_summary["rms"],
            "recent_coverage_ratio": audio_summary["recent_coverage_ratio"],
            "recent_missing_ratio": audio_summary["recent_missing_ratio"],
            "recent_max_gap_seconds": audio_summary["recent_max_gap_seconds"],
            "max_buffer_samples": audio_summary["max_buffer_samples"],
            "max_buffer_seconds": audio_summary["max_buffer_seconds"],
            "frame_energy": frame_energy,
            "frame_sequence": frame_sequence,
            "source_type": source_type,
            "time_quality": time_quality,
            "buffer_uses_receipt_time": buffer_uses_receipt_time,
            "status": "live_ingest_process",
        }
        runner_stats = runner_stats_from_timing_diagnostics(timing_diagnostics)
        if runner_stats is not None:
            summary["runner_stats"] = runner_stats
        summary["ingest_health"] = self._ingest_health.observe(
            node_id=node_id,
            sequence_gap_count=sequence_gap_count,
            timing_diagnostics=timing_diagnostics,
        )
        await self._live_ingest_state.record_audio_summary(node_id=node_id, summary=summary)

    @staticmethod
    def _kalman_1d(x: float, p: float, z: float, q: float, r: float) -> tuple[float, float]:
        p += q
        k = p / (p + r)
        return x + k * (z - x), p * (1.0 - k)

    def _is_gps_outlier(
        self, state: "_NodePositionKalman", raw_local: tuple[float, float, float]
    ) -> bool:
        gate = self._gps_gate_m
        if gate <= 0.0:
            return False
        return (
            abs(raw_local[0] - state.x) > gate
            or abs(raw_local[1] - state.y) > gate
            or abs(raw_local[2] - state.z) > gate
        )

    def _normalize_node_spec(self, spec: NodeSpec, *, position_filter: str | None = None) -> tuple[NodeSpec, GeoPoint]:
        unrecognized_capabilities = spec.metadata.get("unrecognized_capabilities")
        if (
            spec.id not in self._unknown_capability_warning_node_ids
            and isinstance(unrecognized_capabilities, list)
            and unrecognized_capabilities
        ):
            self._unknown_capability_warning_node_ids.add(spec.id)
            _logger.warning(
                "node %s reported unsupported capabilities that were ignored: %s",
                spec.id,
                ", ".join(str(capability) for capability in unrecognized_capabilities),
            )

        gps_metadata = spec.metadata.get("gps") if isinstance(spec.metadata, dict) else None
        gps_position_source = (
            gps_metadata.get("position_source")
            if isinstance(gps_metadata, dict)
            else spec.metadata.get("position_source") if isinstance(spec.metadata, dict) else None
        )
        has_trusted_gps_position = (
            isinstance(gps_position_source, str)
            and gps_position_source.startswith("gps")
            and gps_position_source != "gps_fallback"
        )
        # A 2D fix solves latitude and longitude only — its altitude is not a
        # measurement. Two co-sited nodes were reporting altitudes 25 m apart
        # because a fix_2d node's altitude was accepted and smoothed like any
        # other axis, becoming the vertical baseline for every track it produced.
        gps_signal = gps_metadata.get("signal") if isinstance(gps_metadata, dict) else None
        has_trusted_gps_altitude = isinstance(gps_signal, str) and gps_signal.strip().lower() == "fix_3d"

        if spec.position_geo is not None:
            raw_local = self._coordinate_frame.geo_to_local(spec.position_geo)
            if has_trusted_gps_position and not has_trusted_gps_altitude:
                # Substitute the last altitude that came from a real 3D fix before
                # any filtering, so the horizontal axes still track normally and the
                # vertical filter state is never advanced by an unmeasured value.
                #
                # Only a genuine 3D fix qualifies. A node that has never had one has
                # no better altitude available, so its readings are left to the normal
                # filters — freezing z at the first sample would discard the averaging
                # that at least suppresses noise. The warning below is the signal that
                # such a node's altitude should not be trusted.
                retained_altitude_m = self._last_trusted_altitude_m.get(spec.id)
                if retained_altitude_m is not None:
                    raw_local = (raw_local[0], raw_local[1], retained_altitude_m)
                if spec.id not in self._untrusted_altitude_warned_node_ids:
                    self._untrusted_altitude_warned_node_ids.add(spec.id)
                    _logger.warning(
                        "node %s reports GPS signal %r; altitude is not solved by a 2D fix "
                        "and will be held rather than tracked",
                        spec.id,
                        gps_signal,
                    )
            elif has_trusted_gps_position and has_trusted_gps_altitude:
                self._last_trusted_altitude_m[spec.id] = float(raw_local[2])
                self._untrusted_altitude_warned_node_ids.discard(spec.id)
            if has_trusted_gps_position:
                effective_filter = position_filter or ("kalman" if spec.mobility == "mobile" else "kde")
                previous_filter = self._position_filter_by_node.get(spec.id)
                if effective_filter == "kde" and previous_filter not in (None, "kde"):
                    # Returning a relocated node to fixed operation must not blend
                    # its former installation into the new geometry.
                    self._position_kde.pop(spec.id, None)
                    deleter = getattr(self._storage, "delete_node_position_estimator_state", None)
                    if callable(deleter):
                        asyncio.create_task(deleter(spec.id))
                self._position_filter_by_node[spec.id] = effective_filter
                if effective_filter == "raw":
                    local_pos = raw_local
                elif effective_filter == "kde":
                    local_pos = self._update_stationary_kde(spec.id, raw_local)
                else:
                    # Kalman remains selectable for either mobility class.
                    is_stationary = spec.mobility == "stationary"
                    q = self._kalman_q_stationary if is_stationary else self._kalman_q
                    state = self._position_kalman.get(spec.id)
                    if state is None or not state.initialized:
                        state = _NodePositionKalman(x=raw_local[0], y=raw_local[1], z=raw_local[2], px=self._kalman_init_p, py=self._kalman_init_p, pz=self._kalman_init_p, initialized=True)
                    elif not (is_stationary and self._is_gps_outlier(state, raw_local)):
                        nx, px = self._kalman_1d(state.x, state.px, raw_local[0], q, self._kalman_r)
                        ny, py = self._kalman_1d(state.y, state.py, raw_local[1], q, self._kalman_r)
                        nz, pz = self._kalman_1d(state.z, state.pz, raw_local[2], q, self._kalman_r)
                        state = _NodePositionKalman(x=nx, y=ny, z=nz, px=px, py=py, pz=pz, initialized=True)
                    self._position_kalman[spec.id] = state
                    local_pos = (state.x, state.y, state.z)
            else:
                # No live GPS fix: retain the most recent estimator output.
                state = self._position_kalman.get(spec.id)
                kde_state = self._position_kde.get(spec.id)
                local_pos = (
                    kde_state.estimate if kde_state and kde_state.estimate is not None
                    else (state.x, state.y, state.z) if state and state.initialized else raw_local
                )
            geo = self._coordinate_frame.local_to_geo(local_pos)
            normalized = spec.model_copy(update={"position_m": local_pos, "position_geo": geo})
            return normalized, geo

        if spec.position_m is not None:
            local_pos = (float(spec.position_m[0]), float(spec.position_m[1]), float(spec.position_m[2]))
            geo = self._coordinate_frame.local_to_geo(local_pos)
            normalized = spec.model_copy(update={"position_m": local_pos, "position_geo": geo})
            return normalized, geo

        raise ValueError("NodeSpec must include position_geo")

    def _update_stationary_kde(self, node_id: str, raw_local: tuple[float, float, float]) -> tuple[float, float, float]:
        state = self._position_kde.setdefault(node_id, StationaryKdeState())
        if state.estimate is not None and self._kde_acceptance_radius_m > 0:
            distance = float(np.linalg.norm(np.asarray(raw_local[:2]) - np.asarray(state.estimate[:2])))
            if distance > self._kde_acceptance_radius_m:
                return state.estimate
        state.add(raw_local, self._kde_reservoir_capacity)
        now = time.monotonic()
        due = (state.seen_count >= self._kde_warmup_fixes and (
            state.estimate is None or state.seen_count - state.last_evaluated_seen_count >= 128
            or now - state.last_evaluated_monotonic_s >= self._kde_recompute_seconds
        ))
        if due and node_id not in self._kde_evaluation_tasks:
            samples = list(state.samples)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # Pure-logic callers (not the live audio path) have no event
                # loop; compute deterministically so unit tests remain useful.
                state.estimate, state.horizontal_std_m = compute_stationary_kde(samples, self._kde_bandwidth_m)
                state.last_evaluated_seen_count, state.last_evaluated_monotonic_s = state.seen_count, now
            else:
                self._kde_evaluation_tasks[node_id] = asyncio.create_task(
                    self._evaluate_kde_in_background(node_id, samples), name=f"node-kde-{node_id}"
                )
        self._maybe_checkpoint_kde(node_id, state, now)
        return state.estimate or raw_local

    async def _evaluate_kde_in_background(self, node_id: str, samples: list[tuple[float, float, float]]) -> None:
        try:
            estimate, std = await asyncio.to_thread(compute_stationary_kde, samples, self._kde_bandwidth_m)
            state = self._position_kde.get(node_id)
            if state is not None:
                state.estimate, state.horizontal_std_m = estimate, std
                state.last_evaluated_seen_count, state.last_evaluated_monotonic_s = state.seen_count, time.monotonic()
                await self._registry.set_node_position_std_m(node_id, std)
                self._maybe_checkpoint_kde(node_id, state, time.monotonic(), force=True)
        finally:
            self._kde_evaluation_tasks.pop(node_id, None)

    def _maybe_checkpoint_kde(self, node_id: str, state: StationaryKdeState, now: float, *, force: bool = False) -> None:
        if not force and now - state.last_checkpoint_monotonic_s < self._kde_checkpoint_seconds:
            return
        writer = getattr(self._storage, "upsert_node_position_estimator_state", None)
        if callable(writer):
            state.last_checkpoint_monotonic_s = now
            asyncio.create_task(writer(node_id, state.snapshot()))


# (Supporting dataclasses are defined at the top of the module.)


# ---------------------------------------------------------------------------
# Environment sample extraction (pure logic, no I/O)
# ---------------------------------------------------------------------------


def _decode_audio_frame(
    *,
    samples_b64: str,
    channels: int,
    decoded_audio: np.ndarray | None,
) -> np.ndarray:
    return (
        decode_pcm16le_b64(samples_b64, channels)
        if decoded_audio is None
        else np.asarray(decoded_audio, dtype=np.float32)
    )


def _preprocess_audio_frame(
    audio: np.ndarray,
    sample_rate_hz: int,
    node_id: str,
    preprocessor: AudioPreprocessor,
) -> tuple[np.ndarray, dict[str, float | int]]:
    processed = np.zeros_like(audio, dtype=np.float32)
    for channel_idx in range(audio.shape[0]):
        processed[channel_idx] = preprocessor.process(
            audio[channel_idx],
            sample_rate_hz,
            node_id=node_id,
            channel_idx=channel_idx,
        )
    input_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))
    output_rms = float(np.sqrt(np.mean(np.square(processed), dtype=np.float64) + 1e-12))
    input_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    output_peak = float(np.max(np.abs(processed))) if processed.size else 0.0
    return processed, {
        "input_rms": input_rms,
        "output_rms": output_rms,
        "input_peak": input_peak,
        "output_peak": output_peak,
        "observed_level_delta_db": 20.0 * float(np.log10(output_rms / input_rms)) if input_rms > 1e-12 else 0.0,
        "clipping_risk_sample_count": int(np.count_nonzero(np.abs(processed) > 1.0)),
    }


def _extract_environment_sample(
    *,
    request: IngestFrameRequest,
    node: NodeSpec,
    toa_ns: int,
    tor_ns: int,
) -> dict[str, Any] | None:
    """Extract environment data from an ingest request.

    Tries the explicit ``request.environment`` field first, then falls back to
    legacy environment data embedded in node metadata.
    """
    environment: EnvironmentSampleIn | None = request.environment
    timestamp_ns = toa_ns
    temperature_c: float | None = None
    humidity_fraction: float | None = None
    pressure_pa: float | None = None
    wind_speed_mps: float | None = None
    wind_dir_deg: float | None = None
    solar_lux: float | None = None
    metadata: dict[str, Any] = {}

    if environment is not None and environment.has_any_measurement():
        timestamp_ns = environment.timestamp_ns or toa_ns
        temperature_c = environment.temperature_c
        humidity_fraction = environment.humidity_fraction
        pressure_pa = environment.pressure_pa
        wind_speed_mps = environment.wind_speed_mps
        wind_dir_deg = environment.wind_dir_deg
        solar_lux = environment.solar_lux
        if environment.source:
            metadata["source"] = environment.source
        if environment.metadata:
            metadata.update(environment.metadata)
    else:
        from_metadata = _extract_environment_from_node_metadata(node.metadata)
        if from_metadata is None:
            return None
        temperature_c = from_metadata.get("temperature_c")
        humidity_fraction = from_metadata.get("humidity_fraction")
        pressure_pa = from_metadata.get("pressure_pa")
        wind_speed_mps = from_metadata.get("wind_speed_mps")
        wind_dir_deg = from_metadata.get("wind_dir_deg")
        solar_lux = from_metadata.get("solar_lux")
        metadata = from_metadata.get("metadata", {})

    if all(
        value is None
        for value in (
            temperature_c,
            humidity_fraction,
            pressure_pa,
            wind_speed_mps,
            wind_dir_deg,
            solar_lux,
        )
    ):
        return None

    if humidity_fraction is not None:
        humidity_fraction = max(0.0, min(1.0, float(humidity_fraction)))
    metadata.setdefault("tor_ns", tor_ns)

    return {
        "node_id": node.id,
        "timestamp_ns": timestamp_ns,
        "temperature_c": temperature_c,
        "humidity_fraction": humidity_fraction,
        "pressure_pa": pressure_pa,
        "wind_speed_mps": wind_speed_mps,
        "wind_dir_deg": wind_dir_deg,
        "solar_lux": solar_lux,
        "metadata": metadata,
        "location_m": node.position_m,
    }


def _extract_environment_from_node_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Extract environment fields from legacy node metadata dicts."""
    root = metadata if isinstance(metadata, dict) else {}
    nested = root.get("environment")
    env = nested if isinstance(nested, dict) else {}

    def _value(keys: tuple[str, ...]) -> float | None:
        for key in keys:
            candidate = _coerce_float(env.get(key))
            if candidate is not None:
                return candidate
        for key in keys:
            candidate = _coerce_float(root.get(key))
            if candidate is not None:
                return candidate
        return None

    temperature_c = _value(("temperature_c", "temp_c"))
    humidity_fraction = _value(("humidity_fraction",))
    if humidity_fraction is None:
        humidity_raw = _value(("humidity", "humidity_percent"))
        if humidity_raw is not None:
            humidity_fraction = humidity_raw / 100.0 if humidity_raw > 1.0 else humidity_raw

    pressure_pa = _value(("pressure_pa",))
    wind_speed_mps = _value(("wind_speed_mps",))
    wind_dir_deg = _value(("wind_dir_deg",))
    solar_lux = _value(("solar_lux",))

    if all(
        value is None
        for value in (
            temperature_c,
            humidity_fraction,
            pressure_pa,
            wind_speed_mps,
            wind_dir_deg,
            solar_lux,
        )
    ):
        return None

    source = (
        env.get("temperature_source")
        or root.get("temperature_source")
        or env.get("source")
        or root.get("source")
    )
    out_meta: dict[str, Any] = {"ingest": "node_metadata"}
    if isinstance(source, str) and source.strip():
        out_meta["source"] = source.strip()
    return {
        "temperature_c": temperature_c,
        "humidity_fraction": humidity_fraction,
        "pressure_pa": pressure_pa,
        "wind_speed_mps": wind_speed_mps,
        "wind_dir_deg": wind_dir_deg,
        "solar_lux": solar_lux,
        "metadata": out_meta,
    }


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _buffer_timestamps_for_frame(
    *,
    frame_start_time_ns: int,
    frame_end_time_ns: int | None,
    sample_count: int,
    sample_rate_hz: int,
    time_quality: TimeQuality,
    server_received_ns: int,
    allow_receipt_time_fallback: bool = False,
) -> tuple[int, int | None, bool]:
    # Receipt-time-fallback policy — intentional asymmetry with the Rust sidecar.
    #
    # Python (this function): when the firmware reports FREE_RUNNING time quality
    # *and* declares gps_optional capability *and* the node clock is skewed beyond
    # _MAX_TRUSTED_NODE_CLOCK_SKEW_NS from server wallclock, the Python ingest path
    # *overrides* the firmware timestamp with a server-receipt-derived timestamp so
    # debug audio remains roughly playable while the firmware lacks GPS/NTP lock.
    # Such frames do not generate localization triggers, and localization
    # excludes their cross-node TDOA pairs. Within-node bearings and
    # gain-corrected amplitude ratios remain safe to use.
    #
    # Rust (dsp_worker.rs `should_use_receipt_time_alignment`): never overrides
    # firmware timestamps when they are present, no matter how skewed. TDOA
    # localization correctness depends on packet-time alignment across nodes — a
    # well-defined wrong timestamp degrades gracefully under TDOA, while a
    # silently-corrected one looks correct but corrupts the spatial solution.
    #
    # The asymmetry is *by design*: Python may use receipt alignment only for
    # non-triggering debug audio, while the Rust sidecar is the real-time hot
    # path that always preserves packet time. Do not "fix" by aligning the two — ensure
    # tests assert the intentional divergence on the FREE_RUNNING + skew case.
    duration_ns = int(round((sample_count / sample_rate_hz) * 1_000_000_000))
    node_clock_skew_ns = abs(frame_start_time_ns - server_received_ns)
    if (
        allow_receipt_time_fallback
        and time_quality == TimeQuality.FREE_RUNNING
        and node_clock_skew_ns > _MAX_TRUSTED_NODE_CLOCK_SKEW_NS
    ):
        start_ns = max(1, server_received_ns - duration_ns)
        return start_ns, server_received_ns, True
    return frame_start_time_ns, frame_end_time_ns, False


def _node_boot_session(node: NodeSpec) -> str:
    boot_count = node.metadata.get("boot_count")
    if isinstance(boot_count, bool):
        return ""
    if isinstance(boot_count, int):
        return f"boot-{boot_count}"
    if isinstance(boot_count, str) and boot_count.strip():
        return boot_count.strip()
    return ""
