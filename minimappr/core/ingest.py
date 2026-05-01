"""Ingest processing — PCM decode, preprocessing, buffer insertion, observation
persistence, environment extraction, and trigger evaluation.

Extracted from FusionNode to isolate the ingest-to-trigger path as an
independently testable component.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)

# Log a summary after this many accepted (non-duplicate) frames.
_INGEST_LOG_INTERVAL_FRAMES = 5000
# Firmware clock labels can be stale during GPS/NTP acquisition or while
# recovering from a rejected discipline source. Use receipt-aligned buffering
# for large skews so debug audio remains usable; TDOA triggering stays disabled
# on receipt-time fallback.
_MAX_TRUSTED_NODE_CLOCK_SKEW_NS = 300_000_000_000

import numpy as np

from minimappr.config import FusionConfig, LocalizationConfig
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.preprocessing import NodePreprocessorFactory
from minimappr.interfaces import AudioPreprocessor, StorageBackend
from minimappr.models import (
    EnvironmentSampleIn,
    GeoPoint,
    IngestFrameRequest,
    IngestFrameResponse,
    NodeSpec,
    TimeQuality,
)
from minimappr.utils.audio import decode_pcm16le_b64, trigger_rms


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EnvironmentCounts:
    ingested: int = 0
    persisted: int = 0


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
        environment_updater: EnvironmentUpdater | None = None,
        persist_observations_on_ingest: bool = True,
    ) -> None:
        self._localization_config = localization_config
        self._fusion_config = fusion_config
        self._registry = registry
        self._buffer = buffer
        self._storage = storage
        self._coordinate_frame = coordinate_frame
        self._preprocessor_factory = preprocessor_factory
        self._environment_updater = environment_updater
        self._persist_observations_on_ingest = persist_observations_on_ingest

        self._last_trigger_ns = 0
        self._accepted_frame_count = 0

    @property
    def last_trigger_ns(self) -> int:
        return self._last_trigger_ns

    @property
    def accepted_frame_count(self) -> int:
        return self._accepted_frame_count

    def replace_coordinate_frame(self, coordinate_frame: LocalCoordinateFrame) -> None:
        self._coordinate_frame = coordinate_frame

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
        normalized_node, geo_position = self._normalize_node_spec(raw_node)
        server_received_ns = time.time_ns()
        runtime = await self._registry.upsert(normalized_node, server_received_ns)
        boot_session = _node_boot_session(normalized_node)
        toa_ns = frame.toa_ns or frame.start_time_ns
        tor_ns = frame.tor_ns if frame.tor_ns is not None else server_received_ns

        # -- duplicate check --------------------------------------------------
        duplicate_ingest = await self._storage.has_ingested_frame(
            node_id=normalized_node.id,
            boot_session=boot_session,
            frame_sequence=frame.sequence,
            start_time_ns=frame.start_time_ns,
            utc_end_ns=frame.utc_end_ns,
            start_sample_index=frame.start_sample_index,
            end_sample_index=frame.end_sample_index,
            source_type=frame.source_type,
            time_quality=frame.time_quality.value,
            tor_ns=tor_ns,
        )
        if duplicate_ingest:
            await self._storage.upsert_node(
                spec=normalized_node,
                last_seen_ns=server_received_ns,
                position_geo=geo_position,
            )
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

        # -- decode PCM --------------------------------------------------------
        audio = (
            decode_pcm16le_b64(frame.samples_b64, frame.channels)
            if decoded_audio is None
            else np.asarray(decoded_audio, dtype=np.float32)
        )
        if audio.ndim != 2:
            raise ValueError("Decoded audio must be channels-first")
        if audio.shape[0] != frame.channels:
            raise ValueError("Decoded channel count does not match frame.channels")
        if frame.samples_per_channel is not None and audio.shape[1] != frame.samples_per_channel:
            raise ValueError("Decoded sample count does not match frame.samples_per_channel")

        # -- preprocess --------------------------------------------------------
        preprocessor: AudioPreprocessor = self._preprocessor_factory.for_node(normalized_node)
        processed = np.zeros_like(audio, dtype=np.float32)
        for channel_idx in range(audio.shape[0]):
            processed[channel_idx] = preprocessor.process(
                audio[channel_idx],
                frame.sample_rate_hz,
                node_id=normalized_node.id,
            )

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

        # -- register frame and optionally persist observations ---------------
        observation_ids: list[str] = []
        frame_registered = False

        async with storage_batch_ctx():
            await self._storage.upsert_node(
                spec=normalized_node,
                last_seen_ns=server_received_ns,
                position_geo=geo_position,
            )
            frame_registered = await self._storage.register_ingested_frame(
                node_id=normalized_node.id,
                boot_session=boot_session,
                frame_sequence=frame.sequence,
                start_time_ns=frame.start_time_ns,
                utc_end_ns=frame.utc_end_ns,
                start_sample_index=frame.start_sample_index,
                end_sample_index=frame.end_sample_index,
                toa_ns=toa_ns,
                tor_ns=tor_ns,
                created_ns=server_received_ns,
                source_type=frame.source_type,
                time_quality=frame.time_quality.value,
            )
            if frame_registered:
                if environment_sample is not None:
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

        if not frame_registered:
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

        # -- buffer insertion --------------------------------------------------
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

        # -- trigger evaluation ------------------------------------------------
        frame_energy = trigger_rms(processed)
        frame_duration_ns = int((processed.shape[1] / frame.sample_rate_hz) * 1_000_000_000)
        half_window_ns = int(self._localization_config.localization_window_seconds * 0.5 * 1_000_000_000)
        center_offset_ns = max(0, frame_duration_ns - half_window_ns)
        center_time_ns = frame.start_time_ns + center_offset_ns
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
            persisted=1 if environment_sample is not None and frame_registered else 0,
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
        )

    def _normalize_node_spec(self, spec: NodeSpec) -> tuple[NodeSpec, GeoPoint]:
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
        if spec.position_geo is not None and (spec.position_m is None or has_trusted_gps_position):
            local_pos = self._coordinate_frame.geo_to_local(spec.position_geo)
            normalized = spec.model_copy(update={"position_m": local_pos})
            return normalized, spec.position_geo

        if spec.position_m is not None:
            local_pos = spec.position_m
            geo = spec.position_geo or self._coordinate_frame.local_to_geo(local_pos)
            normalized = spec.model_copy(update={"position_m": local_pos, "position_geo": geo})
            return normalized, geo

        if spec.position_geo is None:
            raise ValueError("NodeSpec must include position_m or position_geo")

        local_pos = self._coordinate_frame.geo_to_local(spec.position_geo)
        normalized = spec.model_copy(update={"position_m": local_pos})
        return normalized, spec.position_geo


# (Supporting dataclasses are defined at the top of the module.)


# ---------------------------------------------------------------------------
# Environment sample extraction (pure logic, no I/O)
# ---------------------------------------------------------------------------


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
    duration_ns = int(round((sample_count / sample_rate_hz) * 1_000_000_000))
    node_clock_skew_ns = abs(frame_start_time_ns - server_received_ns)
    if allow_receipt_time_fallback and node_clock_skew_ns > _MAX_TRUSTED_NODE_CLOCK_SKEW_NS:
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
